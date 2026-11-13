"""Fresh, separately scoped credentials for persistent jobs; never development mocks.

Uses existing SQL users/configured API keys and python-jose. Scoped tokens require
explicit issuance through /auth/token with network-jobs-v1 or batch-jobs-v1. Legacy unscoped
or role-only bearer tokens cannot authorize persistent job access.
"""
from dataclasses import dataclass
import hashlib
import math
import os
import secrets
import time
from fastapi import HTTPException,Request,Security,Depends
from fastapi.security import APIKeyHeader,HTTPBearer
from sqlalchemy.exc import SQLAlchemyError

AUDIENCE='network-jobs-v1'
ISSUER='aneos-network-v1'
BATCH_AUDIENCE='batch-jobs-v1'
ISSUERS={AUDIENCE:ISSUER,BATCH_AUDIENCE:'aneos-batch-v1'}
key_header=APIKeyHeader(name='X-API-Key',auto_error=False)
bearer_header=HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    subject:str
    role:str
    credential_id:str


def credential_id(key):
    return hashlib.sha256(('aneos-network-token-key-v1:'+key).encode()).hexdigest()


def _valid_key(key):
    return isinstance(key,str) and 32<=len(key)<=512 and not key.endswith('-not-configured') and not any(ord(c)<33 for c in key)


def _principals(sessions):
    from .database import User
    rows=[]
    for role in ('admin','analyst','viewer'):
        key=os.getenv('ANEOS_'+role.upper()+'_API_KEY','')
        if _valid_key(key):rows.append(('configured-key:'+role,role,[key]))
    try:
        with sessions() as session:
            users=session.query(User).filter(User.is_active==True).all()
            for u in users:
                if (not isinstance(u.user_id,str) or not u.user_id or len(u.user_id)>100 or
                    not isinstance(u.api_keys,list)):
                    continue
                rows.append(('db-user:'+u.user_id,u.role,[key for key in u.api_keys if _valid_key(key)]))
    except SQLAlchemyError as exc:
        raise HTTPException(503,'Credential store unavailable') from exc
    return rows


def authenticate_key(sessions,key):
    if not _valid_key(key):raise HTTPException(401,'Valid configured credentials required')
    matches=[]
    for subject,role,keys in _principals(sessions):
        for candidate in keys:
            if secrets.compare_digest(candidate.encode(),key.encode()):
                matches.append(Principal(subject,role,credential_id(candidate)))
    unique=set(matches)
    if len(unique)>1:raise HTTPException(503,'Ambiguous credential configuration')
    if not unique:raise HTTPException(401,'Invalid or inactive credentials')
    return next(iter(unique))


def issue_token(sessions,key,*,audience=AUDIENCE):
    from . import auth
    if audience not in ISSUERS:raise HTTPException(422,'Unsupported token audience')
    principal=authenticate_key(sessions,key)
    if not auth._HAS_JOSE or not _valid_key(auth._JWT_SECRET):
        raise HTTPException(503,'Scoped token signing unavailable')
    now=int(time.time())
    return auth._jose_jwt.encode({'sub':principal.subject,'credential_id':principal.credential_id,
        'iat':now,'exp':now+86400,'aud':audience,'iss':ISSUERS[audience]},auth._JWT_SECRET,algorithm='HS256')


def authenticate_token(sessions,token,*,audience=AUDIENCE):
    from . import auth
    if audience not in ISSUERS:raise HTTPException(401,'Unsupported token audience')
    if not auth._HAS_JOSE or not _valid_key(auth._JWT_SECRET):raise HTTPException(401,'Scoped bearer authentication unavailable')
    try:
        payload=auth._jose_jwt.decode(token,auth._JWT_SECRET,algorithms=['HS256'],audience=audience,issuer=ISSUERS[audience],
            options={'require_exp':True,'require_sub':True,'require_iat':True,'require_aud':True,'require_iss':True})
        issued=payload['iat']
        if type(issued) not in (int,float) or not math.isfinite(issued) or issued>time.time():raise ValueError('Invalid issued time')
        digest=payload.get('credential_id')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('No current credential binding')
    except (auth._JWTError,ValueError,TypeError,KeyError) as exc:
        raise HTTPException(401,'Invalid or expired scoped bearer token') from exc
    matches=set()
    for subject,role,keys in _principals(sessions):
        for key in keys:
            if secrets.compare_digest(credential_id(key),digest):matches.add(Principal(subject,role,digest))
    if len(matches)!=1 or next(iter(matches)).subject!=payload['sub']:
        raise HTTPException(401,'Inactive or ambiguous scoped bearer principal')
    return next(iter(matches))


def get_runtime(request):
    from .network_runtime import NetworkRuntime
    runtime=getattr(request.app.state,'network_runtime',None)
    if not isinstance(runtime,NetworkRuntime):raise HTTPException(503,'Persistent execution runtime not started')
    return runtime


def _request_principal(request,api_key,bearer,*,audience):
    if (len(request.headers.getlist('x-api-key'))>1 or len(request.headers.getlist('authorization'))>1 or
        (request.headers.get('x-api-key') is not None and request.headers.get('authorization') is not None)):
        raise HTTPException(401,'Use one credential mechanism')
    runtime=get_runtime(request)
    if api_key:principal=authenticate_key(runtime.store.sessions,api_key)
    elif bearer:principal=authenticate_token(runtime.store.sessions,bearer.credentials,audience=audience)
    else:raise HTTPException(401,'Authentication required')
    if principal.role not in ('viewer','analyst','admin'):raise HTTPException(403,'Unsupported account role')
    return principal


def require_network_principal(request:Request,api_key=Security(key_header),bearer=Security(bearer_header)):
    return _request_principal(request,api_key,bearer,audience=AUDIENCE)


def require_batch_principal(request:Request,api_key=Security(key_header),bearer=Security(bearer_header)):
    return _request_principal(request,api_key,bearer,audience=BATCH_AUDIENCE)


def require_writer(principal):
    if principal.role not in ('admin','analyst'):raise HTTPException(403,'Analyst or admin role required')


def require_persistent_admin(principal=Depends(require_network_principal)):
    """Protect credential-provisioning/admin authority with the same fresh principal."""
    if principal.role!='admin':raise HTTPException(403,'Verified administrator required')
    return {'user_id':principal.subject,'username':principal.subject,'role':'admin','is_active':True}
