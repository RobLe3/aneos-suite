"""
Authentication endpoints for aNEOS API.

Provides a /auth/token endpoint that exchanges a valid API key for a signed JWT
bearer token.  Requires ANEOS_SECRET_KEY to be set in the environment; see auth.py.
"""

from typing import Optional

try:
    from fastapi import APIRouter, HTTPException, Security, Request, Query
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..auth import API_KEY_MAP, create_access_token

if HAS_FASTAPI:
    router = APIRouter(prefix='/auth', tags=['auth'])
    _api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)

    class TokenResponse(BaseModel):
        access_token: str
        token_type: str = 'bearer'
        expires_in_hours: int = 24

    @router.post('/token', response_model=TokenResponse,
                 summary='Exchange API key for JWT bearer token')
    async def get_token(
        request: Request,
        api_key: Optional[str] = Security(_api_key_header),
        audience: Optional[str] = Query(None),
    ) -> TokenResponse:
        """Exchange a valid X-API-Key for a signed JWT bearer token.

        Set ANEOS_SECRET_KEY in your environment before using this endpoint.
        Generate a key with::

            python -c "import secrets; print(secrets.token_hex(32))"
        """
        if audience is not None:
            from ..network_access import ISSUERS, get_runtime, issue_token
            if audience not in ISSUERS:raise HTTPException(422,'Unsupported token audience')
            if len(request.headers.getlist('x-api-key'))!=1 or request.headers.get('authorization') is not None:
                raise HTTPException(401,'Use one API key for scoped token issuance')
            runtime=get_runtime(request)
            return TokenResponse(access_token=issue_token(runtime.store.sessions,api_key,audience=audience))
        if not api_key:
            raise HTTPException(status_code=401, detail='X-API-Key header required')
        user = API_KEY_MAP.get(api_key)
        if not user or not user.get('is_active'):
            raise HTTPException(status_code=401, detail='Invalid or inactive API key')
        try:
            token = create_access_token(
                user_id=str(user['user_id']),
                role=str(user.get('role', 'viewer')),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return TokenResponse(access_token=token)

else:
    # Stub router for environments without FastAPI
    class _StubRouter:
        pass
    router = _StubRouter()
