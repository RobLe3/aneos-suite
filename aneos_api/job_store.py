"""Transactional execution records on aNEOS's existing SQLAlchemy database.

Storage only: no handlers, HTTP routes, scientific verdicts or automatic worker
startup. Owner scopes must come from a trusted caller; this is not authentication.
File SQLite is regression-tested; PostgreSQL requires deployment verification.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
import hashlib
import json
import math
import re
import time
import uuid

from sqlalchemy import Column, Float, Integer, String, Text, UniqueConstraint, select, update, or_, and_, text, inspect
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine

from .database import Base


class DurableQueue(Base):
    __tablename__ = 'durable_job_queue'
    name = Column(String(40), primary_key=True)
    schema_version = Column(Integer, nullable=False)
    capacity = Column(Integer, nullable=False)
    active = Column(Integer, nullable=False)


class DurableJob(Base):
    __tablename__ = 'durable_jobs'
    job_id = Column(String(64), primary_key=True)
    owner_scope = Column(String(64), nullable=False, index=True)
    kind = Column(String(80), nullable=False, index=True)
    idempotency_key = Column(String(128))
    request_json = Column(Text, nullable=False)
    request_sha256 = Column(String(64), nullable=False)
    status = Column(String(24), nullable=False, index=True)
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)
    available_at = Column(Float, nullable=False, index=True)
    attempts = Column(Integer, nullable=False)
    max_attempts = Column(Integer, nullable=False)
    revision = Column(Integer, nullable=False)
    lease_token = Column(String(64))
    lease_until = Column(Float)
    worker_id = Column(String(100))
    result_json = Column(Text)
    result_sha256 = Column(String(64))
    error_code = Column(String(80))
    __table_args__ = (UniqueConstraint('owner_scope','kind','idempotency_key',name='uq_job_idempotency'),)


class JobCheckpoint(Base):
    __tablename__ = 'durable_job_checkpoints'
    job_id = Column(String(64), primary_key=True)
    key = Column(String(160), primary_key=True)
    payload_json = Column(Text, nullable=False)
    payload_sha256 = Column(String(64), nullable=False)
    receipt_sha256 = Column(String(64), nullable=False)
    created_at = Column(Float, nullable=False)
    attempt = Column(Integer, nullable=False)


class JobEvent(Base):
    __tablename__ = 'durable_job_events'
    job_id = Column(String(64), primary_key=True)
    revision = Column(Integer, primary_key=True)
    event = Column(String(32), nullable=False)
    at = Column(Float, nullable=False)
    attempt = Column(Integer, nullable=False)
    details_json = Column(Text, nullable=False)
    details_sha256 = Column(String(64), nullable=False)
    event_sha256 = Column(String(64), nullable=False)


class JobStorageUnavailable(RuntimeError): pass
class QueueFull(RuntimeError): pass
class IdempotencyConflict(ValueError): pass
class ImmutableCheckpoint(ValueError): pass
class LostLease(RuntimeError): pass
class JobIntegrityError(ValueError): pass


@dataclass(frozen=True)
class JobLease:
    job_id: str
    token: str
    attempt: int
    request: dict
    kind: str
    lease_until: float
    max_attempts: int = 3


MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


def _json(value):
    if not isinstance(value, dict):
        raise ValueError('Job payload must be a JSON object')
    def check(item, depth=0):
        if depth > 64:
            raise ValueError('Job JSON nesting exceeds supported depth')
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError('Job JSON keys must be strings')
            for child in item.values():check(child,depth+1)
        elif type(item) is list:
            for child in item:check(child,depth+1)
        elif type(item) not in (str,int,float,bool,type(None)):
            raise ValueError('Only JSON values are supported, not coerced Python types')
    try:
        check(value)
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        encoded = raw.encode()
        if len(encoded)>MAX_PAYLOAD_BYTES:raise ValueError('Job payload exceeds storage limit')
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError('Job payload must be bounded finite JSON with string keys') from exc
    return raw, hashlib.sha256(encoded).hexdigest()


def _read(raw, digest):
    try:
        if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != digest:
            raise ValueError('checksum mismatch')
        result = json.loads(raw)
        if _json(result) != (raw, digest):raise ValueError('not canonical')
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise JobIntegrityError('Stored job payload is invalid or checksum mismatched') from exc


def _identifier(value, label, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(c)<32 for c in value):
        raise ValueError(f'Invalid {label}')
    return value


def _scope(value):
    value = _identifier(value, 'owner scope', 200)
    return hashlib.sha256(('aneos-job-owner-v1:' + value).encode()).hexdigest()


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive')
    return float(value)


class JobStore:
    """Atomic admission, leased execution, immutable checkpoints and result commit.

    No connection/transaction is held during a scientific calculation or provider
    request. Fencing tokens, not worker names, authorize current-attempt writes.
    """
    QUEUE = 'execution-v1'

    def __init__(self, session_factory, *, clock=time.time, capacity=128):
        if type(capacity) is not int or not 1 <= capacity <= 100000:
            raise ValueError('Invalid durable queue capacity')
        if not isinstance(session_factory,sessionmaker):
            raise ValueError('Use an Engine-bound sessionmaker that creates independent sessions')
        self.sessions, self.clock, self.capacity = session_factory, clock, capacity
        with session_factory() as session:
            self.engine = session.get_bind()
        if not isinstance(self.engine,Engine):
            raise ValueError('Durable jobs cannot share a caller-owned connection/transaction')
        if self.engine.dialect.name == 'sqlite':
            if (self.engine.url.database in (None,'',':memory:')
                    or str(self.engine.url.database).startswith('file::memory:')
                    or self.engine.url.query.get('mode')=='memory'
                    or isinstance(self.engine.pool, StaticPool)):
                raise ValueError('Durable jobs require file SQLite with independent connections, not memory/StaticPool')

    @contextmanager
    def _transaction(self, *, allow_integrity=False, read_only=False):
        try:
            with self.sessions() as session:
                with session.begin():
                    if self.engine.dialect.name == 'sqlite':
                        # pysqlite's legacy SELECT/DDL transaction mode is not a
                        # coherent multi-statement snapshot. Serialize these short
                        # job transactions explicitly; no provider work is inside.
                        session.execute(text('BEGIN' if read_only else 'BEGIN IMMEDIATE'))
                    self._check_backend(session)
                    yield session
        except IntegrityError as exc:
            if allow_integrity:raise
            raise JobStorageUnavailable('Durable job transaction integrity failure') from exc
        except SQLAlchemyError as exc:
            raise JobStorageUnavailable('Durable job storage unavailable; no successful transition confirmed') from exc

    def _check_backend(self, session):
        if self.engine.dialect.name == 'sqlite':
            if not self.engine.dialect.server_version_info or self.engine.dialect.server_version_info < (3,35,0):
                raise JobStorageUnavailable('SQLite 3.35 or newer is required for fenced RETURNING transitions')
            synchronous=session.execute(text('PRAGMA synchronous')).scalar_one()
            journal=session.execute(text('PRAGMA journal_mode')).scalar_one()
            if synchronous not in (2,3) or str(journal).lower() not in ('delete','truncate','persist','wal'):
                raise JobStorageUnavailable('SQLite job durability settings must use FULL/EXTRA synchronization and on-disk journaling')

    def _now(self):
        now = self.clock()
        if isinstance(now,bool) or not isinstance(now,(int,float)) or not math.isfinite(now) or now < 0:
            raise ValueError('Invalid job clock')
        return float(now)

    def initialize_schema(self):
        """Explicit non-destructive setup; no worker is started or job reset."""
        tables=[model.__table__ for model in (DurableQueue,DurableJob,JobCheckpoint,JobEvent)]
        try:
            try:
                with self._transaction(allow_integrity=True) as session:
                    Base.metadata.create_all(bind=session.connection(),tables=tables)
                    if session.get(DurableQueue,self.QUEUE) is None:
                        if session.query(DurableJob).count():
                            raise JobIntegrityError('Admission state missing for existing jobs; explicit recovery required')
                        session.add(DurableQueue(name=self.QUEUE,schema_version=1,capacity=self.capacity,active=0))
                        session.flush()
                    self._queue(session)
                    self._verify_schema(session,tables)
            except IntegrityError:
                # A concurrent initializer may have inserted the same budget row.
                with self._transaction() as session:
                    self._queue(session)
                    self._verify_schema(session,tables)
        except SQLAlchemyError as exc:
            raise JobStorageUnavailable('Durable job schema initialization failed') from exc

    def _verify_schema(self, session, tables):
        inspector=inspect(session.connection())
        for table in tables:
            actual={column['name'] for column in inspector.get_columns(table.name)}
            required={column.name for column in table.columns}
            if not required.issubset(actual):
                raise JobIntegrityError('Durable job schema columns are incompatible; explicit migration required')
            expected_pk=[column.name for column in table.primary_key.columns]
            if inspector.get_pk_constraint(table.name).get('constrained_columns')!=expected_pk:
                raise JobIntegrityError('Durable job primary key is incompatible')
        unique={tuple(item['column_names']) for item in inspector.get_unique_constraints(DurableJob.__tablename__)}
        if ('owner_scope','kind','idempotency_key') not in unique:
            raise JobIntegrityError('Durable job idempotency constraint is missing')
        queue=self._queue(session)
        active=session.query(DurableJob).filter(DurableJob.status.in_(['queued','retry_wait','running'])).count()
        if queue.active!=active:
            raise JobIntegrityError('Admission accounting disagrees with stored jobs; explicit recovery required')

    def _queue(self, session):
        queue=session.get(DurableQueue,self.QUEUE)
        if queue is None or queue.schema_version != 1 or queue.capacity != self.capacity or not 0 <= queue.active <= queue.capacity:
            raise JobIntegrityError('Queue not initialized, unsupported schema, inconsistent capacity or invalid admission state')
        return queue

    def _event(self, session, job_id, revision, event, now, attempt, details=None):
        raw,digest=_json(details or {})
        envelope={'job_id':job_id,'revision':revision,'event':event,'at':now,'attempt':attempt,'details_sha256':digest}
        session.add(JobEvent(job_id=job_id,revision=revision,event=event,at=now,attempt=attempt,
                             details_json=raw,details_sha256=digest,event_sha256=_json(envelope)[1]))

    def _event_view(self, row):
        envelope={'job_id':row.job_id,'revision':row.revision,'event':row.event,'at':row.at,
                  'attempt':row.attempt,'details_sha256':row.details_sha256}
        if _json(envelope)[1]!=row.event_sha256:raise JobIntegrityError('Stored event envelope checksum mismatch')
        return {'revision':row.revision,'event':row.event,'at':row.at,'attempt':row.attempt,
                'details':_read(row.details_json,row.details_sha256)}

    def _receipt(self, job_id, key, digest, now, attempt):
        return _json({'job_id':job_id,'key':key,'payload_sha256':digest,'created_at':now,'attempt':attempt})[1]

    def _checkpoint_view(self, row):
        if self._receipt(row.job_id,row.key,row.payload_sha256,row.created_at,row.attempt)!=row.receipt_sha256:
            raise JobIntegrityError('Stored checkpoint binding checksum mismatch')
        return _read(row.payload_json,row.payload_sha256)

    def _view(self, session, row):
        request=_read(row.request_json,row.request_sha256)
        result=_read(row.result_json,row.result_sha256) if row.result_json is not None else None
        if row.status=='completed' and result is None:
            raise JobIntegrityError('Completed job has no committed result')
        if row.status!='completed' and result is not None:
            raise JobIntegrityError('Noncompleted job contains a scientific result')
        count=session.query(JobCheckpoint).filter_by(job_id=row.job_id).count()
        return {'job_id':row.job_id,'kind':row.kind,'request':request,'request_sha256':row.request_sha256,
            'status':row.status,'created_at':row.created_at,'updated_at':row.updated_at,'available_at':row.available_at,
            'attempts':row.attempts,'max_attempts':row.max_attempts,'revision':row.revision,
            'lease_until':row.lease_until,'checkpoint_count':count,'result':result,
            'result_sha256':row.result_sha256,'error_code':row.error_code}

    def submit(self, *, owner_scope, kind, request, idempotency_key=None, max_attempts=3):
        owner=_scope(owner_scope);kind=_identifier(kind,'job kind',80);raw,digest=_json(request)
        if idempotency_key is not None:_identifier(idempotency_key,'idempotency key',128)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:raise ValueError('Invalid attempt limit')
        now=self._now()
        try:
            with self._transaction(allow_integrity=True) as session:
                self._queue(session)
                if idempotency_key is not None:
                    existing=session.query(DurableJob).filter_by(owner_scope=owner,kind=kind,idempotency_key=idempotency_key).first()
                    if existing:
                        if existing.request_sha256!=digest or existing.max_attempts!=max_attempts:
                            raise IdempotencyConflict('Idempotency key is bound to another request/policy')
                        return self._view(session,existing)
                reserved=session.execute(update(DurableQueue).where(DurableQueue.name==self.QUEUE,
                    DurableQueue.active<DurableQueue.capacity).values(active=DurableQueue.active+1)).rowcount
                if reserved!=1:
                    # An identical racing submit may have committed while this
                    # transaction waited for the admission row's write lock.
                    existing=session.query(DurableJob).filter_by(owner_scope=owner,kind=kind,idempotency_key=idempotency_key).first() if idempotency_key else None
                    if existing:
                        if existing.request_sha256!=digest or existing.max_attempts!=max_attempts:
                            raise IdempotencyConflict('Idempotency key is bound to another request/policy')
                        return self._view(session,existing)
                    raise QueueFull('Durable job admission capacity exhausted')
                job_id='job_'+uuid.uuid4().hex
                row=DurableJob(job_id=job_id,owner_scope=owner,kind=kind,idempotency_key=idempotency_key,
                    request_json=raw,request_sha256=digest,status='queued',created_at=now,updated_at=now,
                    available_at=now,attempts=0,max_attempts=max_attempts,revision=1)
                session.add(row);self._event(session,job_id,1,'queued',now,0)
                session.flush()
                return self._view(session,row)
        except IntegrityError as exc:
            # Unique idempotency races roll back admission and the entire insert.
            with self._transaction() as session:
                existing=session.query(DurableJob).filter_by(owner_scope=owner,kind=kind,idempotency_key=idempotency_key).first() if idempotency_key else None
                if existing:
                    if existing.request_sha256!=digest or existing.max_attempts!=max_attempts:
                        raise IdempotencyConflict('Idempotency key is bound to another request/policy') from exc
                    return self._view(session,existing)
            raise JobStorageUnavailable('Job insert was not confirmed') from exc
        except SQLAlchemyError as exc:
            raise JobStorageUnavailable('Job submission not confirmed; retry only with the same idempotency identity') from exc

    def find_submission(self,owner_scope,kind,idempotency_key):
        owner=_scope(owner_scope);kind=_identifier(kind,'kind',80)
        key=_identifier(idempotency_key,'idempotency key',128)
        with self._transaction(read_only=True) as session:
            row=session.query(DurableJob).filter_by(owner_scope=owner,kind=kind,idempotency_key=key).first()
            return self._view(session,row) if row else None

    def get(self, owner_scope, job_id):
        owner=_scope(owner_scope)
        with self._transaction(read_only=True) as session:
            row=session.query(DurableJob).filter_by(job_id=job_id,owner_scope=owner).first()
            return self._view(session,row) if row else None

    def history(self, owner_scope, job_id):
        owner=_scope(owner_scope)
        with self._transaction(read_only=True) as session:
            if not session.query(DurableJob).filter_by(job_id=job_id,owner_scope=owner).first():return None
            return [self._event_view(r) for r in session.query(JobEvent).filter_by(job_id=job_id).order_by(JobEvent.revision)]

    def read_checkpoints(self, owner_scope, job_id):
        """Verified owner-scoped inspection, including completed and partial jobs."""
        owner=_scope(owner_scope)
        with self._transaction(read_only=True) as session:
            row=session.query(DurableJob).filter_by(job_id=job_id,owner_scope=owner).first()
            if row is None:return None
            self._view(session,row)
            return {r.key:self._checkpoint_view(r) for r in session.query(JobCheckpoint).filter_by(job_id=job_id)}

    def _release(self, session):
        changed=session.execute(update(DurableQueue).where(DurableQueue.name==self.QUEUE,
            DurableQueue.active>0,DurableQueue.active<=DurableQueue.capacity,DurableQueue.schema_version==1).values(active=DurableQueue.active-1)).rowcount
        if changed!=1:raise JobIntegrityError('Queue admission accounting is invalid')

    def cancel(self, owner_scope, job_id):
        owner=_scope(owner_scope);now=self._now()
        with self._transaction() as session:
            identity=[DurableJob.job_id==job_id,DurableJob.owner_scope==owner,
                      DurableJob.status.in_(['queued','retry_wait','running'])]
            locked=session.execute(update(DurableJob).where(*identity).values(revision=DurableJob.revision)
                .returning(DurableJob.revision)).first()
            if not locked:return False
            now=self._now()
            returned=session.execute(update(DurableJob).where(*identity,DurableJob.revision==locked.revision)
                .values(status='cancelled',updated_at=now,lease_token=None,lease_until=None,
                        revision=DurableJob.revision+1).returning(DurableJob.revision,DurableJob.attempts)).first()
            if not returned:return False
            self._release(session);self._event(session,job_id,returned.revision,'cancelled',now,returned.attempts)
            return True

    def claim(self, worker_id, *, lease_seconds=30, kinds=None, job_id=None):
        worker_id=_identifier(worker_id,'worker ID',100);duration=_positive(lease_seconds,'lease');now=self._now()
        if kinds is not None:
            if not kinds:raise ValueError('An empty worker kind scope is not a wildcard')
            kinds=[_identifier(kind,'kind',80) for kind in kinds]
        if job_id is not None:_identifier(job_id,'job ID',64)
        eligible=or_(and_(DurableJob.status.in_(['queued','retry_wait']),DurableJob.available_at<=now),
                     and_(DurableJob.status=='running',DurableJob.lease_until<=now))
        for _ in range(20):
            with self._transaction() as session:
                self._queue(session)
                query=select(DurableJob).where(eligible)
                if kinds is not None:query=query.where(DurableJob.kind.in_(kinds))
                if job_id is not None:query=query.where(DurableJob.job_id==job_id)
                row=session.execute(query.order_by(DurableJob.available_at,DurableJob.created_at,DurableJob.job_id).limit(1)).scalar_one_or_none()
                if row is None:return None
                # Lock the selected revision without changing its event revision.
                # A heartbeat/reclaimer that won this race invalidates selection.
                identity=[DurableJob.job_id==row.job_id,DurableJob.revision==row.revision,eligible]
                locked=session.execute(update(DurableJob).where(*identity).values(revision=DurableJob.revision)
                    .returning(DurableJob.revision)).first()
                if not locked:continue
                now=self._now()
                current_eligible=or_(and_(DurableJob.status.in_(['queued','retry_wait']),DurableJob.available_at<=now),
                    and_(DurableJob.status=='running',DurableJob.lease_until<=now))
                criteria=[DurableJob.job_id==row.job_id,DurableJob.revision==locked.revision,current_eligible]
                if row.attempts>=row.max_attempts:
                    rev=session.execute(update(DurableJob).where(*criteria).values(status='failed',error_code='ATTEMPTS_EXHAUSTED',
                        updated_at=now,lease_token=None,lease_until=None,revision=DurableJob.revision+1).returning(DurableJob.revision)).scalar_one_or_none()
                    if rev is not None:
                        self._release(session);self._event(session,row.job_id,rev,'failed',now,row.attempts,{'code':'ATTEMPTS_EXHAUSTED'})
                    continue
                request=_read(row.request_json,row.request_sha256)
                recovering=row.status=='running'
                token=uuid.uuid4().hex
                returned=session.execute(update(DurableJob).where(*criteria).values(status='running',updated_at=now,
                    lease_token=token,lease_until=now+duration,worker_id=worker_id,error_code=None,
                    attempts=DurableJob.attempts+1,revision=DurableJob.revision+1).returning(DurableJob.revision,DurableJob.attempts)).first()
                if returned:
                    self._event(session,row.job_id,returned.revision,'claimed',now,returned.attempts,
                                {'recovered_expired_lease':recovering})
                    return JobLease(row.job_id,token,returned.attempts,request,row.kind,now+duration,row.max_attempts)
        return None  # Contention, not evidence of an empty scientific population.

    def _fenced_update(self, session, lease, now, *, lease_seconds=None, **values):
        if not isinstance(lease,JobLease) or type(lease.max_attempts) is not int:
            raise LostLease('Invalid execution lease')
        request_digest=_json(lease.request)[1]
        identity=[DurableJob.job_id==lease.job_id,DurableJob.status=='running',
            DurableJob.lease_token==lease.token,DurableJob.attempts==lease.attempt,
            DurableJob.request_sha256==request_digest,DurableJob.kind==lease.kind,
            DurableJob.max_attempts==lease.max_attempts]
        # Acquire the row's write lock first. A timestamp sampled before waiting
        # for a database lock must not authorize a now-expired worker to commit.
        locked=session.execute(update(DurableJob).where(*identity)
            .values(revision=DurableJob.revision).returning(DurableJob.lease_until)).first()
        now=self._now()
        if locked is None or locked.lease_until is None or locked.lease_until<=now:
            raise LostLease('Execution lease is stale, expired or cancelled')
        if lease_seconds is not None:values['lease_until']=now+lease_seconds
        if values.get('status')=='failed':values['available_at']=now
        row=session.execute(update(DurableJob).where(*identity,DurableJob.lease_until>now)
            .values(**values,updated_at=now,revision=DurableJob.revision+1)
            .returning(DurableJob.revision,DurableJob.attempts,DurableJob.lease_until)).first()
        if row is None:raise LostLease('Execution lease is stale, expired or cancelled')
        return SimpleNamespace(revision=row.revision,attempts=row.attempts,lease_until=row.lease_until,at=now)

    def renew(self, lease, *, lease_seconds=30):
        duration=_positive(lease_seconds,'lease');now=self._now()
        with self._transaction() as session:
            row=self._fenced_update(session,lease,now,lease_seconds=duration)
            self._event(session,lease.job_id,row.revision,'renewed',row.at,row.attempts)
        return JobLease(lease.job_id,lease.token,lease.attempt,json.loads(_json(lease.request)[0]),lease.kind,row.lease_until)

    def checkpoint(self, lease, key, payload):
        key=_identifier(key,'checkpoint key',160);raw,digest=_json(payload);now=self._now()
        with self._transaction() as session:
            row=self._fenced_update(session,lease,now)
            existing=session.get(JobCheckpoint,(lease.job_id,key))
            if existing:
                self._checkpoint_view(existing)
                if (existing.payload_json,existing.payload_sha256)!=(raw,digest):
                    raise ImmutableCheckpoint('A frozen checkpoint cannot be replaced')
            else:
                session.add(JobCheckpoint(job_id=lease.job_id,key=key,payload_json=raw,payload_sha256=digest,
                    created_at=row.at,attempt=lease.attempt,
                    receipt_sha256=self._receipt(lease.job_id,key,digest,row.at,lease.attempt)))
            self._event(session,lease.job_id,row.revision,'checkpoint',row.at,row.attempts,{'key':key,'sha256':digest})

    def checkpoints(self, lease):
        if not isinstance(lease,JobLease) or type(lease.max_attempts) is not int:
            raise LostLease('Invalid execution lease')
        request_digest=_json(lease.request)[1]
        now=self._now()
        with self._transaction(read_only=True) as session:
            active=session.query(DurableJob).filter(DurableJob.job_id==lease.job_id,DurableJob.status=='running',
                DurableJob.lease_token==lease.token,DurableJob.attempts==lease.attempt,DurableJob.lease_until>now,
                DurableJob.request_sha256==request_digest,DurableJob.kind==lease.kind,
                DurableJob.max_attempts==lease.max_attempts).first()
            if not active:raise LostLease('Execution lease is stale, expired or cancelled')
            return {r.key:self._checkpoint_view(r) for r in session.query(JobCheckpoint).filter_by(job_id=lease.job_id)}

    def finish(self, lease, result):
        raw,digest=_json(result);now=self._now()
        with self._transaction() as session:
            row=self._fenced_update(session,lease,now,status='completed',result_json=raw,result_sha256=digest,
                                    lease_token=None,lease_until=None,error_code=None)
            self._release(session);self._event(session,lease.job_id,row.revision,'completed',row.at,row.attempts,{'result_sha256':digest})

    def fail(self, lease, code, *, retry_at=None):
        if not isinstance(lease,JobLease) or type(lease.max_attempts) is not int:
            raise LostLease('Invalid execution lease')
        code=_identifier(code,'error code',80);now=self._now()
        if not re.fullmatch('[A-Z][A-Z0-9_]*',code):raise ValueError('Use a bounded public error code, not exception/URL/credential text')
        if retry_at is not None and (isinstance(retry_at,bool) or not isinstance(retry_at,(int,float)) or not math.isfinite(retry_at) or retry_at<=now):
            raise ValueError('Retry deadline must be finite and in the future')
        with self._transaction() as session:
            current=session.get(DurableJob,lease.job_id)
            retry=retry_at is not None and current is not None and current.attempts<current.max_attempts
            status='retry_wait' if retry else 'failed'
            row=self._fenced_update(session,lease,now,status=status,error_code=code,available_at=retry_at if retry else now,
                                    lease_token=None,lease_until=None)
            if not retry:self._release(session)
            self._event(session,lease.job_id,row.revision,status,row.at,row.attempts,{'code':code,'retry_at':retry_at if retry else None})
