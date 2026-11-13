"""Atomic candidate-query receipts on the existing acquisition/enrichment database.

Global empty/failed queries have receipts, not invented object designations.
"""
from datetime import datetime, timezone
import json
import base64
from hashlib import sha256
from sqlalchemy import Column, Integer, String, Text, LargeBinary, inspect, select
from sqlalchemy.orm import sessionmaker
from .database import Base
from .enrichment_store import SourceAssertion, ObjectProjection, initialize_schema, schema_ready, enrich, _record
from aneos_core.data.candidate_discovery import replay_receipt
from aneos_core.data.enrichment import dumps, fingerprint, project_assertions
from aneos_core.utils.errors import EvidencePersistenceError


class CandidateQuery(Base):
    __tablename__ = 'neo_candidate_queries_v1'
    sequence = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String(100), unique=True, nullable=False)
    receipt_id = Column(String(64), unique=True, nullable=False)
    receipt_json = Column(Text, nullable=False)
    recorded_at = Column(String(40), nullable=False)
    projection_ids_json = Column(Text, nullable=False)
    assertion_ids_json = Column(Text, nullable=False)
    binding_sha256 = Column(String(64), nullable=False)
    native_body = Column(LargeBinary, nullable=True)


def _ready(engine):
    table = CandidateQuery.__table__
    i = inspect(engine)
    return (table.name in i.get_table_names()
            and {c.name for c in table.columns}.issubset({c['name'] for c in i.get_columns(table.name)})
            and i.get_pk_constraint(table.name).get('constrained_columns') == ['sequence']
            and all([name] in [u['column_names'] for u in i.get_unique_constraints(table.name)]
                    for name in ('query_id','receipt_id')))


def _binding(query_id, receipt_id, projection_ids, assertion_ids, recorded_at):
    """Storage seal excludes derived read-model status fields."""
    return {'query_id': query_id, 'receipt_id': receipt_id,
            'projection_ids': projection_ids, 'assertion_ids': assertion_ids, 'recorded_at': recorded_at}


def _native_body(receipt, archive):
    response = receipt['response']
    if response['body_base64'] is None:
        if archive is not None:
            raise ValueError('Unexpected archive without a source response')
        return None
    prefix = base64.b64decode(response['body_base64'], validate=True)
    if response['body_in_receipt_complete']:
        if archive is not None:
            raise ValueError('Unexpected duplicate native archive')
        return prefix
    if (not isinstance(archive, bytes) or len(archive) != response['received_bytes']
            or sha256(archive).hexdigest() != response['body_sha256']
            or archive[:len(prefix)] != prefix):
        raise ValueError('Full native archive is missing or changed')
    return archive


class CandidateQueryRepository:
    def __init__(self, engine=None):
        if engine is None:
            from .database import db_manager
            engine = db_manager.engine
        self.engine = engine
        self.sessions = sessionmaker(bind=engine)

    def prepare(self):
        try:
            if schema_ready(self.engine) and _ready(self.engine):
                with self.engine.connect() as connection:
                    names = {r[0] for r in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='trigger'")}
                    if all('neo_candidate_queries_v1_no_'+a in names for a in ('update','delete')):
                        return
            initialize_schema(self.engine, include_all=True)
            if not _ready(self.engine):
                raise ValueError('Candidate schema requires explicit migration')
            with self.engine.begin() as connection:
                for action in ('UPDATE','DELETE'):
                    connection.exec_driver_sql(
                        f"CREATE TRIGGER IF NOT EXISTS neo_candidate_queries_v1_no_{action.lower()} "
                        f"BEFORE {action} ON neo_candidate_queries_v1 "
                        "BEGIN SELECT RAISE(ABORT,'immutable candidate query'); END")
        except Exception as exc:
            raise EvidencePersistenceError('Candidate storage prerequisite unavailable; source not queried') from exc

    def load(self, query_id):
        with self.sessions() as db:
            row = db.scalar(select(CandidateQuery).where(CandidateQuery.query_id == query_id))
            if row is None:
                return None
            receipt, packets = replay_receipt(json.loads(row.receipt_json))
            if receipt['receipt_id'] != row.receipt_id or receipt['query_id'] != row.query_id:
                raise ValueError('Candidate receipt binding changed')
            assertions = list(db.scalars(select(SourceAssertion).where(
                SourceAssertion.session_id == query_id, SourceAssertion.kind == 'source_assertion')))
            verified = [_record(a) for a in assertions]
            if sorted(fingerprint(a['payload']) for a in verified) != sorted(receipt['packet_ids']):
                raise ValueError('Candidate object assertions are incomplete or changed')
            native_body = _native_body(receipt, row.native_body)
            projection_ids = json.loads(row.projection_ids_json)
            assertion_ids = json.loads(row.assertion_ids_json)
            binding = _binding(row.query_id, row.receipt_id, projection_ids, assertion_ids, row.recorded_at)
            if fingerprint(binding) != row.binding_sha256 or sorted(assertion_ids) != sorted(a['assertion_id'] for a in verified):
                raise ValueError('Candidate storage binding changed')
            if set(projection_ids) != {p['designation'] for p in packets}:
                raise ValueError('Candidate projections are incomplete')
            for designation, key in projection_ids.items():
                projection = db.scalar(select(ObjectProjection).where(ObjectProjection.projection_id == key))
                if projection is None or projection.designation != designation:
                    raise ValueError('Candidate projection missing')
                payload = json.loads(projection.payload_json)
                inputs = json.loads(projection.input_assertions_json)
                if (fingerprint(payload) != projection.payload_sha256
                        or fingerprint({'inputs': inputs, 'projection': payload}) != key):
                    raise ValueError('Candidate projection fingerprint changed')
                history = [_record(a) for a in db.scalars(select(SourceAssertion).where(
                    SourceAssertion.assertion_id.in_(inputs)).order_by(SourceAssertion.sequence))]
                expected = {a['assertion_id'] for a in verified if a['designation'] == designation}
                if (not expected.issubset(inputs) or [a['assertion_id'] for a in history] != inputs
                        or project_assertions(history, payload['source_priority']) != payload):
                    raise ValueError('Candidate projection does not reproduce')
            return {'receipt': receipt, 'records': packets, 'commit_status': 'committed',
                    'recorded_at': row.recorded_at,
                    'native_body_status': 'complete_entity_verified' if native_body is not None else 'not_available_from_transport'}

    def load_native_body(self, query_id):
        result = self.load(query_id)
        if result is None:
            raise KeyError('Unknown candidate query')
        with self.sessions() as db:
            row = db.scalar(select(CandidateQuery).where(CandidateQuery.query_id == query_id))
            if row is None:
                raise ValueError('Candidate query disappeared')
            return _native_body(result['receipt'], row.native_body)

    def commit(self, receipt, *, native_body=None):
        receipt, packets = replay_receipt(receipt)
        query_id = receipt['query_id']
        def failed(message):
            error = EvidencePersistenceError(message)
            error.query_id = query_id
            return error
        def guard(db):
            old = db.scalar(select(CandidateQuery).where(CandidateQuery.query_id == query_id))
            if old is not None:
                if old.receipt_id != receipt['receipt_id'] or json.loads(old.receipt_json) != receipt:
                    raise ValueError('Query id already binds different data')
                if native_body is not None:
                    _native_body(receipt, native_body)
                return False
        def append(db):
            _native_body(receipt, native_body)
            db.flush()
            projection_ids = {}
            for packet in packets:
                projection = db.scalar(select(ObjectProjection).where(
                    ObjectProjection.designation == packet['designation']).order_by(ObjectProjection.sequence.desc()).limit(1))
                if projection is None:
                    raise ValueError('Candidate projection was not prepared')
                projection_ids[packet['designation']] = projection.projection_id
            assertion_ids = list(db.scalars(select(SourceAssertion.assertion_id).where(
                SourceAssertion.session_id == query_id, SourceAssertion.kind == 'source_assertion').order_by(SourceAssertion.sequence)))
            stamp = datetime.now(timezone.utc).isoformat()
            binding = _binding(query_id, receipt['receipt_id'], projection_ids, assertion_ids, stamp)
            db.add(CandidateQuery(query_id=query_id, receipt_id=receipt['receipt_id'],
                                  receipt_json=dumps(receipt), recorded_at=stamp,
                                  projection_ids_json=dumps(projection_ids), assertion_ids_json=dumps(assertion_ids),
                                  binding_sha256=fingerprint(binding), native_body=native_body))
        with self.sessions() as db:
            result = enrich(db, packets, query_id, _after_begin=guard, _before_commit=append)
        if result.get('commit_status') != 'committed':
            raise failed('Candidate receipt/object transaction not committed')
        try:
            loaded = self.load(query_id)
        except Exception as exc:
            raise failed('Candidate commit readback could not be verified') from exc
        if loaded is None:
            raise failed('Candidate commit readback is missing')
        return loaded
