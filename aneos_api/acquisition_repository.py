"""Existing project SQL persistence adapter for ordinary object acquisitions.

One transaction owns source assertions, query envelope and compatibility projection.
File-SQLite is validated; other backends fail explicitly, not silently ephemeral.
Imports register models only. Schema writes occur during explicit acquisition preparation or commit, never import.
"""

import json
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, select, inspect
from sqlalchemy.orm import sessionmaker

from .database import Base
from .enrichment_store import (
    SourceAssertion,
    ObjectProjection,
    ObjectAcquisition,
    initialize_schema,
    schema_ready,
    require_writable_acquisition_uri,
    enrich,
    _record,
)
from aneos_core.data.acquisition import validate_bundle
from aneos_core.data.enrichment import dumps, fingerprint
from aneos_core.utils.errors import EvidencePersistenceError


def _table_ready(connection):
    i = inspect(connection)
    table = ObjectAcquisition.__table__
    if table.name not in i.get_table_names():
        return False
    if not {c.name for c in table.columns}.issubset(
        {c["name"] for c in i.get_columns(table.name)}
    ):
        return False
    if i.get_pk_constraint(table.name).get("constrained_columns") != ["sequence"]:
        return False
    unique = [u["column_names"] for u in i.get_unique_constraints(table.name)]
    return ["bundle_id"] in unique and ["query_id"] in unique


def initialize_acquisition_schema(engine):
    initialize_schema(engine, include_all=True)
    with engine.begin() as c:
        c.exec_driver_sql("BEGIN IMMEDIATE")
        if not _table_ready(c):
            raise RuntimeError(
                "Incompatible acquisition schema; explicit migration required"
            )
        for action in ("UPDATE", "DELETE"):
            c.exec_driver_sql(
                f"CREATE TRIGGER IF NOT EXISTS neo_acquisitions_v1_no_{action.lower()} BEFORE {action} ON neo_acquisitions_v1 BEGIN SELECT RAISE(ABORT,'immutable acquisition evidence'); END"
            )


def acquisition_schema_ready(engine):
    if not schema_ready(engine):
        return False
    with engine.connect() as c:
        if not _table_ready(c):
            return False
        names = {
            r[0]
            for r in c.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        return all("neo_acquisitions_v1_no_" + a in names for a in ("update", "delete"))


def _payloads(bundle):
    rows = []
    attempts = []
    for o in bundle["outcomes"]:
        if o["status"] == "available":
            rows.append(
                {
                    "designation": bundle["designation"],
                    "data_source": o["source"],
                    "snapshot_schema": "neo-data-v1",
                    "neo_data": o["neo_data"],
                    "acquisition_query_id": bundle["query_id"],
                }
            )
        attempts.append(
            {
                "designation": bundle["designation"],
                "data_source": o["source"],
                "status": o["status"],
                "acquisition_query_id": bundle["query_id"],
                "source_plan": bundle["source_plan"],
                "source_outcome": o,
            }
        )
    return rows, attempts


def _expected_identities(bundle):
    rows, attempts = _payloads(bundle)
    return [
        {
            "designation": bundle["designation"],
            "source": p["data_source"],
            "session_id": "acquisition-" + bundle["query_id"],
            "kind": kind,
            "payload": p,
        }
        for kind, payloads in (("source_assertion", rows), ("query_attempt", attempts))
        for p in payloads
    ]


def _query_record(db, row):
    bundle = validate_bundle(json.loads(row.bundle_json))
    if (
        bundle["query_id"] != row.query_id
        or bundle["bundle_id"] != row.bundle_id
        or bundle["designation"] != row.designation
    ):
        raise ValueError("Stored acquisition identity mismatch")
    expected = _expected_identities(bundle)
    ids = json.loads(row.assertion_ids_json)
    if ids != [fingerprint(i) for i in expected]:
        raise ValueError("Acquisition input assertions do not reproduce")
    for key, identity in zip(ids, expected):
        assertion = db.scalar(
            select(SourceAssertion).where(SourceAssertion.assertion_id == key)
        )
        if assertion is None:
            raise ValueError("Acquisition input assertion missing")
        actual = _record(assertion)
        if any(actual[k] != v for k, v in identity.items()):
            raise ValueError("Acquisition input assertion mismatch")
    if (
        row.projection_id is not None
        and db.scalar(
            select(ObjectProjection).where(
                ObjectProjection.projection_id == row.projection_id
            )
        )
        is None
    ):
        raise ValueError("Acquisition projection missing")
    return {
        "bundle": bundle,
        "receipt": {
            "schema_version": "acquisition-storage-receipt-v1",
            "query_id": row.query_id,
            "bundle_id": row.bundle_id,
            "commit_status": "committed",
            "committed_at": row.committed_at,
            "query_sequence": row.sequence,
            "projection_id": row.projection_id,
            "assertion_count": len(ids),
            "consistency_status": "replayed_not_authenticated",
        },
    }


def query_history(db, designation):
    if ObjectAcquisition.__tablename__ not in inspect(db.get_bind()).get_table_names():
        return []
    if not acquisition_schema_ready(db.get_bind()):
        raise ValueError("Acquisition schema unavailable or invalid")
    history = [
        _query_record(db, row)
        for row in db.scalars(
            select(ObjectAcquisition)
            .where(ObjectAcquisition.designation == designation)
            .order_by(ObjectAcquisition.sequence)
        )
    ]
    seen = {}
    for record in history:
        repeats = []
        for o in record["bundle"]["outcomes"]:
            digest = o["native_response_sha256"]
            if digest is not None:
                key = (o["source"], digest)
                repeats.append(
                    {
                        "source": o["source"],
                        "native_response_sha256": digest,
                        "exact_repeat_of_query": seen.get(key),
                        "semantics": "native response bytes/JSON equality, not independent observations",
                    }
                )
                seen.setdefault(key, record["bundle"]["query_id"])
        record["native_response_repeat_checks"] = repeats
    return history


class SQLEvidenceRepository:
    def __init__(self, engine=None):
        if engine is None:
            from .database import db_manager

            engine = db_manager.engine
        self.engine = engine
        self.sessions = sessionmaker(bind=engine)

    def load_acquisition(self, query_id):
        if not acquisition_schema_ready(self.engine):
            return None
        with self.sessions() as db:
            row = db.scalar(
                select(ObjectAcquisition).where(ObjectAcquisition.query_id == query_id)
            )
            return _query_record(db, row) if row is not None else None

    def prepare_acquisition(self):
        try:
            require_writable_acquisition_uri(self.engine)
        except ValueError as exc:
            raise EvidencePersistenceError("Acquisition URI configuration is not writable; sources not queried") from exc
        if not acquisition_schema_ready(self.engine):
            initialize_acquisition_schema(self.engine)
        return {
            "status": "ready",
            "basis": "file_sqlite_schema_not_future_commit_guarantee",
        }

    def commit_acquisition(self, bundle):
        bundle = validate_bundle(bundle)
        existing = self.load_acquisition(bundle["query_id"])
        if existing is not None:
            if existing["bundle"] != bundle:
                raise EvidencePersistenceError(
                    "Query identifier already binds different evidence"
                )
            return existing["receipt"]
        self.prepare_acquisition()
        rows, attempts = _payloads(bundle)

        def guard(db):
            existing = db.scalar(
                select(ObjectAcquisition).where(
                    ObjectAcquisition.query_id == bundle["query_id"]
                )
            )
            if existing is not None:
                if _query_record(db, existing)["bundle"] != bundle:
                    raise ValueError(
                        "Query identifier already binds different evidence"
                    )
                return False  # Immutable replay must not reapply an older projection policy.
            return True

        def append(db):
            existing = db.scalar(
                select(ObjectAcquisition).where(
                    ObjectAcquisition.query_id == bundle["query_id"]
                )
            )
            if existing is not None:
                if _query_record(db, existing)["bundle"] != bundle:
                    raise ValueError(
                        "Query identifier already binds different evidence"
                    )
                return
            db.flush()
            expected = _expected_identities(bundle)
            projection = db.scalar(
                select(ObjectProjection)
                .where(ObjectProjection.designation == bundle["designation"])
                .order_by(ObjectProjection.sequence.desc())
                .limit(1)
            )
            db.add(
                ObjectAcquisition(
                    query_id=bundle["query_id"],
                    designation=bundle["designation"],
                    bundle_id=bundle["bundle_id"],
                    bundle_json=dumps(bundle),
                    assertion_ids_json=dumps([fingerprint(i) for i in expected]),
                    projection_id=projection.projection_id if projection else None,
                    committed_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        with self.sessions() as db:
            stats = enrich(
                db,
                rows,
                "acquisition-" + bundle["query_id"],
                query_attempts=attempts,
                priority=bundle["source_plan"]["source_priority"] + ["CAD"],
                _before_commit=append,
                _after_begin=guard,
            )
            if stats.get("commit_status") != "committed":
                raise EvidencePersistenceError("Acquisition transaction not committed")
        recorded = self.load_acquisition(bundle["query_id"])
        if recorded is None:
            raise EvidencePersistenceError("Acquisition commit could not be read back")
        return recorded["receipt"]
