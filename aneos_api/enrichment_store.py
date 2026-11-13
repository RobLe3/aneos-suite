"""Append-only acquisition assertions/projections on the existing SQLAlchemy Base.

Explicit additive schema initialization; file-SQLite transaction/immutability tests.
Checksums establish stored consistency, not external authenticity or DB-admin security.
"""

from datetime import datetime, timezone
import json
from sqlalchemy import Column, Integer, String, Text, select, text, inspect, null
from .database import Base, EnrichedNEO
from aneos_core.data.enrichment import (
    clone,
    dumps,
    fingerprint,
    identifier,
    canonical_source,
    normalize_assertion,
    project_assertions,
    SOURCE_PRIORITY,
)


class SourceAssertion(Base):
    __tablename__ = "neo_source_assertions_v1"
    sequence = Column(Integer, primary_key=True, autoincrement=True)
    assertion_id = Column(String(64), unique=True, nullable=False)
    designation = Column(String(100), index=True, nullable=False)
    source = Column(String(100), nullable=False)
    session_id = Column(String(160), nullable=False)
    kind = Column(String(32), nullable=False)
    recorded_at = Column(String(40), nullable=False)
    payload_json = Column(Text, nullable=False)
    payload_sha256 = Column(String(64), nullable=False)
    normalized_json = Column(Text, nullable=False)
    normalized_sha256 = Column(String(64), nullable=False)


class ObjectProjection(Base):
    __tablename__ = "neo_object_projections_v1"
    sequence = Column(Integer, primary_key=True, autoincrement=True)
    projection_id = Column(String(64), unique=True, nullable=False)
    designation = Column(String(100), index=True, nullable=False)
    created_at = Column(String(40), nullable=False)
    input_assertions_json = Column(Text, nullable=False)
    payload_json = Column(Text, nullable=False)
    payload_sha256 = Column(String(64), nullable=False)


class ObjectAcquisition(Base):
    __tablename__ = "neo_acquisitions_v1"
    sequence = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String(100), unique=True, nullable=False)
    designation = Column(String(100), index=True, nullable=False)
    bundle_id = Column(String(64), unique=True, nullable=False)
    bundle_json = Column(Text, nullable=False)
    assertion_ids_json = Column(Text, nullable=False)
    projection_id = Column(String(64), nullable=True)
    committed_at = Column(String(40), nullable=False)


TABLES = (SourceAssertion.__table__, ObjectProjection.__table__)
ACQUISITION_TABLES = (ObjectAcquisition.__table__,)


def _file_sqlite(engine):
    database = engine.url.database
    return (
        engine.dialect.name == "sqlite"
        and database not in (None, "", ":memory:")
        and engine.url.query.get("mode") != "memory"
        and not str(database).startswith("file::memory:")
    )


def require_writable_acquisition_uri(engine):
    """Reject declared read-only/unsupported immutable settings, not future IO faults.

    Read/replay paths deliberately do not call this guard. For fresh acquisition,
    immutable must be omitted or a documented explicit false token; custom boolean
    spellings are not assumed writable. This does not inspect OS permissions/VFS.
    """
    if engine.dialect.name == 'sqlite':
        immutable = engine.url.query.get('immutable', '0')
        writable_flag = isinstance(immutable, str) and immutable.lower() in ('0', 'false', 'off', 'no', '')
        if engine.url.query.get('mode') == 'ro' or not writable_flag:
            raise ValueError('Fresh acquisition requires a writable SQLite URI; read-only/unsupported immutable settings')


def _columns_ready(connection, tables=TABLES):
    inspector = inspect(connection)
    for table in tables:
        if table.name not in inspector.get_table_names():
            return False
        if not {c.name for c in table.columns}.issubset(
            {c["name"] for c in inspector.get_columns(table.name)}
        ):
            return False
        if inspector.get_pk_constraint(table.name).get("constrained_columns") != [
            c.name for c in table.primary_key
        ]:
            return False
        expected = {
            SourceAssertion.__tablename__: "assertion_id",
            ObjectProjection.__tablename__: "projection_id",
            ObjectAcquisition.__tablename__: "bundle_id",
        }[table.name]
        if not any(
            u["column_names"] == [expected]
            for u in inspector.get_unique_constraints(table.name)
        ):
            return False
    return True


def initialize_schema(engine, *, include_all=False):
    if not _file_sqlite(engine):
        raise RuntimeError(
            "File-SQLite enrichment backend required; other deployments unvalidated"
        )
    with engine.begin() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        Base.metadata.create_all(
            connection,
            tables=None if include_all else list(TABLES + ACQUISITION_TABLES),
        )
        if not _columns_ready(connection, TABLES + ACQUISITION_TABLES):
            raise RuntimeError(
                "Incompatible evidence schema; explicit migration required"
            )
        for table in TABLES + ACQUISITION_TABLES:
            for action in ("UPDATE", "DELETE"):
                name = f"{table.name}_no_{action.lower()}"
                connection.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS {name} BEFORE {action} ON {table.name} BEGIN SELECT RAISE(ABORT,'immutable acquisition evidence'); END"
                )


def schema_ready(engine):
    if not _file_sqlite(engine):
        return False
    with engine.connect() as connection:
        if not _columns_ready(connection):
            return False
        names = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        return all(
            f"{t.name}_no_{a}" in names for t in TABLES for a in ("update", "delete")
        )


def _record(row):
    payload = json.loads(row.payload_json)
    normalized = json.loads(row.normalized_json)
    if (
        fingerprint(payload) != row.payload_sha256
        or fingerprint(normalized) != row.normalized_sha256
    ):
        raise ValueError("Stored assertion fingerprint mismatch")
    identity = {
        "designation": row.designation,
        "source": row.source,
        "session_id": row.session_id,
        "kind": row.kind,
        "payload": payload,
    }
    if fingerprint(identity) != row.assertion_id:
        raise ValueError("Stored assertion identity mismatch")
    if row.kind == "source_assertion":
        actual = normalize_assertion(payload, row.recorded_at)
        if actual != normalized:
            raise ValueError("Stored source normalization mismatch")
    return dict(
        identity,
        assertion_id=row.assertion_id,
        recorded_at=row.recorded_at,
        normalized=normalized,
        sequence=row.sequence,
        payload_sha256=row.payload_sha256,
    )


def records(db, designation):
    result = []
    seen = {}
    for row in db.scalars(
        select(SourceAssertion)
        .where(SourceAssertion.designation == designation)
        .order_by(SourceAssertion.sequence)
    ):
        record = _record(row)
        key = (record["source"], record["kind"], record["payload_sha256"])
        record["exact_payload_repeat_of"] = seen.get(key)
        seen.setdefault(key, record["assertion_id"])
        result.append(record)
    return result


def latest_projection(db, designation, *, verify=True):
    row = db.scalar(
        select(ObjectProjection)
        .where(ObjectProjection.designation == designation)
        .order_by(ObjectProjection.sequence.desc())
        .limit(1)
    )
    if row is None:
        return None
    result = json.loads(row.payload_json)
    ids = json.loads(row.input_assertions_json)
    if (
        fingerprint(result) != row.payload_sha256
        or fingerprint({"inputs": ids, "projection": result}) != row.projection_id
    ):
        raise ValueError("Stored projection fingerprint mismatch")
    if verify:
        history = records(db, designation)
        lookup = {r["assertion_id"]: r for r in history}
        inputs = [lookup[i] for i in ids]
        if project_assertions(inputs, result["source_priority"]) != result:
            raise ValueError("Stored projection does not reproduce")
    return dict(
        projection_id=row.projection_id,
        created_at=row.created_at,
        projection=result,
        input_assertions=ids,
    )


def _append(db, designation, source, session_id, kind, payload, stamp, normalized=None):
    identity = {
        "designation": designation,
        "source": source,
        "session_id": session_id,
        "kind": kind,
        "payload": payload,
    }
    key = fingerprint(identity)
    existing = db.scalar(
        select(SourceAssertion).where(SourceAssertion.assertion_id == key)
    )
    if existing is not None:
        _record(existing)
        return False
    normalized = (
        normalize_assertion(payload, stamp)
        if kind == "source_assertion"
        else (
            normalized
            or {
                "status": (
                    "legacy_unverified"
                    if kind == "legacy_projection"
                    else payload["status"]
                )
            }
        )
    )
    db.add(
        SourceAssertion(
            assertion_id=key,
            designation=designation,
            source=source,
            session_id=session_id,
            kind=kind,
            recorded_at=stamp,
            payload_json=dumps(payload),
            payload_sha256=fingerprint(payload),
            normalized_json=dumps(normalized),
            normalized_sha256=fingerprint(normalized),
        )
    )
    db.flush()
    return True


def _legacy_snapshot(row):
    value = {}
    for column in EnrichedNEO.__table__.columns:
        item = getattr(row, column.name)
        value[column.name] = item.isoformat() if isinstance(item, datetime) else item
    return clone(value)


def _import_legacy(db, row, stamp):
    _append(
        db,
        row.designation,
        "LEGACY_ENRICHED",
        "legacy-row-" + str(row.id),
        "legacy_projection",
        _legacy_snapshot(row),
        stamp,
    )
    slots = {
        "SBDB": "nasa_sbdb_data",
        "CAD": "nasa_cad_data",
        "MPC": "mpc_data",
        "NEODyS": "neodys_data",
    }

    def archive_source(source, payload):
        from aneos_core.data.models import NEOData
        from aneos_core.data.acquisition import has_material

        try:
            normalized = normalize_assertion(payload, stamp)
            if not has_material(NEOData.from_dict(normalized["normalized_neo"])):
                raise ValueError("No interpretable legacy fields")
        except (ValueError, TypeError, KeyError, AttributeError):
            _append(
                db,
                row.designation,
                source,
                "legacy-source-" + str(row.id),
                "legacy_source_uninterpretable",
                payload,
                stamp,
                normalized={"status": "legacy_uninterpretable_not_current_measurement"},
            )
            return False
        _append(
            db,
            row.designation,
            source,
            "legacy-source-" + str(row.id),
            "source_assertion",
            payload,
            stamp,
        )
        return True

    imported = False
    for source, column in slots.items():
        original = getattr(row, column)
        if not isinstance(original, dict) or not original:
            continue
        payload = clone(original)
        payload.setdefault("designation", row.designation)
        payload["data_source"] = source
        payload["legacy_import_context"] = {
            "source_column": column,
            "status": "legacy_unverified_import",
        }
        imported = archive_source(source, payload) or imported
    if not imported:
        payload = {
            "designation": row.designation,
            "data_source": "LEGACY_AGGREGATE",
            "orbital_elements": clone(row.orbital_elements),
            "physical_properties": clone(row.physical_properties),
            "close_approaches": clone(row.close_approaches or []),
            "legacy_import_context": {
                "status": "legacy_unverified_aggregate_not_individual_source"
            },
        }
        archive_source("LEGACY_AGGREGATE", payload)


def _update_materialized(row, projection, history, stamp):
    p = projection["neo_data"]
    oe = p["orbital_elements"]
    pp = p["physical_properties"]
    assertions = [r for r in history if r["kind"] == "source_assertion"]
    row.data_sources = list(dict.fromkeys(r["source"] for r in assertions))
    old_sessions = [
        s
        for r in history
        if r["kind"] == "legacy_projection"
        for s in (r["payload"].get("polling_sessions") or [])
        if isinstance(s, str)
    ]
    row.polling_sessions = list(
        dict.fromkeys(
            old_sessions
            + [
                r["session_id"]
                for r in history
                if r["kind"] != "legacy_projection"
                and not r["session_id"].startswith("legacy-source-")
            ]
        )
    )
    row.last_updated = datetime.fromisoformat(stamp)
    row.total_detections = len(assertions)
    row.orbital_elements = clone(oe)
    row.physical_properties = clone(pp)
    row.close_approaches = clone(p["close_approaches"])
    row.orbital_elements_source = (
        p["evidence_metadata"].get("field_sources") or {}
    ).get("orbital_elements")
    row.artificial_probability = null()
    row.data_quality_score = null()
    row.orbital_elements_quality = null()
    row.has_orbital_elements = bool(oe)
    row.has_physical_properties = bool(pp)
    row.has_close_approaches = bool(p["close_approaches"])
    # A named field-presence metric only; not source accuracy or scientific completeness.
    from aneos_core.data.models import NEOData
    from aneos_core.analysis.object_review import build_data_inventory

    row.completeness_score = build_data_inventory(NEOData.from_dict(p))["coverage"][
        "basic_catalogue"
    ]["fraction_present"]
    slots = {
        "SBDB": ("nasa_sbdb_data", "nasa_sbdb_last_update"),
        "CAD": ("nasa_cad_data", "nasa_cad_last_update"),
        "MPC": ("mpc_data", "mpc_last_update"),
        "NEODyS": ("neodys_data", "neodys_last_update"),
    }
    latest = {r["source"]: r for r in assertions}
    for source, (column, updated) in slots.items():
        if source in latest:
            setattr(row, column, clone(latest[source]["payload"]))
            setattr(row, updated, datetime.fromisoformat(latest[source]["recorded_at"]))


def enrich(
    db,
    rows,
    session_id,
    *,
    query_attempts=None,
    priority=SOURCE_PRIORITY,
    _before_commit=None,
    _after_begin=None,
):
    """Own one fresh transaction; acknowledge only after assertions+projection commit."""
    if db.in_transaction():
        return {
            "error": "FRESH_SESSION_REQUIRED",
            "commit_status": "not_committed",
            "inserted_assertions": 0,
        }
    if not schema_ready(db.get_bind()):
        return {
            "error": "ENRICHMENT_SCHEMA_REQUIRED",
            "commit_status": "not_committed",
            "inserted_assertions": 0,
        }
    if not isinstance(rows, list) or (
        query_attempts is not None and not isinstance(query_attempts, list)
    ):
        return {
            "error": "INVALID_ENRICHMENT_INPUT",
            "commit_status": "not_committed",
            "inserted_assertions": 0,
        }
    try:
        session_id = identifier(session_id, "session id", 160)
    except ValueError:
        return {
            "error": "INVALID_SESSION_ID",
            "commit_status": "not_committed",
            "inserted_assertions": 0,
        }
    stamp = datetime.now(timezone.utc).isoformat()
    began = False
    stats = {
        "new_neos": 0,
        "updated_neos": 0,
        "enriched_neos": 0,
        "total_processed": len(rows),
        "session_id": session_id,
        "inserted_assertions": 0,
        "duplicate_assertions": 0,
        "commit_status": "not_committed",
    }
    try:
        prepared = []
        for payload in rows:
            payload = clone(payload)
            normalized = normalize_assertion(payload, stamp)
            prepared.append(
                (
                    normalized["designation"],
                    normalized["source"],
                    "source_assertion",
                    payload,
                )
            )
        for attempt in query_attempts or []:
            attempt = clone(attempt)
            designation = identifier(attempt.get("designation"), "designation")
            source = canonical_source(attempt.get("data_source"))
            if attempt.get("status") not in (
                "available",
                "missing",
                "unavailable",
                "invalid",
                "ambiguous",
                "not_supported",
            ):
                raise ValueError("Invalid query attempt status")
            prepared.append((designation, source, "query_attempt", attempt))
        db.execute(text("BEGIN IMMEDIATE"))
        began = True
        if _after_begin is not None and _after_begin(db) is False:
            db.commit()
            stats["commit_status"] = "committed"
            return stats
        changed = set()
        for designation, source, kind, payload in prepared:
            row = db.scalar(
                select(EnrichedNEO).where(EnrichedNEO.designation == designation)
            )
            if row is not None and not records(db, designation):
                _import_legacy(db, row, stamp)
            inserted = _append(
                db, designation, source, session_id, kind, payload, stamp
            )
            stats["inserted_assertions" if inserted else "duplicate_assertions"] += 1
            if inserted:
                changed.add(designation)
        for designation in {item[0] for item in prepared}:
            previous = latest_projection(db, designation)
            if previous is not None and previous["projection"][
                "source_priority"
            ] != list(priority):
                changed.add(designation)
        for designation in sorted(changed):
            history = records(db, designation)
            if not any(r["kind"] == "source_assertion" for r in history):
                continue
            projection = project_assertions(history, list(priority))
            ids = [
                r["assertion_id"] for r in history if r["kind"] == "source_assertion"
            ]
            key = fingerprint({"inputs": ids, "projection": projection})
            if (
                db.scalar(
                    select(ObjectProjection).where(
                        ObjectProjection.projection_id == key
                    )
                )
                is None
            ):
                db.add(
                    ObjectProjection(
                        projection_id=key,
                        designation=designation,
                        created_at=stamp,
                        input_assertions_json=dumps(ids),
                        payload_json=dumps(projection),
                        payload_sha256=fingerprint(projection),
                    )
                )
            row = db.scalar(
                select(EnrichedNEO).where(EnrichedNEO.designation == designation)
            )
            if row is None:
                row = EnrichedNEO(
                    designation=designation,
                    first_discovered=None,  # Local intake is not an astronomical discovery.
                )
                db.add(row)
                stats["new_neos"] += 1
            else:
                stats["updated_neos"] += 1
            _update_materialized(row, projection, history, stamp)
            stats["enriched_neos"] += 1
        if _before_commit is not None:
            _before_commit(db)
        db.commit()
        stats["commit_status"] = "committed"
        return stats
    except Exception:
        if began:
            db.rollback()
        return {
            "error": "ENRICHMENT_NOT_COMMITTED",
            "commit_status": "not_committed",
            "inserted_assertions": 0,
            "new_neos": 0,
            "updated_neos": 0,
            "enriched_neos": 0,
            "total_processed": len(rows),
            "session_id": session_id,
        }
