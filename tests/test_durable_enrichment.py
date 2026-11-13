"""Actual file-SQLite enrichment, fresh sessions and immutable evidence history."""

import json
import copy
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from aneos_api.database import Base, EnrichedNEO, EnrichedNEOService


@pytest.fixture
def database(tmp_path):
    url = "sqlite:///" + str(tmp_path / "enrichment.sqlite")
    engine = create_engine(
        url, connect_args={"check_same_thread": False, "timeout": 10}
    )
    Base.metadata.create_all(engine)
    from aneos_api.enrichment_store import initialize_schema

    initialize_schema(engine)
    yield engine, sessionmaker(bind=engine), url
    engine.dispose()


def datum(source="NASA_SBDB", **overrides):
    result = {
        "designation": "SYNTHETIC",
        "data_source": source,
        "orbital_elements": {"a": 1.2, "e": 0.2, "i": 1.49, "q": 0.96},
        "physical_properties": {"diameter_km": 0.011},
        "artificial_probability": None,
    }
    result.update(overrides)
    return result


def enrich(database, rows, session):
    _, sessions, _ = database
    with sessions() as db:
        return EnrichedNEOService(db).enrich_neo_data(rows, session)


def test_fresh_sessions_retain_source_sessions_and_correlated_physical_values(database):
    assert "error" not in enrich(
        database, [datum(artificial_probability=0.03)], "initial"
    )
    result = enrich(
        database,
        [
            datum(
                "MPC", physical_properties={"albedo": 0.52}, artificial_probability=0.01
            )
        ],
        "second",
    )
    assert "error" not in result
    with database[1]() as db:
        n = db.scalar(select(EnrichedNEO).where(EnrichedNEO.designation == "SYNTHETIC"))
        assert set(n.data_sources) == {"SBDB", "MPC"}
        assert set(n.polling_sessions) == {"initial", "second"}
        assert n.physical_properties["diameter_km"] == 0.011
        assert n.physical_properties["albedo"] == 0.52


def test_null_origin_update_does_not_roll_back_source_evidence_or_become_zero(database):
    enrich(database, [datum(artificial_probability=0.03)], "initial")
    result = enrich(
        database,
        [datum("Horizons", physical_properties={"rotation_period_hours": 0.09})],
        "canonical-null",
    )
    assert "error" not in result
    with database[1]() as db:
        n = db.scalar(select(EnrichedNEO).where(EnrichedNEO.designation == "SYNTHETIC"))
        assert n.physical_properties["rotation_period_hours"] == 0.09
        assert n.artificial_probability is None


def test_assertions_and_projections_survive_engine_restart_with_native_claims_only_in_history(
    database,
):
    from aneos_api.enrichment_store import records, latest_projection

    engine, sessions, url = database
    enrich(database, [datum(artificial_probability=0.03)], "old")
    enrich(
        database,
        [datum("Horizons", physical_properties={"rotation_period_hours": 0.09})],
        "new",
    )
    engine.dispose()
    fresh = create_engine(url)
    try:
        with sessionmaker(bind=fresh)() as db:
            history = records(db, "SYNTHETIC")
            receipt = latest_projection(db, "SYNTHETIC")
            assert len(history) == 2
            assert history[0]["payload"]["artificial_probability"] == 0.03
            assert history[1]["payload"]["artificial_probability"] is None
            assert receipt["projection"]["artificial_probability"] is None
            assert (
                receipt["projection"]["neo_data"]["physical_properties"][
                    "rotation_period_hours"
                ]
                == 0.09
            )
    finally:
        fresh.dispose()


def test_duplicates_are_idempotent_but_new_session_and_changed_values_are_history(
    database,
):
    from aneos_api.enrichment_store import records, ObjectProjection

    first = enrich(database, [datum(), datum()], "same")
    assert first["inserted_assertions"] == 1 and first["duplicate_assertions"] == 1
    again = enrich(database, [datum()], "same")
    assert again["inserted_assertions"] == 0
    enrich(database, [datum(physical_properties={"diameter_km": 0.012})], "same")
    enrich(database, [datum()], "later")
    with database[1]() as db:
        assert len(records(db, "SYNTHETIC")) == 3
        assert len(list(db.scalars(select(ObjectProjection)))) == 3
        assert db.scalar(select(EnrichedNEO)).total_detections == 3


def test_legacy_row_is_archived_before_current_projection_changes(database):
    from aneos_api.enrichment_store import records

    with database[1]() as db:
        db.add(
            EnrichedNEO(
                designation="SYNTHETIC",
                data_sources=["NASA_SBDB"],
                polling_sessions=["before"],
                orbital_elements={"a": 1.2, "e": 0.2, "i": 1.49},
                physical_properties={"diameter_km": 0.011},
                artificial_probability=0.03,
                nasa_sbdb_data=datum(artificial_probability=0.03),
            )
        )
        db.commit()
    result = enrich(
        database, [datum("MPC", physical_properties={"albedo": 0.52})], "after"
    )
    assert result["commit_status"] == "committed"
    with database[1]() as db:
        history = records(db, "SYNTHETIC")
        legacy = next(r for r in history if r["kind"] == "legacy_projection")
        assert legacy["payload"]["artificial_probability"] == 0.03
        assert legacy["payload"]["polling_sessions"] == ["before"]
        row = db.scalar(select(EnrichedNEO))
        assert row.artificial_probability is None
        assert (
            row.physical_properties["diameter_km"] == 0.011
            and row.physical_properties["albedo"] == 0.52
        )


def test_immutable_sql_tables_refuse_updates_and_deletes(database):
    from sqlalchemy.exc import DBAPIError
    from aneos_api.enrichment_store import SourceAssertion, ObjectProjection

    enrich(database, [datum()], "one")
    for table in (SourceAssertion.__tablename__, ObjectProjection.__tablename__):
        for statement in (
            f"UPDATE {table} SET designation='OTHER'",
            f"DELETE FROM {table}",
        ):
            with database[1]() as db:
                with pytest.raises(DBAPIError):
                    db.execute(text(statement))
                    db.commit()
                db.rollback()


@pytest.mark.parametrize(
    "bad",
    [
        datum(physical_properties={"diameter_km": float("nan")}),
        datum(designation=""),
        datum(physical_properties={"diameter_km": True}),
        datum(physical_properties={"diameter_km": 0.01, "diameter_uncertainty": -0.01}),
    ],
)
def test_invalid_batch_rolls_back_all_data_and_success_counts(database, bad):
    from aneos_api.enrichment_store import SourceAssertion, ObjectProjection

    result = enrich(database, [datum(), bad], "bad")
    assert (
        result["commit_status"] == "not_committed"
        and result["inserted_assertions"] == 0
    )
    with database[1]() as db:
        assert not list(db.scalars(select(SourceAssertion)))
        assert not list(db.scalars(select(ObjectProjection)))
        assert not list(db.scalars(select(EnrichedNEO)))


def test_physical_only_source_and_missing_uncertainty_are_not_discarded_or_mispaired(
    database,
):
    enrich(database, [datum(physical_properties={"diameter_km": 0.011})], "primary")
    enrich(
        database,
        [
            datum(
                "MPC",
                orbital_elements=None,
                physical_properties={
                    "diameter_km": 0.02,
                    "diameter_uncertainty": 0.003,
                    "albedo": 0.52,
                },
            )
        ],
        "secondary",
    )
    with database[1]() as db:
        data = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert data["physical_properties"]["diameter_km"] == 0.011
        assert data["physical_properties"]["diameter_uncertainty"] is None
        assert data["physical_properties"]["albedo"] == 0.52
        assert len(data["evidence_history"]) == 2
        assert data["overall_data_completeness"] is None
        assert data["data_inventory"]["coverage"]["eligible_source_coverage"] is None


def test_query_failure_attempts_survive_without_fabricated_object_data(database):
    from aneos_api.enrichment_store import records

    with database[1]() as db:
        result = EnrichedNEOService(db).enrich_neo_data(
            [],
            "query-only",
            query_attempts=[
                {
                    "designation": "SYNTHETIC",
                    "data_source": "Horizons",
                    "status": "unavailable",
                    "request": {"years": 2},
                }
            ],
        )
        assert result["commit_status"] == "committed"
    with database[1]() as db:
        data = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert data["storage_status"] == "attempts_only"
        assert (
            data["projection_receipt"] is None
            and data["artificial_probability"] is None
        )
        assert records(db, "SYNTHETIC")[0]["payload"]["status"] == "unavailable"


def test_commit_failure_rolls_back_evidence_and_projection_together(
    database, monkeypatch
):
    from aneos_api.enrichment_store import SourceAssertion, ObjectProjection

    with database[1]() as db:

        def fail():
            raise RuntimeError("synthetic commit failure")

        monkeypatch.setattr(db, "commit", fail)
        result = EnrichedNEOService(db).enrich_neo_data([datum()], "fails")
        assert (
            result["commit_status"] == "not_committed"
            and result["inserted_assertions"] == 0
        )
    with database[1]() as db:
        assert not list(db.scalars(select(SourceAssertion)))
        assert not list(db.scalars(select(ObjectProjection)))
        assert not list(db.scalars(select(EnrichedNEO)))


def test_schema_is_explicit_and_missing_schema_never_claims_saved(tmp_path):
    url = "sqlite:///" + str(tmp_path / "legacy-only.sqlite")
    engine = create_engine(url)
    try:
        EnrichedNEO.__table__.create(engine)
        with sessionmaker(bind=engine)() as db:
            r = EnrichedNEOService(db).enrich_neo_data([datum()], "missing-schema")
            assert r["error"] == "ENRICHMENT_SCHEMA_REQUIRED"
        from sqlalchemy import inspect

        assert "neo_source_assertions_v1" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_unrelated_pending_session_changes_are_not_committed_or_rolled_back(database):
    with database[1]() as db:
        unrelated = EnrichedNEO(designation="UNRELATED")
        db.add(unrelated)
        r = EnrichedNEOService(db).enrich_neo_data([datum()], "blocked")
        assert r["error"] == "FRESH_SESSION_REQUIRED" and unrelated in db.new
        db.rollback()
    with database[1]() as db:
        assert not list(db.scalars(select(EnrichedNEO)))


def test_concurrent_writers_do_not_lose_sources_or_counts(database):
    from concurrent.futures import ThreadPoolExecutor

    records_to_save = [datum(), datum("MPC", physical_properties={"albedo": 0.52})]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: enrich(database, [item[1]], "parallel-" + str(item[0])),
                enumerate(records_to_save),
            )
        )
    assert all(r["commit_status"] == "committed" for r in results)
    with database[1]() as db:
        n = db.scalar(select(EnrichedNEO))
        assert set(n.data_sources) == {"SBDB", "MPC"}
        assert n.total_detections == 2
        assert (
            n.physical_properties["diameter_km"] == 0.011
            and n.physical_properties["albedo"] == 0.52
        )


def test_verified_export_is_complete_and_refuses_overwrite(database, tmp_path):
    from aneos_api.enrichment_store import latest_projection

    enrich(
        database,
        [datum(), datum("MPC", physical_properties={"albedo": 0.52})],
        "export",
    )
    path = tmp_path / "object.json"
    with database[1]() as db:
        service = EnrichedNEOService(db)
        service.export_enriched_json("SYNTHETIC", path)
        original = path.read_bytes()
        with pytest.raises(FileExistsError):
            service.export_enriched_json("SYNTHETIC", path)
    assert path.read_bytes() == original
    data = json.loads(original)
    assert len(data["evidence_history"]) == 2
    assert (
        data["physical_properties"]["albedo"] == 0.52
        and data["artificial_probability"] is None
    )
    assert data["projection_receipt"]["projection"]["selected_source_assertions"]


def test_typed_sbdb_source_native_units_sigmas_epoch_and_export_survive_sql(
    database, tmp_path
):
    from tests.test_atomic_physical_evidence import acquired, diameter
    from aneos_core.data.models import NEOData
    from aneos_core.analysis.screening import ScreeningAnalysisService

    neo = acquired([diameter()])
    expected = ScreeningAnalysisService().analyze_neo(neo).to_payload()
    packet = {
        "designation": "SYNTHETIC",
        "data_source": "SBDB",
        "snapshot_schema": "neo-data-v1",
        "neo_data": neo.to_dict(),
    }
    assert enrich(database, [packet], "typed")["commit_status"] == "committed"
    with database[1]() as db:
        result = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        restored = NEOData.from_dict(result["neo_data"])
        assert restored.physical_properties.diameter_uncertainty == 0.002
        assert restored.orbital_elements.native_epoch.time_scale == "TDB"
        # Additional reconciliation provenance changes the evidence id, not raw ranking physics.
        actual = ScreeningAnalysisService().analyze_neo(restored).to_payload()
        assert actual["screening_score"] == expected["screening_score"]
        assert actual["artificial_probability"] is None
        EnrichedNEOService(db).export_enriched_json(
            "SYNTHETIC", tmp_path / "typed.json"
        )
    exported = json.loads((tmp_path / "typed.json").read_text())
    native = exported["evidence_history"][0]["payload"]["neo_data"][
        "evidence_metadata"
    ]["physical_evidence"]["records"][0]
    assert native["native"]["sigma"] == ".002"


def test_existing_legacy_row_is_not_mutated_by_read_and_origin_is_not_current_probability(
    database,
):
    with database[1]() as db:
        db.add(
            EnrichedNEO(
                designation="SYNTHETIC",
                artificial_probability=0.03,
                data_sources=["NASA_SBDB"],
                completeness_score=1.0,
            )
        )
        db.commit()
    with database[1]() as db:
        result = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert result["artificial_probability"] is None
        assert result["legacy_origin_output"]["value"] == 0.03
        assert result["storage_status"] == "legacy_unverified"
        assert db.scalar(select(EnrichedNEO)).artificial_probability == 0.03


def test_incomplete_high_priority_orbit_does_not_discard_a_complete_secondary_solution(
    database,
):
    enrich(
        database,
        [
            datum(orbital_elements={"a": 1.2}),
            datum("Horizons", orbital_elements={"a": 1.3, "e": 0.2, "i": 2.0}),
        ],
        "mixed",
    )
    with database[1]() as db:
        data = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert data["orbital_elements"]["semi_major_axis"] == 1.3
        assert data["orbital_elements_source"] == "Horizons"
        assert (
            data["neo_data"]["evidence_metadata"]["conflicts"]["orbital_elements:SBDB"][
                "semi_major_axis"
            ]
            == 1.2
        )
        assert data["physical_properties"]["diameter_km"] == 0.011


def test_repeated_source_payloads_are_labeled_not_independent_data(database):
    enrich(database, [datum()], "one")
    enrich(database, [datum()], "two")
    with database[1]() as db:
        history = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")[
            "evidence_history"
        ]
        assert history[0]["exact_payload_repeat_of"] is None
        assert history[1]["exact_payload_repeat_of"] == history[0]["assertion_id"]


def test_actual_legacy_polling_write_and_detail_renderer_use_real_ledger(
    database, monkeypatch
):
    import asyncio, io
    from rich.console import Console
    from aneos_menu import ANEOSMenu
    import aneos_api.database as module

    manager = module.DatabaseManager()
    manager.engine = database[0]
    manager.SessionLocal = database[1]
    monkeypatch.setattr(module, "db_manager", manager)
    output = io.StringIO()
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = Console(file=output, width=100, color_system=None)
    menu.console.input = lambda *args: ""
    result = asyncio.run(
        menu._enrich_database_with_polling_results(
            [datum(), datum("MPC", physical_properties={"albedo": 0.52})],
            ["SBDB", "MPC"],
            "synthetic",
        )
    )
    assert result["commit_status"] == "committed"
    with database[1]() as db:
        service = EnrichedNEOService(db)
        menu._show_enriched_neo_details(service, "SYNTHETIC")
        assert len(service.get_enriched_neo("SYNTHETIC")["evidence_history"]) == 2
    text_output = output.getvalue()
    assert (
        "not calibrated" in text_output
        and "Dated source assertion history" in text_output
    )
    assert "0.52" in text_output
    assert "could not be verified" not in text_output


def test_schema_rejects_incompatible_existing_tables_without_replacement(tmp_path):
    from aneos_api.enrichment_store import initialize_schema, schema_ready

    engine = create_engine("sqlite:///" + str(tmp_path / "incompatible.sqlite"))
    try:
        with engine.begin() as c:
            c.exec_driver_sql(
                "CREATE TABLE neo_source_assertions_v1 (sequence INTEGER PRIMARY KEY, marker TEXT)"
            )
        with pytest.raises(RuntimeError, match="explicit migration"):
            initialize_schema(engine)
        with engine.connect() as c:
            assert [
                r[1]
                for r in c.exec_driver_sql(
                    "PRAGMA table_info(neo_source_assertions_v1)"
                )
            ] == ["sequence", "marker"]
        assert not schema_ready(engine)
    finally:
        engine.dispose()


def test_memory_database_cannot_claim_restart_durable_enrichment():
    from aneos_api.enrichment_store import initialize_schema, schema_ready

    engine = create_engine("sqlite:///:memory:")
    try:
        assert not schema_ready(engine)
        with pytest.raises(RuntimeError, match="File-SQLite"):
            initialize_schema(engine)
    finally:
        engine.dispose()


def test_repeated_concurrent_identical_assertion_is_single_durable_write(database):
    from concurrent.futures import ThreadPoolExecutor
    from aneos_api.enrichment_store import records

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: enrich(database, [datum()], "same-concurrent"), range(2))
        )
    assert all(r["commit_status"] == "committed" for r in results)
    assert sum(r["inserted_assertions"] for r in results) == 1
    with database[1]() as db:
        assert len(records(db, "SYNTHETIC")) == 1


@pytest.mark.parametrize(
    "rows,session,expected",
    [(None, "x", "INVALID_ENRICHMENT_INPUT"), ([datum()], "", "INVALID_SESSION_ID")],
)
def test_invalid_operation_schema_is_reported_before_any_write(
    database, rows, session, expected
):
    with database[1]() as db:
        result = EnrichedNEOService(db).enrich_neo_data(rows, session)
        assert (
            result["error"] == expected and result["commit_status"] == "not_committed"
        )


def test_legacy_aggregate_without_source_slots_remains_accessible_after_new_source(
    database,
):
    with database[1]() as db:
        db.add(
            EnrichedNEO(
                designation="SYNTHETIC",
                physical_properties={"diameter_km": 0.011},
                data_sources=["unknown legacy"],
                polling_sessions=["before"],
                artificial_probability=0.03,
            )
        )
        db.commit()
    assert (
        enrich(database, [datum("MPC", physical_properties={"albedo": 0.52})], "after")[
            "commit_status"
        ]
        == "committed"
    )
    with database[1]() as db:
        row = db.scalar(select(EnrichedNEO))
        assert set(row.polling_sessions) == {"before", "after"}
        assert row.physical_properties["diameter_km"] == 0.011
        assert row.physical_properties["albedo"] == 0.52
        history = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")[
            "evidence_history"
        ]
        assert any(
            r["kind"] == "legacy_projection"
            and r["payload"]["artificial_probability"] == 0.03
            for r in history
        )


def test_encounter_source_alternatives_are_preserved_and_exact_duplicates_labeled(
    database,
):
    event = {
        "designation": "SYNTHETIC",
        "epoch_jd_tdb": 2451545.0,
        "time_scale": "TDB",
        "body": "Earth",
        "dist": 0.01,
    }
    other = dict(event, dist=0.02)
    enrich(
        database,
        [
            datum(close_approaches=[event]),
            datum("MPC", close_approaches=[event, other]),
        ],
        "events",
    )
    with database[1]() as db:
        data = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert len(data["close_approaches"]) == 2
        assert (
            len(data["neo_data"]["evidence_metadata"]["encounter_source_assertions"])
            == 3
        )


def test_latest_projection_is_served_over_stale_mutable_compatibility_fields(database):
    enrich(database, [datum()], "one")
    with database[1]() as db:
        row = db.scalar(select(EnrichedNEO))
        row.physical_properties = {"diameter_km": 999.0}
        db.commit()
    with database[1]() as db:
        data = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert data["projection_cache_status"] == "stale_compatibility_columns"
        assert data["physical_properties"]["diameter_km"] == 0.011


@pytest.mark.parametrize(
    "helper",
    [
        "_create_new_neo",
        "_update_existing_neo",
        "_update_consolidated_data",
        "_assess_orbital_quality",
        "_calculate_completeness",
        "_calculate_completeness_from_existing",
    ],
)
def test_unsafe_legacy_write_quality_helpers_are_controlled_deprecations(
    database, helper
):
    with database[1]() as db:
        with pytest.raises(NotImplementedError):
            getattr(EnrichedNEOService(db), helper)({})


def test_atomic_scientific_metadata_and_alternatives_are_not_ranked_by_fake_quality(
    database,
):
    enrich(
        database,
        [
            datum(
                physical_properties={
                    "diameter_km": 0.011,
                    "reference_source": "first-study",
                }
            ),
            datum(
                "MPC",
                physical_properties={
                    "diameter_km": 0.02,
                    "diameter_uncertainty": 0.003,
                    "reference_source": "second-study",
                    "albedo": 0.52,
                },
            ),
        ],
        "conflicts",
    )
    with database[1]() as db:
        result = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert result["data_quality_score"] is None
        assert (
            result["neo_data"]["evidence_metadata"]["conflicts"][
                "physical_properties.diameter_km:MPC"
            ]["value"]
            == 0.02
        )
        assert (
            result["neo_data"]["evidence_metadata"]["physical_selection"][
                "diameter_km"
            ]["reference_source"]
            == "first-study"
        )
        assert result["physical_properties"]["reference_source"] is None


def test_ledger_integrity_failure_is_not_served_as_verified_evidence(database):
    from aneos_api.enrichment_store import latest_projection

    enrich(database, [datum()], "integrity")
    # Adversarial edit on this test's synthetic DB only; no live protections weakened.
    with database[0].begin() as c:
        c.exec_driver_sql("DROP TRIGGER neo_source_assertions_v1_no_update")
        c.exec_driver_sql("UPDATE neo_source_assertions_v1 SET payload_json='{}'")
    from aneos_api.enrichment_store import initialize_schema

    initialize_schema(database[0])
    with database[1]() as db:
        with pytest.raises(ValueError, match="fingerprint"):
            latest_projection(db, "SYNTHETIC")
        with pytest.raises(ValueError):
            EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")


def test_export_absent_object_does_not_create_placeholder_file(database, tmp_path):
    with database[1]() as db:
        with pytest.raises(ValueError, match="No object"):
            EnrichedNEOService(db).export_enriched_json(
                "ABSENT", tmp_path / "none.json"
            )
    assert not (tmp_path / "none.json").exists()


def test_changed_selection_policy_creates_new_projection_not_new_observations(database):
    from aneos_api.enrichment_store import ObjectProjection, records

    data = [datum(), datum("MPC", physical_properties={"diameter_km": 0.02})]
    assert enrich(database, data, "policy")["commit_status"] == "committed"
    with database[1]() as db:
        result = EnrichedNEOService(
            db, source_priority=["MPC", "SBDB"]
        ).enrich_neo_data(data, "policy")
        assert (
            result["commit_status"] == "committed"
            and result["inserted_assertions"] == 0
        )
    with database[1]() as db:
        result = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert result["physical_properties"]["diameter_km"] == 0.02
        assert len(records(db, "SYNTHETIC")) == 2
        assert len(list(db.scalars(select(ObjectProjection)))) == 2


def test_concurrent_additive_initializers_are_idempotent_and_preserve_rows(database):
    from aneos_api.enrichment_store import initialize_schema, records
    from concurrent.futures import ThreadPoolExecutor

    enrich(database, [datum()], "before-init")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: initialize_schema(database[0]), range(2)))
    with database[1]() as db:
        assert len(records(db, "SYNTHETIC")) == 1


def test_real_sql_detail_capture_has_readable_literal_provenance_and_hashes(tmp_path):
    import subprocess, sys
    from pathlib import Path
    from hashlib import sha256

    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "qa"
    command = [
        sys.executable,
        str(root / "scripts/qa_enriched_object.py"),
        "--output",
        str(directory),
    ]
    subprocess.run(command, cwd=root, capture_output=True, check=True)
    manifest = json.loads((directory / "inventory.json").read_text())
    for width in (80, 120, 160):
        text = (directory / f"detail-{width}.txt").read_text()
        assert "not calibrated" in " ".join(text.split())
        assert "Physical selection provenance" in text
        assert "SYNTHETIC primary" in text and "MPC" in text
        assert manifest["metrics"][str(width)]["max_columns"] <= width
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest
    assert not list(directory.glob("*.sqlite"))
    assert subprocess.run(command, cwd=root, capture_output=True).returncode != 0


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_process_kill_before_and_after_commit_has_atomic_recoverable_state(
    database, tmp_path, phase
):
    import os, subprocess, sys, time
    from pathlib import Path
    from aneos_api.enrichment_store import records, ObjectProjection

    root = Path(__file__).resolve().parents[1]
    marker = tmp_path / "transaction-marker"
    code = """
import json,os,sys,time
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.database import EnrichedNEOService
engine=create_engine(sys.argv[1]);db=sessionmaker(bind=engine)()
rows=json.loads(sys.argv[4]);marker=Path(sys.argv[2])
if sys.argv[3]=='before_commit':
 def interrupt():
  db.flush();marker.write_text('flushed-before-commit');time.sleep(120)
 db.commit=interrupt
result=EnrichedNEOService(db).enrich_neo_data(rows,'crash-retry')
if result.get('commit_status')!='committed':raise RuntimeError(str(result))
marker.write_text('committed');time.sleep(120)
"""
    env = dict(os.environ, ANEOS_DATABASE_URL=database[2], PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            database[2],
            str(marker),
            phase,
            json.dumps([datum()]),
        ],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while (
            not marker.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert marker.exists(), process.communicate(timeout=1)
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.communicate()
    with database[1]() as db:
        expected = 0 if phase == "before_commit" else 1
        assert len(records(db, "SYNTHETIC")) == expected
        assert len(list(db.scalars(select(ObjectProjection)))) == expected
        assert len(list(db.scalars(select(EnrichedNEO)))) == expected
    result = enrich(database, [datum()], "crash-retry")
    assert result["commit_status"] == "committed"
    with database[1]() as db:
        assert len(records(db, "SYNTHETIC")) == 1
        assert len(list(db.scalars(select(ObjectProjection)))) == 1
    read_code = """
import json,sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.database import EnrichedNEOService
with sessionmaker(bind=create_engine(sys.argv[1]))() as db:
 data=EnrichedNEOService(db).get_enriched_neo('SYNTHETIC')
 print(json.dumps({'count':len(data['evidence_history']),'diameter':data['physical_properties']['diameter_km'],'origin':data['artificial_probability'],'storage':data['storage_status']}))
"""
    fresh = subprocess.run(
        [sys.executable, "-c", read_code, database[2]],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(fresh.stdout.splitlines()[-1])
    assert observed == {
        "count": 1,
        "diameter": 0.011,
        "origin": None,
        "storage": "verified_ledger_projection",
    }


def test_legacy_postgres_initialization_is_not_blocked_by_sqlite_only_ledger(monkeypatch):
    from unittest.mock import Mock
    from aneos_api.database import DatabaseManager, Base
    from aneos_api.enrichment_store import TABLES
    manager = DatabaseManager.__new__(DatabaseManager)
    manager.engine = Mock()
    manager.engine.dialect.name = "postgresql"
    create = Mock()
    monkeypatch.setattr(Base.metadata, "create_all", create)
    assert manager.init_database() is True
    assert create.call_args.kwargs["bind"] is manager.engine
    tables = create.call_args.kwargs["tables"]
    assert all(t not in tables for t in TABLES)
    assert any(t.name == "enriched_neos" for t in tables)
    # A call-contract regression, not a live PostgreSQL validation.
