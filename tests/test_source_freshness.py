"""Read-only receipt age, typed retry plans and actual legacy menu integration."""

from datetime import datetime, timezone
from unittest.mock import Mock
import asyncio
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.database import Base, EnrichedNEO

NOW = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)


@pytest.fixture
def legacy(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "legacy.sqlite"))
    Base.metadata.create_all(engine, tables=[EnrichedNEO.__table__])
    sessions = sessionmaker(bind=engine)
    yield engine, sessions
    engine.dispose()


def test_old_naive_compatibility_dates_are_unknown_not_utc(legacy):
    from aneos_api.source_freshness import inspect_freshness

    engine, sessions = legacy
    with sessions() as db:
        db.add(
            EnrichedNEO(
                designation="SYNTHETIC", nasa_sbdb_last_update=datetime(2000, 1, 1)
            )
        )
        db.commit()
    with sessions() as db:
        report = inspect_freshness(db, ["NASA_SBDB"], checked_at=NOW)
    assert report["outdated_count"] == 1
    row = report["objects"][0]
    assert row["sources"][0]["freshness"] == "unknown"
    assert row["sources"][0]["last_adapter_read_at"] is None
    assert row["sources"][0]["legacy_timestamp"] == "2000-01-01T00:00:00"
    assert len(row["sources"]) == 5


def test_horizons_and_empty_store_require_no_compatibility_column(legacy):
    from aneos_api.source_freshness import inspect_freshness

    _, sessions = legacy
    with sessions() as db:
        report = inspect_freshness(db, ["Horizons"], checked_at=NOW)
    assert report["outdated_count"] == 0
    assert report["source_plan"]["sources"] == [
        "Horizons",
        "SBDB",
        "NEODyS",
        "MPC",
        "CAD",
    ]


@pytest.mark.parametrize(
    "literal", ["2026-11-13T11:00:00", None, "bad", "2026-11-13", True]
)
def test_unknown_time_is_not_assigned_utc(literal):
    from aneos_core.data.freshness import source_freshness

    report = source_freshness(
        "SBDB",
        [{"status": "available", "read_at": literal, "attempt_at": literal}],
        checked_at=NOW,
    )
    assert report["freshness"] == "unknown"
    assert report["retry_eligible"] is True


def test_old_source_read_does_not_become_fresh_from_local_receipt():
    from aneos_core.data.freshness import source_freshness

    report = source_freshness(
        "SBDB",
        [
            {
                "status": "available",
                "read_at": "2000-01-01T00:00:00+00:00",
                "attempt_at": "2000-01-01T00:00:00Z",
                "ledger_recorded_at": NOW.isoformat(),
            }
        ],
        checked_at=NOW,
    )
    assert report["freshness"] == "stale_adapter_read"
    assert report["retry_eligible"] is True


def test_future_source_clock_is_not_fresh():
    from aneos_core.data.freshness import source_freshness

    report = source_freshness(
        "SBDB",
        [{"status": "available", "read_at": "2027-01-01T00:00:00Z"}],
        checked_at=NOW,
    )
    assert report["freshness"] == "clock_ahead"
    assert report["needs_review"] is True


@pytest.mark.parametrize("status", ["available", "missing"])
@pytest.mark.parametrize("offset", ["+00:00", "+02:00"])
def test_recent_valid_adapter_reads_and_explicit_empty_responses(status, offset):
    from aneos_core.data.freshness import source_freshness

    report = source_freshness(
        "nasa_sbdb",
        [
            {
                "status": status,
                "read_at": f"2026-11-13T11:30:00{offset}",
                "attempt_at": f"2026-11-13T11:30:00{offset}",
            }
        ],
        checked_at=NOW,
    )
    assert report["freshness"] == "recent_adapter_read"
    assert report["retry_eligible"] is False
    assert report["latest_outcome"] == status


@pytest.mark.parametrize(
    "status", ["unavailable", "invalid", "ambiguous", "not_supported"]
)
def test_failures_defer_then_expire_without_overwriting_previous_data(status):
    from aneos_core.data.freshness import source_freshness

    events = [
        {
            "status": "available",
            "read_at": "2026-11-13T09:00:00Z",
            "attempt_at": "2026-11-13T09:00:00Z",
        },
        {"status": status, "attempt_at": "2026-11-13T11:30:00Z"},
    ]
    report = source_freshness("MPC", events, checked_at=NOW)
    assert report["last_adapter_read_at"] == events[0]["read_at"]
    assert report["needs_review"] is True
    assert report["retry_eligible"] is False
    assert report["retry_reason"] == "failed_attempt_cooldown"
    report = source_freshness("MPC", events, checked_at=NOW.replace(hour=13))
    assert report["retry_eligible"] is True


def test_unsupported_is_explicit_not_a_missing_compatibility_column():
    from aneos_core.data.freshness import source_freshness

    report = source_freshness("SENTRY", [], checked_at=NOW)
    assert report["source"] == "Sentry"
    assert report["retry_eligible"] is False
    assert report["retry_reason"] == "unsupported_source"


def test_append_order_and_clock_regression_are_not_sorted_away():
    from aneos_core.data.freshness import source_freshness

    report = source_freshness(
        "CAD",
        [
            {
                "status": "available",
                "read_at": "2026-11-13T10:00:00Z",
                "attempt_at": "2026-11-13T10:00:00Z",
            },
            {
                "status": "missing",
                "read_at": "2026-11-13T09:00:00Z",
                "attempt_at": "2026-11-13T09:00:00Z",
            },
        ],
        checked_at=NOW,
    )
    assert report["clock_regression"] is True
    assert report["latest_outcome"] == "missing"
    assert report["last_adapter_read_at"] == "2026-11-13T09:00:00Z"


@pytest.mark.parametrize(
    "bad",
    [
        "2026-11-13T00:00:00+99:00",
        "2026-99-99T00:00:00Z",
        "2026-11-13T12:00:00.1234567Z",
        "9999-12-31T23:59:59-12:00",
    ],
)
def test_invalid_or_unrepresentable_clocks_are_unknown(bad):
    from aneos_core.data.freshness import instant

    assert instant(bad) is None


@pytest.mark.parametrize("bad", [None, "2026-11-13", datetime(2026, 11, 13)])
def test_caller_clock_must_be_explicit(bad):
    from aneos_core.data.freshness import check_clock

    with pytest.raises(ValueError):
        check_clock(bad)


@pytest.fixture
def repository(tmp_path):
    from aneos_api.acquisition_repository import SQLEvidenceRepository

    engine = create_engine("sqlite:///" + str(tmp_path / "evidence.sqlite"))
    repo = SQLEvidenceRepository(engine)
    repo.prepare_acquisition()
    yield repo
    engine.dispose()


def bundle(
    statuses=None,
    *,
    stamp="2026-11-13T11:30:00+00:00",
    designation="SYNTHETIC",
    source_plan=None,
    read_at=None,
):
    from aneos_core.data.acquisition import plan, outcome, build_bundle
    from aneos_core.data.models import NEOData, OrbitalElements

    p = source_plan or plan([])
    statuses = statuses or {s: "available" for s in p["sources"]}
    outcomes = []
    for source in p["sources"]:
        status = statuses.get(source, "missing")
        neo = None
        if status == "available":
            date = datetime.fromisoformat(read_at or stamp)
            neo = NEOData(
                designation,
                orbital_elements=OrbitalElements(
                    semi_major_axis=1.23, eccentricity=0.2, inclination=1.49
                ),
                sources_used=[source],
                fetched_at=date,
                created_at=date,
                updated_at=date,
            )
        outcomes.append(
            outcome(
                source,
                designation,
                status=status,
                neo=neo,
                started_at=stamp,
                completed_at=stamp,
            )
        )
    return build_bundle(
        designation,
        p,
        outcomes,
        started_at=stamp,
        software={
            "application_version": "2.2.0.dev0",
            "fixture": "synthetic_not_observations",
        },
    )


def test_every_installed_source_and_alias_uses_verified_sql_receipts(repository):
    from aneos_api.source_freshness import inspect_freshness

    b = bundle()
    repository.commit_acquisition(b)
    with repository.sessions() as db:
        report = inspect_freshness(
            db, ["NASA_SBDB", "neodys", "MPC", "horizons", "NASA_CAD"], checked_at=NOW
        )
    assert report["outdated_count"] == 0
    assert len(report["objects"][0]["sources"]) == 5
    for source in report["objects"][0]["sources"]:
        assert source["freshness"] == "recent_adapter_read"
        assert source["last_attempt_reference"]["query_id"] == b["query_id"]


def test_failed_query_without_neo_is_reviewed_and_cooldown_blocks_whole_plan(
    repository,
):
    from aneos_api.source_freshness import inspect_freshness

    b = bundle({s: "unavailable" for s in ("SBDB", "NEODyS", "MPC", "Horizons", "CAD")})
    repository.commit_acquisition(b)
    with repository.sessions() as db:
        assert db.query(EnrichedNEO).count() == 0
        report = inspect_freshness(db, [], checked_at=NOW)
    assert report["checked_count"] == 1
    assert report["outdated_count"] == 1
    assert report["eligible_count"] == 0
    assert len(report["objects"][0]["blocked_sources"]) == 5


def test_old_frozen_query_rehydration_does_not_slide_adapter_age(repository):
    from aneos_api.source_freshness import inspect_freshness

    b = bundle(stamp="2000-01-01T00:00:00+00:00")
    first = repository.commit_acquisition(b)
    assert repository.commit_acquisition(b) == first
    with repository.sessions() as db:
        report = inspect_freshness(db, [], checked_at=NOW)
    assert report["eligible_count"] == 1
    assert all(
        s["freshness"] == "stale_adapter_read" for s in report["objects"][0]["sources"]
    )


def test_typed_intake_preserves_old_read_despite_new_append(repository):
    from aneos_api.source_freshness import inspect_freshness
    from aneos_api.database import EnrichedNEOService

    b = bundle(stamp="2000-01-01T00:00:00+00:00")
    neo = b["outcomes"][0]["neo_data"]
    with repository.sessions() as db:
        result = EnrichedNEOService(db).enrich_neo_data(
            [
                {
                    "designation": "SYNTHETIC",
                    "data_source": "SBDB",
                    "snapshot_schema": "neo-data-v1",
                    "neo_data": neo,
                }
            ],
            "typed-intake",
        )
        assert result.get("error") is None
    with repository.sessions() as db:
        report = inspect_freshness(db, [], checked_at=NOW)
    sbdb = report["objects"][0]["sources"][0]
    assert sbdb["freshness"] == "stale_adapter_read"
    assert sbdb["last_ledger_recorded_at"] != sbdb["last_adapter_read_at"]


def test_sql_read_only_limit_missing_and_unsupported_sources(legacy):
    from aneos_api.source_freshness import inspect_freshness

    engine, sessions = legacy
    with sessions() as db:
        db.add_all([EnrichedNEO(designation=f"SYNTHETIC{i:03}") for i in range(51)])
        db.commit()
    before = __import__("pathlib").Path(engine.url.database).read_bytes()
    with sessions() as db:
        report = inspect_freshness(db, ["SENTRY"], checked_at=NOW)
    assert report["truncated"] is True
    assert report["checked_count"] == 50
    assert report["outdated_count"] == 50
    assert report["source_plan"]["sources"][0] == "Sentry"
    with sessions() as db:
        report = inspect_freshness(
            db, [], checked_at=NOW, designations=["SYNTHETIC000", "ABSENT"]
        )
    assert report["requested_designations_not_found"] == ["ABSENT"]
    assert __import__("pathlib").Path(engine.url.database).read_bytes() == before


@pytest.mark.parametrize("limit", [0, 51, True, "50"])
def test_invalid_object_limits_fail_closed(legacy, limit):
    from aneos_api.source_freshness import inspect_freshness

    with legacy[1]() as db, pytest.raises(ValueError):
        inspect_freshness(db, [], checked_at=NOW, limit=limit)


@pytest.mark.parametrize("names", ["SYNTHETIC", [None], ["x"] * 51])
def test_invalid_designation_lists_fail_closed(legacy, names):
    from aneos_api.source_freshness import inspect_freshness

    with legacy[1]() as db, pytest.raises(ValueError):
        inspect_freshness(db, [], checked_at=NOW, designations=names)


def test_incompatible_evidence_schema_is_not_hidden_as_fresh(legacy):
    from aneos_api.source_freshness import inspect_freshness
    from sqlalchemy import text

    with legacy[0].begin() as c:
        c.execute(text("CREATE TABLE neo_source_assertions_v1 (designation TEXT)"))
    with legacy[1]() as db, pytest.raises(ValueError, match="schema"):
        inspect_freshness(db, [], checked_at=NOW)


def test_actual_menu_empty_positive_and_session_closure(legacy, monkeypatch, capsys):
    from aneos_menu import ANEOSMenu
    from aneos_api.database import db_manager

    db = legacy[1]()
    close = Mock(wraps=db.close)
    monkeypatch.setattr(db, "close", close)
    monkeypatch.setattr(db_manager, "get_db", lambda: db)
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = None
    result = asyncio.run(menu._check_outdated_database_entries(["HORIZONS"], "1m"))
    assert result["checked_count"] == 0
    assert close.call_count == 1
    assert "Objects checked: 0" in capsys.readouterr().out


def test_actual_menu_error_closes_session_and_never_prints_raw_exception(
    legacy, monkeypatch, capsys
):
    from aneos_menu import ANEOSMenu
    from aneos_api.database import db_manager
    import aneos_api.source_freshness as module

    db = legacy[1]()
    close = Mock(wraps=db.close)
    monkeypatch.setattr(db, "close", close)
    monkeypatch.setattr(db_manager, "get_db", lambda: db)
    monkeypatch.setattr(
        module, "inspect_freshness", Mock(side_effect=ValueError("secret-fixture"))
    )
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = None
    result = asyncio.run(menu._check_outdated_database_entries(["NASA_SBDB"], "1m"))
    assert result["error"] == "SOURCE_FRESHNESS_UNAVAILABLE"
    assert close.call_count == 1
    assert "secret-fixture" not in capsys.readouterr().out


def install_synthetic_fetcher(monkeypatch, tmp_path, mode):
    """Actual public fetch/commit path; only its provider collector is synthetic."""
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.cache import CacheManager
    from aneos_core.data.models import NEOData
    from aneos_core.utils.errors import DataSourceUnavailableError

    original = DataFetcher
    instances = []

    def factory(**kwargs):
        f = original(cache_manager=CacheManager(tmp_path / "refresh-cache"), **kwargs)

        def collect(designation, source_plan=None):
            status = "available" if mode == "full" else "unavailable"
            states = {s: status for s in source_plan["sources"]}
            if mode == "partial":
                states["SBDB"] = "available"
            b = bundle(states, designation=designation, source_plan=source_plan)
            if b["neo_data"] is None:
                exc = DataSourceUnavailableError("synthetic no measurements")
                exc.acquisition_bundle = b
                raise exc
            n = NEOData.from_dict(b["neo_data"])
            n._acquisition_bundle = b
            return n

        f._fetch_from_all_sources = Mock(side_effect=collect)
        f.close = Mock(wraps=f.close)
        instances.append(f)
        return f

    monkeypatch.setattr("aneos_core.data.fetcher.DataFetcher", factory)
    return instances


@pytest.mark.parametrize(
    "mode,expected,count",
    [
        ("full", "recorded_full_plan", 5),
        ("partial", "recorded_partial", 1),
        ("none", "recorded_no_data", 0),
    ],
)
def test_explicit_refresh_uses_public_acquisition_and_verifies_sql_commit(
    repository, tmp_path, monkeypatch, mode, expected, count
):
    from aneos_api.source_freshness import refresh_objects, inspect_freshness

    repository.commit_acquisition(bundle(stamp="2000-01-01T00:00:00+00:00"))
    instances = install_synthetic_fetcher(monkeypatch, tmp_path, mode)
    report = refresh_objects(repository.engine, ["SYNTHETIC"], [], checked_at=NOW)
    result = report["results"][0]
    assert result["status"] == expected
    assert result["coverage"]["contributing_sources"] == count
    assert report["recorded_count"] == 1
    assert repository.load_acquisition(result["query_id"]) is not None
    assert instances[0]._fetch_from_all_sources.call_count == 1
    assert instances[0].close.call_count == 1
    with repository.sessions() as db:
        after = inspect_freshness(db, [], checked_at=NOW)
    if count == 5:
        assert after["outdated_count"] == 0
    else:
        assert after["eligible_count"] == 0


def test_latest_failed_attempt_blocks_forged_menu_refresh_selection(
    repository, tmp_path, monkeypatch
):
    from aneos_api.source_freshness import refresh_objects

    b = bundle({s: "unavailable" for s in ("SBDB", "NEODyS", "MPC", "Horizons", "CAD")})
    repository.commit_acquisition(b)
    constructor = Mock(side_effect=AssertionError("must not initialize providers"))
    monkeypatch.setattr("aneos_core.data.fetcher.DataFetcher", constructor)
    report = refresh_objects(repository.engine, ["SYNTHETIC"], [], checked_at=NOW)
    assert report["results"][0]["status"] == "deferred"
    assert report["recorded_count"] == 0
    assert not constructor.called


def test_refresh_no_longer_fresh_records_and_absent_selection_are_not_queried(
    repository, monkeypatch
):
    from aneos_api.source_freshness import refresh_objects

    repository.commit_acquisition(bundle())
    constructor = Mock(side_effect=AssertionError("must not query"))
    monkeypatch.setattr("aneos_core.data.fetcher.DataFetcher", constructor)
    report = refresh_objects(
        repository.engine, ["SYNTHETIC", "ABSENT"], [], checked_at=NOW
    )
    assert report["results"][0]["status"] == "deferred"
    assert report["inspection"]["requested_designations_not_found"] == ["ABSENT"]
    assert not constructor.called


@pytest.mark.parametrize("names", [[], ["x"] * 11, "SYNTHETIC"])
def test_refresh_requires_small_explicit_selection(repository, names):
    from aneos_api.source_freshness import refresh_objects

    with pytest.raises(ValueError):
        refresh_objects(repository.engine, names, [], checked_at=NOW)


def test_refresh_commit_failure_is_not_announced_as_success(
    repository, tmp_path, monkeypatch
):
    from aneos_api.source_freshness import refresh_objects
    from aneos_api.acquisition_repository import SQLEvidenceRepository

    repository.commit_acquisition(bundle(stamp="2000-01-01T00:00:00+00:00"))
    instances = install_synthetic_fetcher(monkeypatch, tmp_path, "full")
    monkeypatch.setattr(
        SQLEvidenceRepository,
        "commit_acquisition",
        Mock(side_effect=OSError("secret-fixture")),
    )
    report = refresh_objects(repository.engine, ["SYNTHETIC"], [], checked_at=NOW)
    assert report["recorded_count"] == 0
    assert report["results"][0]["status"] == "failed"
    assert "secret-fixture" not in str(report)
    assert instances[0].close.call_count == 1


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {"commit_status": "not_committed"},
        {"commit_status": "committed", "query_id": "ABSENT"},
    ],
)
def test_unverified_refresh_receipts_never_become_success(
    repository, monkeypatch, receipt
):
    from aneos_api.source_freshness import refresh_objects

    repository.commit_acquisition(bundle(stamp="2000-01-01T00:00:00+00:00"))
    f = Mock()
    f.fetch_neo_data.return_value.evidence_metadata = {
        "acquisition": {"storage_receipt": receipt}
    }
    monkeypatch.setattr("aneos_core.data.fetcher.DataFetcher", Mock(return_value=f))
    result = refresh_objects(repository.engine, ["SYNTHETIC"], [], checked_at=NOW)
    assert result["results"][0]["status"] == "failed"
    assert f.fetch_neo_data.call_args.kwargs["force_refresh"] is True
    assert f.close.call_count == 1


def test_actual_menu_refresh_integration_uses_shared_path_and_clear_outcomes(
    repository, tmp_path, monkeypatch, capsys
):
    from aneos_menu import ANEOSMenu
    from aneos_api.database import db_manager

    repository.commit_acquisition(bundle(stamp="2000-01-01T00:00:00+00:00"))
    instances = install_synthetic_fetcher(monkeypatch, tmp_path, "partial")
    monkeypatch.setattr(db_manager, "engine", repository.engine)
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = None
    result = asyncio.run(
        menu._repoll_outdated_entries(
            [{"designation": "SYNTHETIC", "retry_eligible": True}]
        )
    )
    assert result["recorded_count"] == 1
    assert result["results"][0]["status"] == "recorded_partial"
    output = capsys.readouterr().out
    assert "1/5 adapters contribute" in output
    assert "scientific completeness" not in output or "not" in output
    assert instances[0].close.call_count == 1


def test_actual_menu_refresh_empty_and_error_paths(monkeypatch, capsys):
    from aneos_menu import ANEOSMenu

    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = None
    assert asyncio.run(menu._repoll_outdated_entries([]))["recorded_count"] == 0
    result = asyncio.run(menu._repoll_outdated_entries([{}]))
    assert result["error"] == "SOURCE_REFRESH_UNAVAILABLE"
    assert "no successful refresh has been verified" in capsys.readouterr().out


@pytest.mark.parametrize("width", [80, 120, 160])
def test_real_menu_freshness_readability_and_safe_labels(legacy, monkeypatch, width):
    from aneos_menu import ANEOSMenu
    from aneos_api.database import db_manager
    from rich.console import Console
    import io

    with legacy[1]() as db:
        db.add(EnrichedNEO(designation="[red]SYNTHETIC[/red]\x1b"))
        db.commit()
    monkeypatch.setattr(db_manager, "get_db", legacy[1])
    stream = io.StringIO()
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = Console(file=stream, width=width, color_system=None)
    result = asyncio.run(
        menu._check_outdated_database_entries(["NASA_SBDB", "Horizons"], "1m")
    )
    assert result["outdated_count"] == 1
    output = stream.getvalue()
    assert "\x1b" not in output
    assert "[red]" in output
    assert "Source receipt review" in output
    assert "Unknown legacy timestamps" in output
    from rich.cells import cell_len

    assert max(map(cell_len, output.splitlines())) <= width


@pytest.mark.parametrize("attempt", [None, "bad", "2026-11-13T11:30:00"])
def test_unknown_failed_attempt_clock_does_not_bypass_cooldown(attempt):
    from aneos_core.data.freshness import source_freshness

    result = source_freshness(
        "SBDB", [{"status": "unavailable", "attempt_at": attempt}], checked_at=NOW
    )
    assert result["retry_eligible"] is False
    assert result["retry_reason"] == "unverified_attempt_clock"


def test_unrepresentable_future_retry_label_fails_closed():
    from aneos_core.data.freshness import source_freshness

    result = source_freshness(
        "SBDB",
        [{"status": "unavailable", "attempt_at": "9999-12-31T23:59:59Z"}],
        checked_at=NOW,
    )
    assert result["retry_eligible"] is False
    assert result["retry_reason"] == "attempt_clock_ahead"
    assert result["retry_after"] is None


def test_an_old_committed_receipt_cannot_masquerade_as_a_new_refresh(
    repository, monkeypatch
):
    from aneos_api.source_freshness import refresh_objects

    b = bundle(stamp="2000-01-01T00:00:00+00:00")
    receipt = repository.commit_acquisition(b)
    f = Mock()
    f.fetch_neo_data.return_value.evidence_metadata = {
        "acquisition": {"storage_receipt": receipt}
    }
    monkeypatch.setattr("aneos_core.data.fetcher.DataFetcher", Mock(return_value=f))
    report = refresh_objects(repository.engine, ["SYNTHETIC"], [], checked_at=NOW)
    assert report["recorded_count"] == 0
    assert report["results"][0]["status"] == "failed"


def test_fetcher_cleanup_failure_does_not_erase_verified_commit(
    repository, tmp_path, monkeypatch
):
    from aneos_api.source_freshness import refresh_objects

    repository.commit_acquisition(bundle(stamp="2000-01-01T00:00:00+00:00"))
    instances = install_synthetic_fetcher(monkeypatch, tmp_path, "full")
    # Inject the failure at construction, retaining the same real acquisition path.
    factory = __import__(
        "aneos_core.data.fetcher", fromlist=["DataFetcher"]
    ).DataFetcher

    def failing_close(**kwargs):
        f = factory(**kwargs)
        old_close = f.close

        def close():
            old_close()
            raise OSError("synthetic cleanup failure")

        f.close = close
        return f

    monkeypatch.setattr("aneos_core.data.fetcher.DataFetcher", failing_close)
    report = refresh_objects(repository.engine, ["SYNTHETIC"], [], checked_at=NOW)
    assert report["recorded_count"] == 1
    assert report["cleanup_status"] == "failed"
    assert instances[0]._fetch_from_all_sources.called


def test_empty_and_truncated_views_and_rich_refresh(capsys):
    from aneos_core.analysis.freshness_view import render_freshness, render_refresh
    from rich.console import Console
    import io

    report = {
        "checked_count": 0,
        "outdated_count": 0,
        "eligible_count": 0,
        "objects": [],
        "truncated": True,
        "object_limit": 50,
    }
    render_freshness(report)
    assert "not reviewed" in capsys.readouterr().out
    output = io.StringIO()
    console = Console(file=output, width=80, color_system=None)
    render_freshness(report, console)
    result = {
        "recorded_count": 0,
        "results": [
            {
                "designation": "SYNTHETIC",
                "status": "deferred",
                "blocked_sources": ["Horizons"],
            }
        ],
        "cleanup_status": "failed",
        "limits": "No new scientific observations established.",
    }
    render_refresh(result, console)
    text = output.getvalue()
    assert "not reviewed" in text and "Horizons" in text and "Cleanup failed" in text


def test_actual_menu_rich_error_paths_close_and_do_not_claim_success(
    legacy, monkeypatch
):
    import io
    from rich.console import Console
    from aneos_menu import ANEOSMenu
    from aneos_api.database import db_manager
    import aneos_api.source_freshness as module

    db = legacy[1]()
    close = Mock(wraps=db.close)
    monkeypatch.setattr(db, "close", close)
    monkeypatch.setattr(db_manager, "get_db", lambda: db)
    monkeypatch.setattr(
        module, "inspect_freshness", Mock(side_effect=OSError("secret-fixture"))
    )
    menu = ANEOSMenu.__new__(ANEOSMenu)
    stream = io.StringIO()
    menu.console = Console(file=stream, width=80, color_system=None)
    assert asyncio.run(menu._check_outdated_database_entries([], ""))["error"]
    assert asyncio.run(menu._repoll_outdated_entries([{}]))["error"]
    assert close.call_count == 1
    assert "secret-fixture" not in stream.getvalue()


def test_qa_artifact_capture_has_all_real_menu_widths_and_hashes(tmp_path):
    from scripts.qa_source_freshness import capture_all
    from hashlib import sha256

    directory = tmp_path / "captures"
    manifest = capture_all(directory)
    assert len(manifest["widths"]) == 9
    for name, entry in manifest["widths"].items():
        assert entry["max_rendered_cells"] <= entry["terminal_width"]
    for path, digest in manifest["artifact_sha256"].items():
        assert sha256((directory / path).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        capture_all(directory)


def test_shared_capture_honors_actual_svg_width_even_under_dumb_terminal(
    tmp_path, monkeypatch
):
    from scripts.qa_menu_entry import capture
    from rich.panel import Panel
    import re

    monkeypatch.setenv("TERM", "dumb")

    class Menu:
        def render(self):
            self.console.print(Panel("SYNTHETIC " * 25))

    menu = Menu()
    observed = []
    for width in (80, 120, 160):
        path = tmp_path / f"width-{width}"
        capture(menu, width, path, method="render")
        svg = path.with_suffix(".svg").read_text()
        view_width = float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))
        assert view_width == __import__("math").ceil(width * 12.2 + 16) + 2
        assert menu.console.width == width
        from html.parser import HTMLParser

        class CodeText(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_code = False
                self.parts = []

            def handle_starttag(self, tag, attrs):
                if tag == "code":
                    self.in_code = True

            def handle_endtag(self, tag):
                if tag == "code":
                    self.in_code = False

            def handle_data(self, data):
                if self.in_code:
                    self.parts.append(data)

        parsed = CodeText()
        parsed.feed(path.with_suffix(".html").read_text())
        assert "".join(parsed.parts) == path.with_suffix(".txt").read_text()
        observed.append(view_width)
    assert len(set(observed)) == 3


def test_failed_fetch_without_commit_receipt_is_not_recorded(repository, monkeypatch):
    from aneos_api.source_freshness import refresh_objects
    from aneos_core.utils.errors import DataSourceUnavailableError

    repository.commit_acquisition(bundle(stamp="2000-01-01T00:00:00+00:00"))
    f = Mock()
    f.fetch_neo_data.side_effect = DataSourceUnavailableError("synthetic no receipt")
    monkeypatch.setattr("aneos_core.data.fetcher.DataFetcher", Mock(return_value=f))
    report = refresh_objects(repository.engine, ["SYNTHETIC"], [], checked_at=NOW)
    assert report["recorded_count"] == 0
    assert report["results"][0]["status"] == "failed"
    assert f.close.call_count == 1


def test_present_but_truncated_requested_object_is_not_reported_absent(legacy):
    from aneos_api.source_freshness import inspect_freshness

    with legacy[1]() as db:
        db.add_all(
            [
                EnrichedNEO(designation="SYNTHETIC_A"),
                EnrichedNEO(designation="SYNTHETIC_B"),
            ]
        )
        db.commit()
    with legacy[1]() as db:
        report = inspect_freshness(
            db,
            [],
            checked_at=NOW,
            limit=1,
            designations=["SYNTHETIC_A", "SYNTHETIC_B", "ABSENT"],
        )
    assert report["truncated"] is True
    assert report["requested_designations_not_found"] == ["ABSENT"]
    assert report["requested_designations_not_reviewed"] == ["SYNTHETIC_B"]
