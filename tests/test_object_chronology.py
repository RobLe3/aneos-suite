"""Calendar labels, astronomical epochs and local receipts have different roles."""

from datetime import datetime, timezone
from unittest.mock import Mock
import json
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from aneos_api.database import Base, EnrichedNEO, EnrichedNEOService
from aneos_api.enrichment_store import initialize_schema
from aneos_core.data.models import NEOData, OrbitalElements


@pytest.fixture
def database(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "dates.sqlite"))
    Base.metadata.create_all(engine)
    initialize_schema(engine)
    yield engine, sessionmaker(bind=engine)
    engine.dispose()


def row():
    n = NEOData(
        "SYNTHETIC",
        orbital_elements=OrbitalElements(
            semi_major_axis=1.23, eccentricity=0.2, inclination=1.49
        ),
        first_observation=datetime(1998, 5, 30),
        last_observation=datetime(2024, 5, 6),
        fetched_at=datetime(2026, 11, 13, 7, tzinfo=timezone.utc),
        evidence_metadata={
            "native_observation_window": {
                "first_observation": {
                    "value": "1998-05-30",
                    "time_scale": "not_declared",
                    "basis": "catalogue_calendar_label_not_instant",
                },
                "last_observation": {
                    "value": "2024-05-06",
                    "time_scale": "not_declared",
                    "basis": "catalogue_calendar_label_not_instant",
                },
            }
        },
    )
    return {
        "designation": "SYNTHETIC",
        "data_source": "SBDB",
        "snapshot_schema": "neo-data-v1",
        "neo_data": n.to_dict(),
    }


def test_storage_date_is_never_exported_as_discovery(database, tmp_path):
    engine, sessions = database
    with sessions() as db:
        EnrichedNEOService(db).enrich_neo_data([row()], "date-control")
    with sessions() as db:
        service = EnrichedNEOService(db)
        x = service.get_enriched_neo("SYNTHETIC")
        assert x["first_discovered"] is None
        assert x["discovery_date"] is None
        assert x["first_ledger_recorded_at"] == x["evidence_history"][0]["recorded_at"]
        assert (
            datetime.fromisoformat(x["first_ledger_recorded_at"])
            .utcoffset()
            .total_seconds()
            == 0
        )
        service.export_enriched_json("SYNTHETIC", tmp_path / "dates.json")
        record = db.scalar(
            select(EnrichedNEO).where(EnrichedNEO.designation == "SYNTHETIC")
        )
        assert record.first_discovered is None
    exported = json.loads((tmp_path / "dates.json").read_text())
    assert exported["first_discovered"] is None
    assert exported["neo_data"]["first_observation"] == "1998-05-30T00:00:00"
    assert exported["chronology"]["schema_version"] == "object-chronology-v1"


def test_legacy_ambiguous_dates_are_preserved_but_not_relabelled_utc(database):
    engine, sessions = database
    with sessions() as db:
        db.add(
            EnrichedNEO(
                designation="LEGACY",
                first_discovered=datetime(2020, 1, 2, 3, 4),
                last_updated=datetime(2021, 2, 3, 4, 5),
            )
        )
        db.commit()
    with sessions() as db:
        x = EnrichedNEOService(db).get_enriched_neo("LEGACY")
        assert x["first_discovered"] is None and x["discovery_date"] is None
        assert x["first_ledger_recorded_at"] is None
        legacy = x["chronology"]["legacy_dates"]["first_discovered"]
        assert legacy["value"] == "2020-01-02T03:04:00"
        assert legacy["status"] == "semantics_and_time_scale_unverified"
        assert legacy["time_scale"] == "not_declared"
        assert db.scalar(
            select(EnrichedNEO).where(EnrichedNEO.designation == "LEGACY")
        ).first_discovered == datetime(2020, 1, 2, 3, 4)


@pytest.mark.parametrize(
    "date,status,value",
    [
        ("1998-May-31", "available", "1998-05-31"),
        (None, "not_available", None),
        ("2026-Feb-29", "invalid", None),
        ("1998-05-31", "invalid", None),
        ("1998-May-31T00:00:00Z", "invalid", None),
    ],
)
def test_source_discovery_is_a_sourced_calendar_label_not_an_instant(
    date, status, value
):
    from aneos_core.data.chronology import build_sbdb_discovery, verify_discovery

    e = build_sbdb_discovery(
        {"date": date, "ref": "SYNTHETIC_REF"},
        {"requested_designation": "SYNTHETIC", "source_response_sha256": "a" * 64},
    )
    assert e["status"] == status
    assert (e["normalized"]["calendar_date"] if e["normalized"] else None) == value
    assert verify_discovery(e) == e
    assert e["native"]["date"] == date
    if e["normalized"]:
        assert e["normalized"]["time_scale"] == "not_declared"
        assert e["normalized"]["precision"] == "day"


def test_sbdb_requests_discovery_and_retains_its_native_provenance(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock
    from aneos_core.config.settings import APIConfig
    from aneos_core.data.sources.sbdb import SBDBSource
    from aneos_core.data.acquisition import adapter_to_neo

    src = SBDBSource(APIConfig())
    payload = {
        "signature": {"version": "1.3"},
        "object": {"des": "SYNTHETIC"},
        "orbit": {
            "elements": [
                {"name": "a", "value": "1.23"},
                {"name": "e", "value": ".2"},
                {"name": "i", "value": "1.49"},
            ],
            "first_obs": "1998-05-30",
        },
        "discovery": {"date": "1998-May-31", "ref": "SYNTHETIC_REF"},
    }
    src._http_get = AsyncMock(return_value=payload)
    r = asyncio.run(src.fetch_orbital_elements("SYNTHETIC"))
    assert r.success
    assert src._http_get.call_args.args[1]["discovery"] == "true"
    n = adapter_to_neo("SBDB", "SYNTHETIC", r.data, "2026-11-13T07:00:00+00:00")
    e = n.evidence_metadata["discovery_evidence"]
    assert e["native"] == payload["discovery"]
    assert e["normalized"]["calendar_date"] == "1998-05-31"


def test_preceding_commit_review_remains_replayable_without_rewriting_assessment():
    from pathlib import Path
    from aneos_core.analysis.object_review import (
        verify_object_review,
        object_review_markdown,
    )

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/object_review_pre_chronology.json"
        ).read_text()
    )
    review = fixture["review"]
    before = json.dumps(review, sort_keys=True)
    assert verify_object_review(review) == review
    assert "Object chronology" in object_review_markdown(review)
    assert json.dumps(review, sort_keys=True) == before


def chrono(**kwargs):
    from aneos_core.data.chronology import build_chronology, source_input

    return build_chronology(
        "SYNTHETIC", [source_input(row()["neo_data"], "SBDB", "/source")], **kwargs
    )


@pytest.mark.parametrize(
    "literal,status,scale",
    [
        ("2026-11-13T07:00:00Z", "explicit_utc_offset", "UTC"),
        ("2026-11-13T07:00:00+00:00", "explicit_utc_offset", "UTC"),
        ("2026-11-13T07:00:00+02:00", "explicit_offset", "explicit_offset"),
        ("2026-11-13T07:00:00", "time_scale_unverified", "not_declared"),
        ("2026-11-13", "invalid", "not_declared"),
        ("bad", "invalid", "not_declared"),
        (None, "not_available", "not_declared"),
    ],
)
def test_adapter_receipts_keep_explicit_offset_or_uncertainty(literal, status, scale):
    from aneos_core.data.chronology import timestamp_event

    e = timestamp_event("adapter_receipt", literal, "SBDB", "/source/fetched_at")
    assert e["value"] == literal and e["status"] == status and e["time_scale"] == scale
    assert "not an observation" in e["meaning"]


@pytest.mark.parametrize(
    "value,status,precision",
    [
        ("1998-05-30", "catalogue_calendar_label", "day"),
        ("1919-??-??", "partial_calendar_label", "partial month/day"),
        ("1919-05-??", "partial_calendar_label", "partial month/day"),
        ("1919-??-15", "partial_calendar_label", "partial month/day"),
        ("1919-13-??", "invalid", "not_established"),
        ("2026-02-29", "invalid", "not_established"),
        ("0000-01-01", "invalid", "not_established"),
        ("1998-05-30T00:00:00", "time_scale_unverified", "literal timestamp precision"),
        (None, "not_available", "not_established"),
    ],
)
def test_observation_precision_is_not_upgraded_to_discovery_or_midnight_utc(
    value, status, precision
):
    from aneos_core.data.chronology import observation_event

    e = observation_event("first_observation", value, "SBDB", "/source")
    assert e["value"] == value and e["status"] == status and e["precision"] == precision
    assert e["time_scale"] == "not_declared"


def test_epochs_fit_date_and_calendar_roles_do_not_share_a_fabricated_time_axis():
    from aneos_core.data.chronology import build_chronology, source_input
    from aneos_core.data.epoch_evidence import NativeEpoch

    n = row()["neo_data"]
    n["orbital_elements"]["native_epoch"] = NativeEpoch(
        "2461200.50000000001", "JD", "TDB", "SBDB"
    ).to_dict()
    n["nongrav"] = {"covariance": {"epoch_jd_tdb": "2457098.500000000001"}}
    n["evidence_metadata"]["solution"] = {
        "schema_version": "1.3",
        "solution_date": "2024-May-07 06:44:30",
    }
    c = build_chronology("SYNTHETIC", [source_input(n, "NASA_SBDB", "/source")])
    events = {e["role"]: e for e in c["events"]}
    assert events["orbit_epoch"]["value"] == "2461200.50000000001"
    assert events["orbit_epoch"]["time_scale"] == "TDB"
    assert events["covariance_epoch"]["value"] == "2457098.500000000001"
    assert (
        events["orbit_fit_date"]["time_scale"] == "Pacific local; offset not supplied"
    )
    assert "No UTC/DST" in events["orbit_fit_date"]["meaning"]
    assert events["first_observation"]["value"] == "1998-05-30"
    assert c["discovery"]["selected"] is None


def receipt(stamp, sequence=1):
    return {
        "recorded_at": stamp,
        "sequence": sequence,
        "reference": "/history/" + str(sequence),
        "assertion_id": str(sequence),
        "kind": "source_assertion",
    }


def test_ledger_sequence_is_not_silently_sorted_by_clock():
    c = chrono(
        ledger_records=[
            receipt("2026-11-13T07:00:00+00:00"),
            receipt("2026-11-13T06:00:00+00:00", 2),
        ]
    )
    assert (
        c["local_receipts"]["first_ledger_recorded_at"] == "2026-11-13T07:00:00+00:00"
    )
    assert c["local_receipts"]["last_ledger_recorded_at"] == "2026-11-13T06:00:00+00:00"
    assert (
        c["local_receipts"]["clock_order"] == "clock_regression_sequence_order_retained"
    )


@pytest.mark.parametrize(
    "literal", ["2026-11-13T07:00:00", "2026-11-13T07:00:00+02:00", "invalid", None]
)
def test_unverified_ledger_timestamps_remain_raw_not_assumed_utc(literal):
    c = chrono(ledger_records=[receipt(literal)])
    assert c["local_receipts"]["first_ledger_recorded_at"] is None
    assert c["local_receipts"]["last_ledger_recorded_at"] is None
    assert c["local_receipts"]["unverified_count"] == 1
    assert c["local_receipts"]["records"][0]["value"] == literal


def test_different_source_discovery_claims_abstain_instead_of_first_or_last_wins():
    from aneos_core.data.chronology import (
        build_chronology,
        build_sbdb_discovery,
        source_input,
    )

    sources = []
    for index, date in enumerate(("1998-May-30", "1998-May-31")):
        n = row()["neo_data"]
        n["evidence_metadata"]["discovery_evidence"] = build_sbdb_discovery(
            {"date": date},
            {
                "requested_designation": "SYNTHETIC",
                "source_response_sha256": str(index) * 64,
            },
        )
        sources.append(
            source_input(n, "SBDB_archive_" + str(index), "/source/" + str(index))
        )
    c = build_chronology("SYNTHETIC", sources)
    assert (
        c["discovery"]["status"] == "ambiguous" and c["discovery"]["selected"] is None
    )
    assert {e["value"] for e in c["discovery"]["claims"]} == {
        "1998-05-30",
        "1998-05-31",
    }


def test_real_ledger_discovery_and_precovery_are_separate_positive_controls(database):
    from aneos_core.data.chronology import build_sbdb_discovery

    engine, sessions = database
    payload = row()
    payload["neo_data"]["evidence_metadata"]["discovery_evidence"] = (
        build_sbdb_discovery(
            {"date": "1998-May-31", "ref": "SYNTHETIC_REF"},
            {"requested_designation": "SYNTHETIC", "source_response_sha256": "a" * 64},
        )
    )
    with sessions() as db:
        EnrichedNEOService(db).enrich_neo_data([payload], "positive-source")
    with sessions() as db:
        x = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert x["discovery_date"] == "1998-05-31"
        assert x["discovery_data"]["date"] == "1998-May-31"
        assert x["first_discovered"] is None
        assert x["neo_data"]["first_observation"] == "1998-05-30T00:00:00"
        events = x["chronology"]["events"]
        assert (
            next(e for e in events if e["role"] == "first_observation")["value"]
            == "1998-05-30"
        )


def test_rejected_discovery_metadata_does_not_become_valid_evidence():
    from copy import deepcopy
    from aneos_core.data.chronology import build_sbdb_discovery, verify_discovery

    e = build_sbdb_discovery(
        {"date": "1998-May-31"},
        {"requested_designation": "SYNTHETIC", "source_response_sha256": "a" * 64},
    )
    bad = deepcopy(e)
    bad["normalized"]["calendar_date"] = "2026-11-13"
    with pytest.raises(ValueError):
        verify_discovery(bad)
    payload = row()["neo_data"]
    payload["evidence_metadata"]["discovery_evidence"] = e
    payload["designation"] = "OTHER"
    with pytest.raises(ValueError, match="object mismatch"):
        NEOData.from_dict(payload)


@pytest.mark.parametrize("width", [80, 120, 160])
def test_actual_chronology_rich_plain_and_markdown_are_readable_and_literal(
    width, capsys
):
    from io import StringIO
    from rich.console import Console
    from aneos_core.analysis.chronology_view import (
        render_chronology,
        chronology_markdown,
    )
    from aneos_core.data.chronology import build_chronology, source_input

    n = row()["neo_data"]
    n["evidence_metadata"]["solution"] = {
        "schema_version": "1.3",
        "solution_date": "<script>[x](javascript:bad)\x1b[2J",
    }
    c = build_chronology(
        "SYNTHETIC",
        [source_input(n, "SBDB", "/source")],
        [receipt("2026-11-13T07:00:00+00:00"), receipt("2026-11-13T06:00:00+00:00", 2)],
        {"first_discovered": "2020-01-02T03:04:00"},
    )
    stream = StringIO()
    render_chronology(c, console=Console(file=stream, width=width, color_system=None))
    text = stream.getvalue()
    assert all(len(line) <= width for line in text.splitlines())
    assert "\x1b" not in text and "\\x1b" in text
    assert "Compatibility" in text and "Receipt clocks regress" in text
    md = chronology_markdown(c)
    assert "<script>" not in md and "&lt;script&gt;" in md
    assert "[x](javascript:" not in md
    render_chronology(c)
    assert capsys.readouterr().out.strip() == md.strip()


@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {"requested_designation": "SYNTHETIC"},
        {"requested_designation": "SYNTHETIC", "source_response_sha256": "bad"},
    ],
)
def test_discovery_requires_object_and_response_context(context):
    from aneos_core.data.chronology import build_sbdb_discovery

    with pytest.raises(ValueError):
        build_sbdb_discovery({"date": "1998-May-31"}, context)


@pytest.mark.parametrize(
    "native", [[], False, {"date": 0}, {"date": False}, {"date": "1998-Mai-31"}]
)
def test_unsupported_discovery_section_is_preserved_without_scalar(native):
    from aneos_core.data.chronology import build_sbdb_discovery

    e = build_sbdb_discovery(
        native,
        {"requested_designation": "SYNTHETIC", "source_response_sha256": "a" * 64},
    )
    assert (
        e["status"] == "invalid" and e["normalized"] is None and e["native"] == native
    )


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        {"schema_version": "future"},
        {"schema_version": "sbdb-discovery-evidence-v1"},
    ],
)
def test_invalid_discovery_records_cannot_be_verified(record):
    from aneos_core.data.chronology import verify_discovery

    with pytest.raises(ValueError):
        verify_discovery(record)


@pytest.mark.parametrize("change", ["schema", "result", "inputs", "missing_inputs"])
def test_chronology_replay_rejects_tampered_or_unknown_data(change):
    from copy import deepcopy
    from aneos_core.data.chronology import verify_chronology

    c = deepcopy(chrono())
    if change == "schema":
        c["schema_version"] = "future"
    elif change == "result":
        c["events"][0]["value"] = "2026-11-13"
    elif change == "inputs":
        c["inputs"]["sources"][0]["designation"] = "OTHER"
    else:
        c.pop("inputs")
    with pytest.raises(ValueError):
        verify_chronology(c)


@pytest.mark.parametrize(
    "record",
    [{}, receipt("2026-11-13T07:00:00Z", 0), receipt("2026-11-13T07:00:00Z", True)],
)
def test_ledger_date_inputs_require_real_ordering_and_shape(record):
    with pytest.raises(ValueError):
        chrono(ledger_records=[record])


def test_ledger_same_sequence_is_not_an_independent_new_receipt():
    with pytest.raises(ValueError):
        chrono(
            ledger_records=[
                receipt("2026-11-13T07:00:00Z"),
                receipt("2026-11-13T08:00:00Z"),
            ]
        )


@pytest.mark.parametrize(
    "native",
    [
        {"native_value": "NaN", "format": "JD"},
        {"native_value": "1e9999", "format": "MJD"},
        False,
    ],
)
def test_uninterpretable_native_epoch_is_not_converted_to_utc(native):
    from aneos_core.data.chronology import build_chronology, source_input

    n = row()["neo_data"]
    n["orbital_elements"]["native_epoch"] = native
    c = build_chronology("SYNTHETIC", [source_input(n, "SBDB", "/source")])
    e = next(e for e in c["events"] if e["role"] == "orbit_epoch")
    assert (
        e["status"] == "invalid"
        and e["value"] == native
        and e["time_scale"] == "not_declared"
    )


def test_legacy_epoch_remains_legacy_not_tdb_or_a_new_observation():
    from aneos_core.data.chronology import build_chronology, source_input

    n = row()["neo_data"]
    n["orbital_elements"]["epoch"] = "2026-11-13T07:00:00"
    c = build_chronology("SYNTHETIC", [source_input(n, "SBDB", "/source")])
    e = next(e for e in c["events"] if e["role"] == "orbit_epoch")
    assert e["status"] == "time_scale_unverified" and e["time_scale"] == "not_declared"
    assert "Legacy epoch" in e["meaning"]


def test_nonselected_source_discovery_survives_primary_orbit_reconciliation():
    from aneos_core.data.chronology import build_object_chronology, build_sbdb_discovery

    n = row()["neo_data"]
    n["sources_used"] = ["NEODyS", "SBDB"]
    n["evidence_metadata"] = {
        "field_sources": {"orbital_elements": "NEODyS"},
        "source_solutions": {
            "NEODyS": {},
            "SBDB": {
                "discovery_evidence": build_sbdb_discovery(
                    {"date": "1998-May-31"},
                    {
                        "requested_designation": "SYNTHETIC",
                        "source_response_sha256": "a" * 64,
                    },
                )
            },
        },
    }
    c = build_object_chronology(n)
    assert c["discovery"]["selected"]["value"] == "1998-05-31"
    assert c["discovery"]["selected"]["source"] == "SBDB"


def test_legacy_dates_and_history_survive_later_enrichment_and_engine_restart(database):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine, sessions = database
    with sessions() as db:
        db.add(
            EnrichedNEO(
                designation="SYNTHETIC",
                first_discovered=datetime(2020, 1, 2, 3, 4),
                last_updated=datetime(2021, 2, 3, 4, 5),
            )
        )
        db.commit()
    with sessions() as db:
        EnrichedNEOService(db).enrich_neo_data([row()], "upgrade")
    fresh = create_engine(str(engine.url))
    try:
        with sessionmaker(bind=fresh)() as db:
            x = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
            archive = next(
                r for r in x["evidence_history"] if r["kind"] == "legacy_projection"
            )
            assert archive["payload"]["first_discovered"] == "2020-01-02T03:04:00"
            assert archive["payload"]["last_updated"] == "2021-02-03T04:05:00"
            assert (
                x["chronology"]["legacy_dates"]["first_discovered"]["value"]
                == "2020-01-02T03:04:00"
            )
            assert x["first_discovered"] is None and x["discovery_date"] is None
            assert (
                x["first_ledger_recorded_at"] == x["evidence_history"][0]["recorded_at"]
            )
    finally:
        fresh.dispose()


def test_actual_date_menu_capture_generator_has_valid_hashes_and_bounded_widths(
    tmp_path,
):
    import subprocess, sys
    from hashlib import sha256
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "renders"
    command = [
        sys.executable,
        str(root / "scripts/qa_object_chronology.py"),
        "--output",
        str(directory),
    ]
    r = subprocess.run(command, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads((directory / "inventory.json").read_text())
    assert len(m["metrics"]) == 9
    assert all(
        v["max_columns"] <= int(k.rsplit("-", 1)[1]) for k, v in m["metrics"].items()
    )
    assert all(
        sha256((directory / k).read_bytes()).hexdigest() == v
        for k, v in m["artifact_sha256"].items()
    )
    assert not list(directory.glob("*.db")) and not list(directory.glob("*.sqlite"))
    for width in (80, 120, 160):
        positive = (directory / f"sourced_discovery-{width}.txt").read_text()
        assert "1998-05-31" in positive and "1998-05-30" in positive
        assert "Pacific" in positive and "TDB" in positive
        assert "could not be verified" not in positive
        assert (
            "1919-??-??" in (directory / f"partial_observation-{width}.txt").read_text()
        )
        assert (
            "2020-01-02" in (directory / f"legacy_unverified-{width}.txt").read_text()
        )
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_actual_stored_menu_plain_fallback_displays_date_roles(database, capsys):
    from aneos_menu import ANEOSMenu
    from aneos_core.data.chronology import build_sbdb_discovery

    engine, sessions = database
    payload = row()
    payload["neo_data"]["evidence_metadata"]["discovery_evidence"] = (
        build_sbdb_discovery(
            {"date": "1998-May-31"},
            {"requested_designation": "SYNTHETIC", "source_response_sha256": "a" * 64},
        )
    )
    with sessions() as db:
        EnrichedNEOService(db).enrich_neo_data([payload], "plain-positive")
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = None
    menu.wait_for_input = Mock()
    with sessions() as db:
        menu._show_enriched_neo_details(EnrichedNEOService(db), "SYNTHETIC")
    text = capsys.readouterr().out
    assert (
        "1998-05-31" in text and "1998-05-30" in text and "First ledger receipt" in text
    )
    assert "could not be verified" not in text
    menu.wait_for_input.assert_called_once()


@pytest.mark.parametrize(
    "literal", ["2026-11-13T07Z", "2026-11-13T07:00Z", "2026-11-13T07:00:00.123456789Z"]
)
def test_unrepresentable_ledger_precision_is_not_silently_truncated(literal):
    c = chrono(ledger_records=[receipt(literal)])
    assert c["local_receipts"]["first_ledger_recorded_at"] is None
    assert c["local_receipts"]["records"][0]["value"] == literal


def test_an_unverified_middle_clock_does_not_get_an_all_clocks_order_claim():
    c = chrono(
        ledger_records=[
            receipt("2026-11-13T07:00:00Z"),
            receipt("unknown", 2),
            receipt("2026-11-13T08:00:00Z", 3),
        ]
    )
    assert c["local_receipts"]["clock_order"] == "unverified_clock_values_present"
    assert c["local_receipts"]["unverified_count"] == 1


def test_covariance_epoch_is_visible_even_without_ng_acceleration_components():
    from aneos_core.data.chronology import build_chronology, source_input

    n = row()["neo_data"]
    n["nongrav"] = None
    n["evidence_metadata"]["covariance"] = {"epoch_jd_tdb": "2457098.500000000001"}
    c = build_chronology("SYNTHETIC", [source_input(n, "SBDB", "/source")])
    e = next(e for e in c["events"] if e["role"] == "covariance_epoch")
    assert e["value"] == "2457098.500000000001" and e["time_scale"] == "TDB"
    assert e["reference"] == "/inputs/sources/0/covariance_epoch_jd_tdb"
    assert (
        c["inputs"]["sources"][0]["covariance_reference"]
        == "/source/evidence_metadata/covariance/epoch_jd_tdb"
    )


def test_reconciled_chronology_references_resolve_to_preserved_inputs_not_missing_root_fields():
    from aneos_core.data.chronology import build_object_chronology, build_sbdb_discovery

    n = row()["neo_data"]
    n["sources_used"] = ["SBDB"]
    n["first_observation"] = None
    metadata = {
        "native_observation_window": {"first_observation": {"value": "1919-??-??"}},
        "solution": {"schema_version": "1.3", "solution_date": "2024-May-07 06:44:30"},
        "covariance": {"epoch_jd_tdb": "2457098.5"},
        "discovery_evidence": build_sbdb_discovery(
            {"date": "1998-May-31"},
            {"requested_designation": "SYNTHETIC", "source_response_sha256": "a" * 64},
        ),
    }
    n["evidence_metadata"] = {
        "field_sources": {"orbital_elements": "SBDB"},
        "source_solutions": {"SBDB": metadata},
    }
    c = build_object_chronology(n)
    assert c["reference_scope"] == "chronology_input_json_pointers"
    for event in c["events"]:
        if event["status"] == "not_available":
            continue
        pointer = event["reference"]
        assert pointer.startswith("/inputs/")
        target = c
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            target = target[int(token)] if isinstance(target, list) else target[token]
        if event["role"] == "discovery":
            assert target["normalized"]["calendar_date"] == event["value"]
        elif event["role"] == "orbit_epoch":
            assert target["native_value"] == event["value"]
        else:
            assert str(target) == str(event["value"])


def test_legacy_context_reference_chronology_replays_without_rewriting():
    from pathlib import Path
    from aneos_core.data.chronology import verify_chronology

    x = json.loads(
        (
            Path(__file__).parent / "fixtures/chronology_legacy_context_refs.json"
        ).read_text()
    )["chronology"]
    assert verify_chronology(x) == x
    from aneos_core.analysis.chronology_view import chronology_markdown
    assert 'Legacy references identify snapshot context' in chronology_markdown(x)


def test_compact_reference_resolver_returns_native_assertion_and_rejects_external_paths():
    from aneos_core.data.chronology import (
        build_chronology,
        source_input,
        resolve_chronology_reference,
    )

    c = build_chronology(
        "SYNTHETIC",
        [source_input(row()["neo_data"], "SBDB", "/context")],
        [receipt("2026-11-13T07:00:00Z")],
        reference_scope="compact_inputs_v1",
    )
    for e in c["events"]:
        if e["role"] == "first_observation":
            assert resolve_chronology_reference(c, e["reference"]) == "1998-05-30"
    assert (
        resolve_chronology_reference(c, c["local_receipts"]["records"][0]["reference"])
        == "2026-11-13T07:00:00Z"
    )
    for bad in (
        "https://example.invalid",
        "/etc/passwd",
        "/inputs/missing",
        "/inputs/sources/999",
        "/inputs/sources/-1",
        "/inputs/sources/00",
        "/inputs/~2",
    ):
        with pytest.raises(ValueError):
            resolve_chronology_reference(c, bad)
