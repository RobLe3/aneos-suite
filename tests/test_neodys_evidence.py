"""Synthetic source contracts; provider availability is a separate network check."""

from unittest.mock import Mock
import pytest

from aneos_core.data.sources.neodys import NEODySSource
from aneos_core.data.acquisition import adapter_to_neo


def oef(identity="1998KY26", mag=" MAG 25.594 0.150", extra=""):
    return (
        "format = 'OEF2.0'\nrectype = 'ML'\nrefsys = ECLM J2000\nEND_OF_HEADER\n"
        + identity
        + "\n EQU 1.23 .1 .1 .01 .01 20\n MJD 55869.089613878 TDT\n"
        + mag
        + "\n"
        + extra
    )


def response(monkeypatch, text=None, status=200):
    import requests

    r = Mock(status_code=status, text=oef() if text is None else text)
    query = Mock(return_value=r)
    monkeypatch.setattr(requests, "get", query)
    return r, query


def neo(payload):
    return adapter_to_neo("NEODyS", "1998 KY26", payload, "2026-11-13T06:00:00+00:00")


def test_provisional_query_does_not_require_or_guess_an_mpc_number(monkeypatch):
    r, query = response(monkeypatch)
    src = NEODySSource()
    src._resolve_number = Mock(
        side_effect=AssertionError("provisional needs no number")
    )
    result = src.fetch_acquisition("1998 KY26")
    assert result.success and result.outcome == "available"
    assert query.call_args.args[0].endswith("/1998KY26.eq0")
    assert result.raw_response["equinoctial_text"] == oef()
    assert neo(result.data).physical_properties.absolute_magnitude_h == 25.594
    assert (
        result.data["_evidence_metadata"]["identity_assessment"]["native_identifier"]
        == "1998KY26"
    )
    r.close.assert_called_once()


@pytest.mark.parametrize("h", ["0", "-1.2", "25.594"])
def test_native_photometry_is_atomic_not_a_fabricated_sbdb_response(h):
    data = NEODySSource()._parse_eq0(oef(mag=f" MAG {h} 0.150"), "1998KY26")
    n = neo(data)
    assert n.physical_properties.absolute_magnitude_h == float(h)
    evidence = n.evidence_metadata["physical_evidence"]
    record = evidence["records"][0]
    assert record["source"] == "NEODyS"
    assert record["adapter"] == "neodys-oef2-photometry-v1"
    assert record["native"]["line"] == f" MAG {h} 0.150"
    assert record["normalized"]["sigma"] is None
    assert record["measurement_epoch"] is None
    assert record["estimation_status"] == "not_provided"
    assert "name" not in record["native"]


@pytest.mark.parametrize("mag", ["", " MAG ? 0.150", " MAG NaN 0.150"])
def test_missing_or_invalid_magnitude_never_becomes_zero(mag):
    n = neo(NEODySSource()._parse_eq0(oef(mag=mag), "1998KY26"))
    assert (
        n.physical_properties is None
        or n.physical_properties.absolute_magnitude_h is None
    )


def test_alternative_photometry_is_not_last_wins():
    n = neo(NEODySSource()._parse_eq0(oef(extra=" MAG 26.0 0.150\n"), "1998KY26"))
    assert (
        n.physical_properties is None
        or n.physical_properties.absolute_magnitude_h is None
    )
    assert (
        n.evidence_metadata["physical_evidence"]["selection"]["absolute_magnitude_h"][
            "status"
        ]
        == "ambiguous"
    )


def test_covariance_and_unknown_records_are_retained_but_not_relabelled():
    data = NEODySSource()._parse_eq0(
        oef(extra=" COV 1 2 3\n LSP 0 0 6\n NEW 42\n"), "1998KY26"
    )
    accounting = data["_evidence_metadata"]["record_accounting"]
    assert {r["tag"] for r in accounting} >= {"EQU", "MJD", "MAG", "COV", "LSP", "NEW"}
    assert (
        next(r for r in accounting if r["tag"] == "COV")["status"]
        == "native_not_normalized"
    )
    assert (
        next(r for r in accounting if r["tag"] == "NEW")["status"]
        == "unsupported_native"
    )
    assert data.get("_nongrav") is None


def test_wrong_native_identity_is_rejected_and_preserved(monkeypatch):
    response(monkeypatch, oef(identity="1999AA"))
    result = NEODySSource().fetch_acquisition("1998 KY26")
    assert not result.success and result.outcome == "invalid"
    assert result.raw_response["equinoctial_text"] == oef(identity="1999AA")


@pytest.mark.parametrize(
    "status,outcome", [(404, "missing"), (503, "unavailable"), (429, "unavailable")]
)
def test_http_statuses_are_not_all_an_empty_result(monkeypatch, status, outcome):
    r, _ = response(monkeypatch, "synthetic source failure", status)
    result = NEODySSource().fetch_acquisition("1998 KY26")
    assert not result.success and result.outcome == outcome
    assert result.raw_response["equinoctial_text"] == "synthetic source failure"
    r.close.assert_called_once()


@pytest.mark.parametrize(
    "bad",
    [
        oef().replace("ECLM J2000", "EQUM J2000"),
        oef().replace("OEF2.0", "OEF3.0"),
        oef().replace("ML", "1L"),
        oef().replace("END_OF_HEADER", ""),
        oef(extra="END_OF_HEADER\n"),
        oef().replace(" MJD 55869.089613878 TDT", ""),
        oef(extra=" MJD 55869.0 TDT\n"),
        oef(extra=" EQU 1.23 .1 .1 .01 .01 20\n"),
        oef().replace("EQU 1.23", "EQU -1.23"),
        oef().replace(".1 .1 .01", "1.1 .1 .01"),
        oef().replace("EQU 1.23", "EQU NaN"),
        oef().replace("MJD 55869.089613878", "MJD NaN"),
    ],
)
def test_unsupported_or_ambiguous_orbits_fail_closed_with_native_packet(
    monkeypatch, bad
):
    response(monkeypatch, bad)
    result = NEODySSource().fetch_acquisition("1998 KY26")
    assert not result.success and result.outcome == "invalid"
    assert result.raw_response["equinoctial_text"] == bad


def test_duplicate_photometry_is_not_independent_evidence():
    n = neo(NEODySSource()._parse_eq0(oef(extra=" MAG 25.594 0.150\n"), "1998KY26"))
    e = n.evidence_metadata["physical_evidence"]
    assert n.physical_properties.absolute_magnitude_h == 25.594
    assert e["duplicate_records"] == {"1": 0}


def test_tampered_photometry_cannot_replay_as_verified():
    from copy import deepcopy
    from aneos_core.data.physical_evidence import (
        verify_physical_evidence,
        PhysicalMeasurement,
    )

    e = NEODySSource()._parse_eq0(oef(), "1998KY26")["_evidence_metadata"][
        "physical_evidence"
    ]
    r = e["records"][0]
    assert PhysicalMeasurement.from_dict(r).to_dict() == r
    bad = deepcopy(e)
    bad["records"][0]["normalized"]["value"] = 0
    with pytest.raises(ValueError):
        verify_physical_evidence(bad)
    bad = deepcopy(r)
    bad["source"] = "SBDB"
    with pytest.raises(ValueError):
        PhysicalMeasurement.from_dict(bad)


def test_fortran_literals_and_unknown_time_scale_are_not_guessed():
    n = neo(
        NEODySSource()._parse_eq0(
            oef(mag=" MAG 2.5594D+01 0.15").replace(" TDT", " XYZ"), "1998KY26"
        )
    )
    assert n.physical_properties.absolute_magnitude_h == 25.594
    assert n.orbital_elements.native_epoch.time_scale == "UNKNOWN"
    assert n.orbital_elements.epoch is None
    assert (
        n.evidence_metadata["physical_evidence"]["context"]["orbital_epoch"][
            "native_value"
        ]
        == "55869.089613878"
    )


def test_requests_outage_is_not_a_missing_object(monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "get", Mock(side_effect=requests.Timeout("synthetic private detail"))
    )
    result = NEODySSource().fetch_acquisition("1998 KY26")
    assert not result.success and result.outcome == "unavailable"
    assert result.raw_response is None
    assert "private" not in str(result)


def test_name_uses_verified_bounded_alias_resolution(monkeypatch):
    from aneos_core.data.sources.mpc import MPCSource

    response(monkeypatch, oef(identity="99942"))
    query = Mock(return_value={"number": 99942, "name": "SYNTHETIC_NAME"})
    monkeypatch.setattr(MPCSource, "_query_mpc", query)
    result = NEODySSource().fetch_acquisition("SYNTHETIC_NAME")
    assert result.success
    assert (
        result.data["_evidence_metadata"]["alias_resolution"]["native_catalogue_row"][
            "number"
        ]
        == 99942
    )
    query.assert_called_once_with("SYNTHETIC_NAME")


def test_legacy_http_function_uses_validated_oef_not_zero_defaults(monkeypatch):
    response(monkeypatch)
    r = NEODySSource()._make_request("1998 KY26")
    assert r["a"] == 1.23 and r["H"] == 25.594
    response(monkeypatch, "not an orbit")
    assert NEODySSource()._make_request("1998 KY26") is None


@pytest.mark.parametrize(
    "mag,expected",
    [
        (" MAG 25.594 0.150", "NEODyS H (OEF MAG)"),
        ("", "NEODyS photometry: not available"),
        (" MAG ? 0.150", "not normalized"),
    ],
)
def test_actual_review_renders_source_specific_photometry_at_three_widths(
    mag, expected
):
    from io import StringIO
    from rich.console import Console
    from aneos_core.analysis.object_review import (
        build_object_review,
        render_object_review,
        object_review_markdown,
    )
    from aneos_core.data.models import NEOData

    n = neo(NEODySSource()._parse_eq0(oef(mag=mag), "1998KY26"))
    n = NEOData.from_dict(n.to_dict())
    review = build_object_review(n)
    for width in (80, 120, 160):
        stream = StringIO()
        console = Console(file=stream, width=width, color_system=None)
        render_object_review(review, console=console)
        text = stream.getvalue()
        assert expected in text
        assert all(len(line) <= width for line in text.splitlines())
        if mag:
            assert "G remains native" in text
    assert expected in object_review_markdown(review)


def test_source_magnitude_survives_public_query_cache_sql_and_export(
    tmp_path, monkeypatch
):
    from sqlalchemy import create_engine
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.cache import CacheManager
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from aneos_api.database import EnrichedNEOService
    from aneos_core.data.models import NEOData
    from aneos_core.data.sources.cad import parse_cad_payload, CloseApproachQuery
    import json

    monkeypatch.chdir(tmp_path)
    r, query = response(monkeypatch)
    engine = create_engine("sqlite:///" + str(tmp_path / "evidence.sqlite"))
    repo = SQLEvidenceRepository(engine)
    fetcher = DataFetcher(
        cache_manager=CacheManager(tmp_path / "cache"), evidence_repository=repo
    )
    # Fixed plan remains five, even when only this source is intentionally installed in a synthetic test.
    fetcher.sources = {"NEODyS": NEODySSource()}
    fetcher._fetch_close_approaches = Mock(return_value=[])
    first = fetcher.fetch_neo_data("1998 KY26")
    second = fetcher.fetch_neo_data("1998 KY26")
    assert first.to_dict() == second.to_dict()
    assert query.call_count == 1
    assert (
        NEOData.from_dict(first.to_dict()).physical_properties.absolute_magnitude_h
        == 25.594
    )
    with repo.sessions() as db:
        svc = EnrichedNEOService(db)
        svc.export_enriched_json("1998 KY26", tmp_path / "export.json")
    stored = json.loads((tmp_path / "export.json").read_text())
    history = stored["acquisition_history"]
    assert len(history) == 1
    outcomes = history[0]["bundle"]["outcomes"]
    admitted = next(x for x in outcomes if x["source"] == "NEODyS")
    assert admitted["status"] == "available"
    assert admitted["adapter_payload"]["_native_response"]["equinoctial_text"] == oef()
    assert history[0]["bundle"]["coverage"]["scientific_completeness"] is None
    engine.dispose()


@pytest.mark.parametrize("native", [True, 0, "0", "../../other", ""])
def test_unsafe_or_non_catalogue_file_identifiers_are_rejected(native):
    from aneos_core.data.neodys_evidence import catalogue_identifier

    with pytest.raises(ValueError):
        catalogue_identifier(native)


@pytest.mark.parametrize(
    "bad",
    [
        oef().replace("rectype = 'ML'", "badheader"),
        oef().replace("rectype = 'ML'", "rectype = 'ML'\nrectype = 'ML'"),
        oef().replace(" EQU 1.23 .1 .1 .01 .01 20", " EQU 1.23 .1"),
        oef().replace(" MJD 55869.089613878 TDT", " MJD 55869.0"),
        "x" * 2_000_001,
    ],
)
def test_header_layout_and_size_admission_is_strict(bad):
    from aneos_core.data.neodys_evidence import parse_oef

    with pytest.raises(ValueError):
        parse_oef(bad, "1998KY26")


@pytest.mark.parametrize("line", [" MAG 25.9", " MAG 25.9 ?", " MAG 25.9 Infinity"])
def test_incomplete_or_nonfinite_hg_pair_does_not_create_valid_photometry(line):
    n = neo(NEODySSource()._parse_eq0(oef(mag=line), "1998KY26"))
    assert (
        n.physical_properties is None
        or n.physical_properties.absolute_magnitude_h is None
    )
    row = next(x for x in n.evidence_metadata["record_accounting"] if x["tag"] == "MAG")
    assert row["status"] == "not_normalized"


@pytest.mark.parametrize(
    "error,outcome",
    [
        (ValueError("synthetic invalid"), "invalid"),
        (ImportError("synthetic unavailable"), "unavailable"),
        (RuntimeError("synthetic unavailable"), "unavailable"),
    ],
)
def test_alias_failures_do_not_become_empty_catalogue_objects(
    monkeypatch, error, outcome
):
    from aneos_core.data.sources.mpc import MPCSource

    monkeypatch.setattr(MPCSource, "_query_mpc", Mock(side_effect=error))
    result = NEODySSource().fetch_acquisition("SYNTHETIC_NAME")
    assert not result.success and result.outcome == outcome


def test_alias_ambiguity_keeps_candidates_and_does_not_poll_file(monkeypatch):
    from aneos_core.data.sources.mpc import MPCSource
    from aneos_core.utils.errors import AmbiguousObjectError

    error = AmbiguousObjectError("synthetic")
    error.native_response = [{"number": 1}, {"number": 2}]
    monkeypatch.setattr(MPCSource, "_query_mpc", Mock(side_effect=error))
    download = Mock(side_effect=AssertionError("must not poll guessed file"))
    src = NEODySSource()
    src._download_eq0 = download
    r = src.fetch_acquisition("SYNTHETIC_NAME")
    assert r.outcome == "ambiguous" and r.raw_response == error.native_response
    download.assert_not_called()


@pytest.mark.parametrize("row", [None, {}, {"number": -1}, {"number": "1998KY26"}])
def test_unusable_alias_resolution_cannot_supply_a_file_number(monkeypatch, row):
    from aneos_core.data.sources.mpc import MPCSource

    monkeypatch.setattr(MPCSource, "_query_mpc", Mock(return_value=row))
    r = NEODySSource().fetch_acquisition("SYNTHETIC_NAME")
    assert not r.success and r.outcome in ("missing", "invalid")


def test_compatibility_source_functions_keep_positive_and_failure_behavior(monkeypatch):
    from aneos_core.data.sources.mpc import MPCSource

    src = NEODySSource()
    assert src.get_base_url() == src.base_url
    response(monkeypatch)
    assert src.fetch_orbital_elements("1998 KY26")["_H"] == 25.594
    assert src._fetch_eq0("1998KY26")["_H"] == 25.594
    assert src.get_object_summary("1998 KY26")["orbital_elements"]["_H"] == 25.594
    assert (
        src._equinoctial_to_keplerian(1.23, 0.1, 0.1, 0.01, 0.01, 20)["semi_major_axis"]
        == 1.23
    )
    response(monkeypatch, "unavailable", 503)
    assert src._fetch_eq0("1998KY26") is None
    assert src.fetch_orbital_elements("1998 KY26") is None
    monkeypatch.setattr(MPCSource, "_query_mpc", Mock(return_value=None))
    assert src._resolve_number("1998 KY26") is None


@pytest.mark.parametrize("status", [200, 503])
def test_directory_health_is_only_directory_reachability(monkeypatch, status):
    import asyncio, aiohttp

    class Context:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return self

    context = Context()
    context.status = status
    monkeypatch.setattr(aiohttp, "ClientSession", Mock(return_value=context))
    assert asyncio.run(NEODySSource().health_check()) is (status == 200)
    monkeypatch.setattr(
        aiohttp, "ClientSession", Mock(side_effect=RuntimeError("synthetic"))
    )
    assert asyncio.run(NEODySSource().health_check()) is False


def test_normalization_revision_partitions_new_cache_without_rewriting_old_queries(
    tmp_path, monkeypatch
):
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.enrichment import fingerprint
    from aneos_core.data.cache import CacheManager

    monkeypatch.chdir(tmp_path)
    old_scope = {
        "workspace": str(tmp_path.resolve()),
        "endpoint_scope": "default-public-v1",
    }
    f = DataFetcher(cache_manager=CacheManager(tmp_path / "cache"))
    assert f.source_plan["context_id"] != fingerprint(old_scope)
    other = DataFetcher(cache_manager=CacheManager(tmp_path / "cache"))
    assert f.source_plan == other.source_plan


def test_undefined_orbfit_magnitude_is_preserved_not_promoted_to_measurement():
    data = NEODySSource()._parse_eq0(oef(mag=" MAG -9.99 0.15"), "1998KY26")
    n = neo(data)
    assert (
        n.physical_properties is None
        or n.physical_properties.absolute_magnitude_h is None
    )
    assert data.get("_H") is None
    record = n.evidence_metadata["physical_evidence"]["records"][0]
    assert "source_missing_value_sentinel" in record["issues"]
    assert record["native"]["line"] == " MAG -9.99 0.15"


def test_actual_neodys_qa_generator_and_hashed_artifacts(tmp_path):
    import subprocess, sys, json
    from hashlib import sha256
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "render"
    command = [
        sys.executable,
        str(root / "scripts/qa_neodys_evidence.py"),
        "--output",
        str(directory),
    ]
    r = subprocess.run(command, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((directory / "inventory.json").read_text())
    assert len(manifest["metrics"]) == 9
    for name, metric in manifest["metrics"].items():
        assert metric["max_columns"] <= int(name.split("-")[-1])
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest
    assert not list(directory.glob("*.sqlite")) and not list(directory.glob("*.db"))
    assert subprocess.run(command, capture_output=True).returncode != 0


@pytest.mark.parametrize("e", [0.0, 0.2, 0.99])
@pytest.mark.parametrize("inclination", [0.0, 60.0, 179.9])
def test_equinoctial_normalization_reconstructs_known_analytic_state(e, inclination):
    import math
    from aneos_core.data.neodys_evidence import equinoctial_to_keplerian

    node = 120.0
    peri = 30.0
    anomaly = 270.0
    lp = math.radians(node + peri)
    ln = math.radians(node)
    tilt = math.tan(math.radians(inclination) / 2)
    original = [
        1.23,
        e * math.sin(lp),
        e * math.cos(lp),
        tilt * math.sin(ln),
        tilt * math.cos(ln),
        (node + peri + anomaly) % 360,
    ]
    k = equinoctial_to_keplerian(*original)
    p = math.radians(k["ra_of_ascending_node"] + k["arg_of_periapsis"])
    n = math.radians(k["ra_of_ascending_node"])
    t = math.tan(math.radians(k["inclination"]) / 2)
    reconstructed = [
        k["semi_major_axis"],
        k["eccentricity"] * math.sin(p),
        k["eccentricity"] * math.cos(p),
        t * math.sin(n),
        t * math.cos(n),
        (k["ra_of_ascending_node"] + k["arg_of_periapsis"] + k["mean_anomaly"]) % 360,
    ]
    assert reconstructed == pytest.approx(original, rel=1e-12, abs=1e-12)
    assert k["inclination"] == pytest.approx(inclination)


def test_orphan_second_object_identifier_is_not_an_unknown_measurement(monkeypatch):
    response(monkeypatch, oef(extra="1999AA\n"))
    r = NEODySSource().fetch_acquisition("1998 KY26")
    assert not r.success and r.outcome == "invalid"
    assert r.raw_response["equinoctial_text"].endswith("1999AA\n")


def test_prior_commit_sbdb_physical_assertions_replay_without_rewriting():
    import json
    from pathlib import Path
    from aneos_core.data.physical_evidence import build_sbdb_evidence, verify_physical_evidence
    fixture=json.loads((Path(__file__).parent/'fixtures/sbdb_physical_v1_frozen.json').read_text())
    assert build_sbdb_evidence(fixture['records'],fixture['context'])==fixture['evidence']
    assert verify_physical_evidence(fixture['evidence'])==fixture['evidence']
