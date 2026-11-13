"""Actual registered-router HTTP contracts, not auth/startup certification."""

from unittest.mock import Mock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from aneos_api.endpoints import analysis
from aneos_core.data.epoch_evidence import NativeEpoch
from aneos_core.analysis.screening import ScreeningAnalysisService


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/v1/analysis")
    with TestClient(app) as client:
        yield client


def native_point(scale="TDB"):
    label = f"2460000.500000001 = A.D. 2023-Feb-25 00:00:00 {scale}"
    epoch = NativeEpoch(
        "2460000.500000001", "JD", scale, "Horizons", "computed_elements", label
    )
    return {
        "epoch": label,
        "native_epoch": epoch.to_dict(),
        "a": 1.2,
        "e": 0.2,
        "i": 1.49,
        "node": 0.0,
        "peri": 0.0,
        "M": 0.0,
    }


def test_real_history_http_response_retains_native_precision_scale_and_computed_basis(
    client, monkeypatch
):
    p = native_point()
    source = Mock(return_value=[p])
    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history", source
    )
    r = client.get(
        "/api/v1/analysis/history", params={"designation": "SYNTHETIC", "years": 2}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["points"][0]["native_epoch"] == p["native_epoch"]
    assert body["evidence_basis"] == "computed_ephemeris_not_observations"
    assert body["returned_points"] == 1
    assert body["epoch_status"] == "native_epochs_preserved"
    assert body["scientific_status"] == "not_independently_validated"


def test_raw_http_albedo_above_one_matches_canonical_service_without_probability(
    client,
):
    request = {
        "designation": "SYNTHETIC",
        "a": 1.2,
        "e": 0.2,
        "i": 1.49,
        "albedo": 1.08,
    }
    response = client.post("/api/v1/analysis/detect", json=request)
    assert response.status_code == 200
    expected = (
        ScreeningAnalysisService()
        .analyze_raw("SYNTHETIC", {"a": 1.2, "e": 0.2, "i": 1.49}, {"albedo": 1.08})
        .to_payload()
    )
    result = response.json()
    for key in (
        "evidence_id",
        "screening_score",
        "candidate",
        "classification",
        "sigma_confidence",
        "artificial_probability",
    ):
        assert result[key] == expected[key]


@pytest.mark.parametrize(
    "status,code", [("unavailable", 503), ("invalid_evidence", 502), ("empty", 404)]
)
def test_history_outage_invalid_and_empty_are_different_http_states(
    client, monkeypatch, status, code
):
    from aneos_core.data.sources.horizons import OrbitalHistoryQuery

    source = Mock(
        return_value=OrbitalHistoryQuery(
            [],
            {
                "status": status,
                "schema_version": "orbital-history-query-v1",
                "designation_requested": "SYNTHETIC",
                "returned_points": 0,
            },
        )
    )
    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history", source
    )
    response = client.get(
        "/api/v1/analysis/history", params={"designation": "SYNTHETIC"}
    )
    assert response.status_code == code
    assert "points" not in response.json()


@pytest.mark.parametrize(
    "mutation", ["native_derived", "header", "role", "nan", "bool", "extra"]
)
def test_bad_history_cannot_be_published_as_valid_http_evidence(
    client, monkeypatch, mutation
):
    p = native_point()
    if mutation == "native_derived":
        p["native_epoch"]["julian_day_same_scale"] = "0"
    elif mutation == "header":
        p["epoch"] = "different header"
    elif mutation == "role":
        p["native_epoch"]["role"] = "observation"
    elif mutation == "nan":
        p["a"] = float("nan")
    elif mutation == "bool":
        p["a"] = True
    else:
        p["unexpected_source_field"] = "unsupported"
    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
        Mock(return_value=[p]),
    )
    r = client.get("/api/v1/analysis/history", params={"designation": "SYNTHETIC"})
    assert r.status_code == 502 and "points" not in r.json()


def test_unknown_native_scale_and_legacy_unbound_history_are_not_invented_utc(
    client, monkeypatch
):
    for point, status in [
        (native_point("UNKNOWN"), "native_epochs_preserved"),
        ({"epoch": "legacy calendar", "a": 1.2}, "legacy_unbound"),
    ]:
        monkeypatch.setattr(
            "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
            Mock(return_value=[point]),
        )
        r = client.get("/api/v1/analysis/history", params={"designation": "SYNTHETIC"})
        assert r.status_code == 200 and r.json()["epoch_status"] == status
        if "native_epoch" in point:
            assert r.json()["points"][0]["native_epoch"]["time_scale"] == "UNKNOWN"
        else:
            assert r.json()["points"][0]["native_epoch"] is None


def test_sync_history_retrieval_runs_outside_event_loop_thread(client, monkeypatch):
    import asyncio

    observed = []

    def retrieve(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            observed.append("event_loop")
        except RuntimeError:
            observed.append("worker_thread")
        return [native_point()]

    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
        retrieve,
    )
    assert (
        client.get(
            "/api/v1/analysis/history", params={"designation": "SYNTHETIC"}
        ).status_code
        == 200
    )
    assert observed == ["worker_thread"]


def test_source_exception_is_redacted_and_not_a_valid_empty_history(
    client, monkeypatch
):
    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
        Mock(side_effect=RuntimeError("private-source-trace")),
    )
    r = client.get("/api/v1/analysis/history", params={"designation": "SYNTHETIC"})
    assert r.status_code == 503 and "private-source-trace" not in r.text


def test_full_app_nonfinite_raw_json_is_client_error_not_500():
    from aneos_api.app import create_app

    http = TestClient(create_app(), raise_server_exceptions=False)
    try:
        response = http.post(
            "/api/v1/analysis/detect",
            content='{"a":1.2,"e":0.2,"i":1.49,"albedo":NaN}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
    finally:
        http.close()


@pytest.mark.parametrize("albedo", [-0.1, "NaN", "Infinity"])
def test_raw_albedo_domain_correction_keeps_negative_and_nonfinite_rejected(
    client, albedo
):
    r = client.post(
        "/api/v1/analysis/detect",
        json={"a": 1.2, "e": 0.2, "i": 1.49, "albedo": albedo},
    )
    assert r.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"designation": ""},
        {"designation": "   "},
        {"designation": "bad\nname"},
        {"designation": "SYNTHETIC", "years": 0},
        {"designation": "SYNTHETIC", "years": 51},
    ],
)
def test_history_request_prerequisites_fail_before_polling(client, monkeypatch, params):
    source = Mock()
    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history", source
    )
    assert client.get("/api/v1/analysis/history", params=params).status_code == 422
    source.assert_not_called()


def test_full_app_validation_does_not_echo_private_input_or_nonfinite_context():
    from aneos_api.app import create_app

    http = TestClient(create_app(), raise_server_exceptions=False)
    try:
        r = http.post(
            "/api/v1/analysis/detect",
            json={"a": "SYNTHETIC-PRIVATE-MARKER", "e": 0.2, "i": 1.49},
        )
        assert r.status_code == 422
        assert "SYNTHETIC-PRIVATE-MARKER" not in r.text
        assert r.json()["detail"][0]["loc"] == ["body", "a"]
    finally:
        http.close()


def provider_text():
    return "$$SOE\n2460000.500000001 = A.D. 2023-Feb-25 00:00:00 TDB\nEC= .2 IN= 1.49 A= 1.2\nOM= 0 W= 0 MA= 0\n$$EOE"


@pytest.mark.parametrize(
    "mode,code",
    [
        ("available", 200),
        ("empty", 404),
        ("bad_version", 502),
        ("bad_block", 502),
        ("invalid_elements", 502),
        ("outage", 503),
    ],
)
def test_real_provider_parser_to_http_outcomes_with_query_scope_and_cleanup(
    client, monkeypatch, mode, code
):
    import requests
    import aneos_core.data.sources.horizons as module

    text = (
        provider_text()
        if mode not in ("empty", "bad_block")
        else "$$SOE\n$$EOE" if mode == "empty" else "unsupported source output"
    )
    if mode == "invalid_elements":
        text = text.replace("A= 1.2", "A= 1e999")
    response = Mock()
    response.json.return_value = {
        "signature": {"version": "9.9" if mode == "bad_version" else "1.3"},
        "result": text,
    }
    provider = (
        Mock(side_effect=requests.ConnectionError("private-source-detail"))
        if mode == "outage"
        else Mock(return_value=response)
    )
    monkeypatch.setattr(module, "provider_get", provider)
    r = client.get(
        "/api/v1/analysis/history", params={"designation": "SYNTHETIC", "years": 2}
    )
    assert r.status_code == code
    assert "private-source-detail" not in r.text
    if mode != "outage":
        response.close.assert_called_once()
    if mode == "available":
        body = r.json()
        q = body["query_evidence"]
        assert q["status"] == "available" and q["result_text_sha256"]
        assert q["request_parameters"]["TIME_TYPE"] == "TDB"
        assert q["request_parameters"]["OUT_UNITS"] == "AU-D"
        assert q["request_parameters"]["CENTER"] == "500@10"
        assert body["points"][0]["native_epoch"]["native_value"] == "2460000.500000001"
        assert q["returned_points"] == body["returned_points"] == 1


@pytest.mark.parametrize(
    "points,status",
    [
        ([native_point()], "native_single_scale_numeric_axis"),
        ([native_point("UNKNOWN")], "unavailable_unbound_or_incompatible_epochs"),
        (
            [native_point(), native_point("TT")],
            "unavailable_unbound_or_incompatible_epochs",
        ),
        (
            [{"epoch": "old label", "a": 1.2}],
            "unavailable_unbound_or_incompatible_epochs",
        ),
    ],
)
def test_history_time_axis_readiness_does_not_convert_or_pool_scales(
    client, monkeypatch, points, status
):
    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
        Mock(return_value=points),
    )
    r = client.get("/api/v1/analysis/history", params={"designation": "SYNTHETIC"})
    assert r.status_code == 200
    assert r.json()["time_axis_status"] == status


def test_actual_horizons_history_keeps_missing_signature_unverified_without_fake_authentication(
    client, monkeypatch
):
    import aneos_core.data.sources.horizons as module

    response = Mock()
    response.json.return_value = {"result": provider_text()}
    monkeypatch.setattr(module, "provider_get", Mock(return_value=response))
    r = client.get("/api/v1/analysis/history", params={"designation": "SYNTHETIC"})
    assert r.status_code == 200
    assert r.json()["query_evidence"]["schema_status"] == "unverified_missing_signature"
    assert r.json()["scientific_status"] == "not_independently_validated"


@pytest.mark.parametrize("years", [0, -1, True, 1001])
def test_direct_history_source_invalid_years_cannot_start_polling(monkeypatch, years):
    import aneos_core.data.sources.horizons as module

    source = Mock()
    monkeypatch.setattr(module, "provider_get", source)
    with pytest.raises(ValueError):
        module.HorizonsSource().fetch_orbital_history("SYNTHETIC", years=years)
    source.assert_not_called()


@pytest.mark.parametrize(
    "field", ["a", "e", "i", "diameter_km", "albedo", "absolute_magnitude"]
)
@pytest.mark.parametrize("bad", [True, "1.2"])
def test_http_numeric_fields_do_not_coerce_unverified_booleans_or_strings(
    client, field, bad
):
    request = {"a": 1.2, "e": 0.2, "i": 1.49}
    request[field] = bad
    assert client.post("/api/v1/analysis/detect", json=request).status_code == 422


@pytest.mark.parametrize(
    "orbit",
    [{"a": 2000.0, "e": 0.9995, "i": 1.49}, {"a": -2000.0, "e": 3.0, "i": 1.49}],
)
def test_arbitrary_http_numeric_caps_do_not_override_canonical_domain_or_abstention(
    client, orbit
):
    response = client.post(
        "/api/v1/analysis/detect", json=dict(orbit, designation="SYNTHETIC")
    )
    expected = ScreeningAnalysisService().analyze_raw("SYNTHETIC", orbit).to_payload()
    assert response.status_code == 200
    for key in (
        "evidence_id",
        "screening_score",
        "candidate",
        "classification",
        "sigma_confidence",
        "artificial_probability",
    ):
        assert response.json()[key] == expected[key]


@pytest.mark.parametrize(
    "orbit",
    [
        {"a": 1.0, "e": 1.5, "i": 1.49},
        {"a": 0.0, "e": 1.5, "i": 1.49},
        {"a": -1.0, "e": 0.2, "i": 1.49},
    ],
)
def test_core_and_http_reject_same_inconsistent_conics(client, orbit):
    with pytest.raises(ValueError, match="Inconsistent"):
        ScreeningAnalysisService().analyze_raw("SYNTHETIC", orbit)
    assert client.post("/api/v1/analysis/detect", json=orbit).status_code == 422


@pytest.mark.parametrize(
    "mutation",
    ["designation", "count", "schema", "unknown_outcome", "empty_with_points"],
)
def test_history_query_envelope_cannot_mismatch_published_points(
    client, monkeypatch, mutation
):
    from aneos_core.data.sources.horizons import OrbitalHistoryQuery

    meta = {
        "schema_version": "orbital-history-query-v1",
        "status": "available",
        "designation_requested": "SYNTHETIC",
        "returned_points": 1,
    }
    if mutation == "designation":
        meta["designation_requested"] = "OTHER"
    elif mutation == "count":
        meta["returned_points"] = 2
    elif mutation == "schema":
        meta["schema_version"] = "unsupported"
    elif mutation == "unknown_outcome":
        meta["status"] = "claimed_complete"
    else:
        meta["status"] = "empty"
    monkeypatch.setattr(
        "aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
        Mock(return_value=OrbitalHistoryQuery([native_point()], meta)),
    )
    r = client.get("/api/v1/analysis/history", params={"designation": "SYNTHETIC"})
    assert r.status_code == 502 and "points" not in r.json()


def test_documented_synthetic_history_example_matches_actual_http_output(
    client, monkeypatch
):
    from pathlib import Path
    import json
    from aneos_core.data.sources.horizons import HorizonsSource

    points = HorizonsSource.__new__(HorizonsSource)._parse_elements_table(
        provider_text()
    )
    monkeypatch.setattr(
        HorizonsSource, "fetch_orbital_history", Mock(return_value=points)
    )
    r = client.get(
        "/api/v1/analysis/history", params={"designation": "SYNTHETIC", "years": 2}
    )
    assert r.status_code == 200
    documented = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/api/examples/history-v2-synthetic.json"
        ).read_text()
    )
    assert documented == r.json()
    assert (
        documented["query_evidence"] is None
    )  # Do not invent provenance for a supplied legacy list.
