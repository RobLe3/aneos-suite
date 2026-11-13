"""Ordinary public queries retain source evidence and a fixed acquisition plan."""

import logging
from unittest.mock import Mock, patch
import pytest
from aneos_core.data.fetcher import DataFetcher
from aneos_core.data.sources.mpc import MPCSource
from aneos_core.data.models import NEOData, PhysicalProperties, OrbitalElements
from aneos_core.config.settings import ANEOSConfig


def test_available_mpc_h_and_native_details_are_not_discarded():
    source = MPCSource()
    source._query_mpc = Mock(
        return_value={
            "semimajor_axis": 1.23,
            "eccentricity": 0.2,
            "inclination": 1.49,
            "absolute_magnitude": 25.9,
            "earth_moid": 0.02,
            "delta_v": 4.0,
            "albedo": 0.52,
        }
    )
    f = DataFetcher.__new__(DataFetcher)
    f.logger = logging.getLogger("acquisition-test")
    neo = f._fetch_from_source("MPC", source, "SYNTHETIC")
    assert neo.physical_properties.absolute_magnitude_h == 25.9
    assert neo.evidence_metadata["native_catalogue_row"]["earth_moid"] == 0.02
    assert source._query_mpc.call_count == 1


def test_configured_plan_survives_source_initialization_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ANEOSConfig()
    cfg.api.data_sources_priority = ["NEODyS", "SBDB", "MPC", "Horizons"]
    with patch("aneos_core.config.settings.get_config", return_value=cfg), patch(
        "aneos_core.data.fetcher.NEODySSource",
        side_effect=RuntimeError("synthetic fail"),
    ):
        f = DataFetcher(cache_manager=Mock())
    assert f.source_priority[0] == "NEODyS"
    assert f.source_plan["sources"] == ["NEODyS", "SBDB", "MPC", "Horizons", "CAD"]
    assert f.source_initialization["NEODyS"] == "unavailable"


def test_physical_only_source_is_retained_and_analysis_can_abstain():
    from aneos_core.analysis.screening import ScreeningAnalysisService

    f = DataFetcher.__new__(DataFetcher)
    f.logger = logging.getLogger("partial-test")
    f.sources = {"MPC": object()}
    f.source_priority = ["MPC"]
    f.max_workers = 1
    f.fetch_stats = {"source_successes": {"MPC": 0}, "source_failures": {"MPC": 0}}
    from aneos_core.data.acquisition import outcome

    f._read_source = Mock(
        return_value=outcome(
            "MPC",
            "SYNTHETIC",
            status="available",
            neo=NEOData(
                "SYNTHETIC", physical_properties=PhysicalProperties(albedo=0.52)
            ),
            basis="controlled_synthetic_fixture",
        )
    )
    f._fetch_close_approaches = Mock(return_value=[])
    n = f._fetch_from_all_sources("SYNTHETIC")
    assert n.physical_properties.albedo == 0.52
    assert n.orbital_elements is None
    assert ScreeningAnalysisService().analyze_neo(n).screening_score is None


@pytest.fixture
def repository(tmp_path):
    from sqlalchemy import create_engine
    from aneos_api.acquisition_repository import SQLEvidenceRepository

    engine = create_engine(
        "sqlite:///" + str(tmp_path / "acquisitions.sqlite"),
        connect_args={"timeout": 10, "check_same_thread": False},
    )
    yield SQLEvidenceRepository(engine)
    engine.dispose()


class Source:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def fetch_orbital_elements(self, designation):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


def fetcher(tmp_path, repository, monkeypatch):
    from aneos_core.data.cache import CacheManager
    from aneos_core.data.sources.cad import parse_cad_payload, CloseApproachQuery

    monkeypatch.chdir(tmp_path)
    f = DataFetcher(
        cache_manager=CacheManager(tmp_path / "cache"), evidence_repository=repository
    )
    f.sources = {
        "SBDB": Source(
            {"semi_major_axis": 1.23, "eccentricity": 0.2, "inclination": 1.49}
        ),
        "MPC": Source(
            {
                "_physical": {"absolute_magnitude_h": 25.9},
                "_evidence_metadata": {
                    "native_catalogue_row": {
                        "absolute_magnitude": "25.9",
                        "earth_moid": 0.02,
                    }
                },
            }
        ),
        "Horizons": Source(error=TimeoutError("private provider exception")),
    }
    meta = parse_cad_payload(
        {
            "signature": {"version": "1.5", "source": "SYNTHETIC"},
            "count": 0,
            "data": [],
        },
        query={"des": "SYNTHETIC", "limit": 10},
        retrieved_at="2026-11-13T00:00:00+00:00",
    )["metadata"]
    f._fetch_close_approaches = Mock(return_value=CloseApproachQuery([], meta))
    return f


def test_ordinary_public_query_commits_fixed_plan_native_evidence_and_export(
    repository, tmp_path, monkeypatch
):
    from sqlalchemy import select, func
    from aneos_api.database import EnrichedNEOService
    from aneos_api.enrichment_store import (
        SourceAssertion,
        ObjectProjection,
        ObjectAcquisition,
    )

    f = fetcher(tmp_path, repository, monkeypatch)
    neo = f.fetch_neo_data("SYNTHETIC")
    assert neo.physical_properties.absolute_magnitude_h == 25.9
    meta = neo.evidence_metadata["acquisition"]
    assert meta["coverage"]["planned_sources"] == 5
    assert meta["coverage"]["contributing_sources"] == 2
    assert meta["coverage"]["validated_response_sources"] == 3
    assert meta["coverage"]["contributing_fraction"] == 0.4
    assert meta["coverage"]["scientific_completeness"] is None
    with repository.sessions() as db:
        assert db.scalar(select(func.count()).select_from(SourceAssertion)) == 7
        assert db.scalar(select(func.count()).select_from(ObjectAcquisition)) == 1
        assert db.scalar(select(func.count()).select_from(ObjectProjection)) == 1
        service = EnrichedNEOService(db)
        data = service.get_enriched_neo("SYNTHETIC")
        assert data["latest_acquisition"]["query_id"] == meta["query_id"]
        assert data["physical_properties"]["absolute_magnitude_h"] == 25.9
        service.export_enriched_json("SYNTHETIC", tmp_path / "object.json")
    import json

    exported = json.loads((tmp_path / "object.json").read_text())
    outcomes = exported["acquisition_history"][0]["bundle"]["outcomes"]
    assert [o["source"] for o in outcomes] == [
        "SBDB",
        "NEODyS",
        "MPC",
        "Horizons",
        "CAD",
    ]
    assert outcomes[1]["reason_code"] == "SOURCE_INITIALIZATION_FAILED"
    assert (
        outcomes[2]["adapter_payload"]["_evidence_metadata"]["native_catalogue_row"][
            "earth_moid"
        ]
        == 0.02
    )
    assert "private provider exception" not in (tmp_path / "object.json").read_text()


def test_cache_hit_reuses_original_query_and_repairs_missing_database_copy_without_repoll(
    repository, tmp_path, monkeypatch
):
    from sqlalchemy import create_engine, select, func
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from aneos_api.enrichment_store import ObjectAcquisition, SourceAssertion

    f = fetcher(tmp_path, repository, monkeypatch)
    first = f.fetch_neo_data("SYNTHETIC")
    second = f.fetch_neo_data("SYNTHETIC")
    assert first.to_dict() == second.to_dict()
    assert sum(s.calls for s in f.sources.values()) == 3
    with repository.sessions() as db:
        assert db.scalar(select(func.count()).select_from(ObjectAcquisition)) == 1
        assert db.scalar(select(func.count()).select_from(SourceAssertion)) == 7
    engine = create_engine("sqlite:///" + str(tmp_path / "another-local.sqlite"))
    f.evidence_repository = SQLEvidenceRepository(engine)
    third = f.fetch_neo_data("SYNTHETIC")
    assert (
        third.evidence_metadata["acquisition"]["query_id"]
        == first.evidence_metadata["acquisition"]["query_id"]
    )
    assert third.physical_properties.to_dict() == first.physical_properties.to_dict()
    assert sum(s.calls for s in f.sources.values()) == 3
    engine.dispose()


def test_all_failed_query_preserves_outcomes_without_fabricated_object(
    repository, tmp_path, monkeypatch
):
    from aneos_core.utils.errors import DataSourceUnavailableError
    from aneos_api.database import EnrichedNEOService
    from sqlalchemy import select, func
    from aneos_api.enrichment_store import ObjectAcquisition

    f = fetcher(tmp_path, repository, monkeypatch)
    f.sources = {
        s: Source(error=TimeoutError("secret")) for s in ("SBDB", "MPC", "Horizons")
    }
    f._fetch_close_approaches = Mock(side_effect=TimeoutError("secret"))
    with pytest.raises(DataSourceUnavailableError) as caught:
        f.fetch_neo_data("SYNTHETIC")
    receipt = caught.value.acquisition_receipt
    assert receipt["commit_status"] == "committed"
    with repository.sessions() as db:
        assert db.scalar(select(func.count()).select_from(ObjectAcquisition)) == 1
        result = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
        assert result["storage_status"] == "attempts_only"
        assert result.get("neo_data") is None
        assert len(result["acquisition_history"][0]["bundle"]["outcomes"]) == 5


def test_persistence_failure_never_returns_success_or_caches_source_evidence(
    tmp_path, monkeypatch
):
    from aneos_core.utils.errors import EvidencePersistenceError

    broken = Mock()
    broken.prepare_acquisition.return_value = {"status": "ready"}
    broken.commit_acquisition.side_effect = OSError("private path")
    f = fetcher(tmp_path, broken, monkeypatch)
    cache = Mock()
    cache.get.return_value = None
    f.cache_manager = cache
    with pytest.raises(
        EvidencePersistenceError, match="could not be committed"
    ) as caught:
        f.fetch_neo_data("SYNTHETIC")
    assert caught.value.acquisition_bundle["neo_data"] is not None
    assert "private path" not in str(caught.value)
    cache.set.assert_not_called()


def fake_mpc(monkeypatch, query):
    import sys, types

    module = types.ModuleType("astroquery.mpc")
    module.MPC = types.SimpleNamespace(query_objects=query)
    monkeypatch.setitem(sys.modules, "astroquery", types.ModuleType("astroquery"))
    monkeypatch.setitem(sys.modules, "astroquery.mpc", module)


def test_provisional_designation_never_falls_back_to_its_year_as_object_number(
    monkeypatch,
):
    calls = []

    def query(*args, **kwargs):
        calls.append(kwargs)
        return (
            [{"number": 1998, "name": "WRONG OBJECT"}]
            if kwargs.get("number") == 1998
            else []
        )

    fake_mpc(monkeypatch, query)
    assert MPCSource()._query_mpc("1998 KY26") is None
    assert calls == [{"limit": 2, "designation": "1998+KY26"}]


def test_ambiguous_mpc_results_are_not_silently_first_selected(monkeypatch):
    from aneos_core.utils.errors import AmbiguousObjectError

    fake_mpc(monkeypatch, lambda *a, **k: [{"number": 1}, {"number": 2}])
    with pytest.raises(AmbiguousObjectError):
        MPCSource()._query_mpc("SYNTHETIC")
    assert MPCSource().fetch_acquisition("SYNTHETIC").outcome == "ambiguous"


def test_legacy_mpc_http_adapter_does_not_manufacture_missing_elements(monkeypatch):
    import requests

    response = Mock()
    response.json.return_value = [{"number": 99942, "H": 0}]
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    result = MPCSource()._make_request("99942")
    assert result["a"] is None and result["e"] is None and result["i"] is None
    assert result["H"] == 0


@pytest.mark.parametrize(
    "designation,row,lookup",
    [
        (
            "1998 KY26",
            {"designation": "1998 KY26", "number": None},
            {"designation": "1998+KY26"},
        ),
        ("99942", {"number": 99942, "name": "Apophis"}, {"number": 99942}),
        ("99942 Apophis", {"number": 99942, "name": "Apophis"}, {"number": 99942}),
        ("Apophis", {"number": 99942, "name": "Apophis"}, {"name": "Apophis"}),
    ],
)
def test_mpc_identity_positive_controls_use_exact_native_binding(
    monkeypatch, designation, row, lookup
):
    query = Mock(return_value=[row])
    fake_mpc(monkeypatch, query)
    result = MPCSource()._query_mpc(designation)
    assert dict(result) == row
    query.assert_called_once_with("asteroid", limit=2, **lookup)
    assert (
        result.identity_assessment["status"]
        == "matched_native_identifier_not_authenticated"
    )


def test_mpc_identity_mismatch_and_ambiguity_are_archived_without_contribution(
    repository, tmp_path, monkeypatch
):
    f = fetcher(tmp_path, repository, monkeypatch)
    query = Mock(return_value=[{"number": 1, "name": "OTHER", "semimajor_axis": 2}])
    fake_mpc(monkeypatch, query)
    f.sources["MPC"] = MPCSource()
    n = f.fetch_neo_data("SYNTHETIC")
    assert n.physical_properties is None
    record = repository.load_acquisition(n.evidence_metadata["acquisition"]["query_id"])
    mpc = next(o for o in record["bundle"]["outcomes"] if o["source"] == "MPC")
    assert (
        mpc["status"] == "invalid"
        and mpc["reason_code"] == "MPC_NATIVE_IDENTITY_MISMATCH"
    )
    assert mpc["adapter_payload"]["_native_response"]["number"] == 1
    query.return_value = [{"number": 1}, {"number": 2}]
    n = f.fetch_neo_data("SYNTHETIC", force_refresh=True)
    record = repository.load_acquisition(n.evidence_metadata["acquisition"]["query_id"])
    mpc = next(o for o in record["bundle"]["outcomes"] if o["source"] == "MPC")
    assert (
        mpc["status"] == "ambiguous"
        and len(mpc["adapter_payload"]["_native_response"]) == 2
    )


def test_cache_write_and_read_failures_do_not_undo_committed_evidence(
    repository, tmp_path, monkeypatch
):
    f = fetcher(tmp_path, repository, monkeypatch)
    cache = Mock()
    cache.get.side_effect = OSError("private cache path")
    cache.set.side_effect = OSError("private cache path")
    f.cache_manager = cache
    neo = f.fetch_neo_data("SYNTHETIC")
    assert (
        repository.load_acquisition(neo.evidence_metadata["acquisition"]["query_id"])
        is not None
    )
    assert f.fetch_stats["cache_errors"] == 2


def test_cache_hit_does_not_extend_original_cache_ttl(
    repository, tmp_path, monkeypatch
):
    f = fetcher(tmp_path, repository, monkeypatch)
    first = f.fetch_neo_data("SYNTHETIC")
    saved = f.cache_manager.get
    f.cache_manager.set = Mock(wraps=f.cache_manager.set)
    assert f.fetch_neo_data("SYNTHETIC").to_dict() == first.to_dict()
    f.cache_manager.set.assert_not_called()


def test_identical_native_mpc_responses_are_labeled_repeats_not_independent_observations(
    repository, tmp_path, monkeypatch
):
    from aneos_api.database import EnrichedNEOService

    f = fetcher(tmp_path, repository, monkeypatch)
    source = MPCSource()
    source._query_mpc = Mock(return_value={"absolute_magnitude": "25.9"})
    f.sources["MPC"] = source
    first = f.fetch_neo_data("SYNTHETIC")
    second = f.fetch_neo_data("SYNTHETIC", force_refresh=True)
    assert (
        first.evidence_metadata["acquisition"]["query_id"]
        != second.evidence_metadata["acquisition"]["query_id"]
    )
    with repository.sessions() as db:
        records = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")[
            "acquisition_history"
        ]
    assert (
        records[0]["native_response_repeat_checks"][0]["exact_repeat_of_query"] is None
    )
    assert (
        records[1]["native_response_repeat_checks"][0]["exact_repeat_of_query"]
        == records[0]["bundle"]["query_id"]
    )


def test_invalid_source_projection_cannot_poison_other_sources(
    repository, tmp_path, monkeypatch
):
    from aneos_core.data.acquisition import outcome

    f = fetcher(tmp_path, repository, monkeypatch)
    old = f._read_source

    def read(name, source, designation):
        value = old(name, source, designation)
        if name == "MPC":
            value["neo_data"]["physical_properties"]["absolute_magnitude_h"] = 1
        return value

    f._read_source = read
    n = f.fetch_neo_data("SYNTHETIC")
    assert n.orbital_elements.semi_major_axis == 1.23
    mpc = next(
        o
        for o in n.evidence_metadata["acquisition"]["outcomes"]
        if o["source"] == "MPC"
    )
    assert mpc["status"] == "invalid" and mpc["reason_code"] == "SOURCE_OUTCOME_INVALID"


def test_cache_replay_never_reapplies_an_older_global_projection_policy(
    repository, tmp_path, monkeypatch
):
    from aneos_core.data.acquisition import plan
    from aneos_api.database import EnrichedNEOService

    f = fetcher(tmp_path, repository, monkeypatch)
    f.sources["MPC"] = Source(
        {"semi_major_axis": 1.5, "eccentricity": 0.3, "inclination": 2}
    )
    first = f.fetch_neo_data("SYNTHETIC")
    newer = fetcher(tmp_path, repository, monkeypatch)
    newer.sources["MPC"] = f.sources["MPC"]
    newer.source_plan = plan(["MPC", "SBDB", "NEODyS", "Horizons"])
    assert (
        newer.fetch_neo_data(
            "SYNTHETIC", force_refresh=True
        ).orbital_elements.semi_major_axis
        == 1.5
    )
    assert f.fetch_neo_data("SYNTHETIC").orbital_elements.semi_major_axis == 1.23
    with repository.sessions() as db:
        data = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
    assert data["orbital_elements"]["semi_major_axis"] == 1.5
    assert len(data["acquisition_history"]) == 2


def test_storage_prerequisite_failure_prevents_source_queries(tmp_path, monkeypatch):
    from aneos_core.utils.errors import EvidencePersistenceError

    broken = Mock()
    broken.prepare_acquisition.side_effect = OSError("private")
    f = fetcher(tmp_path, broken, monkeypatch)
    with pytest.raises(EvidencePersistenceError, match="sources not queried"):
        f.fetch_neo_data("SYNTHETIC")
    assert all(s.calls == 0 for s in f.sources.values())
    f._fetch_close_approaches.assert_not_called()
    broken.commit_acquisition.assert_not_called()


def test_default_repository_composition_persists_without_an_injected_sink(tmp_path):
    import subprocess, sys, os, json
    from sqlalchemy import create_engine
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from pathlib import Path

    path = tmp_path / "automatic.sqlite"
    code = """
import json
from unittest.mock import Mock
from aneos_core.data.fetcher import DataFetcher
from aneos_core.data.sources.cad import CloseApproachQuery
f=DataFetcher(cache_manager=Mock())
f.cache_manager.get.return_value=None
class Source:
 def fetch_orbital_elements(self,d):return {'semi_major_axis':1.2,'eccentricity':.2,'inclination':1.49}
f.sources={'SBDB':Source()}
f._fetch_close_approaches=lambda d:CloseApproachQuery([],{'api_version':'1.5','count':0})
n=f.fetch_neo_data('SYNTHETIC')
import sys
assert 'aneos_api.app' not in sys.modules
print(json.dumps(n.evidence_metadata['acquisition']))
"""
    env = {
        **os.environ,
        "ANEOS_DATABASE_URL": "sqlite:///" + str(path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**env, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(proc.stdout)
    engine = create_engine("sqlite:///" + str(path))
    repo = SQLEvidenceRepository(engine)
    record = repo.load_acquisition(summary["query_id"])
    assert record["bundle"]["neo_data"]["orbital_elements"]["semi_major_axis"] == 1.2
    assert record["receipt"]["commit_status"] == "committed"
    engine.dispose()


def test_cache_disk_keys_do_not_collide_after_long_prefix_or_path_normalization(
    tmp_path,
):
    from aneos_core.data.cache import CacheManager

    cache = CacheManager(tmp_path / "cache")
    values = {"x" * 140 + "A": 1, "x" * 140 + "B": 2, "A/B": 3, "A_B": 4}
    for key, value in values.items():
        cache.set(key, value)
    fresh = CacheManager(tmp_path / "cache")
    assert {k: fresh.get(k) for k in values} == values


def test_cache_initialization_does_not_delete_or_load_legacy_pickle_bytes(tmp_path):
    from aneos_core.data.cache import CacheManager

    directory = tmp_path / "cache"
    directory.mkdir()
    original = directory / "historical.pickle"
    original.write_bytes(b"not executed; preserve original bytes")
    CacheManager(directory)
    assert original.read_bytes() == b"not executed; preserve original bytes"


def test_workspace_and_custom_endpoint_scopes_do_not_share_object_query_cache(
    tmp_path, monkeypatch, repository
):
    from aneos_core.data.cache import CacheManager

    shared = CacheManager(tmp_path / "shared-cache")
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    f = fetcher(one, repository, monkeypatch)
    f.cache_manager = shared
    first = f.fetch_neo_data("SYNTHETIC")
    g = fetcher(two, repository, monkeypatch)
    g.cache_manager = shared
    second = g.fetch_neo_data("SYNTHETIC")
    assert (
        first.evidence_metadata["acquisition"]["query_id"]
        != second.evidence_metadata["acquisition"]["query_id"]
    )
    assert sum(s.calls for s in g.sources.values()) == 3
    cfg = ANEOSConfig()
    cfg.api.sbdb_url = "https://user:secret@example.invalid/private"
    with patch("aneos_core.config.settings.get_config", return_value=cfg):
        a = DataFetcher(cache_manager=shared)
        b = DataFetcher(cache_manager=shared)
    assert a.source_plan["context_id"] != b.source_plan["context_id"]
    import json

    assert "secret" not in json.dumps(a.source_plan)


def test_invalid_cad_response_is_preserved_as_invalid_not_empty_or_zero(
    repository, tmp_path, monkeypatch
):
    f = fetcher(tmp_path, repository, monkeypatch)
    del f._fetch_close_approaches
    response = Mock()
    payload = {
        "signature": {"version": "unexpected", "source": "SYNTHETIC"},
        "count": 0,
        "data": [],
    }
    response.json.return_value = payload
    monkeypatch.setattr(
        "aneos_core.data.fetcher.provider_get", lambda *a, **k: response
    )
    n = f.fetch_neo_data("SYNTHETIC")
    record = repository.load_acquisition(n.evidence_metadata["acquisition"]["query_id"])
    cad = record["bundle"]["outcomes"][-1]
    assert (
        cad["status"] == "invalid"
        and cad["adapter_payload"]["_native_response"] == payload
    )
    assert cad["neo_data"] is None
    response.close.assert_called_once()


def test_query_metadata_records_software_precision_versions_and_code_identity(
    repository, tmp_path, monkeypatch
):
    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    software = repository.load_acquisition(
        n.evidence_metadata["acquisition"]["query_id"]
    )["bundle"]["software"]
    assert software["float_mantissa_bits"] == 53
    assert software["package_versions"]["SQLAlchemy"]
    assert software["repository_commit"]
    assert software["source_hashes_at_report_time"]["aneos_core/data/acquisition.py"]


def test_invalid_legacy_slot_is_preserved_without_blocking_new_acquisition(
    repository, tmp_path, monkeypatch
):
    from aneos_api.database import EnrichedNEO, EnrichedNEOService
    from aneos_api.enrichment_store import initialize_schema

    initialize_schema(repository.engine, include_all=True)
    with repository.sessions() as db:
        row = EnrichedNEO(
            designation="SYNTHETIC",
            data_sources=["SBDB"],
            physical_properties={"diameter_km": 0.011},
            nasa_sbdb_data={
                "designation": "SYNTHETIC",
                "physical_properties": {"diameter_km": -1},
            },
        )
        db.add(row)
        db.commit()
    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    assert n.physical_properties.absolute_magnitude_h == 25.9
    with repository.sessions() as db:
        data = EnrichedNEOService(db).get_enriched_neo("SYNTHETIC")
    raw = next(r for r in data["evidence_history"] if r["kind"] == "legacy_projection")[
        "payload"
    ]
    assert raw["nasa_sbdb_data"]["physical_properties"]["diameter_km"] == -1
    assert any(
        r["kind"] == "legacy_source_uninterpretable" for r in data["evidence_history"]
    )


def test_full_named_catalogue_presence_and_source_responses_are_not_full_scientific_completeness(
    repository, tmp_path, monkeypatch
):
    from aneos_core.data.epoch_evidence import NativeEpoch
    from aneos_core.data.sources.cad import (
        parse_cad_payload,
        close_approach_from_record,
        CloseApproachQuery,
    )
    from aneos_core.analysis.object_review import build_data_inventory
    from aneos_core.analysis.screening import ScreeningAnalysisService

    f = fetcher(tmp_path, repository, monkeypatch)
    basic = {"semi_major_axis": 1.23, "eccentricity": 0.2, "inclination": 1.49}
    full = dict(
        basic,
        ra_of_ascending_node=1.0,
        arg_of_periapsis=2.0,
        mean_anomaly=3.0,
        native_epoch=NativeEpoch("2460000.5", "JD", "TDB", "SBDB").to_dict(),
        first_observation_date="2026-Nov-01",
        last_observation_date="2026-Nov-08",
        _physical={
            "diameter": 0.011,
            "absolute_magnitude_h": 25.9,
            "albedo": 0.52,
            "rot_per": 0.089,
            "spectral_type": "Xe",
        },
    )
    f.sources = {
        name: Source(full if name == "SBDB" else basic)
        for name in ("SBDB", "NEODyS", "MPC", "Horizons")
    }
    native = {
        "signature": {"source": "SYNTHETIC", "version": "1.5"},
        "count": 1,
        "total": 1,
        "fields": [
            "des",
            "orbit_id",
            "jd",
            "cd",
            "dist",
            "dist_min",
            "dist_max",
            "v_rel",
            "v_inf",
            "t_sigma_f",
            "h",
        ],
        "data": [
            [
                "SYNTHETIC",
                "synthetic-orbit",
                "2460000.5",
                "2023-Feb-25 00:00",
                ".1",
                ".09",
                ".11",
                "1.2",
                "1.1",
                "00:05",
                "25.9",
            ]
        ],
    }
    parsed = parse_cad_payload(
        native,
        query={"des": "SYNTHETIC", "limit": 10},
        retrieved_at="2026-11-13T00:00:00+00:00",
    )
    f._fetch_close_approaches = lambda d: CloseApproachQuery(
        [close_approach_from_record(r) for r in parsed["records"]], parsed["metadata"]
    )
    n = f.fetch_neo_data("SYNTHETIC")
    assert n.evidence_metadata["acquisition"]["coverage"]["contributing_fraction"] == 1
    inventory = build_data_inventory(n)
    assert inventory["coverage"]["basic_catalogue"]["present"] == 15
    assert inventory["coverage"]["overall_data_completeness"] is None
    result = ScreeningAnalysisService().analyze_neo(n)
    assert (
        result.sigma_confidence is None
        and result.artificial_probability is None
        and result.false_discovery_rate is None
    )


def test_source_attempts_projection_and_query_record_rollback_together(
    repository, tmp_path, monkeypatch
):
    from sqlalchemy import event, select, func
    from sqlalchemy.orm import Session
    from aneos_api.enrichment_store import (
        SourceAssertion,
        ObjectProjection,
        ObjectAcquisition,
    )
    from aneos_api.database import EnrichedNEO
    from aneos_core.utils.errors import EvidencePersistenceError

    f = fetcher(tmp_path, repository, monkeypatch)

    def fail(session):
        raise RuntimeError("injected commit failure")

    event.listen(Session, "before_commit", fail)
    try:
        with pytest.raises(EvidencePersistenceError):
            f.fetch_neo_data("SYNTHETIC")
    finally:
        event.remove(Session, "before_commit", fail)
    with repository.sessions() as db:
        for table in (
            SourceAssertion,
            ObjectProjection,
            ObjectAcquisition,
            EnrichedNEO,
        ):
            assert db.scalar(select(func.count()).select_from(table)) == 0


def test_concurrent_identical_receipt_retries_are_idempotent(
    repository, tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from sqlalchemy import select, func
    from aneos_api.enrichment_store import ObjectAcquisition, SourceAssertion

    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    record = repository.load_acquisition(n.evidence_metadata["acquisition"]["query_id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(repository.commit_acquisition, [record["bundle"]] * 2))
    assert receipts[0] == receipts[1] == record["receipt"]
    with repository.sessions() as db:
        assert db.scalar(select(func.count()).select_from(ObjectAcquisition)) == 1
        assert db.scalar(select(func.count()).select_from(SourceAssertion)) == 7


def test_changed_query_with_reused_identifier_is_not_committed(
    repository, tmp_path, monkeypatch
):
    from aneos_core.data.acquisition import build_bundle
    from aneos_core.utils.errors import EvidencePersistenceError

    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    record = repository.load_acquisition(n.evidence_metadata["acquisition"]["query_id"])
    b = record["bundle"]
    changed = build_bundle(
        b["designation"],
        b["source_plan"],
        b["outcomes"],
        query_id=b["query_id"],
        started_at="2020-01-01T00:00:00+00:00",
        software=b["software"],
    )
    with pytest.raises(EvidencePersistenceError):
        repository.commit_acquisition(changed)
    assert repository.load_acquisition(b["query_id"]) == record


def test_query_table_is_append_only_and_corrupt_query_is_not_served(
    repository, tmp_path, monkeypatch
):
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    qid = n.evidence_metadata["acquisition"]["query_id"]
    for statement in (
        "UPDATE neo_acquisitions_v1 SET bundle_json='{}'",
        "DELETE FROM neo_acquisitions_v1",
    ):
        with pytest.raises(IntegrityError):
            with repository.engine.begin() as c:
                c.execute(text(statement))
    # Deliberately bypass only this test-owned DB trigger to test read consistency, not admin security.
    with repository.engine.begin() as c:
        c.exec_driver_sql("DROP TRIGGER neo_acquisitions_v1_no_update")
        c.execute(text("UPDATE neo_acquisitions_v1 SET bundle_json='{}'"))
        c.exec_driver_sql(
            "CREATE TRIGGER neo_acquisitions_v1_no_update BEFORE UPDATE ON neo_acquisitions_v1 BEGIN SELECT RAISE(ABORT,'immutable acquisition evidence'); END"
        )
    with pytest.raises(ValueError):
        repository.load_acquisition(qid)


def test_source_status_display_validates_counts_and_renders_rich_and_plain(
    repository, tmp_path, monkeypatch, capsys
):
    import io, copy
    from rich.console import Console
    from aneos_core.analysis.acquisition_view import (
        render_acquisition_summary,
        verify_summary,
    )
    from aneos_core.analysis.object_review import (
        build_object_review,
        object_review_markdown,
        render_object_review,
    )

    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    summary = n.evidence_metadata["acquisition"]
    for width in (80, 120, 160):
        stream = io.StringIO()
        console = Console(file=stream, width=width, color_system=None)
        render_acquisition_summary(summary, console=console)
        text = stream.getvalue()
        assert "2/5" in text and "3/5" in text and "NEODyS" in text
        assert max(map(len, text.splitlines())) <= width
    render_acquisition_summary(summary)
    assert "not field completeness" in capsys.readouterr().out
    bad = copy.deepcopy(summary)
    bad["coverage"]["contributing_sources"] = 5
    with pytest.raises(ValueError):
        verify_summary(bad)
    review = build_object_review(n)
    assert "## Recorded acquisition" in object_review_markdown(review)
    stream = io.StringIO()
    render_object_review(review, console=Console(file=stream, width=120))
    assert "Recorded acquisition outcomes" in stream.getvalue()


def test_stored_projection_flags_earlier_source_values_and_plain_detail_is_readable(
    repository, tmp_path, monkeypatch, capsys
):
    from aneos_api.database import EnrichedNEOService
    from aneos_menu import ANEOSMenu

    f = fetcher(tmp_path, repository, monkeypatch)
    f.fetch_neo_data("SYNTHETIC")
    f.sources["SBDB"] = Source(error=TimeoutError())
    f.fetch_neo_data("SYNTHETIC", force_refresh=True)
    ui = ANEOSMenu.__new__(ANEOSMenu)
    ui.console = None
    ui.wait_for_input = Mock()
    with repository.sessions() as db:
        service = EnrichedNEOService(db)
        data = service.get_enriched_neo("SYNTHETIC")
        assert (
            next(x for x in data["projection_query_bindings"] if x["source"] == "SBDB")[
                "from_latest_query"
            ]
            is False
        )
        ui._show_enriched_neo_details(service, "SYNTHETIC")
    text = capsys.readouterr().out
    assert (
        "earlier query" in text
        and "Recorded acquisition" in text
        and "not calibrated" in text
    )
    assert "'bundle_json':" not in text


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "software",
        "plan",
        "outcome_order",
        "unknown_status",
        "wrong_designation",
        "outcome_timezone",
        "nonmaterial",
        "projection",
        "coverage",
        "bundle_hash",
        "native_hash",
    ],
)
def test_acquisition_envelope_rejects_inconsistent_schema_and_projection(
    repository, tmp_path, monkeypatch, mutation
):
    import copy
    from aneos_core.data.acquisition import validate_bundle

    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    b = copy.deepcopy(
        repository.load_acquisition(n.evidence_metadata["acquisition"]["query_id"])[
            "bundle"
        ]
    )
    if mutation == "schema":
        b["schema_version"] = "unsupported"
    elif mutation == "software":
        b["software"] = {}
    elif mutation == "plan":
        b["source_plan"]["sources"].pop()
    elif mutation == "outcome_order":
        b["outcomes"].reverse()
    elif mutation == "unknown_status":
        b["outcomes"][0]["status"] = "validated_origin"
    elif mutation == "wrong_designation":
        b["outcomes"][0]["designation"] = "OTHER"
    elif mutation == "outcome_timezone":
        b["outcomes"][0]["completed_at"] = "2026-01-01T00:00:00"
    elif mutation == "nonmaterial":
        b["outcomes"][0]["neo_data"]["orbital_elements"] = None
    elif mutation == "projection":
        b["neo_data"]["orbital_elements"]["semi_major_axis"] = 2
    elif mutation == "coverage":
        b["coverage"]["contributing_fraction"] = 1
    elif mutation == "bundle_hash":
        b["bundle_id"] = "0" * 64
    elif mutation == "native_hash":
        b["outcomes"][0]["native_response_sha256"] = "0" * 64
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_bundle(b)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"_designation": "OTHER", "semi_major_axis": 1},
        {"_physical": []},
        {"_physical": {"diameter": 1, "diameter_km": 2}},
        {"_physical": {"albedo": True}},
        {"_physical": {"albedo_uncertainty": -1}},
        {"_physical": {"spectral_type": 123}},
        {"_physical": {"color_indices": {"V-R": True}}},
    ],
)
def test_uninterpretable_adapter_values_never_become_measured_zero(payload):
    from aneos_core.data.acquisition import adapter_to_neo

    with pytest.raises((ValueError, TypeError, KeyError)):
        adapter_to_neo("SBDB", "SYNTHETIC", payload, "2026-01-01T00:00:00+00:00")


def test_observation_and_retrieval_dates_do_not_gain_an_undeclared_timezone():
    original = {
        "designation": "SYNTHETIC",
        "first_observation": "2024-01-01T00:00:00",
        "last_observation": "2024-02-01T00:00:00",
        "fetched_at": "2024-02-02T01:02:03",
    }
    neo = NEOData.from_dict(original)
    for key in ("first_observation", "last_observation", "fetched_at"):
        assert getattr(neo, key).tzinfo is None
        assert neo.to_dict()[key] == original[key]


def test_simultaneous_first_commit_of_same_query_has_one_receipt(
    repository, tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from sqlalchemy import select, func
    from aneos_api.enrichment_store import ObjectAcquisition, SourceAssertion

    f = fetcher(tmp_path, repository, monkeypatch)
    repository.prepare_acquisition()
    bundle = f._fetch_from_all_sources("SYNTHETIC")._acquisition_bundle
    original = repository.load_acquisition
    barrier = threading.Barrier(2)
    state = threading.local()

    def load(qid):
        result = original(qid)
        if not getattr(state, "entered", False):
            state.entered = True
            barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(repository, "load_acquisition", load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(repository.commit_acquisition, [bundle, bundle]))
    assert results[0] == results[1]
    with repository.sessions() as db:
        assert db.scalar(select(func.count()).select_from(ObjectAcquisition)) == 1
        assert db.scalar(select(func.count()).select_from(SourceAssertion)) == 7


def test_unparsed_native_calendar_and_legacy_acceleration_stay_explicit():
    from aneos_core.data.acquisition import adapter_to_neo

    n = adapter_to_neo(
        "SBDB",
        "SYNTHETIC",
        {"_nongrav": {"a3": 1e-12}, "first_observation_date": "date unresolved"},
        "2026-01-01T00:00:00+00:00",
    )
    assert n.first_observation is None
    assert n.nongrav.a3 == 1e-12
    assert (
        n.evidence_metadata["native_observation_window"]["first_observation"]["value"]
        == "date unresolved"
    )


@pytest.mark.parametrize("mutation", ["schema", "order", "status", "receipt"])
def test_source_summary_does_not_promote_edited_counts_or_receipts(
    repository, tmp_path, monkeypatch, mutation
):
    import copy
    from aneos_core.analysis.acquisition_view import verify_summary

    f = fetcher(tmp_path, repository, monkeypatch)
    s = copy.deepcopy(f.fetch_neo_data("SYNTHETIC").evidence_metadata["acquisition"])
    if mutation == "schema":
        s["schema_version"] = "unknown"
    elif mutation == "order":
        s["outcomes"].reverse()
    elif mutation == "status":
        s["outcomes"][0]["status"] = "confirmed"
    else:
        s["storage_receipt"]["commit_status"] = "not_committed"
    with pytest.raises(ValueError):
        verify_summary(s)


@pytest.mark.parametrize(
    "field,value",
    [
        ("assertion_ids_json", "[]"),
        ("projection_id", "0" * 64),
        ("bundle_id", "0" * 64),
    ],
)
def test_stored_query_relation_corruption_fails_closed(
    repository, tmp_path, monkeypatch, field, value
):
    from sqlalchemy import text

    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    qid = n.evidence_metadata["acquisition"]["query_id"]
    with repository.engine.begin() as c:
        c.exec_driver_sql("DROP TRIGGER neo_acquisitions_v1_no_update")
        c.execute(
            text(f"UPDATE neo_acquisitions_v1 SET {field}=:value"), {"value": value}
        )
        c.exec_driver_sql(
            "CREATE TRIGGER neo_acquisitions_v1_no_update BEFORE UPDATE ON neo_acquisitions_v1 BEGIN SELECT RAISE(ABORT,'immutable'); END"
        )
    with pytest.raises(ValueError):
        repository.load_acquisition(qid)


def test_query_missing_assertion_is_not_reported_as_intact(
    repository, tmp_path, monkeypatch
):
    from sqlalchemy import text

    f = fetcher(tmp_path, repository, monkeypatch)
    n = f.fetch_neo_data("SYNTHETIC")
    qid = n.evidence_metadata["acquisition"]["query_id"]
    with repository.engine.begin() as c:
        c.exec_driver_sql("DROP TRIGGER neo_source_assertions_v1_no_delete")
        c.execute(
            text(
                "DELETE FROM neo_source_assertions_v1 WHERE sequence=(SELECT MIN(sequence) FROM neo_source_assertions_v1)"
            )
        )
        c.exec_driver_sql(
            "CREATE TRIGGER neo_source_assertions_v1_no_delete BEFORE DELETE ON neo_source_assertions_v1 BEGIN SELECT RAISE(ABORT,'immutable'); END"
        )
    with pytest.raises(ValueError):
        repository.load_acquisition(qid)


def test_absent_and_incompatible_query_schema_have_distinct_outcomes(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from aneos_api.acquisition_repository import (
        query_history,
        initialize_acquisition_schema,
    )

    engine = create_engine("sqlite:///" + str(tmp_path / "legacy.sqlite"))
    with Session(engine) as db:
        assert query_history(db, "SYNTHETIC") == []
    with engine.begin() as c:
        c.exec_driver_sql("CREATE TABLE neo_acquisitions_v1 (wrong TEXT)")
    with Session(engine) as db:
        with pytest.raises(ValueError):
            query_history(db, "SYNTHETIC")
    with pytest.raises(RuntimeError):
        initialize_acquisition_schema(engine)
    engine.dispose()


def test_incompatible_acquisition_unique_identity_requires_migration(tmp_path):
    from sqlalchemy import create_engine
    from aneos_api.acquisition_repository import initialize_acquisition_schema

    engine = create_engine("sqlite:///" + str(tmp_path / "incompatible.sqlite"))
    with engine.begin() as c:
        c.exec_driver_sql(
            "CREATE TABLE neo_acquisitions_v1 (sequence INTEGER PRIMARY KEY, query_id TEXT, designation TEXT, bundle_id TEXT UNIQUE, bundle_json TEXT, assertion_ids_json TEXT, projection_id TEXT, committed_at TEXT)"
        )
    with pytest.raises(RuntimeError):
        initialize_acquisition_schema(engine)
    engine.dispose()


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_real_process_kill_respects_query_transaction_boundary(
    repository, tmp_path, phase
):
    import subprocess, sys, os, time, json
    from pathlib import Path
    from sqlalchemy import select, func
    from aneos_api.enrichment_store import ObjectAcquisition, SourceAssertion

    marker = tmp_path / "marker"
    url = str(repository.engine.url)
    root = Path(__file__).resolve().parents[1]
    code = """
import os,json,time,sys
from pathlib import Path
from unittest.mock import Mock
from sqlalchemy import create_engine,event
from sqlalchemy.orm import Session
from aneos_api.acquisition_repository import SQLEvidenceRepository
from aneos_core.data.fetcher import DataFetcher
from aneos_core.data.sources.cad import parse_cad_payload,CloseApproachQuery
repo=SQLEvidenceRepository(create_engine(sys.argv[1]))
repo.prepare_acquisition()
f=DataFetcher(cache_manager=Mock(),evidence_repository=repo)
f.cache_manager.get.return_value=None
class Source:
 def fetch_orbital_elements(self,d):return {'semi_major_axis':1.23,'eccentricity':.2,'inclination':1.49}
f.sources={'SBDB':Source()}
meta=parse_cad_payload({'signature':{'version':'1.5','source':'SYNTHETIC'},'count':0,'data':[]},query={'des':'SYNTHETIC','limit':10})['metadata']
f._fetch_close_approaches=lambda d:CloseApproachQuery([],meta)
def stop(session):
 Path(sys.argv[2]).write_text('before_commit')
 time.sleep(120)
if sys.argv[3]=='before_commit':event.listen(Session,'before_commit',stop)
n=f.fetch_neo_data('SYNTHETIC')
Path(sys.argv[2]).write_text(json.dumps(n.evidence_metadata['acquisition']))
time.sleep(120)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, url, str(marker), phase],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "ANEOS_DATABASE_URL": url,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 25
        while (
            not marker.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert marker.exists(), "Child did not reach the transaction marker"
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.communicate(timeout=10)
    with repository.sessions() as db:
        assert db.scalar(select(func.count()).select_from(ObjectAcquisition)) == (
            0 if phase == "before_commit" else 1
        )
        assert db.scalar(select(func.count()).select_from(SourceAssertion)) == (
            0 if phase == "before_commit" else 6
        )
    if phase == "after_commit":
        summary = json.loads(marker.read_text())
        assert (
            repository.load_acquisition(summary["query_id"])["receipt"]["commit_status"]
            == "committed"
        )
        read = "from aneos_api.acquisition_repository import SQLEvidenceRepository;from sqlalchemy import create_engine;import sys,json;print(json.dumps(SQLEvidenceRepository(create_engine(sys.argv[1])).load_acquisition(sys.argv[2])['bundle']['neo_data']['orbital_elements']['semi_major_axis']))"
        result = subprocess.run(
            [sys.executable, "-c", read, url, summary["query_id"]],
            cwd=tmp_path,
            env={
                **os.environ,
                "PYTHONPATH": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "ANEOS_DATABASE_URL": url,
            },
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout) == 1.23


def test_http_detection_storage_failure_is_503_not_scientific_result(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aneos_api.endpoints import analysis
    from aneos_core.utils.errors import EvidencePersistenceError
    import aneos_core.data.fetcher as fetch_module

    class Unwritable:
        def fetch_neo_data(self, *a, **k):
            raise EvidencePersistenceError("private path")

    monkeypatch.setattr(fetch_module, "DataFetcher", Unwritable)
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/v1/analysis")
    with TestClient(app) as client:
        response = client.get("/api/v1/analysis/detect?designation=SYNTHETIC")
    assert response.status_code == 503
    assert "storage unavailable" in response.json()["detail"]
    assert "private path" not in response.text


@pytest.mark.parametrize(
    "rows",
    [
        [{"number": 1998}],
        [{"designation": "1998 KY26"}, {"designation": "1998 KY26"}],
    ],
)
def test_legacy_mpc_http_rejects_wrong_or_ambiguous_identity(monkeypatch, rows):
    import requests

    response = Mock()
    response.json.return_value = rows
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    assert MPCSource()._make_request("1998 KY26") is None


def test_actual_acquisition_render_generator_has_bounded_widths_and_no_database(
    tmp_path,
):
    import subprocess, sys, json
    from pathlib import Path
    from hashlib import sha256

    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "renders"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/qa_object_acquisition.py"),
            "--output",
            str(directory),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((directory / "inventory.json").read_text())
    assert len(manifest["metrics"]) == 9
    for name, metric in manifest["metrics"].items():
        assert metric["max_columns"] <= int(name.split("-")[-1])
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest
    assert not list(directory.glob("*.sqlite"))
    assert not list(directory.glob("*.db"))
    again = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/qa_object_acquisition.py"),
            "--output",
            str(directory),
        ],
        capture_output=True,
    )
    assert again.returncode != 0


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("empty_ng", False),
        ("uncertainty_only", False),
        ("node_only", True),
        ("periapsis_only", True),
    ],
)
def test_material_contribution_requires_a_value_not_an_empty_container(kind, expected):
    from aneos_core.data.acquisition import has_material
    from aneos_core.data.models import NonGravitationalParameters

    values = {
        "empty_ng": NEOData("SYNTHETIC", nongrav=NonGravitationalParameters()),
        "uncertainty_only": NEOData(
            "SYNTHETIC",
            physical_properties=PhysicalProperties(diameter_uncertainty=0.002),
        ),
        "node_only": NEOData(
            "SYNTHETIC", orbital_elements=OrbitalElements(ra_of_ascending_node=1.0)
        ),
        "periapsis_only": NEOData(
            "SYNTHETIC", orbital_elements=OrbitalElements(arg_of_periapsis=2.0)
        ),
    }
    assert has_material(values[kind]) is expected
