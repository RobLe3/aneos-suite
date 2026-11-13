"""Provider-shaped synthetic measurements; not scientific population validation."""

import copy
import json
import pytest
from aneos_core.config.settings import APIConfig
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties
from aneos_core.data.sources.sbdb import SBDBSource
from aneos_core.data.fetcher import DataFetcher


def source_payload(records):
    return {
        "signature": {"version": "1.3", "source": "synthetic-provider"},
        "object": {"des": "SYNTHETIC", "spkid": "synthetic-id"},
        "orbit": {
            "orbit_id": "synthetic-orbit",
            "epoch": "2461200.5",
            "model_pars": [],
            "elements": [
                {"name": k, "value": v}
                for k, v in [("a", "1.2"), ("e", ".2"), ("i", "1.5")]
            ],
        },
        "phys_par": records,
    }


def diameter(value=".011", sigma=".002", units="km", ref="synthetic-study"):
    return {
        "name": "diameter",
        "value": value,
        "sigma": sigma,
        "units": units,
        "ref": ref,
        "notes": "synthetic control",
    }


def parse(records):
    return SBDBSource(APIConfig())._parse_sbdb_response(
        source_payload(records), "SYNTHETIC"
    )


def test_adapter_retains_atomic_value_sigma_units_ref_context_and_projection():
    p = parse([diameter()])
    e = p["_evidence_metadata"]["physical_evidence"]
    r = e["records"][0]
    assert r["native"] == diameter()
    assert r["normalized"] == {
        "field": "diameter_km",
        "value": 0.011,
        "sigma": 0.002,
        "units": "km",
        "uncertainty_kind": "provider_1sigma",
    }
    assert r["context"]["object"]["spkid"] == "synthetic-id"
    assert r["context"]["retrieved_at"]
    assert r["measurement_epoch"] is None
    assert p["_physical"]["diameter_uncertainty"] == 0.002


def test_merge_never_attaches_secondary_sigma_to_primary_value_even_equal_rounded_value():
    f = DataFetcher.__new__(DataFetcher)
    f.source_priority = ["A", "B"]

    def n(value, sigma):
        return NEOData(
            "SYNTHETIC",
            orbital_elements=OrbitalElements(semi_major_axis=1.2),
            physical_properties=PhysicalProperties(
                diameter_km=value, diameter_uncertainty=sigma
            ),
        )

    for value in (0.011, 0.02):
        merged = f._reconcile_sources({"A": n(0.011, None), "B": n(value, 0.003)})
        assert merged.physical_properties.diameter_km == 0.011
        assert merged.physical_properties.diameter_uncertainty is None
        assert (
            merged.evidence_metadata["physical_selection"]["diameter_km"]["source"]
            == "A"
        )
        assert len(merged.evidence_metadata["physical_source_snapshots"]) == 2


@pytest.mark.parametrize(
    "record,issue",
    [
        (diameter(units="furlong"), "unsupported_units"),
        (diameter(value="NaN"), "invalid_value"),
        (diameter(value=True), "invalid_value"),
        (diameter(value="-.01"), "invalid_value"),
    ],
)
def test_unusable_native_values_retained_without_promoting(record, issue):
    p = parse([record])
    e = p["_evidence_metadata"]["physical_evidence"]
    assert e["records"][0]["native"] == record
    assert issue in e["records"][0]["issues"]
    assert "diameter" not in p["_physical"]


@pytest.mark.parametrize("sigma", ["+.002/-.001", "bad", "-.002", True, "inf"])
def test_unsupported_uncertainty_is_never_zero_or_guessed(sigma):
    p = parse([diameter(sigma=sigma)])
    r = p["_evidence_metadata"]["physical_evidence"]["records"][0]
    assert r["native"]["sigma"] == sigma
    assert r["normalized"]["sigma"] is None
    assert r["normalized"]["uncertainty_kind"] == "unsupported"
    assert "diameter_uncertainty" not in p["_physical"]


def test_units_convert_value_and_sigma_together_and_h_sigma_is_not_lost():
    p = parse(
        [
            diameter(value="11", sigma="2", units="m"),
            {
                "name": "rot_per",
                "value": "5.3516",
                "sigma": ".0001",
                "units": "min",
                "ref": "synthetic-spin",
            },
            {
                "name": "H",
                "value": "25.91",
                "sigma": ".03",
                "units": None,
                "ref": "synthetic-H",
            },
        ]
    )
    assert p["_physical"]["diameter"] == 0.011
    assert p["_physical"]["diameter_uncertainty"] == 0.002
    assert p["_physical"]["rot_per"] == pytest.approx(5.3516 / 60)
    assert p["_physical"]["rotation_period_uncertainty"] == pytest.approx(0.0001 / 60)
    e = p["_evidence_metadata"]["physical_evidence"]
    assert e["records"][2]["normalized"]["sigma"] == 0.03
    assert e["records"][2]["normalized"]["units"] == "mag"


def test_alternative_measurements_abstain_instead_of_last_record_wins():
    a = diameter()
    b = diameter(".02", ".003", ref="other-study")
    p = parse([a, b])
    e = p["_evidence_metadata"]["physical_evidence"]
    assert e["selection"]["diameter_km"]["status"] == "ambiguous"
    assert [r["native"] for r in e["records"]] == [a, b]
    assert "diameter" not in p["_physical"]


def test_exact_duplicate_assertions_do_not_create_ambiguous_or_independent_evidence():
    p = parse([diameter(), diameter()])
    e = p["_evidence_metadata"]["physical_evidence"]
    assert e["duplicate_records"] == {"1": 0}
    assert e["selection"]["diameter_km"]["status"] == "selected"
    assert p["_physical"]["diameter"] == 0.011


def test_physical_contract_rejects_modified_sigma_even_rehashed():
    from aneos_core.data.physical_evidence import verify_physical_evidence, fingerprint

    e = parse([diameter()])["_evidence_metadata"]["physical_evidence"]
    assert verify_physical_evidence(json.loads(json.dumps(e))) == e
    changed = copy.deepcopy(e)
    changed["records"][0]["normalized"]["sigma"] = 0.0
    changed["evidence_id"] = fingerprint(
        {k: v for k, v in changed.items() if k != "evidence_id"}
    )
    with pytest.raises(ValueError):
        verify_physical_evidence(changed)


def test_orphan_legacy_sigma_cannot_fill_another_source_measurement():
    f = DataFetcher.__new__(DataFetcher)
    f.source_priority = ["A", "B"]
    a = NEOData(
        "SYNTHETIC",
        orbital_elements=OrbitalElements(semi_major_axis=1.2),
        physical_properties=PhysicalProperties(diameter_uncertainty=0.8),
    )
    b = NEOData(
        "SYNTHETIC",
        orbital_elements=a.orbital_elements,
        physical_properties=PhysicalProperties(diameter_km=0.011),
    )
    m = f._reconcile_sources({"A": a, "B": b})
    assert m.physical_properties.diameter_km == 0.011
    assert m.physical_properties.diameter_uncertainty is None
    assert (
        m.evidence_metadata["physical_source_snapshots"]["A"]["physical_properties"][
            "diameter_uncertainty"
        ]
        == 0.8
    )


def test_geometric_albedo_is_not_bounded_like_a_probability():
    from aneos_core.analysis.screening import ScreeningAnalysisService

    p = parse(
        [
            {
                "name": "albedo",
                "value": "1.08",
                "sigma": ".04",
                "units": None,
                "ref": "synthetic-geometric-albedo",
            }
        ]
    )
    assert p["_physical"]["albedo"] == 1.08
    pp = PhysicalProperties(albedo=1.08)
    n = NEOData(
        "SYNTHETIC",
        orbital_elements=OrbitalElements(
            semi_major_axis=1.2, eccentricity=0.2, inclination=1.0
        ),
        physical_properties=pp,
    )
    a = ScreeningAnalysisService().analyze_neo(n)
    assert a.artificial_probability is None
    with pytest.raises(ValueError):
        PhysicalProperties(albedo=-0.1)


def acquired(records):
    from unittest.mock import AsyncMock

    s = SBDBSource(APIConfig())
    s._http_get = AsyncMock(return_value=source_payload(records))
    f = DataFetcher.__new__(DataFetcher)
    import logging

    f.logger = logging.getLogger("physical-fixture")
    return f._fetch_from_source("SBDB", s, "SYNTHETIC")


def test_actual_source_model_merge_cache_analysis_review_and_exports(tmp_path, capsys):
    from aneos_core.analysis.object_review import (
        build_object_review,
        verify_object_review,
        render_object_review,
    )
    from aneos_core.reporting.exporters import Exporter
    from aneos_core.analysis.screening import ScreeningAnalysisService

    a = acquired(
        [diameter(), {"name": "H", "value": "25.91", "sigma": ".03", "units": None}]
    )
    assert a is not None and a.physical_properties.diameter_uncertainty == 0.002
    f = DataFetcher.__new__(DataFetcher)
    f.source_priority = ["SBDB"]
    n = f._reconcile_sources({"SBDB": a})
    restored = f._deserialize_neo_data(json.loads(json.dumps(f._serialize_neo_data(n))))
    review = build_object_review(restored)
    assert (
        review["inventory"]["physical_measurement_evidence"][0]["evidence"]["records"][
            0
        ]["normalized"]["sigma"]
        == 0.002
    )
    assert (
        ScreeningAnalysisService().analyze_neo(restored).to_payload()
        == review["assessment"]
    )
    render_object_review(review, verbose=False)
    assert "0.002 km" in capsys.readouterr().out
    path = Exporter(str(tmp_path)).export_to_json(
        [{"object_review": review}], output_file="physical.json", exclusive=True
    )
    assert (
        verify_object_review(json.loads(open(path).read())["neos"][0]["object_review"])
        == review
    )
    changed = copy.deepcopy(n.to_dict())
    changed["physical_properties"]["diameter_uncertainty"] = 0.5
    with pytest.raises(ValueError, match="uncertainty"):
        NEOData.from_dict(changed)
    n.physical_properties.diameter_uncertainty = 0.8
    with pytest.raises(ValueError, match="uncertainty"):
        ScreeningAnalysisService().analyze_neo(n)


@pytest.mark.parametrize("records", [None, {}, [None], [{}]])
def test_schema_invalid_physical_response_is_explicit_source_failure(records):
    with pytest.raises(ValueError):
        parse(records)
    assert acquired(records) is None


@pytest.mark.parametrize(
    "field,sigma_field",
    [
        ("diameter_km", "diameter_uncertainty"),
        ("albedo", "albedo_uncertainty"),
        ("rotation_period_hours", "rotation_period_uncertainty"),
    ],
)
def test_all_paired_fields_preserve_provider_pair_and_reference(field, sigma_field):
    f = DataFetcher.__new__(DataFetcher)
    f.source_priority = ["A", "B"]
    orbit = OrbitalElements(semi_major_axis=1.2)
    a = NEOData(
        "SYNTHETIC",
        orbital_elements=orbit,
        physical_properties=PhysicalProperties(
            **{field: 0.5, "reference_source": "study-A"}
        ),
    )
    b = NEOData(
        "SYNTHETIC",
        orbital_elements=orbit,
        physical_properties=PhysicalProperties(
            **{field: 0.6, sigma_field: 0.01, "reference_source": "study-B"}
        ),
    )
    m = f._reconcile_sources({"B": b, "A": a})
    assert getattr(m.physical_properties, field) == 0.5
    assert getattr(m.physical_properties, sigma_field) is None
    assert m.physical_properties.reference_source is None
    assert (
        m.evidence_metadata["physical_selection"][field]["reference_source"]
        == "study-A"
    )
    assert (
        m.evidence_metadata["conflicts"][f"physical_properties.{field}:B"][
            "reference_source"
        ]
        == "study-B"
    )


def test_missing_sigma_remains_missing_zero_sigma_is_preserved():
    for sigma in (None, "0"):
        p = parse([diameter(sigma=sigma)])
        r = p["_evidence_metadata"]["physical_evidence"]["records"][0]
        assert r["normalized"]["sigma"] == (None if sigma is None else 0.0)
        assert r["normalized"]["uncertainty_kind"] == (
            "not_available" if sigma is None else "provider_1sigma"
        )


def test_spectral_assertions_unknown_parameters_and_unsupported_units_retained():
    p = parse(
        [
            {"name": "spec_T", "value": "Xe", "units": None},
            {"name": "spec_B", "value": "X", "sigma": "ambiguous", "units": None},
            {"name": "spin_pole", "value": "unknown"},
            diameter(units=None),
        ]
    )
    e = p["_evidence_metadata"]["physical_evidence"]
    assert (
        p["_physical"]["spectral_type"] == "Xe"
        and p["_physical"]["spectral_class"] == "X"
    )
    assert "unsupported_label_uncertainty" in e["records"][1]["issues"]
    assert "unsupported_parameter" in e["records"][2]["issues"]
    assert "unsupported_units" in e["records"][3]["issues"]
    assert "diameter" not in p["_physical"]


def test_new_fetch_cache_key_does_not_relabel_or_overwrite_old_snapshots(tmp_path):
    from unittest.mock import Mock
    import logging
    from sqlalchemy import create_engine
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from aneos_core.data.acquisition import plan, outcome, build_bundle
    p = plan(['SBDB'])
    source = NEOData('SYNTHETIC', orbital_elements=OrbitalElements(semi_major_axis=1.2,eccentricity=.2,inclination=1.5))
    outcomes = [outcome(name, 'SYNTHETIC', status='available' if name=='SBDB' else 'unavailable',
                        neo=source if name=='SBDB' else None, basis='controlled_synthetic_fixture') for name in p['sources']]
    bundle = build_bundle('SYNTHETIC', p, outcomes)
    n = NEOData.from_dict(bundle['neo_data']);n._acquisition_bundle=bundle
    engine = create_engine('sqlite:///'+str(tmp_path/'cache-receipt.sqlite'))
    f = DataFetcher.__new__(DataFetcher)
    f.logger = logging.getLogger('cache-fixture')
    f.source_plan=p;f.evidence_repository=SQLEvidenceRepository(engine)
    f.fetch_stats = {'total_requests': 0, 'cache_hits': 0, 'cache_misses': 0}
    f.cache_ttl = 30;f.cache_manager=Mock();f.cache_manager.get.return_value=None
    f._fetch_from_all_sources=Mock(return_value=n)
    result=f.fetch_neo_data('SYNTHETIC')
    assert result.orbital_elements.to_dict()==source.orbital_elements.to_dict()
    assert result.evidence_metadata['acquisition']['storage_receipt']['commit_status']=='committed'
    key='neo-query:object-acquisition-v1:'+p['plan_id']+':SYNTHETIC'
    assert f.cache_manager.get.call_args.args[0]==key
    assert f.cache_manager.set.call_args.args[0]==key
    assert f.cache_manager.set.call_args.args[1]['schema_version']=='object-query-cache-v1'
    engine.dispose()


def test_modified_provider_projection_is_not_accepted_as_successful_source():
    from unittest.mock import Mock

    p = parse([diameter()])
    p["_physical"]["diameter_uncertainty"] = 0.8
    source = Mock()
    source.fetch_orbital_elements.return_value = p
    f = DataFetcher.__new__(DataFetcher)
    import logging

    f.logger = logging.getLogger("tampered-fixture")
    assert f._fetch_from_source("SBDB", source, "SYNTHETIC") is None


def test_report_retains_selected_and_alternative_physical_measurements():
    from types import SimpleNamespace
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.analysis.reporting import (
        build_screening_report,
        verify_screening_report,
    )

    n = acquired([diameter()])
    assessment = ScreeningAnalysisService().analyze_neo(n).to_payload()
    report = build_screening_report(
        SimpleNamespace(neo_data=n, screening_assessment=assessment)
    )
    assert verify_screening_report(json.loads(json.dumps(report)))
    record = report["neo_data"]["evidence_metadata"]["physical_evidence"]["records"][0]
    assert record["normalized"]["sigma"] == 0.002
    assert record["native"]["sigma"] == ".002"
    assert (
        report["artificial_probability"] is None
        and report["sigma_statistical_level"] is None
    )


def test_atomic_record_copy_replay_and_malformed_evidence_are_not_silent():
    from aneos_core.data.physical_evidence import (
        PhysicalMeasurement,
        verify_physical_evidence,
    )

    e = parse([diameter()])["_evidence_metadata"]["physical_evidence"]
    r = e["records"][0]
    obj = PhysicalMeasurement.from_dict(r)
    out = obj.to_dict()
    out["native"]["value"] = "forged"
    assert obj.to_dict() == r
    with pytest.raises(ValueError):
        PhysicalMeasurement.from_dict(out)
    for bad in (
        None,
        {"schema_version": "physical-evidence-v1", "context": {}, "records": [None]},
        dict(e, context={"different": "context"}),
    ):
        with pytest.raises(ValueError):
            verify_physical_evidence(bad)


def test_physical_assertion_render_controls_real_rich_at_three_widths(tmp_path):
    import subprocess, sys
    from pathlib import Path
    from hashlib import sha256

    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "physical-qa"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/qa_object_review.py"),
            "--physical-cases",
            "--output",
            str(directory),
        ],
        cwd=root,
        capture_output=True,
        check=True,
    )
    inventory = json.loads((directory / "inventory.json").read_text())
    assert set(inventory["cases"]) == {"selected", "ambiguous", "units-unavailable"}
    for width in (80, 120, 160):
        selected = (directory / f"selected-{width}.txt").read_text()
        assert "0.002 km" in selected
        assert "SYNTHETIC study" in selected
        assert (
            "no scalar selection" in (directory / f"ambiguous-{width}.txt").read_text()
        )
        assert (
            "unsupported_units"
            in (directory / f"units-unavailable-{width}.txt").read_text()
        )
        assert (
            inventory["cases"]["selected"]["widths"][str(width)]["max_columns"] <= width
        )
    for name, digest in inventory["artifact_sha256"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "mutation", ["selected_reference", "duplicate_metadata", "source_lineage"]
)
def test_source_reference_and_duplicate_metadata_cannot_disagree(mutation):
    a = acquired([diameter()])
    f = DataFetcher.__new__(DataFetcher)
    f.source_priority = ["SBDB"]
    n = f._reconcile_sources({"SBDB": a})
    data = n.to_dict()
    meta = data["evidence_metadata"]
    if mutation == "selected_reference":
        meta["physical_selection"]["diameter_km"][
            "reference_source"
        ] = "invented reference"
    elif mutation == "duplicate_metadata":
        meta["source_solutions"]["SBDB"]["physical_evidence"]["context"]["object"][
            "des"
        ] = "another object"
    else:
        meta["field_sources"]["physical_properties.diameter_km"] = "OTHER"
    with pytest.raises(ValueError):
        NEOData.from_dict(data)


def test_duplicate_records_are_labeled_in_actual_plain_detail(capsys):
    from aneos_core.analysis.object_review import (
        build_object_review,
        render_object_review,
    )

    n = acquired([diameter(), diameter()])
    render_object_review(build_object_review(n), verbose=False)
    assert "duplicate of record 0, not additional evidence" in capsys.readouterr().out
