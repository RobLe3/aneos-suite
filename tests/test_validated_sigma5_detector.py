"""Legacy facade contract after withdrawal of unsupported inference.

This is NOT evidence of detector accuracy. Old assertions that demanded sigma5,
a posterior and F1>0.8 from constructor examples were scientifically defective;
see docs/engineering/screening-result-contract.md. Frozen legacy arithmetic is
still executed by test_ky26_baseline, independently of this deployed facade.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy import stats

from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
    ValidatedSigma5ArtificialNEODetector, EvidenceType,
)


@pytest.fixture
def detector():
    return ValidatedSigma5ArtificialNEODetector()


@pytest.mark.parametrize("orbital,physical", [
    ({"a": 1.325, "e": .256, "i": 1.077}, {"diameter": 12, "absolute_magnitude": 28}),
    ({"a": .922, "e": .191, "i": 3.331}, {"diameter": 370, "absolute_magnitude": 19.7}),
    ({"a": 1.126, "e": .204, "i": 6.035}, {"diameter": 492, "absolute_magnitude": 20.9, "spectral_type": "B"}),
])
def test_named_reference_inputs_do_not_substitute_for_origin_evidence(detector, orbital, physical):
    result = detector.analyze_neo_validated(orbital, physical)
    assert result.screening_score is not None
    assert result.is_artificial is None
    assert result.sigma_confidence is None
    assert result.bayesian_probability is None
    assert result.combined_p_value is None
    assert result.false_discovery_rate is None
    assert result.validation_metrics is None
    assert result.analysis_metadata["five_sigma_claim_permitted"] is False
    assert next(e for e in result.evidence_sources if e.evidence_type == EvidenceType.ORBITAL_DYNAMICS).analyzed
    assert all(e.p_value is None for e in result.evidence_sources)


def test_frozen_ky26_rank_preserved_without_reinterpreting_it_as_sigma(detector):
    fixture = json.loads((Path(__file__).parent / "fixtures/ky26_baseline.json").read_text())
    result = detector.analyze_neo_validated(**{
        "orbital_elements": fixture["rounded_input"]["orbital"],
        "physical_data": fixture["rounded_input"]["physical"],
    })
    assert result.screening_score == pytest.approx(
        fixture["reproduction"]["rounded_input_expected"]["sigma_confidence"], rel=1e-12)
    assert result.sigma_confidence is None
    assert result.analysis_metadata["assessment"]["classification"] == "NOT_FLAGGED"


def test_validation_metrics_not_invented_or_enabled_by_legacy_assignment(detector):
    detector.validation_results = SimpleNamespace(f1_score=1)
    report = detector.get_validation_report()
    assert report["validation_performed"] is False
    assert report["performance_metrics"] is None
    assert report["false_discovery_rate"] is None
    assert report["five_sigma_claim_permitted"] is False


def test_ground_truth_database_loaded_but_not_used_as_calibration(detector):
    assert any("Tesla" in o["name"] for o in detector.artificial_objects_db)
    assert any("Apophis" in o["name"] for o in detector.natural_objects_db)
    assert detector.validation_results is None


def test_zero_orbital_elements_are_complete_observations(detector):
    result = detector.analyze_neo_validated({"a": 1, "e": 0, "i": 0})
    assert result.screening_score is not None
    assert result.evidence_sources[0].analyzed
    assert result.evidence_sources[0].quality_score is None


@pytest.mark.parametrize("method,args", [
    ("_bayesian_evidence_fusion", ([],)), ("_perform_cross_validation", ()),
])
def test_defective_inference_methods_have_controlled_deprecation(detector, method, args):
    with pytest.raises(NotImplementedError, match="withdrawn"):
        getattr(detector, method)(*args)


def test_missing_data_has_no_fabricated_intervals_or_significance(detector):
    result = detector.analyze_neo_validated({})
    assert result.screening_score is None
    assert all(not e.analyzed for e in result.evidence_sources)
    assert all(e.confidence_interval == () for e in result.evidence_sources)
    assert all(e.p_value is None for e in result.evidence_sources)


def test_false_discovery_rate_unavailable_even_with_positive_candidate(detector):
    result = detector.analyze_neo_validated({"a": 1, "e": .2, "i": 80})
    assert result.analysis_metadata["assessment"]["candidate"] is True
    assert result.false_discovery_rate is None
    assert result.is_artificial is None


def test_reproducibility_of_complete_assessment(detector):
    results = [detector.analyze_neo_validated({"a": 1.325, "e": .256, "i": 1.077},
               {"diameter": 12}).analysis_metadata["assessment"] for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_gaussian_tail_constant_is_arithmetic_not_permission_to_claim(detector):
    assert detector.SIGMA_5_P_VALUE == pytest.approx(2 * stats.norm.sf(5))
    assert detector.get_validation_report()["five_sigma_claim_permitted"] is False


def test_detector_accuracy_on_ground_truth_is_not_fabricated_from_missing_probability():
    from aneos_core.datasets.ground_truth_validator import GroundTruthValidator
    from aneos_core.detection.detection_manager import DetectionManager
    # Frozen inputs avoid collection-time network probes and in-sample threshold tuning.
    objects = [SimpleNamespace(object_id="synthetic-control", orbital_elements={
        "a": 1, "e": .2, "i": 80}, physical_params={}, is_artificial=True)]
    with pytest.raises(ValueError, match="probability unavailable"):
        GroundTruthValidator().calibrated_run(objects, DetectionManager())


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 2, True])
def test_evaluator_rejects_missing_or_invalid_probability(value):
    from aneos_core.datasets.ground_truth_validator import GroundTruthValidator
    obj = SimpleNamespace(object_id="synthetic-marker", orbital_elements={}, physical_params={})
    detector = SimpleNamespace(analyze_neo=lambda *a: SimpleNamespace(artificial_probability=value))
    with pytest.raises(ValueError):
        GroundTruthValidator()._score_object(obj, detector)


def test_evaluator_accepts_available_probability_positive_control():
    from aneos_core.datasets.ground_truth_validator import GroundTruthValidator
    obj = SimpleNamespace(object_id="synthetic-marker", orbital_elements={}, physical_params={})
    detector = SimpleNamespace(analyze_neo=lambda *a: SimpleNamespace(artificial_probability=.7))
    assert GroundTruthValidator()._score_object(obj, detector) == .7
