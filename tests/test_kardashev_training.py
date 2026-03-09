"""
Tests for the Kardashev ML training pipeline + benefit assessment.

TestFeatureExtraction  (3) — feature vector shape, dtype, named values
TestSyntheticNaturals  (2) — n respected; physical params in natural range
TestPipelineRuns       (3) — pipeline completes; verdict set; scores exist
TestBenefitQuality     (3) — RF scores artificials > naturals; AUC > 0.75; top feature
"""

import math
import numpy as np
import pytest

from aneos_core.datasets.kardashev_training import (
    _params_to_features, FEATURE_NAMES,
    generate_synthetic_naturals,
    GROUND_TRUTH_OBJECTS,
    KardashevMLPipeline,
    BenefitAssessment,
)


# ===========================================================================
# TestFeatureExtraction
# ===========================================================================

class TestFeatureExtraction:

    def test_feature_vector_length(self):
        p = {"a": 1.0, "e": 0.2, "i": 10.0, "albedo": 0.25,
             "density_g_cm3": 2.0, "diameter_km": 0.5, "nongrav_A2": 0.0}
        fv = _params_to_features(p)
        assert len(fv) == len(FEATURE_NAMES)

    def test_feature_vector_dtype(self):
        p = {"a": 1.0, "e": 0.2, "i": 10.0, "albedo": 0.25,
             "density_g_cm3": 2.0, "diameter_km": 0.5, "nongrav_A2": 1e-13}
        fv = _params_to_features(p)
        assert fv.dtype == np.float64
        assert not np.any(np.isnan(fv))
        assert not np.any(np.isinf(fv))

    def test_artificial_a2_elevates_log_feature(self):
        """Large A2 (artificial) should give much higher log10_A2 than natural floor."""
        art = _params_to_features({"nongrav_A2": 1e-11})
        nat = _params_to_features({"nongrav_A2": 1e-15})
        idx = FEATURE_NAMES.index("log10_A2_abs")
        assert art[idx] > nat[idx] + 3   # 4 orders of magnitude → at least 3 feature units


# ===========================================================================
# TestSyntheticNaturals
# ===========================================================================

class TestSyntheticNaturals:

    def test_n_respected(self):
        nats = generate_synthetic_naturals(n=50, seed=0)
        assert len(nats) == 50

    def test_params_in_natural_range(self):
        nats = generate_synthetic_naturals(n=100, seed=1)
        for n in nats:
            assert 0.5 <= n["a"] <= 4.5,         f"a out of range: {n['a']}"
            assert 0.0 <= n["e"] <= 0.95,        f"e out of range: {n['e']}"
            assert 0.0 <= n["i"] <= 65.0,        f"i out of range: {n['i']}"
            assert 0.5 <= n["density_g_cm3"] <= 6.0


# ===========================================================================
# TestPipelineRuns
# ===========================================================================

class TestPipelineRuns:

    @pytest.fixture(scope="class")
    def assessment(self):
        pipeline = KardashevMLPipeline()
        return pipeline.run(n_per_scenario=10, n_naturals=50, seed=0)

    def test_pipeline_returns_assessment(self, assessment):
        assert isinstance(assessment, BenefitAssessment)

    def test_verdict_is_set(self, assessment):
        assert assessment.verdict in {
            "CONFIRMED BENEFIT", "MARGINAL BENEFIT", "NO CONFIRMED BENEFIT"
        }

    def test_all_ground_truth_objects_scored(self, assessment):
        scored_names = {r["name"] for r in assessment.object_scores}
        expected = set(GROUND_TRUTH_OBJECTS.keys())
        assert expected.issubset(scored_names)


# ===========================================================================
# TestBenefitQuality (requires full training — uses scope="module" fixture)
# ===========================================================================

class TestBenefitQuality:

    @pytest.fixture(scope="class")
    def full_assessment(self):
        pipeline = KardashevMLPipeline()
        return pipeline.run(n_per_scenario=50, n_naturals=200, seed=42)

    def test_rf_artificials_score_above_mean_natural(self, full_assessment):
        """RF P(artificial) must be higher for confirmed artificials on average."""
        art_scores = [r["rf_score"] for r in full_assessment.object_scores
                      if r["label"] == "artificial"]
        nat_scores = [r["rf_score"] for r in full_assessment.object_scores
                      if r["label"] == "natural"]
        assert np.mean(art_scores) > np.mean(nat_scores), (
            f"art mean {np.mean(art_scores):.3f} not > nat mean {np.mean(nat_scores):.3f}"
        )

    def test_rf_auc_above_chance(self, full_assessment):
        """RF AUC on ground-truth set must be > 0.75 (well above chance)."""
        assert full_assessment.rf_auc > 0.75, (
            f"RF AUC {full_assessment.rf_auc:.3f} not > 0.75"
        )

    def test_density_is_top_feature(self, full_assessment):
        """log10_density or albedo must be in top-2 features (both discriminate hollow vs rocky)."""
        top2 = sorted(full_assessment.feature_importance.items(),
                      key=lambda x: -x[1])[:2]
        top2_names = {n for n, _ in top2}
        assert top2_names & {"log10_density", "albedo"}, (
            f"Expected density or albedo in top-2; got {top2_names}"
        )
