"""
Tests for the Kardashev Synthetic NEO Generator.

Verifies:
  TestScenarioCatalogue    (3) — 14 scenarios exist; all tiers present; names unique
  TestSampling             (4) — deterministic with seed; values in physics range;
                                  anomaly flags fire correctly; to_feature_dict complete
  TestAnomalyFlags         (3) — density sub-natural, A2 above Yarkovsky, albedo extreme
  TestNEODataConversion    (2) — to_neo_data() returns valid NEOData with orbital elements
  TestBuildFeatureVectors  (3) — n_per_scenario respected; labels all 1; fv has features
"""

import math
import pytest
import numpy as np

from aneos_core.datasets.kardashev_generator import (
    KardashevSyntheticGenerator,
    KardashevTier,
    SCENARIO_CATALOGUE,
    SyntheticNEO,
    build_labeled_feature_vectors,
    _NATURAL_A2_MAX,
    _NATURAL_DENSITY_RANGE,
)


# ===========================================================================
# TestScenarioCatalogue
# ===========================================================================

class TestScenarioCatalogue:

    def test_exactly_14_scenarios(self):
        assert len(SCENARIO_CATALOGUE) == 14

    def test_all_four_tiers_represented(self):
        tiers_present = {s.tier for s in SCENARIO_CATALOGUE}
        assert tiers_present == {
            KardashevTier.K0_5,
            KardashevTier.K1_0,
            KardashevTier.K1_5,
            KardashevTier.K2_0,
        }

    def test_scenario_names_unique_within_tier(self):
        from collections import Counter
        keys = [(s.tier, s.name) for s in SCENARIO_CATALOGUE]
        counts = Counter(keys)
        duplicates = [k for k, v in counts.items() if v > 1]
        assert duplicates == [], f"Duplicate scenarios: {duplicates}"


# ===========================================================================
# TestSampling
# ===========================================================================

class TestSampling:

    def test_deterministic_with_same_seed(self):
        gen1 = KardashevSyntheticGenerator(seed=7)
        gen2 = KardashevSyntheticGenerator(seed=7)
        s1 = gen1.generate(n_per_scenario=1)[0]
        s2 = gen2.generate(n_per_scenario=1)[0]
        assert s1.params["a"] == pytest.approx(s2.params["a"])
        assert s1.params["e"] == pytest.approx(s2.params["e"])

    def test_different_seeds_differ(self):
        gen1 = KardashevSyntheticGenerator(seed=1)
        gen2 = KardashevSyntheticGenerator(seed=2)
        s1 = gen1.generate(n_per_scenario=1)[0]
        s2 = gen2.generate(n_per_scenario=1)[0]
        # Extremely unlikely to be equal
        assert s1.params["a"] != pytest.approx(s2.params["a"])

    def test_orbital_elements_physically_valid(self):
        gen = KardashevSyntheticGenerator(seed=42)
        samples = gen.generate(n_per_scenario=10)
        for s in samples:
            p = s.params
            assert 0.0 < p["a"] < 15.0,       f"a out of range: {p['a']}"
            assert 0.0 <= p["e"] < 1.0,        f"e out of range: {p['e']}"
            assert 0.0 <= p["i"] <= 180.0,     f"i out of range: {p['i']}"
            assert 0.0 <= p["om"] <= 360.0
            assert 0.0 <= p["w"]  <= 360.0
            assert 0.0 <= p["M"]  <= 360.0
            assert p["diameter_km"] > 0
            assert 0.0 < p["albedo"] < 1.0
            assert p["density_g_cm3"] > 0
            assert p["rotation_period_hours"] > 0

    def test_to_feature_dict_contains_all_keys(self):
        gen = KardashevSyntheticGenerator(seed=0)
        s = gen.generate(n_per_scenario=1)[0]
        d = s.to_feature_dict()
        for key in ["a", "e", "i", "diameter_km", "albedo", "density_g_cm3",
                    "nongrav_A2", "nongrav_magnitude", "label", "tier", "scenario"]:
            assert key in d, f"Missing key: {key}"


# ===========================================================================
# TestAnomalyFlags
# ===========================================================================

class TestAnomalyFlags:

    def _make_sample_with_params(self, **overrides) -> SyntheticNEO:
        """Make a SyntheticNEO with controlled params."""
        base = {
            "a": 1.0, "e": 0.2, "i": 10.0, "om": 45.0, "w": 60.0, "M": 90.0,
            "diameter_km": 0.5, "albedo": 0.3, "density_g_cm3": 2.0,
            "rotation_period_hours": 10.0, "absolute_magnitude_h": 18.0,
            "nongrav_A1": 0.0, "nongrav_A2": 0.0, "nongrav_A3": 0.0,
        }
        base.update(overrides)
        flags = KardashevSyntheticGenerator._compute_anomaly_flags(base)
        return SyntheticNEO(
            designation="TEST-0000",
            tier=KardashevTier.K0_5,
            scenario="test",
            scenario_description="",
            params=base,
            anomaly_flags=flags,
        )

    def test_density_sub_natural_flagged(self):
        s = self._make_sample_with_params(density_g_cm3=0.001)
        assert s.anomaly_flags["density_subnatural"] is True

    def test_a2_above_yarkovsky_flagged(self):
        s = self._make_sample_with_params(nongrav_A2=1e-11)
        assert s.anomaly_flags["a2_above_yarkovsky_max"] is True

    def test_extreme_high_albedo_flagged(self):
        s = self._make_sample_with_params(albedo=0.95)
        assert s.anomaly_flags["albedo_extreme_high"] is True


# ===========================================================================
# TestNEODataConversion
# ===========================================================================

class TestNEODataConversion:

    def test_to_neo_data_returns_neo_data(self):
        from aneos_core.data.models import NEOData
        gen = KardashevSyntheticGenerator(seed=3)
        s = gen.generate(n_per_scenario=1)[0]
        neo = s.to_neo_data()
        assert isinstance(neo, NEOData)
        assert neo.designation == s.designation

    def test_to_neo_data_has_orbital_elements(self):
        gen = KardashevSyntheticGenerator(seed=4)
        s = gen.generate(n_per_scenario=1)[0]
        neo = s.to_neo_data()
        oe = neo.orbital_elements
        assert oe is not None
        assert oe.semi_major_axis == pytest.approx(s.params["a"])
        assert oe.eccentricity    == pytest.approx(s.params["e"])
        assert oe.inclination     == pytest.approx(s.params["i"])


# ===========================================================================
# TestBuildFeatureVectors
# ===========================================================================

class TestBuildFeatureVectors:

    def test_n_per_scenario_respected(self):
        fvs, labels, desigs = build_labeled_feature_vectors(n_per_scenario=2, seed=0)
        # 14 scenarios × 2 = 28
        assert len(fvs) == 28

    def test_all_labels_are_one(self):
        _, labels, _ = build_labeled_feature_vectors(n_per_scenario=2, seed=1)
        assert all(l == 1 for l in labels)

    def test_feature_vectors_have_numpy_array(self):
        fvs, _, _ = build_labeled_feature_vectors(n_per_scenario=1, seed=2)
        for fv in fvs:
            assert isinstance(fv.features, np.ndarray)
            assert fv.features.ndim == 1
            assert len(fv.features) > 0
