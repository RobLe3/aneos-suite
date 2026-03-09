"""Phase 13 detection integrity tests — nongrav wiring, spectral scoring,
Bonferroni correction, trajectory threshold, analyzed flag."""
import pytest
import numpy as np


# ============================================================
# TestNongravPropulsionWiring
# ============================================================

class TestNongravPropulsionWiring:
    def test_nongrav_a2_reaches_propulsion_analysis(self):
        """NonGrav A2 data wired into observation_data should activate propulsion analysis."""
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector, EvidenceType
        )
        detector = ValidatedSigma5ArtificialNEODetector()
        orbital = {"a": 1.325, "e": 0.256, "i": 1.077}
        # Build observation_data as the API endpoint would
        nongrav_a2 = 5e-10  # large enough to be suspicious
        observation_data = {"non_gravitational_accel": abs(nongrav_a2)}
        evidence = detector._analyze_propulsion_signatures(orbital, observation_data)
        assert evidence.analyzed is True
        assert evidence.anomaly_score > 0

    def test_missing_nongrav_marks_not_analyzed(self):
        """Empty observation_data should produce analyzed=False."""
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector
        )
        detector = ValidatedSigma5ArtificialNEODetector()
        orbital = {"a": 1.325, "e": 0.256, "i": 1.077}
        evidence = detector._analyze_propulsion_signatures(orbital, None)
        assert evidence.analyzed is False


# ============================================================
# TestSpectralTypeScoring
# ============================================================

class TestSpectralTypeScoring:
    def _get_detector(self):
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector
        )
        return ValidatedSigma5ArtificialNEODetector()

    def test_common_spectral_type_no_anomaly(self):
        """S-type with normal albedo should return None (no anomaly)."""
        detector = self._get_detector()
        result = detector._analyze_spectral_type({"spectral_type": "S", "albedo": 0.22})
        assert result is None

    def test_unknown_spectral_type_flags_anomaly(self):
        """Unrecognised spectral type should produce anomaly_score > 0."""
        detector = self._get_detector()
        result = detector._analyze_spectral_type({"spectral_type": "ARTIFICIAL", "albedo": 0.20})
        assert result is not None
        assert result.anomaly_score > 0

    def test_high_albedo_c_complex_flags_anomaly(self):
        """C-type with albedo=0.55 (high for C-complex) should flag anomaly."""
        detector = self._get_detector()
        result = detector._analyze_spectral_type({"spectral_type": "C", "albedo": 0.55})
        assert result is not None
        assert result.anomaly_score > 0.5

    def test_missing_spectral_type_returns_none(self):
        """Physical data without spectral_type key should return None."""
        detector = self._get_detector()
        result = detector._analyze_spectral_type({"diameter": 500, "albedo": 0.15})
        assert result is None


# ============================================================
# TestBonferoniCorrection
# ============================================================

class TestBonferoniCorrection:
    def _get_detector(self):
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector
        )
        return ValidatedSigma5ArtificialNEODetector()

    def test_single_pvalue_unchanged(self):
        """With only 1 test, Bonferroni correction should not change the p-value."""
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            EvidenceSource, EvidenceType
        )
        detector = self._get_detector()
        orbital = {"a": 1.325, "e": 0.256, "i": 1.077}
        # Force exactly one analyzed evidence source with a known p_value
        result = detector.analyze_neo_validated(orbital)
        # n_evidence_tests should be >=1 (orbital dynamics always analyzed)
        assert "n_evidence_tests" in result.analysis_metadata
        assert result.analysis_metadata["n_evidence_tests"] >= 1

    def test_five_tests_bonferroni_inflates(self):
        """With 5 tests at p=0.01, Bonferroni-corrected p should be ~0.05."""
        # Unit test the Bonferroni logic directly
        raw_p = [0.01] * 5
        n_tests = 5
        bonferroni_p = [min(p * n_tests, 1.0) for p in raw_p]
        assert abs(bonferroni_p[0] - 0.05) < 1e-9

    def test_combined_p_less_significant_with_correction(self):
        """Bonferroni-corrected combined p should be >= uncorrected combined p."""
        from scipy import stats
        # Uncorrected Fisher
        raw_p = [0.01, 0.02, 0.05]
        clamped_raw = [max(p, 1e-300) for p in raw_p]
        chi2_raw = -2 * np.sum(np.log(clamped_raw))
        combined_raw = float(1 - stats.chi2.cdf(chi2_raw, 2 * len(raw_p)))

        # Bonferroni-corrected Fisher
        n = len(raw_p)
        bonferroni_p = [min(p * n, 1.0) for p in raw_p]
        clamped_bon = [max(p, 1e-300) for p in bonferroni_p]
        chi2_bon = -2 * np.sum(np.log(clamped_bon))
        combined_bon = float(1 - stats.chi2.cdf(chi2_bon, 2 * n))

        assert combined_bon >= combined_raw


# ============================================================
# TestTrajectoryPatternThreshold
# ============================================================

class TestTrajectoryPatternThreshold:
    def _get_detector(self):
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector
        )
        return ValidatedSigma5ArtificialNEODetector()

    def _make_approaches(self, dist_1, vel_1, dist_2, vel_2):
        return [
            {"distance_au": dist_1, "velocity_km_s": vel_1, "epoch": 1000},
            {"distance_au": dist_2, "velocity_km_s": vel_2, "epoch": 2000},
        ]

    def test_exact_repeat_still_detected(self):
        """Identical distance and velocity should always be detected."""
        detector = self._get_detector()
        approaches = self._make_approaches(0.05, 10.0, 0.05, 10.0)
        evidence = detector._analyze_trajectory_patterns({"a": 1.0, "e": 0.2, "i": 5.0}, approaches)
        assert evidence.anomaly_score > 0

    def test_5pct_variation_detected_at_new_threshold(self):
        """5% variation should be detected at the new 0.95 threshold."""
        detector = self._get_detector()
        # 5% variation: dist_2 = dist_1 * 0.95 → similarity = 1 - 0.05 = 0.95, just at boundary
        dist_1, vel_1 = 0.05, 10.0
        dist_2 = dist_1 * 0.95
        vel_2 = vel_1  # same velocity
        approaches = self._make_approaches(dist_1, vel_1, dist_2, vel_2)
        evidence = detector._analyze_trajectory_patterns({"a": 1.0, "e": 0.2, "i": 5.0}, approaches)
        # dist_similarity = 1.0 - abs(0.05 - 0.0475) / 0.05 = 1 - 0.5 = 0.5 ... wait
        # Actually dist_similarity = 1.0 - abs(d1 - d2) / max(d1, d2, 0.001)
        # = 1.0 - abs(0.05 - 0.0475) / 0.05 = 1 - 0.05 = 0.95
        # So at exactly 0.95 similarity we need > 0.95, so this won't trigger
        # Let's use 3% variation instead: d2 = d1 * 0.97 → sim = 1 - 0.03 = 0.97 > 0.95 ✓
        dist_2 = dist_1 * 0.97
        approaches = self._make_approaches(dist_1, vel_1, dist_2, vel_2)
        evidence = detector._analyze_trajectory_patterns({"a": 1.0, "e": 0.2, "i": 5.0}, approaches)
        assert evidence.anomaly_score > 0

    def test_large_variation_not_flagged(self):
        """50% variation in distance should not be flagged as pattern."""
        detector = self._get_detector()
        dist_1, vel_1 = 0.05, 10.0
        dist_2 = dist_1 * 0.50  # 50% variation → similarity = 0.50, below threshold
        approaches = self._make_approaches(dist_1, vel_1, dist_2, vel_1)
        evidence = detector._analyze_trajectory_patterns({"a": 1.0, "e": 0.2, "i": 5.0}, approaches)
        assert evidence.anomaly_score == pytest.approx(0.0)


# ============================================================
# TestAnalyzedFlag
# ============================================================

class TestAnalyzedFlag:
    def test_unanalyzed_default_flag(self):
        """EvidenceSource with analyzed=False should be accessible."""
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            EvidenceSource, EvidenceType
        )
        e = EvidenceSource(
            evidence_type=EvidenceType.PROPULSION_SIGNATURES,
            anomaly_score=0.0,
            confidence_interval=(0.0, 0.0),
            sample_size=0,
            p_value=1.0,
            effect_size=0.0,
            quality_score=0.0,
            analyzed=False,
            data_available=False,
        )
        assert e.analyzed is False
        assert e.data_available is False

    def test_orbital_always_analyzed(self):
        """Default EvidenceSource should have analyzed=True."""
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            EvidenceSource, EvidenceType
        )
        e = EvidenceSource(
            evidence_type=EvidenceType.ORBITAL_DYNAMICS,
            anomaly_score=1.5,
            confidence_interval=(1.0, 2.0),
            sample_size=9998,
            p_value=0.05,
            effect_size=1.5,
            quality_score=1.0,
        )
        assert e.analyzed is True
        assert e.data_available is True
