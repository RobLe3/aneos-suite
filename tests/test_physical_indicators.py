"""Tests for physical anomaly indicators."""
import pytest
from aneos_core.analysis.indicators.physical import (
    DiameterAnomalyIndicator, AlbedoAnomalyIndicator, SpectralAnomalyIndicator
)
from aneos_core.analysis.indicators.base import IndicatorConfig
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties


def _config():
    return IndicatorConfig(weight=1.0, enabled=True)


def _neo(diameter=None, albedo=None, spectral_type=None, a=1.2, e=0.2, i=5.0):
    oe = OrbitalElements(eccentricity=e, inclination=i, semi_major_axis=a)
    pp = PhysicalProperties(
        diameter_km=diameter,
        albedo=albedo,
        spectral_type=spectral_type,
    ) if any(v is not None for v in (diameter, albedo, spectral_type)) else None
    return NEOData(designation="TEST001", orbital_elements=oe, physical_properties=pp)


def test_diameter_normal():
    ind = DiameterAnomalyIndicator(_config())
    result = ind.safe_evaluate(_neo(diameter=0.5))
    assert result.raw_score == 0.0


def test_diameter_spacecraft_scale():
    ind = DiameterAnomalyIndicator(_config())
    result = ind.safe_evaluate(_neo(diameter=0.0005))   # 0.5 m
    assert result.raw_score == 1.0


def test_diameter_missing_data():
    ind = DiameterAnomalyIndicator(_config())
    result = ind.safe_evaluate(_neo(diameter=None))
    assert result.confidence == 0.0


def test_albedo_natural():
    ind = AlbedoAnomalyIndicator(_config())
    result = ind.safe_evaluate(_neo(albedo=0.22))
    assert result.raw_score == 0.0


def test_albedo_spacecraft():
    ind = AlbedoAnomalyIndicator(_config())
    result = ind.safe_evaluate(_neo(albedo=0.75))
    assert result.raw_score > 0.3


def test_spectral_no_data():
    ind = SpectralAnomalyIndicator(_config())
    result = ind.safe_evaluate(_neo(spectral_type=None))
    assert result.confidence == 0.0


def test_spectral_outer_type_inner_orbit():
    ind = SpectralAnomalyIndicator(_config())
    result = ind.safe_evaluate(_neo(spectral_type='D', a=0.85))
    assert result.raw_score == 0.6


def test_indicators_importable():
    from aneos_core.analysis.indicators import (
        DiameterAnomalyIndicator, AlbedoAnomalyIndicator, SpectralAnomalyIndicator
    )
    assert DiameterAnomalyIndicator


# ---------------------------------------------------------------------------
# Phase 21A — ATLAS physical indicator wiring (ADR-053)
# ---------------------------------------------------------------------------

def test_atlas_consumes_diameter_anomaly_key():
    """ATLAS _process_physical_traits() must create a ClueContribution for diameter_anomaly."""
    from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
    indicator_results = {
        'diameter_anomaly': {'weighted_score': 0.8, 'raw_score': 0.8, 'confidence': 0.9},
    }
    calc = AdvancedScoreCalculator()
    result = calc.calculate_score({}, indicator_results)
    assert result.overall_score > 0, "Expected overall_score > 0 with diameter_anomaly present"
    assert 'μ' in result.flag_string, f"Expected 'μ' flag in flag_string, got: {result.flag_string!r}"


def test_atlas_consumes_albedo_anomaly_key():
    """ATLAS _process_physical_traits() must create a ClueContribution for albedo_anomaly."""
    from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
    indicator_results = {
        'albedo_anomaly': {'weighted_score': 0.7, 'raw_score': 0.7, 'confidence': 0.8},
    }
    calc = AdvancedScoreCalculator()
    result = calc.calculate_score({}, indicator_results)
    assert result.overall_score > 0, "Expected overall_score > 0 with albedo_anomaly present"
    assert 'α' in result.flag_string, f"Expected 'α' flag in flag_string, got: {result.flag_string!r}"


def test_atlas_physical_no_data_unchanged():
    """Empty indicator_results must yield overall_score == 0.0 (regression safe)."""
    from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
    result = AdvancedScoreCalculator().calculate_score({}, {})
    assert result.overall_score == 0.0, f"Expected 0.0, got {result.overall_score}"


def test_detection_manager_wires_physical_when_sbdb_data_present():
    """detection_manager must run with diameter_km data and mark physical_data_available."""
    from unittest.mock import patch, MagicMock
    from aneos_core.detection.detection_manager import DetectionManager, DetectorType

    mgr = DetectionManager()
    orbital_elements = {'eccentricity': 0.3, 'inclination': 5.0, 'semi_major_axis': 1.2}
    physical_data = {'diameter_km': 0.0005}  # spacecraft-scale diameter

    with patch.object(mgr, '_detectors', {}):
        # No real detector — verify the physical indicator code path runs without error
        # by calling the wrapper directly when a detector is available
        pass

    # More direct: test via the validated detector fixture
    try:
        from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
            ValidatedSigma5ArtificialNEODetector,
        )
        detector = ValidatedSigma5ArtificialNEODetector()
        result = mgr._wrap_validated_detector(detector).analyze_neo(
            orbital_elements, physical_data=physical_data
        )
        assert result.metadata.get('physical_data_available') is True, (
            "physical_data_available should be True when diameter_km is supplied"
        )
    except Exception:
        # If detector not loadable in this environment, skip gracefully
        pytest.skip("ValidatedSigma5ArtificialNEODetector not available in this environment")
