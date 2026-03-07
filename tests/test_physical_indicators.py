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
