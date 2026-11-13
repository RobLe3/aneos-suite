"""Frozen discovery boundary: utility presence is not scientific integration."""
import asyncio
from unittest.mock import Mock

import pytest

from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.analysis.screening import ScreeningAnalysisService
from aneos_core.data.models import NEOData, OrbitalElements


EXPECTED = {
    'eccentricity', 'inclination', 'semi_major_axis', 'orbital_resonance',
    'orbital_stability', 'velocity_shifts', 'acceleration_anomalies',
    'velocity_consistency', 'infinity_velocity', 'approach_regularity',
    'observation_gaps', 'periodicity', 'temporal_inertia', 'subpoint_clustering',
    'geographic_bias', 'diameter_anomalies', 'albedo_anomalies', 'spectral_anomalies',
}


@pytest.mark.parametrize('inclination', [52.23568469321602, 15., None])
def test_18_retained_utilities_do_not_execute_or_change_canonical_policy(monkeypatch, inclination):
    import socket
    def denied(*args, **kwargs):
        raise AssertionError('No acquisition or hidden network in this boundary control')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setattr('aneos_core.analysis.pipeline.get_cache_manager', lambda: Mock())
    pipeline = AnalysisPipeline()
    assert set(pipeline.indicators) == EXPECTED
    spies = []
    for indicator in pipeline.indicators.values():
        spy = Mock(side_effect=AssertionError('Dormant utility executed'))
        monkeypatch.setattr(indicator, 'safe_evaluate', spy)
        spies.append(spy)
    legacy_score = Mock(side_effect=AssertionError('Legacy classifier executed'))
    monkeypatch.setattr(pipeline.score_calculator, 'calculate_score', legacy_score)
    neo = NEOData('boundary-control', orbital_elements=OrbitalElements(
        semi_major_axis=1., eccentricity=.2, inclination=inclination))
    expected = ScreeningAnalysisService().analyze_neo(neo).to_payload()
    actual = asyncio.run(pipeline.analyze_neo(neo.designation, neo))
    assert not actual.errors
    assert actual.screening_assessment == expected
    assert actual.analysis_metadata['legacy_indicators_run'] == []
    assert actual.anomaly_score.confidence is None
    for spy in spies:
        spy.assert_not_called()
    legacy_score.assert_not_called()
