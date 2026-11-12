#!/usr/bin/env python3
"""Simple actual-source screening adapter with explicit unavailable inference.

No name-based orbit, fixed posterior, impact probability or fallback detector is
installed. Structured analysis uses the shared core; legacy numerical contracts
that manufactured scientific certainty are controlled-deprecated.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from aneos_core.analysis.pipeline import create_analysis_pipeline
from aneos_core.analysis.reporting import build_screening_report
from aneos_core.analysis.impact_evidence import describe_impact_prerequisites
from aneos_core.analysis.advanced_aneos_analyzer import _synchronously

logger = logging.getLogger(__name__)

class SimpleNEOAnalyzer:
    """Compatibility facade for actual acquisition and reproducible screening."""

    def __init__(self, pipeline=None):
        """Initialize only the real core; failure never installs a toy detector."""
        self.advanced_analyzer = self.detection_manager = self.detector = None
        self.system_maturity = None
        self.initialization_error = None
        try:
            self.pipeline = pipeline if pipeline is not None else create_analysis_pipeline()
            self.initialized = True
        except (ImportError, RuntimeError, ValueError) as exc:
            self.pipeline = None
            self.initialized = False
            self.initialization_error = str(exc)

    async def analyze_neo_async(self, designation, force_refresh=False):
        """Async library entrypoint through the same source/core/report contract."""
        if self.pipeline is None:
            from types import SimpleNamespace
            core = SimpleNamespace(designation=designation, status='service_unavailable',
                errors=[self.initialization_error or 'Core unavailable'], neo_data=None)
        else:
            core = await self.pipeline.analyze_neo(designation, force_refresh=force_refresh)
        impact = describe_impact_prerequisites(getattr(core, 'neo_data', None))
        return build_screening_report(core, impact_assessment=impact)

    @staticmethod
    def _actual_neo(result):
        from aneos_core.data.models import NEOData
        value = result.get('neo_data') if isinstance(result, dict) else getattr(result, 'neo_data', None)
        if isinstance(value, dict):
            return NEOData.from_dict(value)
        return value if isinstance(value, NEOData) else None

    def calculate_artificial_probability(self, designation: str) -> float:
        """Controlled deprecation: a screening rank cannot supply origin probability."""
        raise NotImplementedError('Artificial-origin probability is not calibrated; use structured screening with explicit unavailable inference')

    def analyze_neo_comprehensive(self, designation: str) -> Dict[str, Any]:
        """Structured real-source screening, not a cascade of fallback detectors."""
        return _synchronously(self.analyze_neo_async, designation)

    def detect_artificial_signatures(self, designation: str) -> Dict[str, Any]:
        """Compatibility name; unusual catalogue properties do not identify origin."""
        return self.analyze_neo_comprehensive(designation)

    def assess_aneos_system_capabilities(self) -> Dict[str, Any]:
        """Implementation presence does not establish a scientific model or maturity."""
        from aneos_core.interfaces.unified_analysis import assess_aneos_system_maturity
        return {**assess_aneos_system_maturity(), **self.get_status()}

    def _get_capability_recommendation(self, capabilities_info: Dict[str, Any]) -> str:
        """No installation/feature count establishes origin-model calibration."""
        return 'Independent observational fits, calibration and scientific review required'

    def _get_mock_orbital_elements(self, designation: str) -> Dict[str, float]:
        """Controlled deprecation of fabricated/fixed scientific inputs or inference."""
        raise NotImplementedError('Mock/fixed inference withdrawn; actual typed evidence and a fitted validated model are required')

    def _generate_test_result(self) -> float:
        """Controlled deprecation of fabricated/fixed scientific inputs or inference."""
        raise NotImplementedError('Mock/fixed inference withdrawn; actual typed evidence and a fitted validated model are required')

    def _fallback_calculation(self, designation: str) -> float:
        """Controlled deprecation of fabricated/fixed scientific inputs or inference."""
        raise NotImplementedError('Mock/fixed inference withdrawn; actual typed evidence and a fitted validated model are required')

    def get_status(self) -> Dict[str, Any]:
        """No provider query or numeric scientific confidence is implied by status."""
        return {'initialized':self.initialized, 'screening_available':self.initialized,
            'detector_available':False, 'detector_type':'canonical_candidate_screening',
            'origin_model_available':False, 'scientific_validation_status':'unavailable',
            'five_sigma_claim_permitted':False, 'initialization_error':self.initialization_error,
            'availability_semantics':'component_initialized_not_provider_health_or_calibrated_science'}

    # Legacy method aliases for backward compatibility
    def analyze_single_neo(self, designation: str) -> Dict[str, Any]:
        """Compatibility alias for the same real-source structured assessment."""
        return self.analyze_neo_comprehensive(designation)

    def get_neo_data(self, designation: str) -> Dict[str, Any]:
        """Return the actual normalized source record or propagate explicit failure."""
        if self.pipeline is None:
            raise RuntimeError('Core analysis unavailable; no mock data fallback exists')
        neo = _synchronously(self.pipeline._fetch_neo_data, designation)
        from aneos_core.data.models import NEOData
        if not isinstance(neo, NEOData) or neo.designation != designation:
            raise ValueError('Source evidence unavailable or object identity mismatch')
        return {**neo.to_dict(), 'data_source':'typed_acquisition', 'artificial_probability':None}

    def _generate_classification_explanations(self, result) -> Dict[str, Any]:
        """No undocumented likelihood or physical-origin explanations are inferred."""
        return {'classification_reasoning':['Uncalibrated ranking for follow-up only; origin remains unresolved'],
            'confidence_factors':['No calibrated null or origin model is installed']}

    def _calculate_calibrated_probabilities(self, result) -> Dict[str, Any]:
        """Compatibility status, not an assumed-prior posterior or significance."""
        return {'status':'not_calibrated', 'calibrated_artificial_probability':None,
            'calibrated_classification':None, 'statistical_significance':None, 'sigma_level':None,
            'prior_artificial_rate':None, 'reason':'No independently calibrated origin/null artifact installed'}

    def _calculate_impact_probability(self, result, designation: str) -> Dict[str, Any]:
        """Inventory actual input, without invoking an unfitted heuristic probability."""
        return describe_impact_prerequisites(self._actual_neo(result))

    def _extract_orbital_elements_for_impact(self, result) -> Optional['OrbitalElements']:
        """Never construct an orbit from a name or an origin probability."""
        neo = self._actual_neo(result)
        return neo.orbital_elements if neo is not None else None

    def _create_mock_orbital_elements_for_impact(self, designation: str) -> 'OrbitalElements':
        """Controlled deprecation of fabricated/fixed scientific inputs or inference."""
        raise NotImplementedError('Mock/fixed inference withdrawn; actual typed evidence and a fitted validated model are required')

    def _extract_close_approaches_for_impact(self, result) -> List:
        """Actual supplied computed encounters only; absence is not fabricated history."""
        neo = self._actual_neo(result)
        return list(neo.close_approaches) if neo is not None else []

    def _is_earth_crossing_orbit_simple(self, orbital_elements) -> bool:
        """Descriptive current radial overlap only, never a future collision probability."""
        if orbital_elements is None or orbital_elements.semi_major_axis is None or orbital_elements.eccentricity is None:
            return None
        a, e = orbital_elements.semi_major_axis, orbital_elements.eccentricity
        if a <= 0 or not 0 <= e < 1:
            return None
        return a * (1-e) <= 1.017 and a * (1+e) >= .983

    def _estimate_observation_arc_quality(self, result) -> float:
        """Historical name: only actual dated arc length, not quality/confidence."""
        return describe_impact_prerequisites(self._actual_neo(result))['input_inventory']['observation_arc_days']

    def _generate_concrete_orbital_explanations(self, orbital_elements, sigma_level: float) -> List[str]:
        """Controlled deprecation of fabricated/fixed scientific inputs or inference."""
        raise NotImplementedError('Mock/fixed inference withdrawn; actual typed evidence and a fitted validated model are required')

    def _add_impact_assessment_rationale(self, impact_assessment: Dict[str, Any],
                                       impact_result, designation: str):
        """Controlled deprecation of fabricated/fixed scientific inputs or inference."""
        raise NotImplementedError('Mock/fixed inference withdrawn; actual typed evidence and a fitted validated model are required')

    def _validate_analysis_output(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """A wrapper cannot certify physics/quality without required model evidence."""
        return {'status':'unavailable', 'overall_validation_passed':False,
            'reason':'Independent observational fits and scientific review are not installed'}

def main(argv=None):
    """Standalone command compatibility, delegated to the shared CLI service."""
    import argparse
    import json
    from aneos_core.analysis.cli import main as analysis_main
    parser = argparse.ArgumentParser(description='Actual-source NEO screening; no calibrated origin probability is installed')
    parser.add_argument('command', choices=['single','analyze','status'])
    args, remaining = parser.parse_known_args(argv)
    if args.command == 'status':
        if remaining:
            parser.error('status does not accept object or analysis options')
        print(json.dumps(SimpleNEOAnalyzer().get_status(), indent=2))
        return 0
    return analysis_main(remaining)

if __name__ == "__main__":
    raise SystemExit(main() or 0)
