"""Compatibility analysis facade using typed acquisition/shared candidate screening.

Advanced naming and installed modules do not establish model calibration,
independent scientific validation, origin identification or impact probability.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc

from ..interfaces.unified_analysis import (
    aNEOSAnalysisInterface, aNEOSAnalysisResult, AdvancedaNEOSCapabilities,
    get_advanced_capabilities, AnalysisCapability
)
from ..interfaces.detection import DetectionResult, OrbitalElementsNormalizer

logger = logging.getLogger(__name__)


def _synchronously(method, *args, **kwargs):
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(method(*args, **kwargs))
    raise RuntimeError('Synchronous analysis cannot run inside an active event loop; use the async analysis method')


class AdvancedaNEOSAnalyzer(aNEOSAnalysisInterface):
    """Preserve the public facade without prototype scoring or source fabrication."""

    def __init__(self, auto_configure: bool = True, pipeline=None):
        """Use the actual core pipeline, not a best-detector fallback cascade."""
        from .pipeline import create_analysis_pipeline
        self.logger = logging.getLogger(__name__)
        self.pipeline = pipeline if pipeline is not None else create_analysis_pipeline()
        self.capabilities = get_advanced_capabilities()
        self.detection_manager = None

    @property
    def data_fetcher(self):
        return self.pipeline.data_fetcher

    @data_fetcher.setter
    def data_fetcher(self, value):
        self.pipeline.data_fetcher = value

    def _auto_configure_capabilities(self):
        """Compatibility inventory only; no automatic model/provider activation."""
        return dict(self.capabilities.component_inventory)

    async def analyze_neo_async(self, designation, orbital_elements=None, physical_data=None,
                                enrichment_sources=None, force_refresh=False):
        """One canonical core result; supplied raw data is never silently supplemented."""
        if enrichment_sources is not None:
            raise NotImplementedError('Per-call source filtering is not implemented; no source selection will be claimed')
        start = time.monotonic()
        supplied = orbital_elements is not None or physical_data is not None
        try:
            neo = self._neo_from_raw(designation, {} if orbital_elements is None else orbital_elements,
                {} if physical_data is None else physical_data) if supplied else None
            core = await self.pipeline.analyze_neo(designation, neo, force_refresh=force_refresh)
        except (ValueError, TypeError, AttributeError) as exc:
            core = self.pipeline._failure_result(designation, 'invalid_evidence', str(exc), start)
        result = self._from_core(core)
        if supplied:
            import json
            def encode(value):
                if isinstance(value, datetime):
                    return value.isoformat()
                raise TypeError('Caller record contains an unsupported value type')
            try:
                result.metadata['caller_supplied_input_record'] = json.loads(json.dumps({
                    'orbital_elements':orbital_elements, 'physical_data':physical_data,
                    'evidence_basis':'caller_supplied_not_independently_verified'}, default=encode, allow_nan=False))
            except (TypeError, ValueError):
                result.metadata['caller_supplied_input_record'] = {
                    'status':'rejected_non_json_or_nonfinite_input',
                    'evidence_basis':'invalid_input_not_used_as_observations'}
        return result

    def _neo_from_raw(self, designation, orbit, physical):
        from ..data.models import NEOData, OrbitalElements, PhysicalProperties
        if not isinstance(orbit, dict) or not isinstance(physical, dict):
            raise ValueError('Raw orbital and physical inputs must be dictionaries or missing')
        # Shared raw validator defines aliases, domain and diameter conversion.
        checked = self.pipeline.screening_service.analyze_raw(designation, orbit, physical)
        raw = checked.evidence_snapshot
        elements = dict(orbit)
        for short, long in {'a':'semi_major_axis','e':'eccentricity','i':'inclination',
            'om':'ascending_node','w':'argument_of_perihelion','M':'mean_anomaly'}.items():
            if short in elements:
                if long in elements and elements[short] != elements[long]:
                    raise ValueError(f'Conflicting orbital aliases: {short}/{long}')
                elements[long] = elements.pop(short)
        epoch = elements.get('epoch')
        if isinstance(epoch, str):
            try:
                epoch = datetime.fromisoformat(epoch.replace('Z', '+00:00'))
            except ValueError as exc:
                raise ValueError('Raw epoch must be a timezone-aware ISO datetime, not an unlabelled JD') from exc
            elements['epoch'] = epoch
        if epoch is not None and (not isinstance(epoch, datetime) or epoch.tzinfo is None):
            raise ValueError('Raw epoch must be a timezone-aware ISO datetime; native TDB epochs require labelled evidence')
        import math
        for key in ('ascending_node','argument_of_perihelion','mean_anomaly','orbital_period'):
            value = elements.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value)):
                raise ValueError(f'{key} must be finite numeric input')
        oe = OrbitalElements(**{key:value for key,value in elements.items()
            if key in OrbitalElements.__dataclass_fields__}) if orbit else None
        phys = raw['physical']
        rotation = physical.get('rotation_period_hours', physical.get('rotation_period'))
        if 'rotation_period_hours' in physical and 'rotation_period' in physical and physical['rotation_period_hours'] != physical['rotation_period']:
            raise ValueError('Conflicting rotation periods')
        if rotation is not None and (isinstance(rotation, bool) or not isinstance(rotation, (int,float)) or not math.isfinite(rotation) or rotation <= 0):
            raise ValueError('Rotation period must be positive, finite hours or missing')
        pp = PhysicalProperties(diameter_km=phys['diameter']/1000 if 'diameter' in phys else None,
            absolute_magnitude_h=phys.get('absolute_magnitude'), albedo=phys.get('albedo'),
            spectral_type=phys.get('spectral_type'), rotation_period_hours=rotation) if physical else None
        return NEOData(designation, orbital_elements=oe, physical_properties=pp)

    def _from_core(self, core):
        from .impact_evidence import describe_impact_prerequisites
        payload = core.screening_assessment
        neo = core.neo_data
        return aNEOSAnalysisResult(designation=core.designation,
            is_artificial=None, artificial_probability=None, confidence_level=None,
            classification=core.anomaly_score.classification, risk_assessment='not_assessed',
            threat_level='not_assessed', sigma_statistical_level=None,
            orbital_analysis=neo.orbital_elements.to_dict() if neo and neo.orbital_elements else None,
            physical_analysis=neo.physical_properties.to_dict() if neo and neo.physical_properties else None,
            temporal_analysis={'status':'not_fitted',
                'first_observation':neo.first_observation.isoformat() if neo and neo.first_observation else None,
                'last_observation':neo.last_observation.isoformat() if neo and neo.last_observation else None},
            data_completeness=core.data_quality.get('completeness_score'), analysis_quality=None,
            validation_status='unavailable_uncalibrated', data_sources_used=list(neo.sources_used) if neo else [],
            detector_used=payload['engine_version'] if payload else 'unavailable',
            analysis_method='canonical_candidate_screening', processing_time_ms=core.processing_time*1000,
            metadata=dict(core.analysis_metadata), screening_assessment=payload, neo_data=neo,
            status=core.status, errors=list(core.errors), impact_assessment=describe_impact_prerequisites(neo),
            data_quality=dict(core.data_quality))

    def analyze_neo_comprehensive(self,
                                 designation: str,
                                 orbital_elements: Optional[Dict[str, Any]] = None,
                                 physical_data: Optional[Dict[str, Any]] = None,
                                 enrichment_sources: Optional[List[str]] = None) -> aNEOSAnalysisResult:
        """Synchronous adapter; use analyze_neo_async inside an active event loop."""
        return _synchronously(self.analyze_neo_async, designation, orbital_elements=orbital_elements,
            physical_data=physical_data, enrichment_sources=enrichment_sources)

    def detect_artificial_signatures(self,
                                   orbital_elements: Dict[str, Any],
                                   physical_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """Raw screening adapter; no origin probability or incomplete-orbit defaults."""
        return self.pipeline.screening_service.analyze_raw('caller-supplied', orbital_elements,
            physical_data).to_detection_result()

    def assess_threat_level(self, analysis_result: aNEOSAnalysisResult) -> str:
        """
        Assess threat level based on comprehensive analysis.

        Uses advanced threat assessment considering multiple factors.
        """
        return self.assess_threat_level_internal(analysis_result)

    def assess_threat_level_internal(self, analysis_input: Union[aNEOSAnalysisResult, DetectionResult]) -> str:
        """Origin rarity is not an impact/strategic threat estimate."""
        return 'not_assessed'

    def fetch_neo_data_multi_source(self, designation: str,
                                   sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Use the existing typed acquisition contract without mock fallback."""
        if sources is not None:
            raise NotImplementedError('Per-call source filtering is not implemented; no source selection will be claimed')
        neo = _synchronously(self.pipeline._fetch_neo_data, designation)
        from ..data.models import NEOData
        if not isinstance(neo, NEOData) or neo.designation != designation:
            raise ValueError('Source evidence unavailable or object identity mismatch')
        data = neo.to_dict()
        data['physical_data'] = dict(self.pipeline.screening_service.analyze_neo(neo).evidence_snapshot['physical'])
        data['data_sources'] = list(neo.sources_used)
        return data

    def _fetch_real_neo_data(self, designation: str) -> Optional[Dict[str, Any]]:
        """Compatibility alias for actual typed acquisition, with explicit failures."""
        return self.fetch_neo_data_multi_source(designation)

    def enrich_neo_data_comprehensive(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve supplied records; no inferred measurements or invented enrichment."""
        import copy
        result = copy.deepcopy(neo_data)
        result['enrichment_status'] = 'not_performed_no_validated_model'
        return result

    def validate_analysis_quality(self, analysis_result: aNEOSAnalysisResult) -> Dict[str, Any]:
        """Field presence is not calibrated scientific quality or validation."""
        return {'status':'unavailable', 'quality_score':None,
            'validation_flags':['independent_scientific_validation_not_installed'],
            'recommendations':['Observation fits, independent calibration and scientific review required']}

    def calculate_anomaly_score_advanced(self, neo_data: Dict[str, Any]) -> Optional[float]:
        """Compatibility scalar is shared ranking units, not a probability."""
        from ..data.models import NEOData
        if isinstance(neo_data, NEOData):
            return self.pipeline.screening_service.analyze_neo(neo_data).screening_score
        if 'physical_properties' in neo_data:
            return self.pipeline.screening_service.analyze_neo(NEOData.from_dict(neo_data)).screening_score
        return self.pipeline.screening_service.analyze_raw(neo_data.get('designation', 'caller-supplied'),
            neo_data.get('orbital_elements', {}), neo_data.get('physical_data')).screening_score

    def generate_comprehensive_report(self, analysis_result: aNEOSAnalysisResult) -> Dict[str, Any]:
        """Use the same replay/input-bound report as CLI and menu adapters."""
        from .reporting import build_screening_report
        return build_screening_report(analysis_result, impact_assessment=analysis_result.impact_assessment)

    # Helper methods

    def _generate_mock_neo_data(self, designation: str) -> Dict[str, Any]:
        """Controlled deprecation of prototype/default scientific inference."""
        raise NotImplementedError('Prototype inference withdrawn; use actual shared screening and explicitly unavailable models')

    def _analyze_orbital_characteristics(self, orbital_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze orbital characteristics in detail."""
        if not orbital_elements:
            return {'status': 'no_data'}

        analysis = {
            'orbit_type': self._classify_orbit_type(orbital_elements),
            'stability_assessment': self._assess_orbital_stability(orbital_elements),
            'anomaly_flags': self._identify_orbital_anomalies(orbital_elements)
        }

        return analysis

    def _analyze_physical_characteristics(self, physical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze physical characteristics in detail."""
        if not physical_data:
            return {'status': 'no_data'}

        analysis = {
            'size_category': self._classify_size_category(physical_data),
            'composition_analysis': self._analyze_composition(physical_data),
            'anomaly_flags': self._identify_physical_anomalies(physical_data)
        }

        return analysis

    def _analyze_temporal_patterns(self, designation: str, orbital_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Controlled deprecation of prototype/default scientific inference."""
        raise NotImplementedError('Prototype inference withdrawn; use actual shared screening and explicitly unavailable models')

    def _extract_anomaly_indicators(self, orbital_analysis: Dict[str, Any],
                                  physical_analysis: Dict[str, Any]) -> List[str]:
        """Extract anomaly indicators from detailed analyses."""
        indicators = []

        if orbital_analysis and orbital_analysis.get('anomaly_flags'):
            indicators.extend(orbital_analysis['anomaly_flags'])

        if physical_analysis and physical_analysis.get('anomaly_flags'):
            indicators.extend(physical_analysis['anomaly_flags'])

        return indicators

    def _calculate_data_completeness(self, orbital_elements: Dict[str, Any],
                                   physical_data: Dict[str, Any]) -> float:
        """Calculate data completeness score."""
        score = 0.0

        # Orbital elements completeness (60% of total)
        if orbital_elements:
            required_orbital = ['a', 'e', 'i']
            orbital_score = sum(1 for elem in required_orbital
                              if orbital_elements.get(elem) or
                                 orbital_elements.get({'a': 'semi_major_axis', 'e': 'eccentricity', 'i': 'inclination'}[elem]))
            score += (orbital_score / len(required_orbital)) * 0.6

        # Physical data completeness (40% of total)
        if physical_data:
            optional_physical = ['diameter', 'albedo', 'mass', 'rotation_period']
            physical_score = sum(1 for elem in optional_physical if physical_data.get(elem))
            score += (physical_score / len(optional_physical)) * 0.4

        return score

    def _assess_analysis_quality(self, detection_result: DetectionResult,
                               orbital_elements: Dict[str, Any]) -> float:
        """Controlled deprecation of prototype/default scientific inference."""
        raise NotImplementedError('Prototype inference withdrawn; use actual shared screening and explicitly unavailable models')

    def _generate_recommendation(self, analysis_result: aNEOSAnalysisResult) -> str:
        """Controlled deprecation of prototype/default scientific inference."""
        raise NotImplementedError('Prototype inference withdrawn; use actual shared screening and explicitly unavailable models')

    # Additional helper methods for detailed analysis
    def _classify_orbit_type(self, orbital_elements: Dict[str, Any]) -> str:
        a = orbital_elements.get('a', orbital_elements.get('semi_major_axis', 1))
        if a < 1.0:
            return "Atira"
        elif a < 1.3:
            return "Aten"
        elif a < 1.52:
            return "Apollo"
        else:
            return "Amor"

    def _assess_orbital_stability(self, orbital_elements: Dict[str, Any]) -> str:
        e = orbital_elements.get('e', orbital_elements.get('eccentricity', 0))
        if e > 0.9:
            return "highly_unstable"
        elif e > 0.6:
            return "unstable"
        else:
            return "stable"

    def _identify_orbital_anomalies(self, orbital_elements: Dict[str, Any]) -> List[str]:
        anomalies = []

        e = orbital_elements.get('e', orbital_elements.get('eccentricity', 0))
        i = orbital_elements.get('i', orbital_elements.get('inclination', 0))
        a = orbital_elements.get('a', orbital_elements.get('semi_major_axis', 1))

        if e > 0.8:
            anomalies.append("extreme_eccentricity")
        if i > 160 or i < 20:
            anomalies.append("unusual_inclination")
        if a > 4.0 or a < 0.5:
            anomalies.append("unusual_semi_major_axis")

        return anomalies

    def _classify_size_category(self, physical_data: Dict[str, Any]) -> str:
        diameter = physical_data.get('diameter', physical_data.get('estimated_diameter', 100))
        if diameter > 1000:
            return "large"
        elif diameter > 100:
            return "medium"
        else:
            return "small"

    def _analyze_composition(self, physical_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'material_type': 'rocky',  # Would be determined from spectral data
            'density_estimate': 'normal',
            'surface_properties': 'typical'
        }

    def _identify_physical_anomalies(self, physical_data: Dict[str, Any]) -> List[str]:
        anomalies = []

        diameter = physical_data.get('diameter', physical_data.get('estimated_diameter'))
        if diameter and (diameter > 1000 or diameter < 1):
            anomalies.append("unusual_size")

        albedo = physical_data.get('albedo')
        if albedo and (albedo > 0.8 or albedo < 0.02):
            anomalies.append("unusual_albedo")

        return anomalies

    def _enrich_orbital_elements(self, orbital_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Controlled deprecation of prototype/default scientific inference."""
        raise NotImplementedError('Prototype inference withdrawn; use actual shared screening and explicitly unavailable models')

    def _enrich_physical_data(self, physical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Controlled deprecation of prototype/default scientific inference."""
        raise NotImplementedError('Prototype inference withdrawn; use actual shared screening and explicitly unavailable models')

    def _compute_additional_characteristics(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Controlled deprecation of prototype/default scientific inference."""
        raise NotImplementedError('Prototype inference withdrawn; use actual shared screening and explicitly unavailable models')


# Global advanced analyzer instance
_advanced_analyzer = None

def get_advanced_aneos_analyzer() -> AdvancedaNEOSAnalyzer:
    """Get global advanced aNEOS analyzer instance."""
    global _advanced_analyzer
    if _advanced_analyzer is None:
        _advanced_analyzer = AdvancedaNEOSAnalyzer()
    return _advanced_analyzer


# Convenience function for direct analysis
def analyze_neo_advanced(designation: str,
                        orbital_elements: Optional[Dict[str, Any]] = None,
                        physical_data: Optional[Dict[str, Any]] = None,
                        enrichment_sources: Optional[List[str]] = None) -> aNEOSAnalysisResult:
    """
    Convenience function for advanced NEO analysis.

    Uses the most sophisticated aNEOS capabilities available.
    """
    analyzer = get_advanced_aneos_analyzer()
    return analyzer.analyze_neo_comprehensive(designation, orbital_elements, physical_data, enrichment_sources)
