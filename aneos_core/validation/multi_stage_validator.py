"""
Multi-Stage Validation Pipeline for aNEOS Scientific Rigor Enhancement.

This module implements the core 5-stage validation pipeline designed to
achieve >90% false positive rejection while preserving all existing
functionality through additive architecture.

The 5-stage pipeline:
1. Data Quality Filter (Target: 60% FP reduction)
2. Known Object Cross-Match (Target: 80% FP reduction)  
3. Physical Plausibility (Target: 90% FP reduction)
4. Statistical Significance (Target: 95% FP reduction)
5. Expert Review Threshold (Target: >98% FP reduction)
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .statistical_testing import StatisticalTesting, StatisticalTestResult, MultipleTestingResult
from .false_positive_prevention import FalsePositivePrevention, FalsePositiveResult, SpaceDebrisMatch
from .delta_bic_analysis import DeltaBICAnalyzer, enhance_stage3_with_delta_bic
from .human_hardware_analysis import HumanHardwareAnalyzer, HumanHardwareMatch
from .spectral_outlier_analysis import SpectralOutlierAnalyzer, SpectralOutlierResult
from .radar_polarization_analysis import RadarPolarizationAnalyzer
from .thermal_ir_analysis import ThermalIRAnalyzer
from .gaia_astrometric_calibration import GaiaAstrometricCalibrator

logger = logging.getLogger(__name__)

@dataclass
class ValidationStageResult:
    """Results from a single validation stage."""
    stage_number: int
    stage_name: str
    passed: bool
    score: Optional[float]
    confidence: Optional[float]
    false_positive_reduction: Optional[float]
    details: Dict[str, Any]
    processing_time_ms: float
    validation_status: Optional[str] = None
    
@dataclass
class EnhancedAnalysisResult:
    """
    Enhanced analysis result that wraps the original result with validation data.
    
    This preserves the original analysis result while adding comprehensive
    validation information without modifying existing code.
    """
    # Original analysis result (unchanged)
    original_result: Any
    
    # Enhanced validation data (new)
    validation_result: 'ValidationResult'
    statistical_tests: Dict[str, StatisticalTestResult]
    uncertainty_analysis: Dict[str, Any]
    
    # Metadata
    enhancement_timestamp: datetime
    enhancement_version: str = "1.0.0"
    screening_assessment: Optional[Dict[str, Any]] = None
    
    def __getattr__(self, name):
        """Expose canonical claims; historical input remains only in original_result."""
        state = object.__getattribute__(self, '__dict__')
        assessment = state.get('screening_assessment') or {}
        guarded = {'is_artificial', 'confidence', 'artificial_probability', 'sigma_confidence',
                   'sigma_level', 'combined_p_value', 'false_discovery_rate', 'classification',
                   'candidate', 'screening_score', 'human_origin_status', 'origin_status',
                   'bayesian_probability', 'statistical_certainty', 'artificial_confidence',
                   'artificial_object_likelihood', 'overall_false_positive_probability', 'overall_confidence'}
        if name in guarded:
            return assessment.get(name)
        if name == 'analysis':
            return {'assessment': assessment, 'legacy_analysis_status': 'retained_only_in_original_result'}
        if name in {'metadata', 'analysis_metadata'}:
            return assessment.get('analysis_metadata', {'scientific_validation_status': 'unavailable'})
        if name == 'data_quality':
            result = state.get('validation_result')
            stages = getattr(result, 'stage_results', [])
            return stages[0].details if stages else {'status': 'unavailable'}
        return getattr(state.get('original_result'), name)

@dataclass 
class ValidationResult:
    """Comprehensive validation result from all stages."""
    overall_validation_passed: bool
    overall_false_positive_probability: Optional[float]
    overall_confidence: Optional[float]
    
    # Individual stage results
    stage_results: List[ValidationStageResult]
    
    # Aggregated results
    space_debris_matches: List[SpaceDebrisMatch]
    synthetic_population_percentile: Optional[float]
    statistical_significance_summary: Optional[MultipleTestingResult]
    
    # Enhanced ΔBIC orbital dynamics analysis (Stage 3 enhancement)
    delta_bic_analysis: Optional[Dict[str, Any]] = None
    orbital_anomaly_detection: Optional[Dict[str, Any]] = None
    artificial_object_likelihood: Optional[float] = None
    non_gravitational_evidence: Optional[Dict[str, Any]] = None
    
    # THETA SWARM Human Hardware Analysis Integration
    human_hardware_analysis: Optional[HumanHardwareMatch] = None
    hardware_classification: Optional[str] = None  # satellite, debris, launch_vehicle, unknown
    hardware_classification_confidence: Optional[float] = None
    material_analysis_result: Optional[Dict[str, Any]] = None
    constellation_match_result: Optional[Dict[str, Any]] = None
    
    # IOTA SWARM Spectral Analysis Integration
    spectral_analysis_result: Optional[Dict[str, Any]] = None
    spectral_classification: Optional[str] = None  # asteroid spectral class (C, S, M, etc.)
    spectral_classification_confidence: Optional[float] = None
    artificial_material_probability: Optional[float] = None
    spectral_outlier_detection: Optional[Dict[str, Any]] = None
    
    # KAPPA SWARM Radar Polarization Analysis Integration
    radar_polarization_result: Optional[Dict[str, Any]] = None
    radar_material_classification: Optional[str] = None  # 'metallic', 'rocky', 'icy', 'mixed', 'artificial'
    radar_surface_type: Optional[str] = None  # 'regolith', 'solid_rock', 'metal', 'composite'
    radar_artificial_probability: Optional[float] = None
    radar_surface_roughness: Optional[str] = None  # 'mirror', 'smooth', 'moderate', 'rough', 'chaotic'
    radar_quality_score: Optional[float] = None
    radar_population_percentile: Optional[float] = None
    
    # LAMBDA SWARM Thermal-IR Analysis Integration
    thermal_ir_result: Optional[Dict[str, Any]] = None
    thermal_beaming_parameter: Optional[float] = None  # η parameter for thermal emission
    thermal_artificial_probability: Optional[float] = None
    thermal_inertia: Optional[float] = None  # J m^-2 K^-1 s^-1/2
    thermal_surface_type: Optional[str] = None  # 'regolith', 'solid_rock', 'metal', 'ice', 'composite', 'artificial'
    thermal_conductivity: Optional[float] = None  # W m^-1 K^-1
    yarkovsky_significance: Optional[float] = None  # Statistical significance of non-gravitational acceleration
    thermal_analysis_reliability: Optional[float] = None
    thermal_anomaly_significance: Optional[float] = None
    
    # MU SWARM Gaia Astrometric Precision Calibration Integration
    gaia_astrometric_result: Optional[Dict[str, Any]] = None
    gaia_validation_passed: Optional[bool] = None
    gaia_quality_score: Optional[float] = None  # Overall astrometric quality (0-1)
    gaia_artificial_probability: Optional[float] = None  # Gaia-based artificial object likelihood
    gaia_position_precision_mas: Optional[float] = None  # Position precision in milliarcseconds
    gaia_proper_motion_significance: Optional[float] = None  # Proper motion statistical significance
    gaia_parallax_significance: Optional[float] = None  # Parallax statistical significance
    gaia_reference_frame_quality: Optional[str] = None  # 'excellent', 'good', 'fair'
    gaia_artificial_indicators: Optional[List[str]] = None  # List of artificial object indicators
    gaia_processing_time_ms: Optional[float] = None
    
    # Final recommendations
    recommendation: str = 'expert_review'  # 'accept', 'reject', 'expert_review'
    expert_review_priority: str = 'medium'  # 'low', 'medium', 'high', 'urgent'
    
    # Processing metadata
    total_processing_time_ms: float = 0.0
    validation_timestamp: datetime = None
    contract_version: Optional[str] = None
    validation_status: Optional[str] = None
    evidence_id: Optional[str] = None
    policy_id: Optional[str] = None
    validation_evidence_id: Optional[str] = None
    screening_score: Optional[float] = None

class MultiStageValidator:
    """Evidence inventory and required-prerequisite status, not installed science.

    Existing function names remain available. Fixed source/physical/null inputs
    and asserted probability reductions have been withdrawn (ADR-068).
    Standalone research analyzers remain in their own modules, not registered
    here as validated likelihood or origin evidence.
    """
    def __init__(self, config=None):
        import copy
        self.config = self._default_config()
        self.config.update(copy.deepcopy(config or {}))
        self.logger = logging.getLogger(__name__)
        self.statistical_testing = StatisticalTesting(alpha_level=self.config.get('alpha_level', .05))
        self._initialize_null_distributions()
        self.false_positive_prevention = None
        self.delta_bic_analyzer = None
        self.spectral_analyzer = None
        self.radar_analyzer = None
        self.thermal_ir_analyzer = None
        self.gaia_calibrator = None
        self.capability_status = {'screening': 'available_uncalibrated',
            'scientific_validation': 'unavailable', 'legacy_analyzers': 'not_registered_as_validated_evidence'}

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for validation pipeline."""
        return {
            'alpha_level': 0.05,
            'stage1_thresholds': {
                'completeness_threshold': 0.7,
                'source_reliability': 0.8,
                'observation_span_days': 30,
                'orbital_uncertainty': 0.1
            },
            'stage2_thresholds': {
                'max_delta_v': 25,  # m/s
                'max_epoch_diff': 30,  # days
                'min_confidence': 0.8
            },
            'stage3_thresholds': {
                'plausibility_threshold': 0.6,
                'physics_consistency': 0.7,
                'delta_bic_threshold': 10.0,
                'artificial_likelihood_threshold': 0.7
            },
            'stage4_thresholds': {
                'min_significant_indicators': 3,
                'max_corrected_p_value': 0.01,
                'min_effect_size': 0.5
            },
            'stage5_thresholds': {
                'overall_score': 0.8,
                'confidence': 0.9,
                'min_anomalous_categories': 3
            },
            'synthetic_population_size': 1000,
            'cache_dir': None,
            'enable_delta_bic': True,
            'delta_bic_config': {
                'bic_threshold': 10.0,
                'anomaly_threshold': 0.95,
                'monte_carlo_samples': 5000,
                'force_models': ['yarkovsky', 'yorp', 'srp']
            },
            'hardware_analysis_config': {
                'processing_timeout_seconds': 2.0,
                'min_confidence_threshold': 0.7,
                'constellation_detection': {
                    'known_constellations': ['starlink', 'oneweb', 'kuiper', 'globalstar', 'iridium']
                },
                'performance_targets': {
                    'max_processing_time_ms': 2000,
                    'min_accuracy_rate': 0.95
                }
            },
            'enable_spectral_analysis': True,
            'spectral_analysis_config': {
                'wavelength_range': (0.4, 2.5),
                'spectral_resolution': 0.01,
                'outlier_contamination': 0.05,
                'classification_confidence_threshold': 0.7,
                'artificial_probability_threshold': 0.5,
                'processing_timeout_ms': 500,
                'max_processing_time_ms': 500,
                'enable_detailed_analysis': True,
                'quality_threshold': 0.6
            },
            'enable_radar_analysis': True,
            'radar_analysis_config': {
                'frequency_bands': ['S', 'X', 'C'],
                'polarization_modes': ['linear', 'circular', 'full_stokes'],
                'min_snr': 10.0,
                'artificial_probability_threshold': 0.7,
                'natural_consistency_threshold': 0.6,
                'material_confidence_threshold': 0.8,
                'max_processing_time_ms': 300,
                'target_accuracy': 0.92,
                'enable_arecibo_db': True,
                'enable_goldstone_db': True,
                'enable_green_bank_db': True,
                'ml_model_types': ['random_forest', 'isolation_forest', 'pca'],
                'cross_validation_folds': 5,
                'integrate_with_spectral': True,
                'integrate_with_orbital': True,
                'enable_statistical_validation': True,
                'significance_level': 0.05,
                'monte_carlo_samples': 1000
            },
            'enable_thermal_ir_analysis': True,
            'thermal_ir_analysis_config': {
                'thermal_models': {
                    'preferred_model': 'NEATM',
                    'model_selection_threshold': 0.8,
                    'chi_squared_threshold': 2.0
                },
                'beaming_parameter': {
                    'natural_range': (0.5, 1.5),
                    'artificial_threshold': 1.5,
                    'high_confidence_threshold': 0.9
                },
                'thermal_inertia': {
                    'rock_range': (200, 2000),
                    'regolith_range': (5, 200),
                    'metal_range': (1000, 10000),
                    'ice_range': (10, 100)
                },
                'yarkovsky_analysis': {
                    'significance_threshold': 3.0,
                    'min_orbital_arc_years': 10.0,
                    'drift_rate_threshold': 1e-4
                },
                'artificial_detection': {
                    'thermal_conductivity_threshold': 10.0,
                    'homogeneity_threshold': 0.9,
                    'metal_probability_threshold': 0.7,
                    'overall_artificial_threshold': 0.6
                },
                'thermal_databases': {
                    'neowise': {'enabled': True, 'weight': 0.4},
                    'spitzer': {'enabled': True, 'weight': 0.3},
                    'iras': {'enabled': True, 'weight': 0.2},
                    'akari': {'enabled': True, 'weight': 0.1}
                },
                'processing_limits': {
                    'max_processing_time_ms': 200,
                    'min_observations': 3,
                    'max_phase_angle_deg': 30.0
                },
                'uncertainty_analysis': {
                    'monte_carlo_samples': 1000,
                    'confidence_levels': [0.68, 0.95, 0.99]
                }
            },
            'enable_gaia_astrometry': True,
            'gaia_astrometric_config': {
                'gaia_table': 'gaiadr3.gaia_source',
                'data_release': 'EDR3',
                'max_sources': 100,
                'default_radius_arcsec': 30.0,
                'position_precision_mas': 0.03,
                'proper_motion_precision_mas_yr': 0.02,
                'parallax_precision_mas': 0.04,
                'systematic_accuracy_mas': 0.1,
                'pm_significance_threshold': 3.0,
                'parallax_significance_threshold': 3.0,
                'artificial_probability_threshold': 0.7,
                'ruwe_threshold': 1.4,
                'stellar_pm_percentiles': (5, 95),
                'excess_noise_threshold': 0.5,
                'color_outlier_sigma': 3.0,
                'parallax_consistency_tolerance': 2.0,
                'query_timeout_sec': 10.0,
                'max_processing_time_ms': 100,
                'enable_cache': True,
                'cache_expiry_hours': 24,
                'enable_parallel_queries': True,
                'reference_epoch': 2016.0,
                'target_epoch': None,
                'reference_frame': 'ICRS',
                'enable_simbad_crossmatch': True,
                'simbad_radius_arcsec': 5.0,
                'enable_local_catalog': False,
                'local_catalog_path': None
            }
        }
    
    def _initialize_null_distributions(self):
        """No calibrated null artifact is installed; defaults are not measurements."""
        self.null_distributions = {}

    @staticmethod
    def _stage(number, name, status, details):
        return ValidationStageResult(number, name, False, None, None, None,
            details, 0., validation_status=status)

    def _data_quality_inventory(self, neo_data):
        orbit = self._extract_orbital_elements(neo_data)
        pp = getattr(neo_data, 'physical_properties', None)
        fields = [orbit.get(k) for k in ('a', 'e', 'i')] + [getattr(pp, k, None)
            for k in ('diameter_km', 'absolute_magnitude_h', 'albedo')]
        import math
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value)) for value in fields):
            raise ValueError('Invalid screening-field measurement')
        available = sum(value is not None for value in fields)
        return {'available_ranking_orbit_fields': [k for k in ('a', 'e', 'i') if k in orbit],
            'screening_field_coverage': {'available_count': available, 'field_count': 6,
                'fraction': available / 6, 'semantics': 'field_presence_not_quality_or_independent_evidence'},
            'source_labels': list(getattr(neo_data, 'sources_used', [])),
            'observation_span_days': self._assess_observation_span(neo_data),
            'source_reliability': None, 'orbital_uncertainty_status': 'no_generic_mixed_unit_scalar',
            'orbital_covariance_available': bool(getattr(neo_data, 'nongrav', None) and neo_data.nongrav.covariance),
            'reason': 'Available inputs do not establish calibrated observational fitness'}

    async def validate_analysis_result(self, neo_data, analysis_result):
        """All callers use canonical screening; preserve legacy input separately."""
        from ..data.models import NEOData
        from ..analysis.screening import ScreeningAnalysisService
        try:
            if isinstance(neo_data, dict):
                neo_data = NEOData.from_dict(neo_data)
            if not isinstance(neo_data, NEOData):
                raise ValueError('NEO evidence unavailable or invalid')
            original = analysis_result
            payload = getattr(analysis_result, 'screening_assessment', None) or (getattr(analysis_result, 'analysis', {}) or {}).get('assessment')
            service = ScreeningAnalysisService()
            if isinstance(payload, dict) and payload.get('schema_version') == 'screening-assessment-v1':
                replay = service.replay(payload['evidence_snapshot'], payload['evidence_id'])
                if replay.policy_id != payload['policy_id']:
                    raise ValueError('Screening policy mismatch')
                # Recompute inference fields rather than forwarding caller-edited claims.
                payload = replay.to_payload()
            else:
                payload = service.analyze_neo(neo_data).to_payload()
            enhanced = self._validate_uncalibrated_screening(neo_data, analysis_result, payload)
            enhanced.original_result = original
            enhanced.screening_assessment = payload
            return enhanced
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            stages = [self._stage(n, f'Required prerequisite {n}', 'unavailable', {'error': reason}) for n in range(1, 6)]
            validation = self._aggregate_validation_stages(stages)
            return EnhancedAnalysisResult(analysis_result, validation, {},
                {'status': 'unavailable', 'overall_uncertainty': None, 'reason': reason},
                datetime.now(), enhancement_version='required-review-v1', screening_assessment=None)

    def _validate_uncalibrated_screening(self, neo_data, analysis_result, payload):
        from ..analysis.screening import ScreeningAnalysisService
        from .review_evidence import freeze_review_inputs
        service = ScreeningAnalysisService()
        replay = service.replay(payload['evidence_snapshot'], payload['evidence_id'])
        if replay.policy_id != payload['policy_id'] or neo_data.designation != replay.designation:
            raise ValueError('Object/policy mismatch in required validation')
        current = service.analyze_neo(neo_data, payload['evidence_snapshot']['additional'])
        if current.evidence_id != replay.evidence_id:
            raise ValueError('Ranking evidence no longer matches NEO input')
        _, review_id = freeze_review_inputs(neo_data)
        nongrav = neo_data.nongrav.evidence_report() if neo_data.nongrav else None
        details = self._data_quality_inventory(neo_data)
        stages = [self._stage(1, 'Data quality fitness', 'inconclusive', details),
            self._stage(2, 'Known-object identity', 'unavailable', {
                'reason': 'No verified cross-catalogue trajectory identification supplied; name lookup is insufficient'}),
            self._stage(3, 'Comparative physical fit', 'unavailable', {
                'non_gravitational_evidence': nongrav,
                'reason': 'Compatible observation residual fits and physical comparison unavailable'}),
            self._stage(4, 'Calibrated statistical significance', 'unavailable', {
                'p_value': None, 'gaussian_equivalent_sigma': None,
                'reason': 'No calibrated null or search-wide error procedure installed'}),
            self._stage(5, 'Scientific review eligibility', 'inconclusive', {
                'screening_candidate': payload['candidate'],
                'reason': 'Screening candidacy does not satisfy the preceding scientific prerequisites'})]
        validation = self._aggregate_validation_stages(stages)
        validation.screening_score = payload.get("screening_score")
        validation.non_gravitational_evidence = nongrav
        validation.evidence_id, validation.policy_id = replay.evidence_id, replay.policy_id
        validation.validation_evidence_id = review_id
        return EnhancedAnalysisResult(analysis_result, validation, {},
            {'status': 'unavailable', 'overall_uncertainty': None,
             'reason': 'No observation covariance propagation or calibrated uncertainty model'},
            datetime.now(), enhancement_version='required-review-v1', screening_assessment=payload)

    async def stage1_data_quality_filter(self, neo_data):
        try:
            return self._stage(1, 'Data quality fitness', 'inconclusive', self._data_quality_inventory(neo_data))
        except Exception as exc:
            return self._stage(1, 'Data quality fitness', 'unavailable', {'error': str(exc)})

    async def stage2_known_object_crossmatch(self, neo_data, analysis_result):
        return self._stage(2, 'Known-object identity', 'unavailable', {
            'reason': 'Verified catalogue trajectory identification not supplied; no fabricated crossmatch orbit'})

    async def stage3_physical_plausibility(self, neo_data, analysis_result):
        try:
            ng = getattr(neo_data, 'nongrav', None)
            return self._stage(3, 'Comparative physical fit', 'unavailable', {
                'non_gravitational_evidence': ng.evidence_report() if ng else None,
                'reason': 'Fitted observational likelihood and competing physical models unavailable'})
        except Exception as exc:
            return self._stage(3, 'Comparative physical fit', 'unavailable', {'error': str(exc)})

    async def stage4_statistical_significance(self, analysis_result):
        return self._stage(4, 'Calibrated statistical significance', 'unavailable', {
            'p_value': None, 'gaussian_equivalent_sigma': None,
            'reason': 'No installed calibrated null or search-wide procedure; heuristic indicators are not statistical tests'})

    async def stage5_expert_review_threshold(self, analysis_result):
        return self._stage(5, 'Scientific review eligibility', 'inconclusive', {
            'reason': 'Cannot infer review eligibility from invented confidence, category counts or impact flags'})

    def _aggregate_validation_stages(self, stage_results):
        """Aggregate explicit prerequisite outcomes, never asserted FP reductions."""
        import json
        valid = False
        if not isinstance(stage_results, (list, tuple)):
            stage_results = []
        try:
            valid = len(stage_results) == 5 and {s.stage_number for s in stage_results} == set(range(1, 6))
            valid = valid and all(type(s.stage_number) is int and type(s.passed) is bool and
                s.validation_status in {'passed', 'failed', 'inconclusive', 'unavailable'} and
                s.passed == (s.validation_status == 'passed') and isinstance(s.details, dict) and
                not s.details.get('error') and not s.details.get('error_message') for s in stage_results)
            json.dumps([asdict(s) for s in stage_results], allow_nan=False)
        except (AttributeError, TypeError, ValueError):
            valid = False
        passed = valid and all(s.passed for s in stage_results)
        status = 'unavailable' if not valid else 'passed' if passed else 'failed' if any(
            s.validation_status == 'failed' for s in stage_results) else 'inconclusive'
        return ValidationResult(overall_validation_passed=bool(passed),
            overall_false_positive_probability=None, overall_confidence=None,
            stage_results=stage_results, space_debris_matches=[], synthetic_population_percentile=None,
            statistical_significance_summary=None, recommendation='expert_review' if passed else
                'additional_evidence_required' if status == 'inconclusive' else 'do_not_advance',
            contract_version='required-review-v1', validation_status=status, validation_timestamp=datetime.now())

    def _run_statistical_tests(self, analysis_result):
        """No installed null artifact: no probability tests are performed."""
        return {}

    async def _calculate_uncertainties(self, neo_data, analysis_result):
        return {'status': 'unavailable', 'uncertainty_method': None, 'overall_uncertainty': None,
                'indicator_uncertainties': {}, 'confidence_bounds': None}

    def _assess_data_completeness(self, neo_data):
        return self._data_quality_inventory(neo_data)['screening_field_coverage']['fraction']

    def _assess_source_reliability(self, neo_data):
        return None  # No source-validation artifact; provider name is not measured reliability.

    def _assess_observation_span(self, neo_data):
        start, end = getattr(neo_data, 'first_observation', None), getattr(neo_data, 'last_observation', None)
        if start is None or end is None:
            return None
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValueError('Observation dates must be datetime values')
        try:
            days = (end - start).total_seconds() / 86400
        except TypeError as exc:
            raise ValueError('Incompatible observation time conventions') from exc
        if days < 0:
            raise ValueError('Observation arc is reversed')
        return days

    def _assess_orbital_uncertainty(self, neo_data):
        return None  # A mixed-unit covariance has no generic scalar uncertainty.

    def _extract_orbital_elements(self, neo_data):
        from ..data.models import NEOData
        if not isinstance(neo_data, NEOData) or neo_data.orbital_elements is None:
            return {}
        oe = neo_data.orbital_elements
        mapping = {'a': 'semi_major_axis', 'e': 'eccentricity', 'i': 'inclination',
                   'Omega': 'ra_of_ascending_node', 'omega': 'arg_of_periapsis', 'M': 'mean_anomaly'}
        import math
        result = {key: getattr(oe, name) for key, name in mapping.items() if getattr(oe, name) is not None}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in result.values()):
            raise ValueError('Invalid orbital measurement')
        return result

    def _extract_indicator_scores(self, analysis_result):
        """Explicit diagnostic values only; never interpreted as null probabilities."""
        import math
        scores = {}
        for name, value in getattr(analysis_result, 'indicator_results', {}).items():
            raw = getattr(value, 'raw_score', None)
            if raw is not None:
                if isinstance(raw, bool) or not isinstance(raw, (float, int)) or not math.isfinite(raw):
                    raise ValueError(f'Invalid diagnostic indicator {name}')
                scores[name] = raw
        return scores

    def _count_anomalous_categories(self, analysis_result):
        return None  # No validated category decision policy is installed.

    def _check_high_impact_indicators(self, analysis_result):
        return None  # No validated high-impact criterion is installed.
