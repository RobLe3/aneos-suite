"""Attach scoped authoritative impact evidence without an origin/risk proxy.

The existing enhanced screening result remains intact. Sentry lookup failures are
explicit and do not turn screening scores or catalogue orbits into impact fits.
"""
import asyncio
import copy
import logging
from typing import Any, Optional
from types import MappingProxyType

from .enhanced_pipeline import EnhancedAnalysisPipeline
from . import impact_evidence
from ..data.models import NEOData, OrbitalElements
from ..data.provider_http import ProviderUnavailableError
from ..reporting.evidence_archive import EvidenceArchive


class ImpactEnhancedAnalysisResult:
    def __init__(self, enhanced_result, impact_assessment=None, *, impact_status='not_requested', persistence=None):
        self.enhanced_result = enhanced_result
        self.impact_assessment = copy.deepcopy(impact_assessment)
        self.impact_status = impact_status
        self.impact_persistence = persistence or {'status':'not_attempted'}
        for attr in dir(enhanced_result):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(enhanced_result, attr))

    def to_dict(self):
        result = copy.deepcopy(self.enhanced_result.to_dict()) if hasattr(self.enhanced_result, 'to_dict') else {}
        result.update(impact_assessment=copy.deepcopy(self.impact_assessment),
                      impact_status=self.impact_status, impact_persistence=copy.deepcopy(self.impact_persistence))
        return result

    def get_impact_summary(self):
        if self.impact_assessment is None:
            return {'status':self.impact_status, 'collision_probability':None,
                    'risk_level':'not_assessed', 'calculation_confidence':None}
        report = impact_evidence.require_sentry_report(self.impact_assessment, self.impact_assessment['designation'])
        return {key:copy.deepcopy(report[key]) for key in
                ('designation','status','collision_probability','risk_level','calculation_confidence','time_scope','source','limitations','report_sha256')}


class ImpactEnhancedPipeline:
    def __init__(self, enhanced_pipeline: EnhancedAnalysisPipeline):
        self.enhanced_pipeline = enhanced_pipeline
        self.logger = logging.getLogger(__name__)
        self.impact_calculator = None  # Compatibility attribute; no heuristic engine.
        self.impact_config = MappingProxyType({'source':'JPL Sentry', 'assessment_policy':'explicit_request_not_origin_or_orbit_proxy'})

    async def analyze_neo_comprehensive(self, designation, neo_data=None, include_impact_assessment=True):
        if type(include_impact_assessment) is not bool:
            raise ValueError('include_impact_assessment must be boolean')
        enhanced = await self.enhanced_pipeline.analyze_neo_with_validation(designation, neo_data)
        report, status, persistence = None, 'not_requested', {'status':'not_attempted'}
        if include_impact_assessment:
            try:
                report = await self._calculate_impact_assessment(designation, enhanced, neo_data)
                status = report['status']
            except ValueError:
                status = 'invalid_evidence'
                self.logger.warning('Impact evidence invalid or mismatched; no probability inferred')
            except ProviderUnavailableError:
                status = 'source_unavailable'
                self.logger.warning('Impact source unavailable; no probability inferred')
            except Exception:
                status = 'failed'
                self.logger.exception('Impact evidence acquisition failed; no probability inferred')
            if report is not None:
                try:
                    path = await asyncio.to_thread(EvidenceArchive().put, report)
                    persistence = {'status':'archived_verified', 'path':str(path)}
                except Exception:
                    persistence = {'status':'not_confirmed_saved'}
                    self.logger.error('Impact evidence archive not confirmed; source result remains in memory')
        return ImpactEnhancedAnalysisResult(enhanced, report, impact_status=status, persistence=persistence)

    def _should_assess_impact(self, enhanced_result, neo_data=None):
        """No heuristic suppression: an explicitly requested lookup may proceed."""
        return True

    async def _calculate_impact_assessment(self, designation, enhanced_result=None, neo_data=None):
        report = await asyncio.to_thread(impact_evidence.get_sentry_impact_service().assess, designation)
        return impact_evidence.require_sentry_report(report, designation)

    def _extract_orbital_elements(self, enhanced_result, neo_data=None):
        """Retained inventory helper; never manufactures an orbit or fit."""
        if isinstance(neo_data, NEOData):
            return neo_data.orbital_elements
        original = getattr(enhanced_result, 'original_result', None)
        actual = getattr(original, 'neo_data', None)
        return actual.orbital_elements if isinstance(actual, NEOData) else None

    def _extract_close_approaches(self, enhanced_result, neo_data=None):
        return list(neo_data.close_approaches) if isinstance(neo_data, NEOData) else []

    def _estimate_observation_arc_length(self, enhanced_result, neo_data=None):
        """Actual dated inventory only; no quality-score-to-duration conversion."""
        return impact_evidence.describe_impact_prerequisites(neo_data if isinstance(neo_data, NEOData) else None)['input_inventory']['observation_arc_days']

    def _is_earth_crossing_orbit(self, orbital_elements):
        if not isinstance(orbital_elements, OrbitalElements):
            return None
        a, e = orbital_elements.semi_major_axis, orbital_elements.eccentricity
        if a is None or e is None:
            return None
        return a * (1-e) < 1.02 and a * (1+e) > .98  # Geometric overlap only, not risk.

    def _has_significant_close_approaches(self, close_approaches):
        return any(item.distance_au is not None and item.distance_au < .2 for item in close_approaches)

    def _has_high_orbital_uncertainty(self, enhanced_result):
        return None  # A quality score cannot supply propagated uncertainty.

    def _is_artificial_candidate(self, enhanced_result):
        return None  # Origin screening does not establish propulsion or impact risk.

    def _log_impact_assessment_results(self, designation, impact_assessment):
        report = impact_evidence.require_sentry_report(impact_assessment, designation)
        self.logger.info('Scoped Sentry evidence status=%s method=%s', report['status'], report['source']['method'])

    async def analyze_neo(self, designation, neo_data=None):
        return await self.analyze_neo_comprehensive(designation, neo_data)

    async def analyze_neo_with_validation(self, designation, neo_data=None):
        return await self.analyze_neo_comprehensive(designation, neo_data)

    def get_pipeline_status(self):
        return {**self.enhanced_pipeline.get_validation_status(), 'impact_enhancement':{
            'source_adapter_implemented':True, 'provider_health':'not_probed',
            'local_fitted_impact_probability_available':False, 'source':'JPL Sentry',
            'archive':'append_only_local_reports_not_a_durable_job_queue',
            'assessment_policy':'explicit_request_not_origin_or_orbit_proxy'}}


def create_impact_enhanced_pipeline(original_pipeline: Any):
    enhanced = original_pipeline if isinstance(original_pipeline, EnhancedAnalysisPipeline) else EnhancedAnalysisPipeline(original_pipeline)
    return ImpactEnhancedPipeline(enhanced)
