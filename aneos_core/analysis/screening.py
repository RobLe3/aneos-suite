"""Canonical candidate screening contract.

A screening score is a deterministic ranking heuristic, not a Gaussian sigma,
posterior, natural-origin verdict or artificial-origin confirmation. No calibrated
scientific model is installed. Unavailable inference is represented by null,
never zero significance, a fabricated prior or a successful denial-only gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Any, Optional

from aneos_core.data.models import NEOData

SCHEMA_VERSION = 'screening-assessment-v1'
ENGINE_VERSION = 'catalogue-ranking-v1'


def _canonical(value):
    # Reject NaN/Infinity rather than allowing non-standard JSON or unstable IDs.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _ranking_term(effect, weight=1.0):
    effect, weight = float(effect), float(weight)
    if not math.isfinite(effect) or not math.isfinite(weight) or weight < 0:
        raise ValueError('Non-finite or invalid ranking contribution')
    try:
        term = effect ** 2 * weight
    except OverflowError as exc:
        raise ValueError('Ranking contribution exceeds numerical range') from exc
    if not math.isfinite(term):
        raise ValueError('Ranking contribution exceeds numerical range')
    return term


@dataclass(frozen=True)
class ScreeningPolicy:
    version: str = 'follow-up-ranking-v1'
    candidate_threshold: float = 5.0

    def __post_init__(self):
        if isinstance(self.candidate_threshold, bool) or not math.isfinite(self.candidate_threshold) or self.candidate_threshold <= 0:
            raise ValueError('candidate_threshold must be finite and positive')


@dataclass
class ScreeningAssessment:
    designation: str
    screening_score: Optional[float]
    candidate: bool
    classification: str
    evidence_sources: list[dict]
    evidence_id: str
    policy_id: str
    evidence_snapshot: dict
    limitations: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    engine_version: str = ENGINE_VERSION
    significance_status: str = 'unavailable_uncalibrated_model'
    origin_status: str = 'inconclusive'
    human_origin_status: str = 'not_assessed'
    # These are intentionally nullable: absence of inference is not a negative.
    sigma_confidence: Optional[float] = None
    combined_p_value: Optional[float] = None
    false_discovery_rate: Optional[float] = None
    artificial_probability: Optional[float] = None
    is_artificial: Optional[bool] = None
    confidence: Optional[float] = None

    @property
    def evidence_count(self):
        return sum(e['analyzed'] for e in self.evidence_sources)

    def ranking_trace(self):
        """Owned calculation trace; deliberately not added to historical v1 payloads."""
        if not hasattr(self, '_ranking_trace_data'):
            raise ValueError('No recorded ranking trace is available')
        return json.loads(_canonical(self._ranking_trace_data))

    def to_payload(self):
        payload = asdict(self)
        payload.update(
            evidence_count=self.evidence_count,
            sigma_tier='UNAVAILABLE',
            interpretation=(f'{self.designation}: {self.classification}; '
                            'screening score is an uncalibrated ranking heuristic. '
                            'Statistical significance and origin probability are unavailable.'),
            non_gravitational_evidence=self.evidence_snapshot['additional'].get('non_gravitational_evidence'),
            spacecraft_veto=False,
            veto_reason=None,
            analysis_metadata={
                'schema_version': self.schema_version, 'engine_version': self.engine_version,
                'evidence_id': self.evidence_id, 'policy_id': self.policy_id,
                'significance_status': self.significance_status,
                'origin_status': self.origin_status, 'human_origin_status': self.human_origin_status,
                'five_sigma_claim_permitted': False,
                'score_basis': 'legacy_uncalibrated_dependent_features',
                'evidence_count_semantics': 'analyzed_proxy_channels_not_independent_observations',
                'reference_population_status': 'fixed_global_prototype_not_conditioned',
                'limitations': self.limitations,
            },
        )
        return payload

    def to_detection_result(self):
        from aneos_core.interfaces.detection import DetectionResult
        payload = self.to_payload()
        result = DetectionResult(
            is_artificial=None, confidence=None, sigma_level=None,
            artificial_probability=None, classification=self.classification,
            analysis={'assessment': payload, 'evidence_breakdown': self.evidence_sources},
            risk_factors=[e['type'] for e in self.evidence_sources if e['analyzed']],
            metadata={**payload['analysis_metadata'], 'combined_p_value': None,
                      'false_discovery_rate': None, 'validation_available': False,
                      'evidence_count': self.evidence_count,
                      'non_gravitational_evidence': payload['non_gravitational_evidence'],
                      'physical_data_available': any(e['type'] == 'physical_properties' and e['analyzed']
                                                     for e in self.evidence_sources)},
        )
        result.screening_score = self.screening_score
        result.candidate = self.candidate
        return result


class ScreeningAnalysisService:
    """One synchronous, side-effect-free scientific path for every adapter.

    Acquisition, persistence and execution scheduling are separate boundaries.
    Inputs (including provenance and unavailable rich evidence) are versioned.
    """
    def __init__(self, policy=None, detector=None):
        self.policy = policy or ScreeningPolicy()
        if detector is None:
            from aneos_core.detection.validated_sigma5_artificial_neo_detector import ValidatedSigma5ArtificialNEODetector
            detector = ValidatedSigma5ArtificialNEODetector()
        self.detector = detector

    def analyze_neo(self, neo: NEOData, additional_data=None):
        from aneos_core.data.physical_evidence import verify_neo_physical
        verify_neo_physical(neo)
        oe = neo.orbital_elements
        orbital = {} if oe is None else {'a': oe.semi_major_axis, 'e': oe.eccentricity, 'i': oe.inclination}
        physical = {}
        pp = neo.physical_properties
        if pp is not None:
            physical = {'diameter': pp.diameter_km * 1000 if pp.diameter_km is not None else None,
                        'albedo': pp.albedo, 'absolute_magnitude': pp.absolute_magnitude_h,
                        'spectral_type': pp.spectral_type}
        extra = dict(additional_data or {})
        if neo.close_approaches:
            extra['close_approach_history'] = [
                {'epoch': getattr(ca, 'epoch_jd_tdb', None), 'time_scale': ca.time_scale,
                 'catalogue_date': ca.close_approach_date.isoformat() if ca.close_approach_date else None,
                 'distance_au': ca.distance_au,
                 'velocity_km_s': ca.relative_velocity_km_s, 'kind': 'computed_encounter',
                 'provenance': ca.evidence_metadata, 'body': ca.body}
                for ca in neo.close_approaches]
        if neo.nongrav is not None:
            extra['non_gravitational_evidence'] = neo.nongrav.evidence_report()
        # Provenance travels with the complete evidence snapshot, not with a
        # claim of calibrated measurement accuracy.
        return self.analyze_raw(neo.designation, orbital, physical, extra, neo.evidence_metadata)

    def analyze_raw(self, designation, orbital, physical=None, additional_data=None, provenance=None):
        aliases = {'a': 'semi_major_axis', 'e': 'eccentricity', 'i': 'inclination'}
        orbit = {}
        for key, alias in aliases.items():
            value = orbital.get(key, orbital.get(alias))
            if key in orbital and alias in orbital and orbital[key] != orbital[alias]:
                raise ValueError(f'Conflicting orbital aliases: {key}/{alias}')
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f'{key} must be a finite number')
                orbit[key] = float(value)
        if ('e' in orbit and orbit['e'] < 0) or ('i' in orbit and not 0 <= orbit['i'] <= 180):
            raise ValueError('Orbital elements outside physical bounds')
        if 'a' in orbit and orbit['a'] <= 0 and orbit.get('e', 0) < 1:
            raise ValueError('Bound orbit requires positive semi-major axis')
        phys = {k: v for k, v in (physical or {}).items() if v is not None and not k.startswith('_')}
        for key in ('diameter', 'diameter_km', 'albedo', 'absolute_magnitude'):
            if key in phys and (isinstance(phys[key], bool) or not isinstance(phys[key], (int, float))):
                raise ValueError(f'{key} must be a finite number')
        if 'diameter_km' in phys:
            meters = phys.pop('diameter_km') * 1000
            if 'diameter' in phys and not math.isclose(phys['diameter'], meters):
                raise ValueError('Conflicting diameter units')
            phys['diameter'] = meters
        for key, value in phys.items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        if 'diameter' in phys and phys['diameter'] <= 0:
            raise ValueError('diameter must be positive; missing is null, not zero')
        if 'albedo' in phys and phys['albedo'] < 0:
            raise ValueError('albedo must be nonnegative; geometric albedo is not a probability')
        extra = {k: v for k, v in (additional_data or {}).items() if v is not None}
        snapshot = {'schema_version': 'screening-evidence-v1', 'designation': designation,
                    'orbital': orbit, 'physical': phys, 'additional': extra,
                    'provenance': provenance or {}}
        snapshot = json.loads(_canonical(snapshot))  # own an immutable-by-alias replay copy
        evidence_id = sha256(_canonical(snapshot).encode()).hexdigest()
        policy_id = sha256(_canonical({'policy': asdict(self.policy), 'engine_version': ENGINE_VERSION,
                                      'reference_parameters': self.detector.neo_population_stats}).encode()).hexdigest()
        limitations = ['No calibrated null or artificial-origin model is installed.',
                       'No search-wide significance claim is available.',
                       'Human origin has not been excluded.',
                       'Legacy ranking includes dependent size/photometry features and hand-set inclination bonuses; it is retained for comparison, not calibrated inference.']
        evidence = []
        scores = []
        terms = []
        def add_term(name,effect,weight=1.0,dependency='orbital_catalogue'):
            value=_ranking_term(effect,weight)
            scores.append(value)
            terms.append({'channel':name,'effect':float(effect),'weight':float(weight),
                          'weighted_square':value,'dependency_group':dependency,
                          'basis':'heuristic_proxy'})
        if set(orbit) == {'a', 'e', 'i'} and orbit['a'] > 0 and orbit['e'] < 1 and orbit['a'] * (1 - orbit['e']) < 1.3:
            raw = self.detector._calculate_orbital_anomaly_score(orbit)
            add_term('orbital_dynamics',raw.effect_size)  # Complete zeros remain valid.
            evidence.append(self._summary('orbital_dynamics', float(raw.effect_size), True))
        else:
            limitations.append('Incomplete or out-of-domain orbit: ranking abstained.')
            evidence.append(self._summary('orbital_dynamics', None, False, 'Incomplete or out-of-domain orbit'))
        if phys:
            raw = self.detector._calculate_physical_anomaly_score(phys)
            available = bool(raw.quality_score > 0)
            evidence.append(self._summary('physical_properties', float(raw.effect_size) if available else None,
                                          available, None if available else 'No supported physical measurements'))
            if available:
                add_term('physical_properties',raw.effect_size,raw.quality_score,'physical_size_photometry')
            spectral = self.detector._analyze_spectral_type(phys)
            if spectral is not None:
                add_term('spectral_analysis',spectral.effect_size,spectral.quality_score,'physical_size_photometry')
                evidence.append(self._summary('spectral_analysis', float(spectral.effect_size), True))
        for name, present, reason in [
            ('course_corrections', bool(extra.get('orbital_history')), 'Requires observed residuals and a validated dynamical comparison; ephemeris changes are not manoeuvres'),
            ('trajectory_patterns', bool(extra.get('close_approach_history')), 'Predicted/repeated encounters are not independent artificial-origin evidence'),
            ('propulsion_signatures', bool(extra.get('observation_data') or extra.get('non_gravitational_evidence')), 'Requires a force model, uncertainty and natural-alternative tests'),
        ]:
            evidence.append(self._summary(name, None, False, reason, present))
        complete = evidence[0]['analyzed']
        total = sum(scores)
        if not math.isfinite(total):
            raise ValueError("Ranking sum exceeds numerical range")
        score = float(math.sqrt(total)) if complete else None
        candidate = bool(score is not None and score >= self.policy.candidate_threshold)
        classification = 'FOLLOW_UP_CANDIDATE' if candidate else 'NOT_FLAGGED' if complete else 'INCONCLUSIVE'
        result = ScreeningAssessment(designation, score, candidate, classification, evidence,
                                   evidence_id, policy_id, snapshot, limitations)
        for term in terms:
            term['share_of_squared_rank'] = term['weighted_square']/total if score is not None and total>0 else None
        result._ranking_trace_data=json.loads(_canonical({
            'schema_version':'ranking-trace-v1','terms':terms,'total_squared_rank':total,
            'policy':asdict(self.policy),'reference_parameters':self.detector.neo_population_stats,
            'manual_inclination_enhancement':bool(complete and orbit['i']<5),
            'overall_rank_available':score is not None,
            'interpretation':'Computation trace and assumed prototype reference parameters, not calibrated probabilities, independent evidence or population validation.'}))
        return result

    def replay(self, snapshot, expected_evidence_id=None):
        if snapshot.get('schema_version') != 'screening-evidence-v1':
            raise ValueError('Unsupported evidence schema')
        assessment = self.analyze_raw(snapshot['designation'], snapshot['orbital'],
                                      snapshot.get('physical'), snapshot.get('additional'),
                                      snapshot.get('provenance'))
        if expected_evidence_id is not None and assessment.evidence_id != expected_evidence_id:
            raise ValueError('Evidence fingerprint mismatch')
        return assessment

    @staticmethod
    def _summary(kind, score, analyzed, reason=None, data_available=None):
        return {'type': kind, 'evidence_type': kind, 'anomaly_score': score,
                'effect_size': score, 'p_value': None, 'quality_score': None,
                'confidence_interval': [], 'sample_size': 0, 'analyzed': analyzed,
                'data_available': analyzed if data_available is None else data_available,
                'unavailable_reason': reason, 'evidence_basis': 'heuristic_proxy',
                'dependency_group': 'physical_size_photometry' if kind in {'physical_properties', 'spectral_analysis'} else 'orbital_catalogue',
                'score_semantics': 'uncalibrated_ranking_heuristic'}


class CanonicalAnalysisAdapter:
    """Async original-pipeline interface backed by real acquisition/shared screening."""
    def __init__(self, service=None, fetcher=None):
        self.service = service or ScreeningAnalysisService()
        self.fetcher = fetcher

    async def analyze_neo(self, designation, neo_data=None):
        if neo_data is None:
            import asyncio
            if self.fetcher is None:
                from ..data.fetcher import DataFetcher
                self.fetcher = DataFetcher()
            neo_data = await asyncio.to_thread(self.fetcher.fetch_neo_data, designation)
        if neo_data is None or neo_data.designation != designation:
            raise ValueError('Acquired evidence unavailable or object identity mismatch')
        result = self.service.analyze_neo(neo_data).to_detection_result()
        result.neo_data = neo_data
        return result
