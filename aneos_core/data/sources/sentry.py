"""Strict Sentry 2.0 object response transport, not a local impact solver.

Primary contract: https://ssd-api.jpl.nasa.gov/doc/sentry.html (2026-11-12).
Keep native inputs, method and event scope. No listing is not zero impact risk.
"""
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeout
import copy
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
import threading
import time

from .. import provider_http

URL = 'https://ssd-api.jpl.nasa.gov/sentry.api'
DOCUMENTATION = 'https://ssd-api.jpl.nasa.gov/doc/sentry.html'
CONTRACT = 'sentry-object-evidence-v1'
LIMITATIONS = [
    'External Sentry model output, not an aNEOS observational fit or independent validation.',
    'JPL states impact probabilities may be inaccurate by factors of a few, occasionally a factor of ten or more.',
    'Probability covers tabulated virtual impactors, not all future times or every possible trajectory.',
    'Not listed, removed, unavailable and invalid responses do not establish zero probability or negligible risk.',
    'No lunar impact probability, calibrated aNEOS confidence, Gaussian discovery significance or origin inference is supplied.',
    'Source physical quantities may be assumed estimates; they are not automatically independent measurements.',
]


def normalize_designation(value):
    """Canonical number/provisional selector only; names require separate resolution."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Provide a primary asteroid number or provisional designation')
    compact = ''.join(value.upper().split())
    if re.fullmatch(r'A?[1-9]\d{0,8}', compact):
        return compact.lstrip('A')
    match = re.fullmatch(r'(\d{4})([A-HJ-Y][A-HJ-Z]\d{0,4})', compact)
    if match:
        return f'{match[1]} {match[2]}'
    raise ValueError('Use a primary asteroid number or unpacked provisional designation; aliases/names are not identity-verified')


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonstandard JSON number')


def _numeric(value, *, probability=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('Missing or invalid numeric value')
    try:
        native = Decimal(str(value))
        converted = float(native)
        if not native.is_finite() or not math.isfinite(converted) or (native != 0 and converted == 0):
            raise ValueError('Nonfinite or unrepresentable value')
        if probability and not 0 <= native <= 1:
            raise ValueError('Probability outside [0,1]')
        if nonnegative and native < 0:
            raise ValueError('Negative quantity')
        return converted
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError('Invalid numeric value') from exc


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Missing timestamp')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('Retrieval timestamp must have a timezone')
    return stamp.astimezone(timezone.utc).isoformat()


def _source_date(value, *, event=False, with_time=False):
    if not isinstance(value, str):
        raise ValueError('Missing source date')
    fractional = re.fullmatch(r'(\d{4})-(\d{1,2})-(\d{1,2})(\.\d+)?', value)
    if event or (fractional and not with_time):
        if not fractional:
            raise ValueError('Invalid fractional-day UTC date')
        day = date(int(fractional[1]), int(fractional[2]), int(fractional[3]))
        fraction = Decimal(fractional[4] or '0')
        return (day.toordinal(), fraction)
    # Preserve native clocks; Pacific Local is ambiguous and not guessed as UTC.
    parsed = datetime.fromisoformat(value)
    if with_time and (' ' not in value and 'T' not in value):
        raise ValueError('Missing source time')
    if parsed.tzinfo is not None:
        raise ValueError('Unexpected source date timezone convention')
    if with_time:
        return parsed
    seconds = Decimal(parsed.hour * 3600 + parsed.minute * 60 + parsed.second)
    seconds += Decimal(parsed.microsecond) / Decimal(1000000)
    return (parsed.date().toordinal(), seconds / Decimal(86400))


def _optional_number(mapping, field, **options):
    return _numeric(mapping[field], **options) if mapping.get(field) is not None else None


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def parse_sentry_response(raw, *, designation, retrieved_at):
    """Parse exactly one unfiltered Mode O result; retain exact UTF-8 bytes/hash."""
    canonical = normalize_designation(designation)
    retrieved = _timestamp(retrieved_at)
    if not isinstance(raw, bytes):
        raise ValueError('Sentry evidence requires raw response bytes')
    try:
        text = raw.decode('utf-8')
        data = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid Sentry JSON') from exc
    if not isinstance(data, dict) or data.get('signature') != {
            'source': 'NASA/JPL Sentry Data API', 'version': '2.0'}:
        raise ValueError('Missing or unsupported Sentry signature/version')
    result = {
        'schema_version': CONTRACT, 'designation': canonical,
        'status': None, 'assessment_basis': 'authoritative_external_model_output',
        'identity': {'query_designation': designation, 'canonical_designation': None,
                     'verification': 'query_bound_no_object_row'},
        'collision_probability': None, 'probability_uncertainty': None,
        'calculation_confidence': None, 'moon_collision_probability': None,
        'moon_earth_ratio': None, 'risk_level': 'not_assessed',
        'impact_energy_mt': None, 'crater_diameter_km': None, 'damage_radius_km': None,
        'time_to_impact_years': None, 'peak_risk_period': None,
        'collision_probability_per_year': None, 'comparative_risk': None,
        'keyhole_passages': None, 'primary_risk_factors': None,
        'impact_probability_by_decade': None, 'source_torino_max': None,
        'source_palermo_cumulative': None,
        'source': {'name': 'JPL Sentry', 'api_version': '2.0', 'endpoint': URL,
            'query': {'des': canonical}, 'retrieved_at': retrieved, 'method': None,
            'documentation': DOCUMENTATION, 'computation_time': None,
            'palermo_computation_time': None, 'removed_at_utc': None},
        'time_scope': None, 'virtual_impactors': [], 'physical_estimate_basis': None,
        'source_snapshot': {'raw_utf8': text, 'sha256': hashlib.sha256(raw).hexdigest()},
        'limitations': list(LIMITATIONS),
    }
    if 'error' in data:
        if 'summary' in data or 'data' in data:
            raise ValueError('Inconsistent Sentry error and object data')
        result['assessment_basis'] = 'authoritative_external_catalogue_lookup'
        error = data['error']
        if error == 'specified object removed' and data.get('removed'):
            _source_date(data['removed'], with_time=True)
            result['status'] = 'removed'
            result['source']['removed_at_utc'] = data['removed']
        elif error == 'specified object not found' and 'removed' not in data:
            result['status'] = 'not_listed'
        else:
            raise ValueError('Unrecognized or incomplete Sentry error')
    else:
        if 'removed' in data:
            raise ValueError('Inconsistent removal and object data')
        summary, rows = data.get('summary'), data.get('data')
        if not isinstance(summary, dict) or not isinstance(rows, list) or not rows:
            raise ValueError('Missing object summary or virtual impactors')
        if normalize_designation(summary.get('des')) != canonical:
            raise ValueError('Sentry identity does not match query')
        method = summary.get('method')
        diagnostics = {'IOBS': ('sigma_vi',), 'MC': ('sigma_mc',),
                       'LOV': ('dist', 'width', 'sigma_imp', 'sigma_lov', 'stretch')}
        if method not in diagnostics:
            raise ValueError('Unsupported or missing Sentry analysis method')
        n_imp = _numeric(summary.get('n_imp'), nonnegative=True)
        if not n_imp.is_integer() or n_imp != len(rows):
            raise ValueError('Incomplete or inconsistent virtual-impactor count')
        if _source_date(summary.get('first_obs')) > _source_date(summary.get('last_obs')):
            raise ValueError('Reversed observation dates')
        for field in ('cdate', 'pdate'):
            _source_date(summary.get(field), with_time=True)
        events = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Invalid virtual-impactor record')
            if 'method' in row and row['method'] != method:
                raise ValueError('Mixed virtual-impactor methods')
            if 'des' in row and normalize_designation(row['des']) != canonical:
                raise ValueError('Mixed object identities')
            _source_date(row.get('date'), event=True)
            probability = _numeric(row.get('ip'), probability=True)
            # Native method diagnostics are fit-coordinate metrics, not discovery sigma.
            native_diagnostics = {key: _numeric(row.get(key)) for key in diagnostics[method]}
            events.append({'date_utc': row['date'], 'collision_probability': probability,
                'energy_mt': _optional_number(row, 'energy', nonnegative=True),
                'source_torino': _optional_number(row, 'ts', nonnegative=True),
                'source_palermo': _optional_number(row, 'ps'),
                'method_diagnostics': native_diagnostics,
                'diagnostic_semantics': 'source_orbit_fit_coordinates_not_discovery_significance'})
        result.update(status='available', risk_level='not_classified_by_aneos',
            collision_probability=_numeric(summary.get('ip'), probability=True), virtual_impactors=events,
            impact_energy_mt=_optional_number(summary, 'energy', nonnegative=True),
            source_torino_max=_optional_number(summary, 'ts_max', nonnegative=True),
            source_palermo_cumulative=_optional_number(summary, 'ps_cum'),
            physical_estimate_basis={
                'diameter': 'source_estimate_not_assumed_independent_measurement',
                'diameter_km': _optional_number(summary, 'diameter', nonnegative=True),
                'mass_kg': _optional_number(summary, 'mass', nonnegative=True),
                'mass': 'source_estimate_not_assumed_independent_measurement',
                'energy': 'source_kinetic_energy_estimate_not_a_validated_damage_prediction'})
        result['identity'].update(canonical_designation=canonical, verification='matching_primary_designation')
        result['source'].update(method=method,
            computation_time={'native': summary['cdate'], 'time_scale': 'Pacific Local'},
            palermo_computation_time={'native': summary['pdate'], 'time_scale': 'UTC'},
            first_observation_utc=summary['first_obs'], last_observation_utc=summary['last_obs'])
        dates = sorted((row['date'] for row in rows), key=lambda value: _source_date(value, event=True))
        result['time_scope'] = {'semantics': 'tabulated_virtual_impactors_only',
            'first_listed_event_utc': dates[0], 'last_listed_event_utc': dates[-1], 'event_count': len(rows)}
    result['report_sha256'] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def verify_sentry_report(report):
    """Reparse the raw snapshot and compare every reported scientific field."""
    try:
        replay = parse_sentry_response(report['source_snapshot']['raw_utf8'].encode('utf-8'),
            designation=report['identity']['query_designation'], retrieved_at=report['source']['retrieved_at'])
        return replay == report
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


class SentryResponseError(ValueError):
    """The source responded, but its evidence cannot safely be attributed/parsed."""


class SentryImpactService:
    """Thread-safe per-process TTL reuse and single-flight acquisition per object.

    The provider transport serializes participating processes; this cache does not
    deduplicate queries across hosts/processes or provide durable case history.
    """
    def __init__(self, ttl_seconds=3600., max_entries=128):
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError('Invalid Sentry cache limits')
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache = OrderedDict()
        self._pending = {}
        self._lock = threading.Lock()

    def assess(self, designation, *, force_refresh=False):
        canonical = normalize_designation(designation)
        with self._lock:
            pending = self._pending.get(canonical)
            if pending is not None:
                owner = False
            else:
                cached = self._cache.get(canonical)
                if cached and time.monotonic() < cached[0] and not force_refresh:
                    self._cache.move_to_end(canonical)
                    return copy.deepcopy(cached[1])
                pending = Future()
                self._pending[canonical] = pending
                owner = True
        if not owner:
            try:
                return copy.deepcopy(pending.result(timeout=60))
            except FutureTimeout as exc:
                raise provider_http.ProviderUnavailableError('Sentry in-flight wait budget exhausted') from exc
        try:
            response = provider_http.get(URL, params={'des': canonical}, timeout=20, _max_wait=30)
            try:
                report = parse_sentry_response(response.content, designation=canonical,
                    retrieved_at=datetime.now(timezone.utc).isoformat())
            except ValueError as exc:
                raise SentryResponseError('Sentry response schema, identity or content is invalid') from exc
            with self._lock:
                self._cache[canonical] = (time.monotonic() + self.ttl, copy.deepcopy(report))
                self._cache.move_to_end(canonical)
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
            pending.set_result(copy.deepcopy(report))
            return report
        except BaseException as exc:
            # Every waiter observes this same attempt, including its failure.
            # A failed refresh never substitutes an older cached success.
            pending.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(canonical, None)
