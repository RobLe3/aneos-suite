"""JPL CAD 1.5 labelled encounter evidence, not an orbital-elements source.

Primary contract: https://ssd-api.jpl.nasa.gov/doc/cad.html (checked2026-11-12).
All times remain native TDB; missing measurements are not synthetic defaults.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from copy import deepcopy

CAD_VERSION = '1.5'


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _number(value, name, positive=False, nonnegative=False):
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError(f'CAD {name} must be numeric, not boolean')
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid CAD {name}') from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f'Invalid CAD {name}')
    return result


def parse_cad_row(row, fields, context=None):
    if (not isinstance(fields, list) or not fields or any(not isinstance(f, str) for f in fields)
            or len(set(fields)) != len(fields) or 'des' not in fields):
        raise ValueError('CAD fields are missing, duplicated or invalid')
    if not isinstance(row, list) or len(row) != len(fields):
        raise ValueError('CAD row dimension does not match field labels')
    values = dict(zip(fields, row))
    des = values['des']
    if not isinstance(des, str) or not des.strip():
        raise ValueError('CAD designation unavailable')
    des = des.strip()
    epoch = _number(values.get('jd'), 'jd', positive=True)
    distance = _number(values.get('dist'), 'dist', nonnegative=True)
    velocity = _number(values.get('v_rel'), 'v_rel', nonnegative=True)
    diameter = _number(values.get('diameter'), 'diameter', positive=True)
    sigma = _number(values.get('diameter_sigma'), 'diameter_sigma', nonnegative=True)
    h = _number(values.get('h'), 'h')
    cd = values.get('cd')
    if cd is not None and not isinstance(cd, str):
        raise ValueError('Invalid CAD native calendar string')
    body = values.get('body') or (context or {}).get('query', {}).get('body', 'Earth')
    native = {'epoch_jd_tdb': epoch, 'calendar_tdb': cd or None, 'time_scale': 'TDB',
        'distance': distance, 'distance_min_3sigma_au': _number(values.get('dist_min'), 'dist_min', nonnegative=True),
        'distance_max_3sigma_au': _number(values.get('dist_max'), 'dist_max', nonnegative=True),
        'velocity': velocity, 'v_inf_km_s': _number(values.get('v_inf'), 'v_inf', nonnegative=True),
        'time_uncertainty_3sigma_formatted': values.get('t_sigma_f'), 'body': body,
        'orbit_id': values.get('orbit_id')}
    evidence = {**deepcopy(context or {}), 'raw_fields': list(fields), 'raw_row': deepcopy(row),
        'evidence_basis': 'catalogue_prediction', 'units': {'distance': 'au', 'velocity': 'km/s',
        'diameter': 'km', 'diameter_sigma': 'km', 'h': 'mag', 'epoch': 'JD TDB'}}
    from decimal import Decimal, Context
    raw_epoch = Decimal(str(values['jd'])) if epoch is not None else None
    native_epoch_key = str(raw_epoch.normalize(Context(prec=len(raw_epoch.as_tuple().digits)))) if raw_epoch is not None else None
    identity = {'designation': des, 'body': body, 'epoch_jd_tdb': native_epoch_key, 'calendar_tdb': cd if epoch is None else None}
    if epoch is None and not cd:
        identity['undated_row_hash'] = _hash(values)
    return {'designation': des, 'orbital_elements': None, 'discovery_date': None,
        'miss_distance_au': distance, 'relative_velocity_km_s': velocity,
        'diameter_km': diameter, 'diameter_uncertainty': sigma,
        'absolute_magnitude_h': h, 'h_magnitude': h, 'close_approach': native,
        'close_approach_date': cd or None, 'data_source': 'NASA_CAD',
        'encounter_id': _hash(identity), 'cad_evidence': evidence}


def parse_cad_payload(payload, query=None, retrieved_at=None, require_complete=True):
    if not isinstance(payload, dict) or payload.get('error') or payload.get('message'):
        raise ValueError('CAD response unavailable or contains a provider error')
    payload = json.loads(json.dumps(payload, allow_nan=False))  # own source evidence, never caller aliases
    signature = payload.get('signature')
    if not isinstance(signature, dict) or signature.get('version') != CAD_VERSION:
        raise ValueError('Unsupported or missing CAD API signature/version')
    count = payload.get('count')
    if type(count) is not int or count < 0:
        raise ValueError('Missing/invalid CAD record count')
    rows = payload.get('data', [] if count == 0 else None)
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError('CAD record count/data mismatch')
    total = payload.get('total', count)
    if type(total) is not int or total < count:
        raise ValueError('Invalid CAD total')
    if require_complete and total != count:
        raise ValueError(f'Incomplete CAD response: {count} of {total} records; narrow query or paginate')
    metadata = {'api_version': CAD_VERSION, 'signature': signature,
        'query': dict(query or {}), 'retrieved_at': retrieved_at,
        'received_at': datetime.now(timezone.utc).isoformat(),
        'retrieval_timestamp_status': 'available' if retrieved_at else 'not_available_from_adapter',
        'canonical_response_sha256': _hash(payload), 'count': count, 'total': total,
        'completeness_status': 'complete' if count == total else 'partial',
        'source_documentation': 'https://ssd-api.jpl.nasa.gov/doc/cad.html'}
    fields = payload.get('fields')
    if rows and (not isinstance(fields, list) or not {'des', 'orbit_id', 'jd', 'cd', 'dist',
            'dist_min', 'dist_max', 'v_rel', 'v_inf', 't_sigma_f', 'h'}.issubset(fields)):
        raise ValueError('Required CAD schema fields unavailable')
    if rows and 'total' not in payload and query and query.get('limit') == count:
        raise ValueError('CAD completeness unverified at request limit')
    records = [parse_cad_row(row, fields, metadata) for row in rows]
    return {'records': records, 'metadata': metadata}


def close_approach_from_record(record):
    from ..models import CloseApproach
    native = record['close_approach']
    # A native, naive TDB calendar is retained for legacy calendar consumers,
    # never attached to UTC or treated as an observation/discovery timestamp.
    date = None
    if native.get('calendar_tdb'):
        try:
            date = datetime.strptime(native['calendar_tdb'].strip(), '%Y-%b-%d %H:%M')
        except ValueError:
            pass  # Native literal/JD are preserved; no guessed date conversion.
    return CloseApproach(record['designation'], orbit_id=native.get('orbit_id'),
        close_approach_date=date, distance_au=native['distance'],
        relative_velocity_km_s=native['velocity'], infinity_velocity_km_s=native.get('v_inf_km_s'),
        epoch_jd_tdb=native['epoch_jd_tdb'], time_scale='TDB', body=native.get('body'),
        native_calendar=native.get('calendar_tdb'), evidence_metadata=record['cad_evidence'])


class CloseApproachQuery(list):
    """List-compatible encounters carrying the declared query coverage."""
    def __init__(self, approaches, query_metadata):
        super().__init__(approaches)
        self.query_metadata = query_metadata
