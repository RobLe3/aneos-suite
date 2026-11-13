"""Bounded CAD collection, distinct from per-object enrichment or origin inference.

A query receipt owns the native response. Existing object assertions own grouped
source snapshots. Only declared selection coverage is reported; no hidden paging.
"""
import base64
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import re
from uuid import uuid4

import requests
from . import provider_http
from .enrichment import clone, fingerprint, identifier
from .models import NEOData
from .sources.cad import parse_cad_payload, close_approach_from_record

URL = 'https://ssd-api.jpl.nasa.gov/cad.api'
SCHEMA = 'cad-candidate-acquisition-v1'
MAX_BODY = 1024 * 1024  # JSON parse/receipt-prefix budget, not an entity-archive limit.
MAX_ROWS = 1000


def query_parameters(start, end, limit):
    for value in (start, end):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
            raise ValueError('Use explicit native calendar dates YYYY-MM-DD')
        date.fromisoformat(value)
    if start > end or type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        raise ValueError('Ordered dates and an explicit 1..1000 row limit are required')
    return {'date-min': start, 'date-max': end, 'limit': limit, 'sort': 'date',
            'body': 'Earth', 'neo': 'true', 'dist-max': '0.05',
            'diameter': 'true', 'fullname': 'true'}


def _stamp(value):
    if not isinstance(value, str):
        raise ValueError('An explicit UTC receipt timestamp is required')
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError('Receipt timestamps require a zero UTC offset')
    return stamp


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate native JSON key')
        result[key] = value
    return result


class _DeclaredSynthetic(ValueError):
    pass


def _declared_synthetic(value):
    """Known explicit markers only; names/absence of a marker are not authenticity tests."""
    markers = ('synthetic', 'is_synthetic', 'mock', 'demo')
    if isinstance(value, dict):
        if any(value.get(k) is True for k in markers):
            return True
        fields, rows = value.get('fields'), value.get('data')
        if isinstance(fields, list) and isinstance(rows, list):
            for i, name in enumerate(fields):
                if name in markers and any(isinstance(row, list) and len(row)>i and row[i] is True for row in rows):
                    return True
        return any(_declared_synthetic(v) for v in value.values())
    return isinstance(value, list) and any(_declared_synthetic(v) for v in value)


def capture_response(response):
    """Describe buffered entity bytes; large complete bodies are archived separately in SQL."""
    if response is None:
        return {'http_status': None, 'body_base64': None, 'body_sha256': None,
                'received_bytes': None, 'retained_bytes': 0, 'body_in_receipt_complete': False}
    status = response.status_code
    body = response.content
    if type(status) is not int or not 100 <= status <= 599 or not isinstance(body, bytes):
        raise ValueError('Invalid transport response')
    prefix = body[:MAX_BODY]
    return {'http_status': status, 'body_base64': base64.b64encode(prefix).decode('ascii'),
            'body_sha256': sha256(body).hexdigest(), 'received_bytes': len(body),
            'retained_bytes': len(prefix), 'body_in_receipt_complete': len(body) <= MAX_BODY}


def _body(response):
    if response['body_base64'] is None:
        if response != capture_response(None):
            raise ValueError('Inconsistent absent response')
        return None
    raw = base64.b64decode(response['body_base64'], validate=True)
    size = response['received_bytes']
    status = response['http_status']
    if (type(status) is not int or not 100 <= status <= 599 or type(size) is not int or size < 0
            or len(raw) != min(size, MAX_BODY) or type(response['retained_bytes']) is not int
            or response['retained_bytes'] != len(raw)
            or type(response['body_in_receipt_complete']) is not bool
            or response['body_in_receipt_complete'] != (size <= MAX_BODY)
            or not re.fullmatch(r'[a-f0-9]{64}', response['body_sha256'])):
        raise ValueError('Inconsistent response retention')
    if response['body_in_receipt_complete'] and sha256(raw).hexdigest() != response['body_sha256']:
        raise ValueError('Native response hash mismatch')
    return raw


def make_receipt(query_id, parameters, started_at, received_at, response, transport_error=None, *, software=None):
    if not isinstance(query_id, str) or not re.fullmatch(r'cad-[a-f0-9]{32}', query_id):
        raise ValueError('Candidate query id requires a canonical CAD UUID')
    _stamp(started_at)
    stamp = _stamp(received_at)
    parameters, response = clone(parameters), clone(response)
    if parameters != query_parameters(parameters['date-min'], parameters['date-max'], parameters['limit']):
        raise ValueError('Unsupported candidate selection')
    if transport_error not in (None, 'TRANSPORT_UNAVAILABLE'):
        raise ValueError('Unknown transport outcome')
    raw = _body(response)
    outcome, diagnostic, parsed = 'unavailable', transport_error, None
    if raw is None:
        diagnostic = 'TRANSPORT_UNAVAILABLE'
    elif transport_error is not None or response['http_status'] != 200:
        diagnostic = transport_error or 'HTTP_UNAVAILABLE'
    elif not response['body_in_receipt_complete']:
        outcome, diagnostic = 'not_processed', 'BODY_PARSE_LIMIT'
    else:
        try:
            payload = json.loads(raw, object_pairs_hook=_unique_pairs)
            if _declared_synthetic(payload):
                raise _DeclaredSynthetic('Declared synthetic response')
            parsed = parse_cad_payload(payload, query=parameters, retrieved_at=received_at,
                                       received_at=received_at, require_complete=False)
            if parsed['metadata']['count'] > parameters['limit']:
                raise ValueError('Source exceeded requested row limit')
            for row in parsed['records']:
                identifier(row['designation'], 'native designation')
                for key in ('orbit_id', 'time_uncertainty_3sigma_formatted'):
                    value = row['close_approach'][key]
                    if value is not None and not isinstance(value, str):
                        raise ValueError('Invalid native reference/uncertainty label')
                if row['miss_distance_au'] is not None and row['miss_distance_au'] > 0.05:
                    raise ValueError('Distance/selection mismatch')
                if row['close_approach']['body'] != 'Earth':
                    raise ValueError('Body/selection mismatch')
            outcome = ('partial' if parsed['metadata']['completeness_status'] == 'partial' else
                       'missing' if not parsed['records'] else 'available')
            diagnostic = None
        except _DeclaredSynthetic:
            outcome, diagnostic, parsed = 'invalid', 'DECLARED_SYNTHETIC_SOURCE', None
        except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError):
            outcome, diagnostic, parsed = 'invalid', 'CAD_PAYLOAD_INVALID', None
    records = parsed['records'] if parsed else []
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record['designation']].append((index, record))
    packets = []
    for designation, entries in grouped.items():
        neo = NEOData(designation, created_at=stamp, updated_at=stamp, fetched_at=stamp,
                      sources_used=['CAD'], close_approaches=[close_approach_from_record(r) for _, r in entries],
                      evidence_metadata={
                          'candidate_query_id': query_id, 'candidate_selection': parameters,
                          'cad_candidate_row_indices': [i for i, _ in entries],
                          'cad_candidate_records': [r for _, r in entries],
                          'evidence_basis': 'catalogue_prediction_not_observation',
                          'physical_projection_status': 'native_unit_labelled_catalogue_fields_retained_in_rows_not_promoted',
                      })
        packets.append({'designation': designation, 'data_source': 'CAD',
                        'snapshot_schema': 'neo-data-v1', 'neo_data': neo.to_dict(),
                        'candidate_query_id': query_id})
    row_hashes = [fingerprint(r['cad_evidence']['raw_row']) for r in records]
    event_variants = defaultdict(set)
    for r, h in zip(records, row_hashes):
        event_variants[r['encounter_id']].add(h)
    receipt = {
        'schema_version': SCHEMA, 'query_id': query_id, 'source': 'CAD',
        'request': {'method': 'GET', 'url': URL, 'parameters': parameters},
        'budget': {'max_transport_invocations': 1, 'max_native_rows': parameters['limit'],
                   'receipt_prefix_bytes': MAX_BODY, 'max_parsed_body_bytes': MAX_BODY, 'enrichment_requests': 0},
        'started_at': started_at, 'received_at': received_at,
        'receipt_clock_order': 'ordered' if stamp >= _stamp(started_at) else 'nonmonotonic_local_clock',
        'response': response, 'transport_error': transport_error,
        'outcome': outcome, 'diagnostic': diagnostic,
        'native_count': parsed['metadata']['count'] if parsed else None,
        'native_total': parsed['metadata']['total'] if parsed else None,
        'selection_coverage': parsed['metadata']['completeness_status'] if parsed else 'not_established',
        'object_count': len(packets), 'exact_native_repeat_rows': len(row_hashes)-len(set(row_hashes)),
        'events_with_alternative_rows': sum(len(v)>1 for v in event_variants.values()),
        'packet_ids': [fingerprint(p) for p in packets],
        'scientific_completeness': None, 'independent_observation_count': None,
        'limits': ['CAD selection is predicted NEO Earth encounters within 0.05 au, not all NEOs or discovery dates.',
                   'One shared-transport invocation; no automatic paging, retries or per-object enrichment.',
                   'Native JD/calendar remain TDB; receipt timestamps are local UTC, not observations.',
                   'Catalogue H/diameter/sigma remain unit-labelled in native rows; physical projection is pending.',
                   'Python float normalization retains native strings and response bytes; no origin inference.',
                   'Over-budget JSON is not normalized; its full buffered entity must be archived in SQL and checked on readback.'],
    }
    if software is not None:
        if not isinstance(software, dict):
            raise ValueError('Software context must be finite JSON metadata')
        receipt['software'] = clone(software)
    receipt['receipt_id'] = fingerprint(receipt)
    return clone(receipt), clone(packets)


def replay_receipt(receipt):
    if not isinstance(receipt, dict) or receipt.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported candidate receipt')
    actual, packets = make_receipt(receipt['query_id'], receipt['request']['parameters'],
                                   receipt['started_at'], receipt['received_at'], receipt['response'],
                                   receipt['transport_error'], software=receipt.get('software'))
    if actual != receipt:
        raise ValueError('Candidate receipt does not reproduce')
    return actual, packets


class CADCandidateService:
    """Explicit acquisition action; constructing/importing performs no SQL or HTTP."""
    def __init__(self, repository=None):
        self.repository = repository

    def poll(self, start, end, *, limit=100):
        parameters = query_parameters(start, end, limit)
        if self.repository is None:
            from aneos_api.candidate_repository import CandidateQueryRepository
            self.repository = CandidateQueryRepository()
        self.repository.prepare()
        from ..utils.reproducibility import software_metadata
        software = software_metadata(source_files=[
            'aneos_version.py', 'aneos_core/data/candidate_discovery.py',
            'aneos_core/data/provider_http.py', 'aneos_core/data/sources/cad.py',
            'aneos_api/candidate_repository.py', 'aneos_api/enrichment_store.py',
            'aneos_core/data/models.py', 'aneos_core/data/enrichment.py',
            'aneos_core/data/reconciliation.py', 'aneos_core/data/physical_evidence.py',
            'aneos_core/data/epoch_evidence.py', 'aneos_core/data/orbit_evidence.py',
            'aneos_core/utils/reproducibility.py'])
        started = datetime.now(timezone.utc).isoformat()
        response, error = None, None
        try:
            response = provider_http.get(URL, params=parameters, timeout=30)
        except requests.RequestException as exc:
            response, error = getattr(exc, 'response', None), 'TRANSPORT_UNAVAILABLE'
        captured = capture_response(response)
        received = datetime.now(timezone.utc).isoformat()
        receipt, _ = make_receipt('cad-'+uuid4().hex, parameters, started, received, captured, error, software=software)
        archive = response.content if response is not None and not captured['body_in_receipt_complete'] else None
        return self.repository.commit(receipt, native_body=archive)
