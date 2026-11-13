"""Native-shaped synthetic CAD outcomes with real disposable file-SQLite commits."""
import base64
import copy
import json
from pathlib import Path
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

FIELDS = ['des','orbit_id','jd','cd','dist','dist_min','dist_max','v_rel','v_inf','t_sigma_f','h']


def native(rows=None, **overrides):
    if rows is None:
        rows = [['SYNTHETIC','1','2461357.5','2026-Nov-13 00:00','.01','.009','.011','5','4','00:01','25']]
    return dict(signature={'version':'1.5','source':'NASA/JPL SBDB Close Approach Data API'},
                fields=list(FIELDS), count=len(rows), total=len(rows), data=rows, **overrides)


class Response:
    def __init__(self, payload=None, *, body=None, status=200):
        self.content = body if body is not None else json.dumps(native() if payload is None else payload).encode()
        self.status_code = status
        self.headers = {'Content-Type':'application/json'}


@pytest.fixture
def service(tmp_path, monkeypatch):
    from aneos_api.candidate_repository import CandidateQueryRepository
    from aneos_core.data.candidate_discovery import CADCandidateService
    engine = create_engine('sqlite:///' + str(tmp_path/'candidates.sqlite'))
    repo = CandidateQueryRepository(engine)
    response = {'value': Response()}
    calls = []
    def get(url, **kwargs):
        calls.append((url, copy.deepcopy(kwargs)))
        value = response['value']
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr('aneos_core.data.provider_http.get', get)
    yield CADCandidateService(repo), repo, response, calls, engine
    engine.dispose()


def poll(service):
    return service[0].poll('2026-11-12','2026-11-13',limit=10)


def test_native_success_commits_receipt_and_existing_object_ledger(service):
    from aneos_api.database import EnrichedNEO
    from aneos_api.enrichment_store import records
    result = poll(service)
    assert result['commit_status'] == 'committed'
    assert result['receipt']['outcome'] == 'available'
    assert len(result['records']) == 1 and len(service[3]) == 1
    neo = result['records'][0]['neo_data']
    assert neo['orbital_elements'] is None
    assert neo['close_approaches'][0]['dist'] == .01
    assert neo['close_approaches'][0]['time_scale'] == 'TDB'
    with sessionmaker(bind=service[4])() as db:
        assert db.scalar(select(EnrichedNEO)).designation == 'SYNTHETIC'
        assert len(records(db,'SYNTHETIC')) == 1
    assert service[1].load(result['receipt']['query_id']) == result
    assert base64.b64decode(result['receipt']['response']['body_base64']) == service[2]['value'].content


def test_valid_empty_is_stored_without_fake_object(service):
    from aneos_api.database import EnrichedNEO
    service[2]['value'] = Response(native(rows=[]))
    result = poll(service)
    assert result['receipt']['outcome'] == 'missing' and result['records'] == []
    assert service[1].load(result['receipt']['query_id']) == result
    with sessionmaker(bind=service[4])() as db:
        assert db.scalar(select(EnrichedNEO)) is None


def test_http_error_and_network_failure_are_not_empty_success(service):
    import requests
    service[2]['value'] = Response(body=b'upstream unavailable',status=503)
    result = poll(service)
    assert result['receipt']['outcome'] == 'unavailable'
    assert result['receipt']['native_count'] is None and result['records'] == []
    service[2]['value'] = requests.Timeout('synthetic transport failure')
    other = poll(service)
    assert other['receipt']['outcome'] == 'unavailable'
    assert other['receipt']['response']['body_base64'] is None
    assert len(service[3]) == 2


def test_multiple_encounters_repeats_alternatives_and_native_details_remain(service):
    data = native()['data'][0]
    second = copy.deepcopy(data); second[2] = '2461357.51'; second[3] = '2026-Nov-13 00:14'
    alternative = copy.deepcopy(data); alternative[1] = '2'; alternative[7] = '6'
    other = copy.deepcopy(data); other[0] = 'OTHER'
    payload = native(rows=[data, second, copy.deepcopy(data), alternative, other])
    service[2]['value'] = Response(payload)
    result = poll(service)
    receipt = result['receipt']
    assert receipt['native_count'] == 5 and receipt['object_count'] == 2
    assert receipt['exact_native_repeat_rows'] == 1
    assert receipt['events_with_alternative_rows'] == 1
    packet = result['records'][0]['neo_data']
    assert len(packet['close_approaches']) == 4
    assert packet['evidence_metadata']['cad_candidate_row_indices'] == [0,1,2,3]
    assert [r['cad_evidence']['raw_row'] for r in packet['evidence_metadata']['cad_candidate_records']] == payload['data'][:4]
    assert packet['physical_properties'] is None
    assert packet['evidence_metadata']['cad_candidate_records'][0]['absolute_magnitude_h'] == 25
    # Returned snapshots do not alias one another or the stored receipt.
    result['records'][0]['neo_data']['evidence_metadata']['candidate_selection']['limit'] = 999
    assert receipt['request']['parameters']['limit'] == 10
    assert service[1].load(receipt['query_id'])['receipt'] == receipt


def test_all_nullable_values_stay_missing_not_zero(service):
    row = ['SYNTHETIC',None,None,None,None,None,None,None,None,None,None]
    service[2]['value'] = Response(native(rows=[row]))
    result = poll(service)
    encounter = result['records'][0]['neo_data']['close_approaches'][0]
    assert encounter['dist'] is None and encounter['v_rel'] is None and encounter['epoch_jd_tdb'] is None
    assert result['records'][0]['neo_data']['first_observation'] is None
    assert result['records'][0]['neo_data'].get('discovery_date') is None


def test_reordered_fields_are_bound_by_labels(service):
    payload = native()
    payload['fields'] = list(reversed(payload['fields']))
    payload['data'] = [list(reversed(row)) for row in payload['data']]
    service[2]['value'] = Response(payload)
    result = poll(service)
    assert result['records'][0]['neo_data']['close_approaches'][0]['dist'] == .01


@pytest.mark.parametrize('mutation', ['version','count','duplicate_fields','row_dimension','negative_distance','outside_distance','control_identity'])
def test_invalid_source_response_is_retained_without_object_projection(service, mutation):
    payload = native()
    if mutation == 'version': payload['signature']['version'] = 'unexpected'
    elif mutation == 'count': payload['count'] = 2
    elif mutation == 'duplicate_fields': payload['fields'][1] = 'des'
    elif mutation == 'row_dimension': payload['data'][0].pop()
    elif mutation == 'negative_distance': payload['data'][0][4] = '-1'
    elif mutation == 'outside_distance': payload['data'][0][4] = '1'
    else: payload['data'][0][0] = 'SYNTHETIC\nOTHER'
    service[2]['value'] = Response(payload)
    result = poll(service)
    assert result['receipt']['outcome'] == 'invalid' and result['records'] == []
    assert result['receipt']['native_count'] is None
    assert base64.b64decode(result['receipt']['response']['body_base64']) == service[2]['value'].content


@pytest.mark.parametrize('body', [b'not JSON', b'{"count":0,"count":1}', b'\xff\xfeinvalid', b'[]', b'{"count":NaN}'])
def test_invalid_json_and_duplicate_keys_preserve_original_bytes(service, body):
    service[2]['value'] = Response(body=body)
    result = poll(service)
    assert result['receipt']['outcome'] == 'invalid'
    assert base64.b64decode(result['receipt']['response']['body_base64']) == body


def test_partial_is_not_complete_and_makes_no_hidden_followup_requests(service):
    payload = native(); payload['total'] = 100
    service[2]['value'] = Response(payload)
    result = poll(service)
    assert result['receipt']['outcome'] == 'partial'
    assert result['receipt']['selection_coverage'] == 'partial'
    assert result['receipt']['native_total'] == 100 and len(service[3]) == 1
    assert result['receipt']['scientific_completeness'] is None
    assert result['receipt']['budget']['enrichment_requests'] == 0


def test_oversize_retention_limit_is_explicit_and_does_not_normalize_prefix(service):
    from aneos_core.data.candidate_discovery import MAX_BODY
    body = b'x'*(MAX_BODY+1)
    service[2]['value'] = Response(body=body)
    result = poll(service)
    r = result['receipt']
    assert r['outcome'] == 'not_processed' and r['diagnostic'] == 'BODY_PARSE_LIMIT'
    assert not r['response']['body_in_receipt_complete']
    assert len(base64.b64decode(r['response']['body_base64'])) == MAX_BODY
    assert r['response']['received_bytes'] == MAX_BODY+1 and result['records'] == []
    assert service[1].load_native_body(r['query_id']) == body
    assert service[1].commit(r) == result


@pytest.mark.parametrize('start,end,limit', [('2026-11-14','2026-11-13',1),('20261112','2026-11-13',1),
    ('2026-02-30','2026-11-13',1),('2026-11-12','2026-11-13',True),('2026-11-12','2026-11-13',0),
    ('2026-11-12','2026-11-13',1001)])
def test_invalid_requests_fail_before_storage_or_http(service, start, end, limit):
    with pytest.raises(ValueError): service[0].poll(start,end,limit=limit)
    assert service[3] == []
    from sqlalchemy import inspect
    assert inspect(service[4]).get_table_names() == []


def test_idempotent_receipt_recommit_and_different_binding_rejection(service):
    from aneos_api.enrichment_store import records
    from aneos_core.data.candidate_discovery import make_receipt
    from aneos_core.utils.errors import EvidencePersistenceError
    result = poll(service); receipt = result['receipt']
    assert service[1].commit(receipt) == result
    with sessionmaker(bind=service[4])() as db: assert len(records(db,'SYNTHETIC')) == 1
    changed, _ = make_receipt(receipt['query_id'],receipt['request']['parameters'],receipt['started_at'],
                             receipt['received_at'],receipt['response'],'TRANSPORT_UNAVAILABLE')
    with pytest.raises(EvidencePersistenceError): service[1].commit(changed)
    assert service[1].load(receipt['query_id']) == result


def test_sql_failure_rolls_back_receipt_and_object_assertions(service):
    from aneos_api.candidate_repository import CandidateQuery
    from aneos_api.enrichment_store import SourceAssertion
    from aneos_api.database import EnrichedNEO
    from aneos_core.utils.errors import EvidencePersistenceError
    service[1].prepare()
    with service[4].begin() as db:
        db.execute(text("CREATE TRIGGER synthetic_failure BEFORE INSERT ON neo_candidate_queries_v1 "
                        "BEGIN SELECT RAISE(ABORT,'synthetic disk failure'); END"))
    with pytest.raises(EvidencePersistenceError): poll(service)
    with sessionmaker(bind=service[4])() as db:
        assert db.scalar(select(CandidateQuery)) is None
        assert db.scalar(select(SourceAssertion)) is None
        assert db.scalar(select(EnrichedNEO)) is None


def test_sql_immutability_has_authorized_insert_positive_and_update_delete_denials(service):
    result = poll(service)
    for action in ("UPDATE neo_candidate_queries_v1 SET receipt_id='changed'", "DELETE FROM neo_candidate_queries_v1"):
        with pytest.raises(Exception, match='immutable candidate query'):
            with service[4].begin() as db: db.execute(text(action))
    assert service[1].load(result['receipt']['query_id']) == result


def test_file_reopen_restores_receipt_and_each_source_object(service):
    from aneos_api.candidate_repository import CandidateQueryRepository
    result = poll(service); url = str(service[4].url)
    service[4].dispose()
    engine = create_engine(url)
    try: assert CandidateQueryRepository(engine).load(result['receipt']['query_id']) == result
    finally: engine.dispose()


def test_nonpersistent_backend_fails_before_http(monkeypatch):
    from aneos_api.candidate_repository import CandidateQueryRepository
    from aneos_core.data.candidate_discovery import CADCandidateService
    from aneos_core.utils.errors import EvidencePersistenceError
    calls = []
    monkeypatch.setattr('aneos_core.data.provider_http.get',lambda *a,**k: calls.append(a))
    engine = create_engine('sqlite:///:memory:')
    try:
        with pytest.raises(EvidencePersistenceError): CADCandidateService(CandidateQueryRepository(engine)).poll('2026-11-12','2026-11-13')
        assert calls == []
    finally: engine.dispose()


def test_default_repository_is_lazy_and_uses_configured_database(service, monkeypatch):
    from types import SimpleNamespace
    import aneos_api.database as database
    from aneos_core.data.candidate_discovery import CADCandidateService
    from sqlalchemy import inspect
    monkeypatch.setattr(database, 'db_manager', SimpleNamespace(engine=service[4]))
    client = CADCandidateService()
    assert client.repository is None and inspect(service[4]).get_table_names() == []
    assert client.poll('2026-11-12','2026-11-13',limit=10)['commit_status'] == 'committed'
    assert len(service[3]) == 1


def test_schema_collision_is_not_implicitly_migrated(service):
    from aneos_core.utils.errors import EvidencePersistenceError
    with service[4].begin() as db:
        db.execute(text('CREATE TABLE neo_candidate_queries_v1 (sequence INTEGER PRIMARY KEY, query_id TEXT)'))
        db.execute(text("INSERT INTO neo_candidate_queries_v1 VALUES (1,'existing')"))
    with pytest.raises(EvidencePersistenceError): poll(service)
    assert service[3] == []
    with service[4].connect() as db:
        assert db.execute(text('SELECT * FROM neo_candidate_queries_v1')).all() == [(1,'existing')]


def test_unknown_query_and_frozen_old_projection_after_new_query(service):
    first = poll(service)
    assert service[1].load('cad-'+'0'*32) is None
    next_row = native()['data'][0]; next_row[4] = '.02'
    service[2]['value'] = Response(native(rows=[next_row]))
    second = poll(service)
    assert first['receipt']['query_id'] != second['receipt']['query_id']
    assert service[1].load(first['receipt']['query_id']) == first
    assert service[1].load(second['receipt']['query_id']) == second


def test_zero_returned_with_nonzero_total_stays_partial_not_empty(service):
    payload = native(rows=[]); payload['total'] = 100
    service[2]['value'] = Response(payload)
    result = poll(service)
    assert result['receipt']['outcome'] == 'partial' and result['records'] == []


@pytest.mark.parametrize('mutation', ['schema','query_id','query_parameters','start_naive','received_nonstring',
    'received_nonzero_offset','transport_code','absent_inconsistent','retention_size','retention_hash','body_type'])
def test_receipt_integrity_guards_reject_changes_without_writing(service, mutation):
    from aneos_core.data.candidate_discovery import replay_receipt
    result = poll(service)
    altered = copy.deepcopy(result['receipt'])
    if mutation == 'schema': altered['schema_version'] = 'future'
    elif mutation == 'query_id': altered['query_id'] = 'different'
    elif mutation == 'query_parameters': altered['request']['parameters']['body'] = 'Mars'
    elif mutation == 'start_naive': altered['started_at'] = '2026-11-13T00:00:00'
    elif mutation == 'received_nonstring': altered['received_at'] = 0
    elif mutation == 'received_nonzero_offset': altered['received_at'] = '2026-11-13T01:00:00+01:00'
    elif mutation == 'transport_code': altered['transport_error'] = 'unexpected'
    elif mutation == 'absent_inconsistent': altered['response']['body_base64'] = None
    elif mutation == 'retention_size': altered['response']['received_bytes'] += 1
    elif mutation == 'retention_hash': altered['response']['body_sha256'] = '0'*64
    else: altered['response']['retained_bytes'] = True
    with pytest.raises((ValueError,TypeError)): replay_receipt(altered)
    assert service[1].load(result['receipt']['query_id']) == result


@pytest.mark.parametrize('mutation', ['binding','receipt_binding','projection_set','missing_assertion',
    'assertion_hash','missing_projection','projection_hash','projection_inputs'])
def test_admin_fault_injection_is_detected_not_a_database_admin_security_claim(service, mutation):
    # Only this test's owned disposable DB: deliberately bypass normal SQL triggers
    # to test readback integrity independently from the update/delete-denial test.
    from aneos_api.candidate_repository import CandidateQuery
    from aneos_api.enrichment_store import ObjectProjection
    from aneos_core.data.enrichment import fingerprint, dumps
    result = poll(service); query_id = result['receipt']['query_id']
    with sessionmaker(bind=service[4])() as db:
        row = db.scalar(select(CandidateQuery))
        projections = json.loads(row.projection_ids_json)
        assertion_ids = json.loads(row.assertion_ids_json)
        binding = {'query_id':row.query_id,'receipt_id':row.receipt_id,'projection_ids':projections,
                   'assertion_ids':assertion_ids,'recorded_at':row.recorded_at}
        projection = db.scalar(select(ObjectProjection))
        projection_payload = json.loads(projection.payload_json)
    with service[4].begin() as db:
        for table in ('neo_candidate_queries_v1','neo_source_assertions_v1','neo_object_projections_v1'):
            for action in ('update','delete'):
                db.exec_driver_sql(f'DROP TRIGGER IF EXISTS {table}_no_{action}')
        if mutation == 'binding':
            db.execute(text("UPDATE neo_candidate_queries_v1 SET binding_sha256='changed'"))
        elif mutation == 'receipt_binding':
            db.execute(text("UPDATE neo_candidate_queries_v1 SET receipt_id='changed'"))
        elif mutation == 'projection_set':
            binding['projection_ids'] = {}
            db.execute(text('UPDATE neo_candidate_queries_v1 SET projection_ids_json=:ids, binding_sha256=:seal'),
                       {'ids':'{}','seal':fingerprint(binding)})
        elif mutation == 'missing_assertion':
            db.execute(text('DELETE FROM neo_source_assertions_v1'))
        elif mutation == 'assertion_hash':
            db.execute(text("UPDATE neo_source_assertions_v1 SET normalized_sha256='changed'"))
        elif mutation == 'missing_projection':
            db.execute(text('DELETE FROM neo_object_projections_v1'))
        elif mutation == 'projection_hash':
            db.execute(text("UPDATE neo_object_projections_v1 SET payload_sha256='changed'"))
        else:
            key = fingerprint({'inputs':[],'projection':projection_payload})
            binding['projection_ids'] = {'SYNTHETIC':key}
            db.execute(text('UPDATE neo_object_projections_v1 SET input_assertions_json=:inputs, projection_id=:key'),
                       {'inputs':'[]','key':key})
            db.execute(text('UPDATE neo_candidate_queries_v1 SET projection_ids_json=:ids,binding_sha256=:seal'),
                       {'ids':dumps(binding['projection_ids']),'seal':fingerprint(binding)})
    with pytest.raises(ValueError): service[1].load(query_id)


def test_commit_readback_failure_is_not_acknowledged_and_committed_data_survives(service, monkeypatch):
    from aneos_core.utils.errors import EvidencePersistenceError
    result = poll(service)
    original = service[1].load
    monkeypatch.setattr(service[1], 'load', lambda _: None)
    with pytest.raises(EvidencePersistenceError, match='missing'): service[1].commit(result['receipt'])
    def fail(_): raise ValueError('synthetic readback failure')
    monkeypatch.setattr(service[1], 'load', fail)
    with pytest.raises(EvidencePersistenceError, match='verified'): service[1].commit(result['receipt'])
    monkeypatch.setattr(service[1], 'load', original)
    assert service[1].load(result['receipt']['query_id']) == result


def test_shared_transport_error_preserves_buffered_closed_body_and_cooldown(tmp_path, monkeypatch):
    import requests
    from aneos_core.data import provider_http
    from aneos_core.data.candidate_discovery import capture_response, URL
    monkeypatch.setenv('ANEOS_PROVIDER_STATE_DIR',str(tmp_path/'state'))
    response = requests.Response(); response.status_code = 503
    response._content = b'{"error":"synthetic upstream unavailable"}'; response._content_consumed = True
    response.headers['Retry-After'] = '10'
    calls = []
    class Clock:
        def time(self): return 1000.
        def monotonic(self): return 0.
        def sleep(self, value): raise AssertionError('No retry or wait in this test')
    def send(*args,**kwargs): calls.append(args); return response
    with pytest.raises(provider_http.ProviderUnavailableError) as captured:
        provider_http.get(URL,_send=send,_clock=Clock(),_max_wait=0,timeout=1)
    assert captured.value.response is response and captured.value.status_code == 503
    assert base64.b64decode(capture_response(captured.value.response)['body_base64']) == response.content
    with pytest.raises(provider_http.ProviderUnavailableError):
        provider_http.get(URL,_send=send,_clock=Clock(),_max_wait=0,timeout=1)
    assert len(calls) == 1


def test_invalid_transport_object_fails_explicitly():
    from aneos_core.data.candidate_discovery import capture_response
    with pytest.raises(ValueError): capture_response(Response(body=b'',status=True))


@pytest.mark.parametrize('kind', ['row_limit','body_mismatch','orbit_type','uncertainty_type'])
def test_response_selection_and_native_label_guards(service, kind):
    payload = native()
    if kind == 'row_limit':
        payload['data'] *= 11; payload['count'] = payload['total'] = 11
    elif kind == 'body_mismatch':
        payload['fields'].append('body'); payload['data'][0].append('Mars')
    elif kind == 'orbit_type': payload['data'][0][1] = {'unparsed':'id'}
    else: payload['data'][0][9] = True
    service[2]['value'] = Response(payload)
    result = poll(service)
    assert result['receipt']['outcome'] == 'invalid' and result['records'] == []


def test_nonmonotonic_receipt_clock_is_retained_not_retagged_and_output_tamper_rejected(service):
    from aneos_core.data.candidate_discovery import make_receipt, replay_receipt
    result = poll(service); original = result['receipt']
    altered, _ = make_receipt(original['query_id'],original['request']['parameters'],
                             '2030-01-01T00:00:00+00:00',original['received_at'],original['response'])
    assert altered['receipt_clock_order'] == 'nonmonotonic_local_clock'
    altered['object_count'] = 100
    with pytest.raises(ValueError,match='reproduce'): replay_receipt(altered)


def test_readback_error_exposes_query_id_for_recovery_without_false_commit_ack(service, monkeypatch):
    from aneos_core.utils.errors import EvidencePersistenceError
    result = poll(service)
    monkeypatch.setattr(service[1],'load',lambda _:None)
    with pytest.raises(EvidencePersistenceError) as failed:
        service[1].commit(result['receipt'])
    assert failed.value.query_id == result['receipt']['query_id']


def test_full_oversize_archive_is_required_before_acknowledgement_and_recoverable(service):
    from uuid import uuid4
    from aneos_core.data.candidate_discovery import MAX_BODY, make_receipt
    from aneos_core.utils.errors import EvidencePersistenceError
    body = b'x'*(MAX_BODY+1)
    service[2]['value'] = Response(body=body)
    old = poll(service)['receipt']
    new, _ = make_receipt('cad-'+uuid4().hex,old['request']['parameters'],old['started_at'],old['received_at'],old['response'])
    with pytest.raises(EvidencePersistenceError): service[1].commit(new)
    assert service[1].load(new['query_id']) is None
    with pytest.raises(EvidencePersistenceError): service[1].commit(new,native_body=body[:-1]+b'y')
    restored = service[1].commit(new,native_body=body)
    assert restored['commit_status'] == 'committed'
    assert service[1].load_native_body(new['query_id']) == body


def test_http_error_priority_and_native_body_lookup_are_not_empty_success(service):
    from aneos_core.data.candidate_discovery import MAX_BODY
    body = b'x'*(MAX_BODY+1)
    service[2]['value'] = Response(body=body,status=503)
    result = poll(service)
    assert result['receipt']['outcome'] == 'unavailable'
    assert service[1].load_native_body(result['receipt']['query_id']) == body
    with pytest.raises(KeyError): service[1].load_native_body('cad-'+'0'*32)


def test_unexpected_duplicate_archive_does_not_create_new_receipt(service):
    from uuid import uuid4
    from aneos_core.data.candidate_discovery import make_receipt, capture_response
    from aneos_core.utils.errors import EvidencePersistenceError
    old = poll(service)['receipt']
    for response in (old['response'],capture_response(None)):
        new, _ = make_receipt('cad-'+uuid4().hex,old['request']['parameters'],old['started_at'],old['received_at'],response)
        with pytest.raises(EvidencePersistenceError): service[1].commit(new,native_body=b'unexpected')
        assert service[1].load(new['query_id']) is None


def test_native_archive_admin_fault_is_detected(service):
    from aneos_core.data.candidate_discovery import MAX_BODY
    service[2]['value'] = Response(body=b'x'*(MAX_BODY+1))
    result = poll(service)
    with service[4].begin() as db:
        db.exec_driver_sql('DROP TRIGGER neo_candidate_queries_v1_no_update')
        db.execute(text('UPDATE neo_candidate_queries_v1 SET native_body=:body'),{'body':b'changed'})
    with pytest.raises(ValueError,match='archive'): service[1].load(result['receipt']['query_id'])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def deny(*args,**kwargs): raise AssertionError('Offline candidate tests cannot contact providers')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'getaddrinfo',deny)


def test_native_catalogue_fields_and_uncertainty_units_are_retained_without_orbit_invention(service):
    payload = native()
    payload['fields'] += ['diameter','diameter_sigma','fullname']
    payload['data'][0] += ['0.011','0.002','  (SYNTHETIC)']
    service[2]['value'] = Response(payload)
    result = poll(service)
    record = result['records'][0]['neo_data']['evidence_metadata']['cad_candidate_records'][0]
    assert record['diameter_km'] == .011 and record['diameter_uncertainty'] == .002
    assert record['cad_evidence']['units']['diameter_sigma'] == 'km'
    assert record['cad_evidence']['raw_row'][-1] == '  (SYNTHETIC)'
    from aneos_core.data.models import NEOData
    from aneos_core.analysis.screening import ScreeningAnalysisService
    analysis = ScreeningAnalysisService().analyze_neo(NEOData.from_dict(result['records'][0]['neo_data']))
    assert analysis.screening_score is None and analysis.artificial_probability is None


def test_fresh_process_replays_without_default_database_creation_or_source_access(service, tmp_path):
    import os,subprocess,sys,hashlib
    result = poll(service)
    database_path = Path(service[4].url.database)
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    script = '''
import json,socket,sys
from sqlalchemy import create_engine
from aneos_api.candidate_repository import CandidateQueryRepository
def deny(*a,**k):raise AssertionError('No network during stored replay')
socket.socket.connect=deny;socket.getaddrinfo=deny
engine=create_engine(sys.argv[1])
try:
 result=CandidateQueryRepository(engine).load(sys.argv[2]);print(json.dumps(result))
finally:engine.dispose()
'''
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).parents[1]),
               ANEOS_DATABASE_URL='sqlite:///'+str(tmp_path/'unused-default.sqlite'))
    run = subprocess.run([sys.executable,'-c',script,str(service[4].url),result['receipt']['query_id']],
                         cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert run.returncode == 0,run.stderr
    assert json.loads(run.stdout) == result
    assert hashlib.sha256(database_path.read_bytes()).hexdigest() == before
    assert not (tmp_path/'unused-default.sqlite').exists()


def test_import_and_construction_do_not_create_database(tmp_path):
    import os,subprocess,sys
    script = 'from aneos_core.data.candidate_discovery import CADCandidateService; from aneos_api.candidate_repository import CandidateQueryRepository; CADCandidateService(); CandidateQueryRepository()'
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).parents[1]),
               ANEOS_DATABASE_URL='sqlite:///'+str(tmp_path/'not-created.sqlite'))
    run = subprocess.run([sys.executable,'-c',script],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert run.returncode == 0,run.stderr
    assert list(tmp_path.iterdir()) == []


def test_query_records_frozen_software_identity_and_old_metadata_free_receipts_still_replay(service):
    from aneos_version import APPLICATION_VERSION
    from aneos_core.data.candidate_discovery import make_receipt, replay_receipt
    receipt = poll(service)['receipt']
    assert receipt['software']['application_version'] == APPLICATION_VERSION
    assert receipt['software']['float_mantissa_bits'] > 0
    assert 'aneos_core/data/candidate_discovery.py' in receipt['software']['source_hashes_at_report_time']
    assert replay_receipt(receipt)[0]['software'] == receipt['software']
    old, _ = make_receipt(receipt['query_id'],receipt['request']['parameters'],receipt['started_at'],receipt['received_at'],receipt['response'])
    assert 'software' not in old and replay_receipt(old)[0] == old
    with pytest.raises(ValueError):
        make_receipt(receipt['query_id'],receipt['request']['parameters'],receipt['started_at'],receipt['received_at'],receipt['response'],software=[])


def test_idempotent_recommit_does_not_silently_ignore_contradictory_archive_argument(service):
    from aneos_core.data.candidate_discovery import MAX_BODY
    from aneos_core.utils.errors import EvidencePersistenceError
    body = b'x'*(MAX_BODY+1)
    service[2]['value'] = Response(body=body)
    result = poll(service)
    assert service[1].commit(result['receipt'],native_body=body) == result
    with pytest.raises(EvidencePersistenceError): service[1].commit(result['receipt'],native_body=b'changed')
    assert service[1].load_native_body(result['receipt']['query_id']) == body
