"""Native-shaped synthetic controls; actual controller, temporary SQLite, no providers."""
import copy
import json
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from aneos_core.data.candidate_discovery import CADCandidateService
from aneos_api.candidate_repository import CandidateQueryRepository
from neo_poller import NEOPoller

FIELDS = ['des','orbit_id','jd','cd','dist','dist_min','dist_max','v_rel','v_inf','t_sigma_f','h','diameter','diameter_sigma']
ROW = ['CONTROL','1','2461357.0','2026-Nov-12 12:00','.01','.009','.011','5','4','00:01','25','.011','.002']


def payload(rows=None):
    rows = [copy.deepcopy(ROW)] if rows is None else rows
    return {'signature':{'version':'1.5','source':'NASA/JPL SBDB Close Approach Data API'},
            'fields':FIELDS,'count':len(rows),'total':len(rows),'data':rows}


@pytest.fixture
def env(tmp_path, monkeypatch):
    import socket
    def denied(*a, **kw): raise AssertionError('Hidden network')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket,'getaddrinfo',denied)
    engine = create_engine('sqlite:///'+str(tmp_path/'evidence.sqlite'))
    repo = CandidateQueryRepository(engine)
    state = {'payload':payload()}; calls = []
    def get(*a, **kw):
        calls.append((a,kw)); return type('Response',(),{'content':json.dumps(state['payload']).encode(),'status_code':200})()
    monkeypatch.setattr('aneos_core.data.provider_http.get',get)
    p = NEOPoller(); p.console = None
    from datetime import date
    p._poll_date=lambda:date(2026,11,13)
    p.candidate_service = CADCandidateService(repo)
    monkeypatch.chdir(tmp_path)
    yield p,repo,state,calls
    engine.dispose()


def test_native_atomic_admission_is_same_row_and_replayable(env):
    p,repo,_,calls = env
    result = p.poll_and_analyze('NASA_CAD','1d',10)
    assert len(result)==1 and len(calls)==1
    neo = result[0]['neo_data']
    assert neo['orbital_elements'] is None
    assert neo['physical_properties']['diameter_km']==.011
    assert neo['physical_properties']['diameter_uncertainty']==.002
    assert neo['physical_properties']['absolute_magnitude_h']==25
    evidence=neo['evidence_metadata']['physical_evidence']
    assert all(r['source']=='CAD' and r['measurement_epoch'] is None for r in evidence['records'])
    assert evidence['records'][0]['native']['row']==ROW
    saved=p.last_poll_result
    assert saved['receipt']['object_policy']=='cad-physical-v1'
    assert repo.load(saved['receipt']['query_id'])==saved
    replay=p.replay_candidate_query(saved['receipt']['query_id'])
    assert replay==result and len(calls)==1
    assert result[0]['screening_assessment']['artificial_probability'] is None

@pytest.mark.parametrize('which', ['null','sigma_only','zero_sigma','zero_h','ambiguous','repeat','reordered'])
def test_atomic_missing_repeat_conflict_and_native_labels(env, which):
    p,repo,state,_=env
    rows=[copy.deepcopy(ROW)]
    if which=='null': rows[0][10:]=[None,None,None]
    if which=='sigma_only': rows[0][11]=None
    if which=='zero_sigma': rows[0][12]='0'
    if which=='zero_h': rows[0][10]='0'
    if which in ('ambiguous','repeat'):
        rows.append(copy.deepcopy(ROW)); rows[1][2]='2461357.01'; rows[1][3]='2026-Nov-12 12:14'
        if which=='ambiguous': rows[1][12]='.003'
    state['payload']=payload(rows)
    if which=='reordered':
        state['payload']['fields']=list(reversed(FIELDS))
        state['payload']['data']=[list(reversed(rows[0]))]
    records=p.poll_and_analyze('NASA_CAD','1d',10)
    n=records[0]['neo_data']; pp=n['physical_properties'] or {}
    e=n['evidence_metadata']['physical_evidence']
    if which in ('null','sigma_only'):
        assert pp.get('diameter_km') is None and pp.get('diameter_uncertainty') is None
    elif which=='ambiguous':
        assert e['selection']['diameter_km']['status']=='ambiguous'
        assert pp.get('diameter_km') is None and pp.get('diameter_uncertainty') is None
        assert pp['absolute_magnitude_h']==25
    else:
        assert pp['diameter_km']==.011
        assert pp['diameter_uncertainty']==(0 if which=='zero_sigma' else .002)
    if which=='zero_h': assert pp['absolute_magnitude_h']==0
    if which=='repeat':
        assert len(n['close_approaches'])==2 and len(e['duplicate_records'])==2
    assert repo.load(p.last_poll_result['receipt']['query_id'])==p.last_poll_result


@pytest.mark.parametrize('case', ['empty','partial','invalid','synthetic'])
def test_collection_outcomes_stay_archived_and_exported_without_fake_objects(env,case):
    p,repo,state,calls=env
    if case=='empty': state['payload']=payload([])
    if case=='partial': state['payload']['total']=5
    if case=='invalid': state['payload']['fields']=['des']
    if case=='synthetic': state['payload']['synthetic']=True
    rows=p.poll_and_analyze('NASA_CAD','1d',1)
    expected={'empty':'missing','partial':'partial','invalid':'invalid','synthetic':'invalid'}[case]
    assert p.last_poll_result['receipt']['outcome']==expected
    assert len(rows)==(1 if case=='partial' else 0)
    d=p.display_results(rows)
    assert d.metrics.total_records==len(rows)
    path=p.save_results(rows,'NASA_CAD','1d')
    packet=json.loads((path/'report.json').read_text())
    assert packet['invocation_context']['candidate_collection']==p.last_poll_result
    assert len(calls)==1


@pytest.mark.parametrize('source',['MPC','NEODyS','NASA_SBDB','bad'])
def test_unsupported_discovery_never_queries(env,source):
    p,_,_,calls=env
    with pytest.raises(NotImplementedError): p.poll_and_analyze(source,'1d',10)
    assert calls==[] and p.last_poll_result is None


@pytest.mark.parametrize('period,limit,budget',[('0d',1,0),('201y',1,0),('bad',1,0),('1d',0,0),('1d',1001,0),('1d',True,0),('1d',1,-1),('1d',1,21),('1d',1,True)])
def test_invalid_budgets_fail_before_network(env,period,limit,budget):
    p,_,_,calls=env
    with pytest.raises(ValueError): p.poll_and_analyze('NASA_CAD',period,limit,enrich_objects=budget)
    assert calls==[]


def test_existing_raw_only_receipts_replay_unchanged(env):
    p,repo,_,calls=env
    old=p.candidate_service.poll('2026-11-12','2026-11-13',limit=10)
    assert 'object_policy' not in old['receipt']
    assert old['records'][0]['neo_data']['physical_properties'] is None
    assert repo.load(old['receipt']['query_id'])==old
    assert p.replay_candidate_query(old['receipt']['query_id'])[0]['neo_data']['physical_properties'] is None
    assert len(calls)==1


def test_store_failure_cannot_return_analysis_or_previous_receipt(env,monkeypatch):
    from aneos_core.utils.errors import EvidencePersistenceError
    p,repo,_,_=env
    p.poll_and_analyze('NASA_CAD','1d',10)
    def fail(*a,**kw): raise EvidencePersistenceError('controlled commit failure')
    monkeypatch.setattr(repo,'commit',fail)
    with pytest.raises(EvidencePersistenceError): p.poll_and_analyze('NASA_CAD','1d',10)
    assert p.last_poll_result is None and p._last_candidate_rows is None


def test_explicit_enrichment_uses_actual_fetcher_and_committed_readback(env,tmp_path,monkeypatch):
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.cache import CacheManager
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    p,repo,state,calls=env
    second=copy.deepcopy(ROW);second[0]='OTHER'
    state['payload']=payload([ROW,second])
    f=DataFetcher(cache_manager=CacheManager(tmp_path/'cache'),evidence_repository=SQLEvidenceRepository(repo.engine))
    class Source:
        def fetch_orbital_elements(self, designation):
            return {'semi_major_axis':1.23,'eccentricity':.2,'inclination':1.49}
    f.sources={'SBDB':Source()}
    f._fetch_close_approaches=Mock(side_effect=AssertionError('Committed CAD must be reused, not polled again'))
    p.fetcher_factory=lambda:f
    rows=p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    assert len(calls)==1
    assert rows[0]['enrichment']['status']=='committed'
    receipt=rows[0]['enrichment']['receipt']
    assert f.evidence_repository.load_acquisition(receipt['query_id'])['receipt']==receipt
    assert rows[0]['neo_data']['orbital_elements']['semi_major_axis']==1.23
    assert rows[0]['candidate_neo_data']['physical_properties']['diameter_km']==.011
    assert rows[0]['neo_data']['physical_properties']['diameter_km']==.011
    assert rows[0]['local_projection']['projection']['selected_source_assertions']['CAD']
    assert rows[1]['enrichment']['status']=='outside_object_budget'
    assert rows[1]['neo_data']['orbital_elements'] is None
    assert rows[0]['screening_assessment']['artificial_probability'] is None


def test_receipt_without_actual_enrichment_commit_cannot_bypass(env):
    from aneos_core.data.models import NEOData
    from aneos_core.utils.errors import EvidencePersistenceError
    p,_,_,_=env
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    fake=Mock();fake.evidence_repository=SQLEvidenceRepository(env[1].engine)
    fake.fetch_neo_data.return_value=NEOData('CONTROL')
    p.fetcher_factory=lambda:fake
    with pytest.raises(EvidencePersistenceError):p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)


def make_fetcher(env,tmp_path):
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.cache import CacheManager
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    f=DataFetcher(cache_manager=CacheManager(tmp_path/'extra-cache'),evidence_repository=SQLEvidenceRepository(env[1].engine))
    class Source:
        def fetch_orbital_elements(self, designation):
            return {'semi_major_axis':1.23,'eccentricity':.2,'inclination':1.49}
    f.sources={'SBDB':Source()}
    f._fetch_close_approaches=Mock(side_effect=AssertionError('Committed CAD must be reused, not polled again'))
    return f


def test_default_fetcher_path_is_same_shared_committed_adapter(env,tmp_path,monkeypatch):
    f=make_fetcher(env,tmp_path)
    monkeypatch.setattr('aneos_core.data.fetcher.DataFetcher',lambda:f)
    rows=env[0].poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    assert rows[0]['enrichment']['status']=='committed'


def test_real_source_failure_stays_explicit_and_durable(env,tmp_path):
    f=make_fetcher(env,tmp_path);f.sources={}
    env[0].fetcher_factory=lambda:f
    rows=env[0].poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    assert rows[0]['enrichment']['status']=='no_new_source_data'
    assert rows[0]['enrichment']['coverage']['status_counts']['available']==1
    assert rows[0]['enrichment']['new_source_contributions']==[]
    r=rows[0]['enrichment']['receipt']
    assert f.evidence_repository.load_acquisition(r['query_id'])['bundle']['source_plan']['candidate_query_id']==env[0].last_poll_result['receipt']['query_id']
    assert rows[0]['screening_assessment']['screening_score'] is None


@pytest.mark.parametrize('fault',['identity','readback','tampered_snapshot','projection','missing_failure_receipt','cross_database'])
def test_enrichment_prerequisites_fail_closed_without_losing_candidate_receipt(env,tmp_path,monkeypatch,fault):
    from aneos_core.data.models import NEOData
    from aneos_core.utils.errors import DataSourceUnavailableError,EvidencePersistenceError
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    f=make_fetcher(env,tmp_path);p=env[0];p.fetcher_factory=lambda:f
    if fault=='identity': monkeypatch.setattr(f,'fetch_neo_data',lambda name:NEOData('WRONG'))
    if fault=='readback': monkeypatch.setattr(f.evidence_repository,'load_acquisition',lambda q:None)
    if fault=='tampered_snapshot':
        actual=f.fetch_neo_data
        def tamper(name):
            n=actual(name);n.orbital_elements.inclination=17.;return n
        monkeypatch.setattr(f,'fetch_neo_data',tamper)
    if fault=='projection': monkeypatch.setattr('aneos_api.enrichment_store.latest_projection',lambda *a,**kw:None)
    if fault=='missing_failure_receipt':
        def fail(*a,**kw):raise DataSourceUnavailableError('no usable receipt')
        monkeypatch.setattr(f,'fetch_neo_data',fail)
    other_engine=None
    if fault=='cross_database':
        other_engine=create_engine('sqlite:///'+str(tmp_path/'wrong.sqlite'))
        f.evidence_repository=SQLEvidenceRepository(other_engine)
        f.fetch_neo_data=Mock(side_effect=AssertionError('Cross-store acquisition must not start'))
    with pytest.raises((ValueError,EvidencePersistenceError,DataSourceUnavailableError)) as raised:
        p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    assert raised.value.candidate_query['commit_status']=='committed'
    assert p.last_poll_result['commit_status']=='committed'
    assert p._last_candidate_rows is None
    assert env[1].load(p.last_poll_result['receipt']['query_id'])==p.last_poll_result
    if other_engine is not None:
        f.fetch_neo_data.assert_not_called();other_engine.dispose()


def test_default_repository_replay_and_unknown_query_do_not_poll(env,monkeypatch):
    p,repo,_,calls=env
    rows=p.poll_and_analyze('NASA_CAD','1d',10)
    key=p.last_poll_result['receipt']['query_id']
    p.candidate_service=None
    monkeypatch.setattr('aneos_api.candidate_repository.CandidateQueryRepository',lambda:repo)
    assert p.replay_candidate_query(key)==rows
    with pytest.raises(KeyError):p.replay_candidate_query('cad-'+'0'*32)
    assert len(calls)==1


def test_uncommitted_candidate_cannot_be_consumed():
    from aneos_core.polling.root_candidate_controller import RootCandidateController
    from aneos_core.utils.errors import EvidencePersistenceError
    with pytest.raises(EvidencePersistenceError):RootCandidateController().consume({'commit_status':'failed'})


def test_actual_compatibility_fetch_archives_native_empty_and_positive(env):
    p,repo,state,calls=env
    for data in [payload(),payload([])]:
        state['payload']=data
        assert p.fetch_cad_data('2026-11-12','2026-11-13',10)==data
        assert repo.load(p.last_poll_result['receipt']['query_id'])==p.last_poll_result
    assert len(calls)==2


def test_actual_plain_menu_uses_native_controller_and_preserves_export(env,monkeypatch,capsys):
    p,_,_,calls=env
    answers=iter(['1','1d','10','y','n'])
    monkeypatch.setattr('builtins.input',lambda *a:next(answers))
    p.run_interactive_menu()
    assert len(calls)==1
    assert p.last_poll_result['receipt']['outcome']=='available'
    text=capsys.readouterr().out
    assert 'workflow finished' in text and 'Inconclusive' in text
    from pathlib import Path
    assert len(list(Path('dashboard_results').glob('*/complete.json')))==1


@pytest.mark.parametrize('empty,invalid',[(False,False),(True,False),(False,True)])
def test_actual_cli_controller_export_and_exit_status(env,monkeypatch,empty,invalid):
    import sys,neo_poller
    p,_,state,calls=env
    if empty:state['payload']=payload([])
    if invalid:state['payload']['signature']['version']='unsupported'
    monkeypatch.setattr(neo_poller,'NEOPoller',lambda:p)
    monkeypatch.setattr(sys,'argv',['neo_poller.py','--api','NASA_CAD','--period','1d','--max-results','10'])
    status=neo_poller.main()
    assert status==(2 if invalid else None)
    assert len(calls)==1


@pytest.mark.parametrize('width',[80,120,160])
def test_actual_cad_detail_and_summary_render_at_each_width(env,width,tmp_path):
    import io
    from rich.console import Console
    from rich.text import Text
    p=env[0];out=io.StringIO()
    p.console=Console(file=out,width=width,force_terminal=True,record=True)
    rows=p.poll_and_analyze('NASA_CAD','1d',10)
    dashboard=p.display_results(rows)
    p._show_detailed_analysis(dashboard.classifications[0])
    text=Text.from_ansi(out.getvalue()).plain
    assert 'CAD catalogue' in text and 'measurement date' in text
    assert 'uncertainty not available' in text and '0.002' in text
    assert all(Text(line).cell_len<=width for line in text.splitlines())
    p.console.save_svg(str(tmp_path/f'cad-{width}.svg'))


def test_reuse_preserves_original_read_time_and_skips_cad_transport(env,tmp_path):
    from aneos_core.analysis.screening import ScreeningAnalysisService
    p,repo,_,calls=env
    f=make_fetcher(env,tmp_path);p.fetcher_factory=lambda:f
    rows=p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    receipt=rows[0]['enrichment']['receipt']
    bundle=f.evidence_repository.load_acquisition(receipt['query_id'])['bundle']
    cad=next(o for o in bundle['outcomes'] if o['source']=='CAD')
    assert cad['neo_data']==p.last_poll_result['records'][0]['neo_data']
    assert cad['neo_data']['fetched_at']==p.last_poll_result['receipt']['received_at']
    assert bundle['source_plan']['roles']['CAD']=='replayed_committed_collection_no_new_cad_request'
    assert cad['adapter_payload']['candidate_receipt']==p.last_poll_result['receipt']
    f._fetch_close_approaches.assert_not_called()
    # Frozen evidence is replayable offline, including the parent physical policy.
    from aneos_core.data.acquisition import validate_bundle
    assert validate_bundle(bundle)==bundle
    assert ScreeningAnalysisService().analyze_neo(__import__('aneos_core.data.models',fromlist=['NEOData']).NEOData.from_dict(rows[0]['neo_data'])).to_payload()==rows[0]['screening_assessment']
    assert len(calls)==1


@pytest.mark.parametrize('reference',['invalid','cad-'+'0'*32])
def test_invalid_or_absent_committed_reference_blocks_all_provider_work(env,tmp_path,reference):
    from aneos_core.utils.errors import EvidencePersistenceError
    f=make_fetcher(env,tmp_path);f.candidate_query_id=reference
    source=Mock(side_effect=AssertionError('Providers must not start'))
    f.sources['SBDB'].fetch_orbital_elements=source
    with pytest.raises((ValueError,EvidencePersistenceError)):f.fetch_neo_data('CONTROL')
    source.assert_not_called();assert env[3]==[]


@pytest.mark.parametrize('fault',['wrong_object','missing_reference','wrong_reference','unexpected_override','bad_plan'])
def test_candidate_source_reference_and_frozen_plan_cannot_be_substituted(env,tmp_path,fault):
    from aneos_core.data.acquisition import plan,validate_bundle,adapter_to_neo
    f=make_fetcher(env,tmp_path);p=env[0]
    p.poll_and_analyze('NASA_CAD','1d',10)
    query_id=p.last_poll_result['receipt']['query_id']
    if fault=='wrong_object':
        with pytest.raises(ValueError):adapter_to_neo('CAD','OTHER',{'candidate_receipt':p.last_poll_result['receipt']},'2026-11-13T00:00:00+00:00')
        return
    source_plan=plan(['SBDB'],candidate_query_id=query_id)
    supplied=f._committed_candidate_outcome('CONTROL',query_id,f.evidence_repository)
    if fault=='missing_reference': supplied=None
    if fault=='wrong_reference': source_plan=plan(['SBDB'],candidate_query_id='cad-'+'0'*32)
    if fault=='unexpected_override': source_plan=plan(['SBDB'])
    if fault=='bad_plan':
        f.source_plan={'source_priority':['SBDB'],'context_id':None,'wrong':True}
        with pytest.raises(ValueError):f._query_plan()
        return
    with pytest.raises(ValueError):f._fetch_from_all_sources('CONTROL',source_plan=source_plan,candidate_outcome=supplied)


@pytest.mark.parametrize('fault',['bad_index','bad_field','bad_identity','not_array','missing_column','bad_adapter','tampered_value','unknown_policy'])
def test_physical_policy_and_native_evidence_fail_closed(env,fault):
    from aneos_core.data.physical_evidence import PhysicalMeasurement,build_cad_evidence,verify_physical_evidence
    native={'fields':FIELDS,'row':copy.deepcopy(ROW),'row_index':0,'field':'diameter'}
    context={'designation':'CONTROL'}
    if fault=='bad_index': native['row_index']=True
    if fault=='bad_field': native['field']='albedo'
    if fault=='bad_identity': context['designation']='OTHER'
    if fault=='missing_column': native['field']='h';native['fields']=FIELDS[:10]+FIELDS[11:];native['row']=ROW[:10]+ROW[11:]
    if fault=='not_array':
        with pytest.raises(ValueError):build_cad_evidence(None,context)
    elif fault in ('bad_adapter','tampered_value'):
        e=build_cad_evidence([native],context)
        if fault=='bad_adapter':e['records'][0]['adapter']='unsupported'
        else:e['records'][0]['normalized']['value']=999
        with pytest.raises(ValueError):verify_physical_evidence(e)
    elif fault=='unknown_policy':
        with pytest.raises(ValueError):env[0].candidate_service.poll('2026-11-12','2026-11-13',limit=1,object_policy='unknown')
        assert env[3]==[]
    else:
        with pytest.raises(ValueError):PhysicalMeasurement.from_cad(native,context)


def test_compatibility_fetch_returns_unavailable_not_empty_for_invalid_payload(env):
    env[2]['payload']['signature']['version']='unsupported'
    assert env[0].fetch_cad_data('2026-11-12','2026-11-13',10) is None
    assert env[0].last_poll_result['receipt']['outcome']=='invalid'


@pytest.mark.parametrize('epoch',['2461356.499999','2461357.500001'])
def test_source_rows_outside_native_query_window_remain_archived_not_admitted(env,epoch):
    p,repo,state,_=env
    state['payload']['data'][0][2]=epoch
    assert p.poll_and_analyze('NASA_CAD','1d',10)==[]
    saved=p.last_poll_result
    assert saved['receipt']['outcome']=='invalid'
    assert repo.load(saved['receipt']['query_id'])==saved
    import base64
    raw=json.loads(base64.b64decode(saved['receipt']['response']['body_base64']))
    assert raw['data'][0][2]==epoch


def test_cached_enrichment_cannot_bypass_parent_readback(env,tmp_path,monkeypatch):
    from aneos_core.utils.errors import EvidencePersistenceError
    p=env[0];f=make_fetcher(env,tmp_path);p.fetcher_factory=lambda:f
    p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    def bad(*a,**kw):raise ValueError('controlled parent corruption')
    monkeypatch.setattr('aneos_api.candidate_repository.CandidateQueryRepository.load',bad)
    f.sources['SBDB'].fetch_orbital_elements=Mock(side_effect=AssertionError('Must not reacquire'))
    with pytest.raises(EvidencePersistenceError):f.fetch_neo_data('CONTROL')
    f.sources['SBDB'].fetch_orbital_elements.assert_not_called()


def test_replay_does_not_append_assertions_or_change_projection(env):
    from sqlalchemy import select,func
    from aneos_api.enrichment_store import SourceAssertion,ObjectProjection
    from aneos_api.candidate_repository import CandidateQuery
    p,repo,_,calls=env
    p.poll_and_analyze('NASA_CAD','1d',10)
    def counts():
        with repo.sessions() as db:
            return [db.scalar(select(func.count()).select_from(cls)) for cls in (SourceAssertion,ObjectProjection,CandidateQuery)]
    before=counts()
    p.replay_candidate_query(p.last_poll_result['receipt']['query_id'])
    assert counts()==before and len(calls)==1


def test_frozen_bundle_reference_substitution_is_rejected_even_with_new_hashes(env,tmp_path):
    from aneos_core.data.acquisition import plan,validate_bundle
    from aneos_core.data.enrichment import fingerprint
    p=env[0];f=make_fetcher(env,tmp_path);p.fetcher_factory=lambda:f
    rows=p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    bundle=f.evidence_repository.load_acquisition(rows[0]['enrichment']['receipt']['query_id'])['bundle']
    old=bundle['source_plan']
    bundle['source_plan']=plan(old['source_priority'],context_id=old['context_id'],candidate_query_id='cad-'+'0'*32)
    bundle['coverage']['profile']=bundle['source_plan']['profile']
    bundle['bundle_id']=fingerprint({k:v for k,v in bundle.items() if k!='bundle_id'})
    with pytest.raises(ValueError,match='bound to committed'):validate_bundle(bundle)


def test_actual_cli_rejects_invalid_budget_without_source_call(env,monkeypatch,capsys):
    import sys,neo_poller
    monkeypatch.setattr(neo_poller,'NEOPoller',lambda:env[0])
    monkeypatch.setattr(sys,'argv',['neo_poller.py','--api','NASA_CAD','--max-results','1001'])
    assert neo_poller.main()==2
    assert env[3]==[] and 'unavailable' in capsys.readouterr().err


def test_actual_rich_menu_has_source_aware_native_success(env,monkeypatch):
    import io,neo_poller
    from rich.console import Console
    p=env[0];out=io.StringIO();p.console=Console(file=out,width=80,record=True)
    monkeypatch.setattr(neo_poller,'console',p.console)
    answers=iter(['NASA_CAD','1d'])
    monkeypatch.setattr('rich.prompt.Prompt.ask',lambda *a,**kw:next(answers))
    monkeypatch.setattr('rich.prompt.IntPrompt.ask',lambda *a,**kw:10)
    confirms=iter([True,False])
    monkeypatch.setattr('rich.prompt.Confirm.ask',lambda *a,**kw:next(confirms))
    p.run_interactive_menu()
    assert p.last_poll_result['receipt']['outcome']=='available'
    assert len(env[3])==1
    text=p.console.export_text()
    assert text.count('Unavailable')>=3 and 'Bounded CAD' in text and 'Inconclusive' in text
    assert all(name in text for name in ('NASA_CAD','NASA_SBDB','MPC','NEODyS'))


def test_real_sql_enrichment_abort_preserves_collection_and_returns_no_analysis(env,tmp_path):
    from sqlalchemy import text,select,func
    from aneos_api.enrichment_store import SourceAssertion,ObjectAcquisition
    from aneos_core.utils.errors import EvidencePersistenceError
    p,repo,_,_=env
    repo.prepare()
    with repo.engine.begin() as db:
        db.execute(text("CREATE TRIGGER controlled_sbdb_abort BEFORE INSERT ON neo_source_assertions_v1 WHEN NEW.source = 'SBDB' BEGIN SELECT RAISE(ABORT,'controlled enrichment abort'); END"))
    f=make_fetcher(env,tmp_path);p.fetcher_factory=lambda:f
    with pytest.raises(EvidencePersistenceError):p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    assert p._last_candidate_rows is None
    assert repo.load(p.last_poll_result['receipt']['query_id'])==p.last_poll_result
    with repo.sessions() as db:
        assert db.scalar(select(func.count()).select_from(SourceAssertion))==1  # Original CAD succeeded.
        assert db.scalar(select(func.count()).select_from(ObjectAcquisition))==0
        assert db.scalar(select(SourceAssertion)).source=='CAD'


@pytest.mark.parametrize('kind',['timeout','http503'])
def test_root_unavailable_collection_is_not_valid_empty_or_missing_as_zero(env,monkeypatch,capsys,kind):
    import requests
    p,repo,_,_=env
    def fail(*a,**kw):
        if kind=='timeout':raise requests.Timeout('controlled provider timeout')
        return type('Response',(),{'status_code':503,'content':b'controlled provider failure'})()
    controlled=Mock(side_effect=fail)
    monkeypatch.setattr('aneos_core.data.provider_http.get',controlled)
    assert p.poll_and_analyze('NASA_CAD','1d',10)==[]
    saved=p.last_poll_result
    assert saved['receipt']['outcome']=='unavailable'
    assert saved['receipt']['native_count'] is None
    assert repo.load(saved['receipt']['query_id'])==saved
    text=capsys.readouterr().out
    assert 'rows not available' in text and 'Recorded diagnostic:' in text
    assert controlled.call_count==1


def test_authorized_positive_root_candidate_still_advances(env,tmp_path):
    p=env[0];f=make_fetcher(env,tmp_path)
    f.sources['SBDB'].fetch_orbital_elements=lambda name:{'semi_major_axis':1.,'eccentricity':.2,'inclination':52.23568469321602}
    p.fetcher_factory=lambda:f
    result=p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)[0]['screening_assessment']
    assert result['classification']=='FOLLOW_UP_CANDIDATE'
    assert result['artificial_probability'] is None
    assert result['analysis_metadata']['five_sigma_claim_permitted'] is False


def test_valid_cache_reuse_and_consistently_rehashed_wrong_parent_control(env,tmp_path):
    from aneos_core.data.acquisition import build_bundle,adapter_to_neo,outcome
    from aneos_core.data.candidate_discovery import make_receipt,capture_response
    from aneos_core.utils.errors import EvidencePersistenceError
    p=env[0];f=make_fetcher(env,tmp_path);p.fetcher_factory=lambda:f
    source=Mock(wraps=f.sources['SBDB'].fetch_orbital_elements)
    f.sources['SBDB'].fetch_orbital_elements=source
    rows=p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    original=f.evidence_repository.load_acquisition(rows[0]['enrichment']['receipt']['query_id'])['bundle']
    assert f.fetch_neo_data('CONTROL').evidence_metadata['acquisition']['storage_receipt']==rows[0]['enrichment']['receipt']
    assert source.call_count==1  # Authorized cache positive, not merely a disabled cache.
    parent=p.last_poll_result['receipt']
    changed=payload();changed['data'][0][11]='.02'
    response=type('Response',(),{'status_code':200,'content':json.dumps(changed).encode()})()
    forged,_=make_receipt(parent['query_id'],parent['request']['parameters'],parent['started_at'],parent['received_at'],
                         capture_response(response),object_policy=parent['object_policy'],software=parent['software'])
    stamp=original['started_at'];packet={'candidate_receipt':forged}
    replacement=outcome('CAD','CONTROL',status='available',neo=adapter_to_neo('CAD','CONTROL',packet,stamp),
                        adapter_payload=packet,started_at=stamp,completed_at=stamp)
    outcomes=[replacement if o['source']=='CAD' else o for o in original['outcomes']]
    forged_bundle=build_bundle('CONTROL',original['source_plan'],outcomes,software=original['software'])
    key='neo-query:object-acquisition-v1:'+original['source_plan']['plan_id']+':CONTROL'
    f.cache_manager.set(key,{'schema_version':'object-query-cache-v1','bundle':forged_bundle},ttl=3600)
    with pytest.raises(EvidencePersistenceError,match='conflicts with committed parent'):f.fetch_neo_data('CONTROL')
    assert source.call_count==1 and len(env[3])==1


def test_empty_cli_retains_requested_enrichment_budget_without_object_queries(env,monkeypatch):
    import sys,neo_poller
    p=env[0];env[2]['payload']=payload([])
    p.fetcher_factory=Mock(side_effect=AssertionError('No objects to enrich'))
    monkeypatch.setattr(neo_poller,'NEOPoller',lambda:p)
    monkeypatch.setattr(sys,'argv',['neo_poller.py','--api','NASA_CAD','--enrich-objects','2'])
    assert neo_poller.main() is None
    p.fetcher_factory.assert_not_called()
    from pathlib import Path
    report=json.loads(next(Path('dashboard_results').glob('*/report.json')).read_text())
    assert report['invocation_context']['requested_additional_object_budget']==2
    assert report['invocation_context']['candidate_collection']['receipt']['budget']['enrichment_requests']==0


def test_collection_replay_export_is_labelled_not_a_new_poll(env):
    p=env[0];p.poll_and_analyze('NASA_CAD','1d',10)
    rows=p.replay_candidate_query(p.last_poll_result['receipt']['query_id'])
    path=p.save_results(rows,'NASA_CAD','caller-supplied-label')
    context=json.loads((path/'report.json').read_text())['invocation_context']
    assert context['mode']=='read_only_collection_replay_not_original_invocation'
    assert context['candidate_collection']==p.last_poll_result
    assert len(env[3])==1


def test_acquisition_software_context_includes_driver_and_controller_hashes(env,tmp_path):
    from pathlib import Path
    from hashlib import sha256
    p=env[0];f=make_fetcher(env,tmp_path);p.fetcher_factory=lambda:f
    rows=p.poll_and_analyze('NASA_CAD','1d',10,enrich_objects=1)
    bundle=f.evidence_repository.load_acquisition(rows[0]['enrichment']['receipt']['query_id'])['bundle']
    root=Path(__file__).resolve().parents[1]
    for software in (p.last_poll_result['receipt']['software'],bundle['software']):
        for name in ('neo_poller.py','aneos_core/polling/root_candidate_controller.py'):
            assert software['source_hashes_at_report_time'][name]==sha256((root/name).read_bytes()).hexdigest()
