"""Owned file-SQLite controls: read-only replay remains useful, fresh polling must stop."""
import json
from unittest.mock import Mock
import pytest
from sqlalchemy import create_engine
from aneos_api.candidate_repository import CandidateQueryRepository
from aneos_api.acquisition_repository import SQLEvidenceRepository,initialize_acquisition_schema
from aneos_core.data.candidate_discovery import CADCandidateService
from aneos_core.data.fetcher import DataFetcher
from aneos_core.utils.errors import EvidencePersistenceError


@pytest.fixture
def stored(tmp_path,monkeypatch):
    import socket
    def denied(*a,**kw):raise AssertionError('No live network in storage preflight controls')
    monkeypatch.setattr(socket.socket,'connect',denied)
    path=tmp_path/'evidence.sqlite';engine=create_engine('sqlite:///'+str(path))
    payload={'signature':{'version':'1.5'},'count':1,'total':1,
             'fields':['des','orbit_id','jd','cd','dist','dist_min','dist_max','v_rel','v_inf','t_sigma_f','h'],
             'data':[['CONTROL','1','2461357.5','2026-Nov-13 00:00','.01','.009','.011','5','4','00:01','25']]}
    response=type('Response',(),{'content':json.dumps(payload).encode(),'status_code':200})()
    transport=Mock(return_value=response)
    monkeypatch.setattr('aneos_core.data.provider_http.get',transport)
    repo=CandidateQueryRepository(engine)
    saved=CADCandidateService(repo).poll('2026-11-12','2026-11-13',limit=1)
    initialize_acquisition_schema(engine)
    transport.reset_mock()
    yield path,repo,saved,transport
    engine.dispose()


@pytest.mark.parametrize('flags',['mode=ro','immutable=1','immutable=true','immutable=YES','immutable=on'])
def test_readonly_query_replay_succeeds_but_new_collection_never_queries(stored,flags):
    path,_,saved,transport=stored
    engine=create_engine('sqlite:///file:'+str(path)+'?'+flags+'&uri=true')
    try:
        repo=CandidateQueryRepository(engine)
        assert repo.load(saved['receipt']['query_id'])==saved
        with pytest.raises(EvidencePersistenceError):
            CADCandidateService(repo).poll('2026-11-12','2026-11-13',limit=1)
        transport.assert_not_called()
        assert repo.load(saved['receipt']['query_id'])==saved
    finally:engine.dispose()


@pytest.mark.parametrize('flags',['mode=ro','immutable=1'])
def test_named_object_preflight_rejects_readonly_before_provider_work(stored,flags):
    path,_,_,_=stored
    engine=create_engine('sqlite:///file:'+str(path)+'?'+flags+'&uri=true')
    try:
        f=DataFetcher(cache_manager=Mock(),evidence_repository=SQLEvidenceRepository(engine))
        f._fetch_from_all_sources=Mock(side_effect=AssertionError('Provider work began'))
        with pytest.raises(EvidencePersistenceError):f.fetch_neo_data('CONTROL',force_refresh=True)
        f._fetch_from_all_sources.assert_not_called()
    finally:engine.dispose()


@pytest.mark.parametrize('flags',['mode=rw','mode=rwc','immutable=0','immutable=false','immutable=OFF','immutable=no','immutable='])
def test_explicit_writable_positive_still_acquires_and_readonly_replay_is_not_disabled(stored,flags):
    path,_,old,transport=stored
    engine=create_engine('sqlite:///file:'+str(path)+'?'+flags+'&uri=true')
    try:
        repo=CandidateQueryRepository(engine)
        new=CADCandidateService(repo).poll('2026-11-12','2026-11-13',limit=1)
        assert new['commit_status']=='committed' and transport.call_count==1
        assert repo.load(old['receipt']['query_id'])==old
        assert SQLEvidenceRepository(engine).prepare_acquisition()['status']=='ready'
    finally:engine.dispose()


def test_named_object_historical_readback_remains_available_in_readonly_mode(stored):
    from aneos_core.data.acquisition import plan,outcome,build_bundle
    from aneos_core.data.models import NEOData,OrbitalElements
    path,writer,_,_=stored
    p=plan(['SBDB'])
    neo=NEOData('CONTROL',orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,inclination=1.49))
    outcomes=[outcome(s,'CONTROL',status='available' if s=='SBDB' else 'unavailable',
                      neo=neo if s=='SBDB' else None) for s in p['sources']]
    bundle=build_bundle('CONTROL',p,outcomes)
    repo=SQLEvidenceRepository(writer.engine);receipt=repo.commit_acquisition(bundle)
    engine=create_engine('sqlite:///file:'+str(path)+'?mode=ro&uri=true')
    try:
        readonly=SQLEvidenceRepository(engine)
        assert readonly.load_acquisition(bundle['query_id'])['receipt']==receipt
        with pytest.raises(EvidencePersistenceError):readonly.prepare_acquisition()
        assert readonly.load_acquisition(bundle['query_id'])['bundle']==bundle
    finally:engine.dispose()


@pytest.mark.parametrize('value',['2','0x1','ambiguous'])
def test_nonstandard_or_enabled_immutable_settings_fail_explicitly_before_poll(stored,value):
    path,_,_,transport=stored
    engine=create_engine('sqlite:///file:'+str(path)+'?immutable='+value+'&uri=true')
    try:
        with pytest.raises(EvidencePersistenceError):CADCandidateService(CandidateQueryRepository(engine)).poll('2026-11-12','2026-11-13',limit=1)
        transport.assert_not_called()
    finally:engine.dispose()
