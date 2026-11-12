"""BC11 compatibility routes, now authenticated persistent execution."""
import pytest
from tests.test_durable_network_api import api,headers,wait,URL,Source


@pytest.mark.parametrize('designations',[[],[f'D{i}' for i in range(501)]])
def test_network_endpoint_designation_limits(api,designations):
    client,*_=api
    assert client.post(URL,json={'designations':designations},headers=headers()).status_code==422


def test_valid_request_is_durable_and_returns_registered_status_url(api):
    client,*_=api
    post=client.post(URL,json={'designations':['A'],'harmonics':False,'clustering':False},headers=headers())
    assert post.status_code==200
    result=wait(client,post.json())
    assert result['execution_status']=='completed'
    assert result['network_sigma'] is None and result['combined_p_value'] is None
    assert result['network_tier']=='UNAVAILABLE'


def test_unknown_job_owner_filtered_404(api):
    client,*_=api
    assert client.get(URL+'/unknown/status',headers=headers()).status_code==404


def test_persistent_status_schema_complete(api):
    client,*_=api
    post=client.post(URL,json={'designations':['A'],'harmonics':False},headers=headers()).json()
    result=wait(client,post)
    for key in ('job_id','status','network_sigma','network_tier','combined_p_value','clusters',
                'harmonic_signals','sub_module_p_values','analysis_metadata','execution_result'):
        assert key in result


@pytest.mark.parametrize('mode',['positive','partial','unavailable','exception','wrong_identity'])
def test_real_sql_worker_preserves_missing_outage_identity_and_candidate_semantics(api,mode):
    client,runtime,*_=api
    calls=[]
    class Synthetic(Source):
        def fetch_neo_data(self,d):
            calls.append(d)
            if mode=='exception':raise RuntimeError('secret-provider-detail')
            if mode=='wrong_identity':return super().fetch_neo_data('other')
            return super().fetch_neo_data(d) if mode=='positive' or (mode=='partial' and d=='A') else None
    runtime.worker.handler.source_factory=Synthetic
    # Keep real bounded retry behavior but shorten only this test's backoff.
    runtime.worker.retry_seconds=.001
    post=client.post(URL,json={'designations':['A','B'],'harmonics':False,'rendezvous':True},headers=headers()).json()
    result=wait(client,post)
    assert 'secret-provider-detail' not in str(result)
    assert result['network_sigma'] is None and result['combined_p_value'] is None
    if mode=='positive':
        assert result['execution_status']=='completed'
        assert result['scientific_report']['object_assessments'][0]['assessment']['candidate']
        assert result['scientific_report']['modules']['rendezvous']['status']=='unavailable_not_implemented'
    else:
        assert result['execution_status']=='failed' and result['scientific_report'] is None
        assert result['error']==('INVALID_EVIDENCE' if mode=='wrong_identity' else 'ACQUISITION_UNAVAILABLE')
        if mode=='partial':
            assert calls.count('A')==1
            assert result['checkpoint_count']==1


def test_duplicate_acquisition_retains_request_order_in_sql_report(api):
    client,runtime,*_=api
    calls=[]
    class Count(Source):
        def fetch_neo_data(self,d):calls.append(d);return super().fetch_neo_data(d)
    runtime.worker.handler.source_factory=Count
    post=client.post(URL,json={'designations':['A','B','A'],'harmonics':False},headers=headers()).json()
    result=wait(client,post)
    assert calls==['A','B']
    assert [n['designation'] for n in result['scientific_report']['input_snapshots']]==['A','B','A']


def test_old_daemon_entrypoint_controlled_deprecation():
    from aneos_api.endpoints.analysis import _run_network_analysis
    with pytest.raises(NotImplementedError,match='SQL NetworkWorker'):_run_network_analysis('old',{})


@pytest.mark.asyncio
@pytest.mark.parametrize('name',['start_network_analysis','get_network_status'])
async def test_legacy_direct_route_functions_controlled_deprecation(name):
    from aneos_api.endpoints import analysis
    with pytest.raises(NotImplementedError,match='HTTP route'):await getattr(analysis,name)()
