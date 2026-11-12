"""BC11 API endpoint tests — POST /analyze/network + GET status."""
import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")


@pytest.fixture(scope="module")
def client():
    from aneos_api.app import create_app
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def enqueue_only_worker(monkeypatch):
    """These tests cover submission/status shape, not live population execution."""
    from unittest.mock import Mock
    from aneos_api.endpoints import analysis
    worker = Mock()
    monkeypatch.setattr(analysis, '_run_network_analysis', worker)
    return worker


class TestNetworkEndpointValidation:
    def test_empty_designations_returns_422(self, client):
        r = client.post("/api/v1/analysis/analyze/network",
                        json={"designations": []})
        assert r.status_code == 422

    def test_too_many_designations_returns_422(self, client):
        r = client.post("/api/v1/analysis/analyze/network",
                        json={"designations": [f"D{i}" for i in range(501)]})
        assert r.status_code == 422

    def test_valid_request_returns_queued(self, client):
        r = client.post("/api/v1/analysis/analyze/network",
                        json={"designations": ["synthetic_test_neo"],
                              "clustering": False, "harmonics": False})
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"
        assert "status_url" in body
        status = client.get(body['status_url'])
        assert status.status_code == 200
        assert status.json()['network_sigma'] is None
        assert status.json()['combined_p_value'] is None
        assert status.json()['network_tier'] == 'UNAVAILABLE'


class TestNetworkStatusEndpoint:
    def test_unknown_job_returns_404(self, client):
        r = client.get("/api/v1/analysis/analyze/network/nonexistent_job_xyz/status")
        assert r.status_code == 404

    def test_status_schema_complete(self, client):
        post = client.post("/api/v1/analysis/analyze/network",
                           json={"designations": ["synthetic_test_neo"],
                                 "clustering": False, "harmonics": False})
        job_id = post.json()["job_id"]
        status = client.get(f"/api/v1/analysis/analyze/network/{job_id}/status")
        assert status.status_code == 200
        body = status.json()
        for key in ["job_id", "status", "network_sigma", "network_tier",
                    "combined_p_value", "clusters", "harmonic_signals",
                    "sub_module_p_values", "analysis_metadata"]:
            assert key in body, f"Missing response key: {key}"


@pytest.mark.parametrize('mode', ['positive','partial','unavailable','exception','wrong_identity'])
def test_actual_worker_preserves_report_and_acquisition_outcomes(client,monkeypatch,mode):
    from aneos_api.endpoints import analysis
    from aneos_api.schemas.network import NetworkRequest
    from aneos_core.data.models import NEOData,OrbitalElements
    from aneos_core.pattern_analysis.session import verify_network_report
    from aneos_core.data import fetcher
    n=NEOData(designation='A',orbital_elements=OrbitalElements(
        semi_major_axis=1.,eccentricity=.2,inclination=180.))
    class SyntheticSource:
        def fetch_neo_data(self,designation):
            if mode=='exception':raise RuntimeError('secret-provider-detail')
            import copy
            if mode=='wrong_identity':
                bad=copy.deepcopy(n);bad.designation='other';return bad
            if mode=='positive' or (mode=='partial' and designation=='A'):
                valid=copy.deepcopy(n);valid.designation=designation;return valid
            return None
    monkeypatch.setattr(fetcher,'DataFetcher',SyntheticSource)
    job='actual-worker-'+mode
    analysis._network_store[job]={'status':'queued'}
    # The enqueue-only seam is patched by the fixture; execute the real function saved below.
    REAL_WORKER(job,NetworkRequest(designations=['A','B'],rendezvous=True,correlation=True))
    response=client.get('/api/v1/analysis/analyze/network/'+job+'/status')
    assert response.status_code==200
    result=response.json()
    assert len(result['acquisition_outcomes'])==2
    assert result['network_sigma'] is None and result['combined_p_value'] is None
    report=verify_network_report(result['scientific_report'])
    assert report['modules']['rendezvous']['status']=='unavailable_not_implemented'
    if mode in ('positive','partial'):
        assert result['status']=='complete'
        assert report['object_assessments'][0]['assessment']['candidate'] is True
        assert result['analysis_metadata']['acquisition_status']==('available' if mode=='positive' else 'partial')
    else:
        assert result['status']=='error' and result['error']=='NO_AVAILABLE_INPUTS'
        assert result['analysis_metadata']['acquisition_status']=='unavailable'
    assert 'secret-provider-detail' not in response.text
    if mode=='wrong_identity':
        assert all(o['status']=='identity_mismatch' for o in result['acquisition_outcomes'])


from aneos_api.endpoints.analysis import _run_network_analysis as REAL_WORKER


def test_actual_worker_acquires_duplicate_identity_once_and_preserves_request_order(client,monkeypatch):
    from aneos_api.endpoints import analysis
    from aneos_api.schemas.network import NetworkRequest
    from aneos_core.data import fetcher
    from aneos_core.data.models import NEOData
    calls=[]
    class SyntheticSource:
        def fetch_neo_data(self,designation):
            calls.append(designation)
            return NEOData(designation=designation)
    monkeypatch.setattr(fetcher,'DataFetcher',SyntheticSource)
    job='duplicates-real-worker';analysis._network_store[job]={'status':'queued'}
    REAL_WORKER(job,NetworkRequest(designations=['A','B','A']))
    assert calls==['A','B']
    result=client.get('/api/v1/analysis/analyze/network/'+job+'/status').json()
    r=result['scientific_report']
    assert [n['designation'] for n in r['input_snapshots']]==['A','B','A']
    assert r['input_snapshots'][0]==r['input_snapshots'][2]
    assert [o['input_index'] for o in result['acquisition_outcomes']]==[0,1,2]
    assert all(a['assessment']['classification']=='INCONCLUSIVE' for a in r['object_assessments'])
