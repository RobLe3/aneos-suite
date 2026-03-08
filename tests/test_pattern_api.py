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
