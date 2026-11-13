"""Optional live HTTP transport contract; no startup/auth/scientific certification."""

import json
import os
from hashlib import sha256
from pathlib import Path
from importlib.metadata import version
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from aneos_api.endpoints import analysis


@pytest.mark.network
def test_real_horizons_provider_through_history_http_preserves_query_and_epochs(
    tmp_path, monkeypatch
):
    import aneos_core.data.sources.horizons as source_module

    original = source_module.provider_get
    responses = []

    def capture(*args, **kwargs):
        response = original(*args, **kwargs)
        responses.append(response.json())
        return response

    monkeypatch.setattr(source_module, "provider_get", capture)
    app = FastAPI()
    app.include_router(analysis.router, prefix="/api/v1/analysis")
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/analysis/history", params={"designation": "1998 KY26", "years": 2}
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["epoch_status"] == "native_epochs_preserved"
    assert body["time_axis_status"] == "native_single_scale_numeric_axis"
    assert body["query_evidence"]["status"] == "available"
    assert all(p["native_epoch"]["time_scale"] == "TDB" for p in body["points"])
    assert body["scientific_status"] == "not_independently_validated"
    assert len(responses) == 1, "No hidden retries or extra reference-object fetches"
    directory = Path(
        os.environ.get(
            "ANEOS_API_LIVE_EVIDENCE_DIR", str(tmp_path / "live-api-history")
        )
    )
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    raw = json.dumps(
        responses[0], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    text = responses[0]["result"]
    assert (
        sha256(text.encode()).hexdigest()
        == body["query_evidence"]["result_text_sha256"]
    )
    (directory / "canonical-provider-response.json").write_text(raw)
    (directory / "http-response.json").write_text(
        json.dumps(body, indent=2, allow_nan=False)
    )
    from aneos_core.analysis.reporting import _software_metadata

    software = _software_metadata()
    software["package_versions"].update(
        {name: version(name) for name in ("fastapi", "starlette", "httpx")}
    )
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        name: sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "aneos_api/app.py",
            "aneos_api/endpoints/analysis.py",
            "aneos_api/schemas/history.py",
            "aneos_api/schemas/detection.py",
            "aneos_core/data/sources/horizons.py",
            "aneos_core/data/epoch_evidence.py",
        )
    }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "software": software,
                "source_sha256": source_hashes,
                "retrieved_at": body["query_evidence"]["retrieved_at"],
                "files_sha256": {
                    p.name: sha256(p.read_bytes()).hexdigest()
                    for p in directory.iterdir()
                },
                "limitations": "One live public-router transport check, not startup/auth isolation, independent accuracy, full observational coverage or scientific validation.",
            },
            indent=2,
        )
    )
