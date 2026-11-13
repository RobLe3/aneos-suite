"""Optional one-object live contract, never part of the offline quality gates."""

import logging
import json
import os
from pathlib import Path
from hashlib import sha256
import pytest
from aneos_core.config.settings import APIConfig
from aneos_core.data.sources.sbdb import SBDBSource
from aneos_core.data.fetcher import DataFetcher
from aneos_core.data.models import NEOData
from aneos_core.data.physical_evidence import verify_physical_evidence
from aneos_core.analysis.object_review import build_object_review, verify_object_review


@pytest.mark.network
def test_real_ky26_physical_evidence_survives_acquisition_and_review(tmp_path):
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.logger = logging.getLogger(__name__)

    class CapturedSBDB(SBDBSource):
        async def _http_get(self, *args, **kwargs):
            result = await super()._http_get(*args, **kwargs)
            self.captured = result
            return result

    source = CapturedSBDB(APIConfig())
    neo = fetcher._fetch_from_source("SBDB", source, "1998 KY26")
    assert (
        neo is not None
    ), "Live provider unavailable or contract changed; inspect, never fabricate."
    evidence = verify_physical_evidence(neo.evidence_metadata["physical_evidence"])
    assert evidence["context"]["response_sha256"]
    assert evidence["context"]["retrieved_at"]
    assert any(r["native"]["name"] == "diameter" for r in evidence["records"])
    assert neo.orbital_elements.native_epoch is not None
    assert neo.orbital_elements.native_epoch.native_value == str(
        evidence["context"]["container_orbit_epoch_jd_tdb"]
    )
    assert neo.orbital_elements.native_epoch.time_scale == "TDB"
    assert neo.orbital_elements.epoch is None  # No invented UTC conversion.
    restored = NEOData.from_dict(neo.to_dict())
    review = verify_object_review(build_object_review(restored))
    assert review["scientific_status"] == "not_independently_validated"
    directory = Path(
        os.environ.get("ANEOS_LIVE_EVIDENCE_DIR", str(tmp_path / "live-contract"))
    )
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    raw = json.dumps(
        source.captured, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert sha256(raw.encode()).hexdigest() == evidence["context"]["response_sha256"]
    (directory / "canonical-response.json").write_text(raw)
    (directory / "neo-snapshot.json").write_text(
        json.dumps(neo.to_dict(), indent=2, allow_nan=False)
    )
    (directory / "object-review.json").write_text(
        json.dumps(review, indent=2, allow_nan=False)
    )
    files = {p.name: sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    from aneos_core.analysis.reporting import _software_metadata

    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "software": _software_metadata(),
                "retrieved_at": evidence["context"]["retrieved_at"],
                "files_sha256": files,
                "limitations": "Optional provider contract only, not scientific validation. Response is canonical JSON, not original HTTP bytes.",
            },
            indent=2,
        )
    )


@pytest.mark.network
def test_real_horizons_history_rows_have_native_epochs(tmp_path):
    from datetime import datetime
    from aneos_core.data.sources.horizons import HorizonsSource
    from aneos_core.data.epoch_evidence import NativeEpoch

    rows = HorizonsSource(APIConfig()).fetch_orbital_history(
        "1998 KY26", years=2
    )
    assert rows, "Live Horizons unavailable or returned no usable bound history rows"
    assert all(
        NativeEpoch.from_dict(row["native_epoch"]).time_scale == "TDB" for row in rows
    )
    assert all({"a", "e", "i", "epoch", "native_epoch"}.issubset(row) for row in rows)
    (tmp_path / "computed-native-history.json").write_text(json.dumps(rows, indent=2))
