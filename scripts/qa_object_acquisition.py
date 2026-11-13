#!/usr/bin/env python3
"""Render actual public-query→SQL→stored-detail flows with synthetic sources only."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys, tempfile
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    directory = Path(parser.parse_args().output)
    directory.mkdir(parents=True, exist_ok=False)
    from sqlalchemy import create_engine
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.cache import CacheManager
    from aneos_core.data.sources.cad import (
        parse_cad_payload,
        close_approach_from_record,
        CloseApproachQuery,
    )
    from aneos_core.data.epoch_evidence import NativeEpoch
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from aneos_api.database import EnrichedNEOService
    from aneos_menu import ANEOSMenu
    from scripts.qa_menu_entry import capture

    class Source:
        def __init__(self, body):
            self.body = body

        def fetch_orbital_elements(self, d):
            if self.body is None:
                raise TimeoutError("synthetic outage")
            return self.body

    with tempfile.TemporaryDirectory(prefix="aneos-acquisition-qa-") as temporary:
        metrics = {}
        for mode in ("complete", "partial", "unavailable"):
            engine = create_engine(
                "sqlite:///" + str(Path(temporary) / (mode + ".sqlite"))
            )
            repo = SQLEvidenceRepository(engine)
            f = DataFetcher(
                cache_manager=CacheManager(Path(temporary) / (mode + "-cache")),
                evidence_repository=repo,
            )
            basic = {
                "semi_major_axis": 1.23,
                "eccentricity": 0.2,
                "inclination": 1.49,
                "_evidence_metadata": {"synthetic": True},
            }
            full = dict(
                basic,
                ra_of_ascending_node=1.0,
                arg_of_periapsis=2.0,
                mean_anomaly=3.0,
                native_epoch=NativeEpoch("2460000.5", "JD", "TDB", "SBDB").to_dict(),
                first_observation_date="2026-Nov-01",
                last_observation_date="2026-Nov-08",
                _physical={
                    "diameter": 0.011,
                    "absolute_magnitude_h": 25.9,
                    "albedo": 0.52,
                    "rot_per": 0.089,
                    "spectral_type": "Xe",
                },
            )
            if mode == "complete":
                f.sources = {
                    name: Source(full if name == "SBDB" else basic)
                    for name in ("SBDB", "NEODyS", "MPC", "Horizons")
                }
            elif mode == "partial":
                f.sources = {
                    "SBDB": Source(basic),
                    "MPC": Source(
                        {
                            "_physical": {"absolute_magnitude_h": 25.9},
                            "_evidence_metadata": {"synthetic": True},
                        }
                    ),
                    "Horizons": Source(None),
                }
            else:
                f.sources = {
                    name: Source(None) for name in ("SBDB", "NEODyS", "MPC", "Horizons")
                }
            native = {
                "signature": {"source": "SYNTHETIC QA", "version": "1.5"},
                "count": 1 if mode == "complete" else 0,
                "total": 1 if mode == "complete" else 0,
                "fields": [
                    "des",
                    "orbit_id",
                    "jd",
                    "cd",
                    "dist",
                    "dist_min",
                    "dist_max",
                    "v_rel",
                    "v_inf",
                    "t_sigma_f",
                    "h",
                ],
                "data": (
                    [
                        [
                            "SYNTHETIC",
                            "synthetic-orbit",
                            "2460000.5",
                            "2023-Feb-25 00:00",
                            ".1",
                            ".09",
                            ".11",
                            "1.2",
                            "1.1",
                            "00:05",
                            "25.9",
                        ]
                    ]
                    if mode == "complete"
                    else []
                ),
            }
            parsed = parse_cad_payload(
                native,
                query={"des": "SYNTHETIC", "limit": 10},
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            )
            f._fetch_close_approaches = (
                Mock(side_effect=TimeoutError("synthetic outage"))
                if mode == "unavailable"
                else Mock(
                    return_value=CloseApproachQuery(
                        [close_approach_from_record(r) for r in parsed["records"]],
                        parsed["metadata"],
                    )
                )
            )
            from aneos_core.utils.errors import DataSourceUnavailableError

            try:
                f.fetch_neo_data("SYNTHETIC")
            except DataSourceUnavailableError:
                if mode != "unavailable":
                    raise

            class View:
                def _print_banner(self):
                    ui = ANEOSMenu.__new__(ANEOSMenu)
                    ui.console = self.console
                    ui.console.input = lambda *a: ""
                    with repo.sessions() as db:
                        ui._show_enriched_neo_details(
                            EnrichedNEOService(db), "SYNTHETIC"
                        )

            for width in (80, 120, 160):
                capture(View(), width, directory / f"{mode}-{width}")
                lines = (directory / f"{mode}-{width}.txt").read_text().splitlines()
                metrics[f"{mode}-{width}"] = {
                    "lines": len(lines),
                    "max_columns": max(map(len, lines), default=0),
                }
            with repo.sessions() as db:
                EnrichedNEOService(db).export_enriched_json(
                    "SYNTHETIC", directory / f"{mode}-object.json"
                )
            engine.dispose()
    manifest = {
        "schema_version": "object-acquisition-menu-qa-v1",
        "synthetic": True,
        "metrics": metrics,
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "aneos_core/data/acquisition.py",
                "aneos_core/data/fetcher.py",
                "aneos_core/data/models.py",
                "aneos_core/data/cache.py",
                "aneos_core/data/sources/mpc.py",
                "aneos_core/data/sources/sbdb.py",
                "aneos_core/data/sources/horizons.py",
                "aneos_core/data/sources/neodys.py",
                "aneos_core/analysis/reporting.py",
                "aneos_core/analysis/acquisition_view.py",
                "aneos_api/acquisition_repository.py",
                "aneos_api/enrichment_store.py",
                "aneos_api/database.py",
                "aneos_menu.py",
                "scripts/qa_object_acquisition.py",
            )
        },
        "artifact_sha256": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        },
        "limits": "Actual public query and real disposable file-SQLite, synthetic source packets only. Complete means this named profile/field presence, never full science, authentic sources, all-source eligibility, independent calibration or all-menu certification.",
    }
    (directory / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
