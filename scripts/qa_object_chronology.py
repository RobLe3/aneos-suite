#!/usr/bin/env python3
"""Capture actual stored-object date views using synthetic provider data and own SQL."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, AsyncMock

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
    from aneos_core.data.sources.sbdb import SBDBSource
    from aneos_core.config.settings import APIConfig
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from aneos_api.database import EnrichedNEOService, EnrichedNEO, Base
    from aneos_menu import ANEOSMenu
    from scripts.qa_menu_entry import capture

    metrics = {}
    with tempfile.TemporaryDirectory(prefix="aneos-chronology-qa-") as temporary:
        for mode in ("sourced_discovery", "partial_observation", "legacy_unverified"):
            engine = create_engine(
                "sqlite:///" + str(Path(temporary) / (mode + ".sqlite"))
            )
            repo = SQLEvidenceRepository(engine)
            if mode == "legacy_unverified":
                Base.metadata.create_all(engine, tables=[EnrichedNEO.__table__])
                with repo.sessions() as db:
                    db.add(
                        EnrichedNEO(
                            designation="SYNTHETIC",
                            first_discovered=datetime(2020, 1, 2, 3, 4),
                            last_updated=datetime(2021, 2, 3, 4, 5),
                        )
                    )
                    db.commit()
            else:
                payload = {
                    "signature": {"version": "1.3"},
                    "object": {
                        "des": "SYNTHETIC",
                        "kind": "an" if mode == "sourced_discovery" else "au",
                    },
                    "orbit": {
                        "orbit_id": "SYNTHETIC_SOLUTION",
                        "epoch": "2461200.50000000001",
                        "soln_date": "2024-May-07 06:44:30",
                        "elements": [
                            {"name": "a", "value": "1.23"},
                            {"name": "e", "value": ".2"},
                            {"name": "i", "value": "1.49"},
                        ],
                        "first_obs": (
                            "1998-05-30"
                            if mode == "sourced_discovery"
                            else "1919-??-??"
                        ),
                        "last_obs": "2024-05-06",
                        "covariance": {
                            "epoch": "2457098.500000000001",
                            "labels": ["e", "q", "tp", "node", "peri", "i"],
                            "data": [
                                [1e-8 if i == j else 0 for j in range(6)]
                                for i in range(6)
                            ],
                        },
                    },
                    "phys_par": [],
                }
                if mode == "sourced_discovery":
                    payload["discovery"] = {
                        "date": "1998-May-31",
                        "ref": "SYNTHETIC_REF",
                        "who": "SYNTHETIC_GROUP",
                    }
                source = SBDBSource(APIConfig())
                source._http_get = AsyncMock(return_value=payload)
                f = DataFetcher(
                    cache_manager=CacheManager(Path(temporary) / (mode + "-cache")),
                    evidence_repository=repo,
                )
                f.sources = {"SBDB": source}
                f._fetch_close_approaches = Mock(return_value=[])
                f.fetch_neo_data("SYNTHETIC")

            class View:
                def _print_banner(self):
                    from rich.text import Text

                    self.console.print(
                        Text(
                            "SYNTHETIC CHRONOLOGY QA — NOT LIVE OBSERVATIONS",
                            style="bold yellow",
                        )
                    )
                    menu = ANEOSMenu.__new__(ANEOSMenu)
                    menu.console = self.console
                    menu.console.input = lambda *a: ""
                    with repo.sessions() as db:
                        menu._show_enriched_neo_details(
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
                    "SYNTHETIC", directory / (mode + "-object.json")
                )
            engine.dispose()
    sources = (
        "aneos_core/data/chronology.py",
        "aneos_core/analysis/chronology_view.py",
        "aneos_core/data/models.py",
        "aneos_core/data/sources/sbdb.py",
        "aneos_core/data/fetcher.py",
        "aneos_core/analysis/object_review.py",
        "aneos_core/analysis/reporting.py",
        "aneos_api/enrichment_store.py",
        "aneos_api/database.py",
        "aneos_menu.py",
        "scripts/qa_object_chronology.py",
    )
    manifest = {
        "schema_version": "object-chronology-qa-v1",
        "synthetic": True,
        "metrics": metrics,
        "source_sha256": {
            s: sha256((ROOT / s).read_bytes()).hexdigest() for s in sources
        },
        "artifact_sha256": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        },
        "limits": "Actual public fetch/SQL/export and stored-object menu paths with synthetic sourced/partial/legacy dates; no live observations, independent source certification, time-scale conversion or all-menu certification.",
    }
    (directory / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
