#!/usr/bin/env python3
"""Capture actual public query→SQL/export→object review with synthetic OEF packets."""

import argparse
from pathlib import Path
from hashlib import sha256
import json
import sys
import tempfile
from unittest.mock import Mock, patch

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
    from aneos_core.data.sources.neodys import NEODySSource
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from aneos_api.database import EnrichedNEOService
    from aneos_core.analysis.object_review import (
        build_object_review,
        render_object_review,
        object_review_markdown,
    )
    from scripts.qa_menu_entry import capture

    metrics = {}
    with tempfile.TemporaryDirectory(prefix="aneos-neodys-qa-") as temporary:
        for mode in ("available", "ambiguous", "missing_photometry"):
            text = "format = 'OEF2.0'\nrectype = 'ML'\nrefsys = ECLM J2000\nEND_OF_HEADER\n1998KY26\n EQU 1.23 .1 .1 .01 .01 20\n MJD 55869.089613878 TDT\n"
            if mode != "missing_photometry":
                text += " MAG 25.594 0.150\n"
            if mode == "ambiguous":
                text += " MAG 26.1 0.150\n"
            text += " COV 1 2 3\n NEW 42\n"
            response = Mock(status_code=200, text=text)
            engine = create_engine(
                "sqlite:///" + str(Path(temporary) / (mode + ".sqlite"))
            )
            repo = SQLEvidenceRepository(engine)
            f = DataFetcher(
                cache_manager=CacheManager(Path(temporary) / (mode + "-cache")),
                evidence_repository=repo,
            )
            f.sources = {"NEODyS": NEODySSource()}
            f._fetch_close_approaches = Mock(return_value=[])
            with patch("requests.get", return_value=response):
                result = f.fetch_neo_data("1998 KY26")
            review = build_object_review(result)
            (directory / (mode + "-review.json")).write_text(
                json.dumps(review, indent=2, allow_nan=False) + "\n"
            )
            (directory / (mode + "-review.md")).write_text(
                object_review_markdown(review)
            )

            class View:
                def _print_banner(self):
                    from rich.text import Text

                    self.console.print(
                        Text(
                            "SYNTHETIC OEF SOURCE — NOT A LIVE KY26 ASSESSMENT",
                            style="bold yellow",
                        )
                    )
                    render_object_review(review, console=self.console)

            for width in (80, 120, 160):
                capture(View(), width, directory / f"{mode}-{width}")
                lines = (directory / f"{mode}-{width}.txt").read_text().splitlines()
                metrics[f"{mode}-{width}"] = {
                    "lines": len(lines),
                    "max_columns": max(map(len, lines), default=0),
                }
            with repo.sessions() as db:
                EnrichedNEOService(db).export_enriched_json(
                    "1998 KY26", directory / (mode + "-object.json")
                )
            engine.dispose()
    sources = (
        "aneos_core/data/neodys_evidence.py",
        "aneos_core/data/physical_evidence.py",
        "aneos_core/data/sources/neodys.py",
        "aneos_core/data/fetcher.py",
        "aneos_core/analysis/object_review.py",
        "aneos_core/analysis/reporting.py",
        "scripts/qa_neodys_evidence.py",
    )
    manifest = {
        "schema_version": "neodys-evidence-qa-v1",
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
        "limits": "Actual public-query, disposable file-SQLite/export and object review paths; synthetic OEF packets only, not live-provider, all-menu, covariance-fit or scientific calibration certification.",
    }
    (directory / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
