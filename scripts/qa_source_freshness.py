#!/usr/bin/env python3
"""Capture actual SQL-backed receipt review; synthetic inputs, no provider requests."""

import argparse
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def capture_all(directory):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from aneos_api.database import Base, EnrichedNEO, db_manager
    from aneos_api.acquisition_repository import SQLEvidenceRepository
    from aneos_api.source_freshness import inspect_freshness
    from aneos_core.data.acquisition import plan, outcome, build_bundle
    from aneos_core.data.models import NEOData, OrbitalElements
    from aneos_menu import ANEOSMenu
    from scripts.qa_menu_entry import capture
    from rich.cells import cell_len

    directory.mkdir(parents=True, exist_ok=False)
    clock = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)
    source_files = [
        "aneos_core/data/freshness.py",
        "aneos_api/source_freshness.py",
        "aneos_core/analysis/freshness_view.py",
        "aneos_menu.py",
        "scripts/qa_source_freshness.py",
        "scripts/qa_menu_entry.py",
    ]
    widths = {}
    with tempfile.TemporaryDirectory(prefix="aneos-freshness-qa-") as temporary:
        for mode in ("recent_adapter_reads", "stale_with_cooldown", "legacy_unknown"):
            engine = create_engine(
                "sqlite:///" + str(Path(temporary) / (mode + ".sqlite"))
            )
            sessions = sessionmaker(bind=engine)
            if mode == "legacy_unknown":
                Base.metadata.create_all(engine, tables=[EnrichedNEO.__table__])
                with sessions() as db:
                    db.add(
                        EnrichedNEO(
                            designation="SYNTHETIC",
                            nasa_sbdb_last_update=datetime(2000, 1, 1),
                        )
                    )
                    db.commit()
            else:
                repo = SQLEvidenceRepository(engine)
                p = plan([])
                stamp = (
                    "2026-11-13T11:30:00+00:00"
                    if mode == "recent_adapter_reads"
                    else "2000-01-01T00:00:00+00:00"
                )
                date = datetime.fromisoformat(stamp)
                outcomes = []
                for s in p["sources"]:
                    n = NEOData(
                        "SYNTHETIC",
                        orbital_elements=OrbitalElements(
                            semi_major_axis=1.23, eccentricity=0.2, inclination=1.49
                        ),
                        fetched_at=date,
                        created_at=date,
                        updated_at=date,
                        sources_used=[s],
                    )
                    outcomes.append(
                        outcome(
                            s,
                            "SYNTHETIC",
                            status="available",
                            neo=n,
                            started_at=stamp,
                            completed_at=stamp,
                        )
                    )
                repo.commit_acquisition(
                    build_bundle(
                        "SYNTHETIC",
                        p,
                        outcomes,
                        started_at=stamp,
                        software={
                            "application_version": "2.2.0.dev0",
                            "basis": "synthetic QA",
                        },
                    )
                )
                if mode == "stale_with_cooldown":
                    failed = [
                        outcome(
                            s,
                            "SYNTHETIC",
                            status="unavailable" if s == "Horizons" else "missing",
                            started_at="2026-11-13T11:30:00+00:00",
                            completed_at="2026-11-13T11:30:00+00:00",
                        )
                        for s in p["sources"]
                    ]
                    repo.commit_acquisition(
                        build_bundle(
                            "SYNTHETIC",
                            p,
                            failed,
                            started_at="2026-11-13T11:30:00+00:00",
                            software={
                                "application_version": "2.2.0.dev0",
                                "basis": "synthetic QA",
                            },
                        )
                    )
            menu = ANEOSMenu.__new__(ANEOSMenu)
            menu._qa_source_receipts = lambda: asyncio.run(
                menu._check_outdated_database_entries(["NASA_SBDB", "Horizons"], "1m")
            )

            def fixed_inspection(db, priority):
                return inspect_freshness(db, priority, checked_at=clock)

            with patch.object(db_manager, "get_db", sessions), patch(
                "aneos_api.source_freshness.inspect_freshness",
                side_effect=fixed_inspection,
            ):
                # First actual invocation also becomes the full machine-readable report.
                menu.console = None
                report = asyncio.run(
                    menu._check_outdated_database_entries(
                        ["NASA_SBDB", "Horizons"], "1m"
                    )
                )
                (directory / (mode + ".json")).write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                for width in (80, 120, 160):
                    name = f"{mode}-{width}"
                    capture(menu, width, directory / name, method="_qa_source_receipts")
                    lines = (directory / (name + ".txt")).read_text().splitlines()
                    max_width = max(map(cell_len, lines), default=0)
                    assert max_width <= width
                    widths[name] = {
                        "terminal_width": width,
                        "max_rendered_cells": max_width,
                    }
            engine.dispose()
    artifacts = {
        p.name: sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }
    manifest = {
        "schema_version": "source-freshness-qa-v1",
        "clock": clock.isoformat(),
        "cases": ["recent_adapter_reads", "stale_with_cooldown", "legacy_unknown"],
        "widths": widths,
        "source_files_sha256": {
            p: sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files
        },
        "artifact_sha256": artifacts,
        "limits": "Actual legacy receipt-inspection menu with disposable SQL and synthetic typed/legacy data. No live provider request, automatic scheduler, all-menu certification or scientific readiness.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    capture_all(Path(args.output))
