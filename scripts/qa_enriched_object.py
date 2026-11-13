#!/usr/bin/env python3
"""Capture actual SQL-backed object detail, using synthetic data only."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    directory = Path(parser.parse_args().output)
    directory.mkdir(parents=True, exist_ok=False)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from aneos_api.database import EnrichedNEOService
    from aneos_api.enrichment_store import initialize_schema

    # The explicit additive initializer creates a new synthetic DB, not a user database.
    temporary = tempfile.TemporaryDirectory(prefix="aneos-synthetic-qa-db-")
    engine = create_engine(
        "sqlite:///" + str(Path(temporary.name) / "synthetic.sqlite")
    )
    initialize_schema(engine, include_all=True)
    sessions = sessionmaker(bind=engine)
    rows = [
        {
            "designation": "SYNTHETIC stored object",
            "data_source": "SBDB",
            "orbital_elements": {"a": 1.2, "e": 0.2, "i": 1.49},
            "physical_properties": {
                "diameter_km": 0.011,
                "reference_source": "SYNTHETIC primary",
            },
        },
        {
            "designation": "SYNTHETIC stored object",
            "data_source": "MPC",
            "physical_properties": {
                "diameter_km": 0.02,
                "diameter_uncertainty": 0.003,
                "albedo": 0.52,
                "reference_source": "SYNTHETIC alternative",
            },
            "artificial_probability": None,
        },
    ]
    with sessions() as db:
        result = EnrichedNEOService(db).enrich_neo_data(rows, "synthetic-capture")
        if result["commit_status"] != "committed":
            raise RuntimeError("Synthetic enrichment failed")
    from aneos_menu import ANEOSMenu
    from scripts.qa_menu_entry import capture

    class View:
        def _print_banner(self):
            menu = ANEOSMenu.__new__(ANEOSMenu)
            menu.console = self.console
            menu.console.input = lambda *args: ""
            with sessions() as db:
                menu._show_enriched_neo_details(
                    EnrichedNEOService(db), "SYNTHETIC stored object"
                )

    metrics = {}
    for width in (80, 120, 160):
        capture(View(), width, directory / f"detail-{width}")
        lines = (directory / f"detail-{width}.txt").read_text().splitlines()
        metrics[str(width)] = {"lines": len(lines), "max_columns": max(map(len, lines))}
    with sessions() as db:
        EnrichedNEOService(db).export_enriched_json(
            "SYNTHETIC stored object", directory / "object-evidence.json"
        )
    engine.dispose()
    temporary.cleanup()  # Only this generator's disposable synthetic database.
    hashes = {
        p.name: sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }
    manifest = {
        "schema_version": "sql-enriched-detail-qa-v1",
        "synthetic": True,
        "metrics": metrics,
        "artifact_sha256": hashes,
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "aneos_api/enrichment_store.py",
                "aneos_api/database.py",
                "aneos_core/data/enrichment.py",
                "aneos_core/data/reconciliation.py",
                "aneos_menu.py",
                "scripts/qa_enriched_object.py",
            )
        },
        "limits": "Synthetic file-SQLite and actual Rich detail only; no live providers, full acquisition polling, scientific validation, DB-admin tamper guarantee or all-menu certification.",
    }
    (directory / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
