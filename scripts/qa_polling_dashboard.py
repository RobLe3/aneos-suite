#!/usr/bin/env python3
"""Actual dashboard/browser/detail renders over explicit synthetic controls, no providers."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json, math, re, sys
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def capture_all(directory):
    from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties
    from aneos_core.analysis.artificial_neo_dashboard import (
        ArtificialNEODashboard,
        classify_record,
    )
    from aneos_menu import ANEOSMenu
    from scripts.qa_menu_entry import capture
    from rich.cells import cell_len

    directory.mkdir(parents=True, exist_ok=False)
    stamp = datetime(2026, 11, 13, 8, tzinfo=timezone.utc)
    positive = NEOData(
        "SYNTHETIC_FOLLOW_UP",
        orbital_elements=OrbitalElements(
            semi_major_axis=2.5, eccentricity=0.9, inclination=170
        ),
        physical_properties=PhysicalProperties(
            diameter_km=0.011, albedo=0.52, absolute_magnitude_h=25.9
        ),
        fetched_at=stamp,
        created_at=stamp,
        updated_at=stamp,
        sources_used=["SBDB"],
    )
    partial = NEOData(
        "SYNTHETIC_PARTIAL",
        orbital_elements=OrbitalElements(semi_major_axis=1.23),
        fetched_at=stamp,
        created_at=stamp,
        updated_at=stamp,
        sources_used=["SBDB"],
    )
    cases = {
        "typed_follow_up": positive,
        "typed_partial": partial,
        "legacy_unverified": {
            "designation": "SYNTHETIC_LEGACY",
            "artificial_probability": 0.99,
            "confidence": 0.95,
            "classification": "ARTIFICIAL",
        },
    }
    widths = {}
    records = {}
    for mode, raw in cases.items():
        item = classify_record(raw)
        d = ArtificialNEODashboard()
        d.classifications = [item]
        menu = ANEOSMenu.__new__(ANEOSMenu)
        menu.wait_for_input = Mock()
        menu.display_header = Mock()

        def dashboard():
            d.console = menu.console
            d.display_dashboard(save_results=False)

        def browser():
            with patch("rich.prompt.Prompt.ask", return_value="Q"):
                menu._norton_style_browser([item], "Synthetic fixture — " + mode)

        menu._qa_dashboard = dashboard
        menu._qa_browser = browser
        menu._qa_detail = lambda: menu._show_detailed_neo_view(item)
        records[mode] = item.to_payload()
        (directory / (mode + ".json")).write_text(
            json.dumps(records[mode], indent=2) + "\n"
        )
        for view in ("dashboard", "browser", "detail"):
            for width in (80, 120, 160):
                name = f"{mode}-{view}-{width}"
                capture(menu, width, directory / name, method="_qa_" + view)
                lines = (directory / (name + ".txt")).read_text().splitlines()
                max_width = max(map(cell_len, lines), default=0)
                assert max_width <= width
                svg = (directory / (name + ".svg")).read_text()
                geometry = float(re.search(r'viewBox="0 0 ([\d.]+)', svg)[1])
                assert geometry == math.ceil(width * 12.2 + 16) + 2
                widths[name] = {
                    "requested_columns": width,
                    "actual_text_max_cells": max_width,
                    "svg_viewbox_width": geometry,
                }
    sources = [
        "aneos_core/analysis/artificial_neo_dashboard.py",
        "aneos_core/analysis/dashboard_browser.py",
        "aneos_core/analysis/dashboard_view.py",
        "aneos_menu.py",
        "artificial_neo_dashboard_launcher.py",
        "scripts/qa_menu_entry.py",
        "scripts/qa_polling_dashboard.py",
    ]
    manifest = {
        "schema_version": "polling-dashboard-qa-v1",
        "cases": list(cases),
        "views": ["dashboard", "browser", "detail"],
        "widths": widths,
        "source_files_sha256": {
            p: sha256((ROOT / p).read_bytes()).hexdigest() for p in sources
        },
        "artifact_sha256": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        },
        "limits": "Synthetic typed computational positives and incomplete/legacy controls. Actual dashboard/browser/detail render paths, not provider acquisition, source authenticity, calibrated significance, physical model fitting or full-menu certification. Local screening receipt timestamps are generated during capture, not source observation dates.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    capture_all(Path(parser.parse_args().output))
