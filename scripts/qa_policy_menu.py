#!/usr/bin/env python3
"""Capture actual policy-menu inspection, controls and unavailable prerequisites."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    directory = Path(parser.parse_args().output)
    directory.mkdir(parents=True, exist_ok=False)
    from aneos_menu import ANEOSMenu
    from aneos_core.analysis.policy_inspection import build_policy_snapshot
    from scripts.qa_menu_entry import capture

    snapshot = build_policy_snapshot()

    class View:
        def __init__(self, mode):
            self.mode = mode

        def _print_banner(self):
            ui = ANEOSMenu.__new__(ANEOSMenu)
            ui.console = self.console
            ui.wait_for_input = lambda: None
            if self.mode == "controls":
                ui._test_validated_detector()
            elif self.mode == "unavailable":
                with patch(
                    "aneos_core.analysis.policy_inspection.build_policy_snapshot",
                    side_effect=RuntimeError("synthetic missing prerequisite"),
                ):
                    ui.configure_analysis()
            else:
                # Freeze only this capture's inspection timestamp. Policy is actual.
                with patch(
                    "aneos_core.analysis.policy_inspection.build_policy_snapshot",
                    return_value=snapshot,
                ), patch("rich.prompt.Prompt.ask", return_value="0"):
                    ui.configure_analysis()

    metrics = {}
    for mode in ("inspection", "controls", "unavailable"):
        for width in (80, 120, 160):
            capture(View(mode), width, directory / f"{mode}-{width}")
            lines = (directory / f"{mode}-{width}.txt").read_text().splitlines()
            metrics[f"{mode}-{width}"] = {
                "lines": len(lines),
                "max_columns": max(map(len, lines), default=0),
            }
    (directory / "policy-snapshot.json").write_text(
        json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    )
    manifest = {
        "schema_version": "policy-menu-qa-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "policy_id": snapshot["policy_description"]["policy_id"],
        "frozen_inspection_timestamp": snapshot["created_at"],
        "metrics": metrics,
        "source_sha256": {
            n: sha256((ROOT / n).read_bytes()).hexdigest()
            for n in (
                "aneos_menu.py",
                "aneos_core/analysis/screening.py",
                "aneos_core/analysis/policy_inspection.py",
                "scripts/qa_policy_menu.py",
            )
        },
        "artifact_sha256": {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        },
        "limits": "Actual policy menu with frozen inspection timestamp, synthetic controls and injected missing prerequisite. No network, mutation, live-job audit, independent calibration or all-menu certification.",
    }
    (directory / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
