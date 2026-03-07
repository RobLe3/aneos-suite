"""Preflight health-check utilities for aNEOS entry points."""

import os
from pathlib import Path
from typing import Dict, Any


def preflight_check() -> Dict[str, Dict[str, str]]:
    """
    Run 8 preflight checks and return a status dict.

    Keys: pipeline, analysis, sbdb, neodys, mpc, horizons, cache_dir, results_dir
    Each value: {"status": "ok"|"error", "detail": str}
    """
    results: Dict[str, Dict[str, str]] = {}

    # --- Component checks ---
    try:
        import aneos_core.pipeline.automatic_review_pipeline  # noqa: F401
        results["pipeline"] = {"status": "ok", "detail": "automatic_review_pipeline imported"}
    except Exception as exc:
        results["pipeline"] = {"status": "error", "detail": str(exc)}

    try:
        import aneos_core.analysis.pipeline  # noqa: F401
        results["analysis"] = {"status": "ok", "detail": "analysis.pipeline imported"}
    except Exception as exc:
        results["analysis"] = {"status": "error", "detail": str(exc)}

    # --- API reachability checks ---
    import requests  # type: ignore

    api_targets = {
        "sbdb": "https://ssd-api.jpl.nasa.gov/sbdb.api",
        "neodys": "https://newton.spacedys.com/neodys/api/",
        "mpc": "https://www.minorplanetcenter.net/",
        "horizons": "https://ssd.jpl.nasa.gov/api/horizons.api",
    }

    for name, url in api_targets.items():
        try:
            resp = requests.head(url, timeout=3)
            results[name] = {"status": "ok", "detail": f"HTTP {resp.status_code}"}
        except Exception as exc:
            results[name] = {"status": "error", "detail": str(exc)}

    # --- Directory checks ---
    cache_dir = Path("neo_data/cache")
    if cache_dir.exists() and os.access(cache_dir, os.W_OK):
        results["cache_dir"] = {"status": "ok", "detail": str(cache_dir)}
    else:
        results["cache_dir"] = {
            "status": "error",
            "detail": f"{cache_dir} missing or not writable",
        }

    results_dir = Path("neo_data/pipeline_results")
    if results_dir.exists() and os.access(results_dir, os.W_OK):
        results["results_dir"] = {"status": "ok", "detail": str(results_dir)}
    else:
        results["results_dir"] = {
            "status": "error",
            "detail": f"{results_dir} missing or not writable",
        }

    return results


def print_preflight_table(results: Dict[str, Dict[str, str]]) -> None:
    """Render preflight results as a Rich table, with plain-text fallback."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="aNEOS Preflight Check")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Detail", style="dim")

        for check, info in results.items():
            status = info["status"]
            icon = "[green]PASS[/green]" if status == "ok" else "[red]FAIL[/red]"
            table.add_row(check, icon, info["detail"])

        console.print(table)

    except ImportError:
        # Plain-text fallback
        print(f"{'Check':<18} {'Status':<6} Detail")
        print("-" * 60)
        for check, info in results.items():
            status = "PASS" if info["status"] == "ok" else "FAIL"
            print(f"{check:<18} {status:<6} {info['detail']}")
