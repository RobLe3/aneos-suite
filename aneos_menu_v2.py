"""
ANEOSMenuV2 — Redesigned interactive menu for the aNEOS platform.

Covers the full feature set of the legacy 121-option menu, organized into
12 real options across 4 groups. No theater: every option invokes real backend
code with honest result labelling.

GROUP A — Detection & Classification
  1  Detect NEO (single)           — validated σ-5 artificial-NEO classifier
  2  Multi-Evidence Analysis        — same detector + per-source evidence breakdown
  3  Batch Detection               — concurrent file-driven classification
  4  Orbital History Analysis      — course corrections / smoking-gun via Horizons

GROUP B — Impact Assessment
  5  Impact Probability            — planetary-defence collision calculation
  6  Close Approach History        — historical Earth close-approaches (CAD API)

GROUP C — Monitoring & Polling
  7  Live Pipeline Dashboard       — 200-year historical polling via pipeline
  8  Population Pattern Analysis   — BC11 network-sigma clustering / harmonics

GROUP D — Results & Reports
  9  Browse Results                — in-session + DB-persisted results
 10  Export Results                — JSON / CSV via Exporter

GROUP E — System
 11  Health Check                  — component imports + API reachability
 12  Start API Server              — uvicorn :8000 subprocess
 13  Detection Analytics           — session stats, σ-tier breakdown, JSON export
 14  Scientific Help               — methodology + σ thresholds from docs

  0  Exit

Design principles
-----------------
- Every option calls real, tested backend code (no fake progress bars)
- σ < 2 → INCONCLUSIVE (not "natural")
- Data failures are hard errors shown to the user, not silent fallbacks
- Results from options 1–5 are persisted to DB in background threads
- Inherits only stateless UI helpers from ANEOSMenuBase
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aneos_menu_base import ANEOSMenuBase


class ANEOSMenuV2(ANEOSMenuBase):
    """Full-featured 14-option menu for the aNEOS research platform."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self._detection_results: Dict[str, Any] = {}
        self._impact_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main event loop — runs until user selects 0."""
        while True:
            self._print_banner()
            choice = self._prompt_choice()
            if choice == "0":
                self.show_info("Exiting aNEOS. Goodbye.")
                break
            handlers = {
                "1": self._detect_single,
                "2": self._detect_multi_evidence,
                "3": self._detect_batch,
                "4": self._orbital_history_analysis,
                "5": self._impact_assessment,
                "6": self._close_approach_history,
                "7": self._live_pipeline,
                "8": self._population_pattern_analysis,
                "9": self._results_browser,
                "10": self._export_results,
                "11": self._system_health,
                "12": self._start_api_server,
                "13": self._detection_analytics,
                "14": self._show_help,
            }
            handler = handlers.get(choice)
            if handler:
                handler()
            else:
                self.show_error(f"Unknown option: {choice!r}")
            self.wait_for_input()

    # ------------------------------------------------------------------
    # Banner / prompt
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        content = (
            "\n"
            "  [bold]aNEOS — Artificial NEO Analysis Platform[/bold]\n"
            "  [dim]Research tool — not an operational warning system[/dim]\n\n"
            "  [bold cyan]── Detection & Classification ──────────────────[/bold cyan]\n"
            "  [cyan] 1[/cyan]  Detect NEO (single)          [σ-5 validated]\n"
            "  [cyan] 2[/cyan]  Multi-Evidence Analysis       [per-source breakdown]\n"
            "  [cyan] 3[/cyan]  Batch Detection               [concurrent, from file]\n"
            "  [cyan] 4[/cyan]  Orbital History Analysis      [smoking gun / course corrections]\n\n"
            "  [bold cyan]── Impact Assessment ───────────────────────────[/bold cyan]\n"
            "  [cyan] 5[/cyan]  Impact Probability            [planetary defence]\n"
            "  [cyan] 6[/cyan]  Close Approach History        [historical CAD data]\n\n"
            "  [bold cyan]── Monitoring & Polling ─────────────────────────[/bold cyan]\n"
            "  [cyan] 7[/cyan]  Live Pipeline Dashboard       [200-year historical poll]\n"
            "  [cyan] 8[/cyan]  Population Pattern Analysis   [BC11 network sigma]\n\n"
            "  [bold cyan]── Results & Reports ───────────────────────────[/bold cyan]\n"
            "  [cyan] 9[/cyan]  Browse Results                [in-session + DB]\n"
            "  [cyan]10[/cyan]  Export Results                [JSON / CSV]\n\n"
            "  [bold cyan]── System ─────────────────────────────────────[/bold cyan]\n"
            "  [cyan]11[/cyan]  Health Check                  [component status]\n"
            "  [cyan]12[/cyan]  Start API Server              [uvicorn :8000]\n"
            "  [cyan]13[/cyan]  Detection Analytics           [session stats + export]\n"
            "  [cyan]14[/cyan]  Scientific Help               [methodology + σ thresholds]\n\n"
            "  [cyan] 0[/cyan]  Exit\n"
        )
        self.display_panel(content, title="aNEOS v2", style="blue")

    def _prompt_choice(self) -> str:
        if self.console:
            from rich.prompt import Prompt
            return Prompt.ask("[bold]Choice[/bold]", default="0").strip()
        return input("Choice [0]: ").strip() or "0"

    # ==================================================================
    # GROUP A — Detection & Classification
    # ==================================================================

    def _detect_single(self) -> None:
        """Option 1: Run validated σ-5 detector on one designation."""
        designation = self._ask("NEO designation (e.g. '99942', '2020 SO', 'tesla')").strip()
        if not designation:
            self.show_error("No designation provided.")
            return

        neo_data = self._fetch_neo_data(designation)
        if neo_data is None:
            return

        result = self._run_detection(designation, neo_data)
        if result is None:
            return

        self._detection_results[designation.upper()] = result
        self._display_detection_result(designation, result, verbose=False)
        self._persist_detection_result_async(designation, result)

    def _detect_multi_evidence(self) -> None:
        """Option 2: Detection + full per-source evidence breakdown."""
        designation = self._ask("NEO designation").strip()
        if not designation:
            self.show_error("No designation provided.")
            return

        neo_data = self._fetch_neo_data(designation)
        if neo_data is None:
            return

        result = self._run_detection(designation, neo_data)
        if result is None:
            return

        self._detection_results[designation.upper()] = result
        self._display_detection_result(designation, result, verbose=True)
        self._persist_detection_result_async(designation, result)

    def _detect_batch(self) -> None:
        """Option 3: Concurrent batch detection from a file of designations."""
        path_str = self.browse_files(
            search_dirs=[".", "neo_data", "data", "tests"],
            extensions=[".txt", ".csv"],
            prompt="Designation file",
        )
        if not path_str:
            self.show_error("No path provided.")
            return

        p = Path(path_str)
        if not p.exists():
            self.show_error(f"File not found: {p}")
            return

        lines = [l.strip() for l in p.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        if not lines:
            self.show_error("File is empty or contains only comments.")
            return

        self.show_info(f"Fetching {len(lines)} designations concurrently…")
        try:
            from aneos_core.data.fetcher import DataFetcher
            from aneos_core.detection.detection_manager import DetectionManager, DetectorType
            fetcher = DataFetcher()
            manager = DetectionManager(preferred_detector=DetectorType.VALIDATED)
        except Exception as exc:
            self.show_error(f"Failed to initialise detector: {exc}")
            return

        # Concurrent fetch — show spinner (bulk operation, not per-item trackable)
        if self.console:
            with self.console.status(f"[green]Fetching {len(lines)} NEOs…", spinner="dots"):
                neo_map = fetcher.fetch_multiple(lines)
        else:
            neo_map = fetcher.fetch_multiple(lines)

        fetched = sum(1 for v in neo_map.values() if v)
        self.show_info(f"Fetched {fetched}/{len(lines)} — running detector…")

        # Detection loop with progress bar
        rows = []
        valid = [d for d in lines if neo_map.get(d) and neo_map[d].orbital_elements]
        skipped = [d for d in lines if not neo_map.get(d) or not neo_map[d].orbital_elements]
        for s in skipped:
            self.show_info(f"{s}: no orbital data — skipped")

        for designation in self.track_progress(valid, f"Detecting {len(valid)} NEOs"):
            neo_data = neo_map[designation]
            try:
                result = self._run_detection(designation, neo_data, manager=manager)
                if result is None:
                    continue
                self._detection_results[designation.upper()] = result
                sigma = getattr(result, "sigma_level", 0.0)
                prob = getattr(result, "artificial_probability", 0.0)
                cls = getattr(result, "classification", "—")
                rows.append([designation, f"{sigma:.2f}", f"{prob:.4f}", cls])
                self._persist_detection_result_async(designation, result)
            except Exception as exc:
                self.show_error(f"{designation}: {exc}")

        if rows:
            self.display_table(
                headers=["Designation", "σ", "P(artificial)", "Classification"],
                rows=rows,
                title=f"Batch Results — {len(rows)}/{len(lines)} processed",
            )
        else:
            self.show_info("No results produced.")

    def _orbital_history_analysis(self) -> None:
        """Option 4: Fetch orbital history via Horizons; detect course corrections."""
        designation = self._ask("NEO designation").strip()
        if not designation:
            self.show_error("No designation provided.")
            return

        years_str = self._ask("Years of orbital history to fetch (default 10)").strip()
        try:
            years = int(years_str) if years_str else 10
        except ValueError:
            years = 10

        # Fetch Horizons multi-epoch elements
        try:
            from aneos_core.data.sources.horizons import HorizonsSource
            from aneos_core.config.settings import APIConfig
            horizons = HorizonsSource(APIConfig())
            if self.console:
                with self.console.status(
                    f"[green]Fetching {years}-year orbital history from Horizons…", spinner="dots"
                ):
                    history = horizons.fetch_orbital_history(designation, years=years)
            else:
                history = horizons.fetch_orbital_history(designation, years=years)
        except Exception as exc:
            self.show_error(f"Horizons fetch failed: {exc}")
            return

        if not history:
            self.show_info(
                f"No orbital history returned for {designation!r}. "
                "The object may not be in the Horizons database or the designation format "
                "differs (try the JPL SPKID or numeric provisional designation)."
            )
            return

        self.show_success(f"Retrieved {len(history)} orbital epochs for {designation}.")

        # Display the time-series
        rows = []
        for epoch in history:
            rows.append([
                str(epoch.get("epoch", ""))[:10],
                f"{epoch.get('a', '?'):.4f}" if isinstance(epoch.get("a"), float) else "?",
                f"{epoch.get('e', '?'):.5f}" if isinstance(epoch.get("e"), float) else "?",
                f"{epoch.get('i', '?'):.3f}" if isinstance(epoch.get("i"), float) else "?",
            ])
        self.display_table(
            headers=["Epoch", "a (AU)", "e", "i (°)"],
            rows=rows,
            title=f"Orbital History — {designation} ({len(history)} epochs)",
        )

        # Also run detection with orbital_history passed through
        neo_data = self._fetch_neo_data(designation)
        if neo_data is not None:
            if self.console:
                with self.console.status("[green]Running detector with orbital history…", spinner="dots"):
                    result = self._run_detection(
                        designation, neo_data, extra_additional={"orbital_history": history}
                    )
            else:
                result = self._run_detection(
                    designation, neo_data, extra_additional={"orbital_history": history}
                )
            if result:
                self._detection_results[designation.upper()] = result
                self._display_detection_result(designation, result, verbose=False)
                self._persist_detection_result_async(designation, result)

    # ==================================================================
    # GROUP B — Impact Assessment
    # ==================================================================

    def _impact_assessment(self) -> None:
        """Option 5: Calculate impact probability for one NEO."""
        designation = self._ask("NEO designation").strip()
        if not designation:
            self.show_error("No designation provided.")
            return

        neo_data = self._fetch_neo_data(designation)
        if neo_data is None:
            return

        oe = neo_data.orbital_elements
        if oe is None:
            self.show_error(f"No orbital elements for {designation!r}.")
            return

        close_approaches = getattr(neo_data, "close_approaches", None) or []
        pp = getattr(neo_data, "physical_properties", None)

        # Derive observation arc from actual SBDB first/last observation dates
        from datetime import datetime as _dt
        arc_days = 30.0  # default
        first_obs = getattr(neo_data, "first_observation", None)
        last_obs  = getattr(neo_data, "last_observation", None)
        if isinstance(first_obs, _dt) and isinstance(last_obs, _dt):
            arc_days = max(1.0, (last_obs - first_obs).days)
        _dates_available = isinstance(first_obs, _dt) and isinstance(last_obs, _dt)
        arc_note = (
            f"Arc: {arc_days:.0f} days ({arc_days/365.25:.1f} yr)"
            if _dates_available
            else "Arc: 30-day default (observation dates unavailable)"
        )

        try:
            from aneos_core.analysis.impact_probability import ImpactProbabilityCalculator
            calc = ImpactProbabilityCalculator()
            if self.console:
                with self.console.status("[green]Computing impact probability…", spinner="dots"):
                    impact = calc.calculate_comprehensive_impact_probability(
                        orbital_elements=oe,
                        close_approaches=close_approaches or None,
                        observation_arc_days=arc_days,
                        physical_properties=pp,
                    )
            else:
                impact = calc.calculate_comprehensive_impact_probability(
                    orbital_elements=oe,
                    close_approaches=close_approaches or None,
                    observation_arc_days=arc_days,
                    physical_properties=pp,
                )
            if self.console:
                self.console.print(f"[dim]{arc_note}[/dim]")
            else:
                print(arc_note)
        except Exception as exc:
            self.show_error(f"Impact calculation failed: {exc}")
            return

        self._impact_results[designation.upper()] = impact
        self._display_impact_result(designation, impact)
        self._persist_impact_result_async(designation, impact)

    def _close_approach_history(self) -> None:
        """Option 6: Fetch historical close-approach data (CAD API)."""
        designation = self._ask("NEO designation").strip()
        if not designation:
            self.show_error("No designation provided.")
            return

        years_str = self._ask("Years back to search (default 30)").strip()
        try:
            years = int(years_str) if years_str else 30
        except ValueError:
            years = 30

        try:
            from aneos_core.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            if self.console:
                with self.console.status(
                    f"[green]Fetching {years}-year close-approach history…", spinner="dots"
                ):
                    approaches = fetcher.fetch_historical_approaches(designation, years_back=years)
            else:
                approaches = fetcher.fetch_historical_approaches(designation, years_back=years)
        except Exception as exc:
            self.show_error(f"CAD fetch failed: {exc}")
            return

        if not approaches:
            self.show_info(
                f"No close-approach records found for {designation!r} in the last {years} years. "
                "The object may not have had approaches < 0.2 AU, or it may not be in SBDB."
            )
            return

        rows = []
        for ca in approaches:
            dist = getattr(ca, "distance_au", None) or getattr(ca, "dist", None)
            vel = getattr(ca, "relative_velocity_km_s", None) or getattr(ca, "v_rel", None)
            date = getattr(ca, "date", None) or getattr(ca, "t_ca", "?")
            rows.append([
                str(date)[:10],
                f"{dist:.5f} AU" if dist is not None else "?",
                f"{vel:.2f} km/s" if vel is not None else "?",
            ])

        self.display_table(
            headers=["Date", "Distance", "Velocity"],
            rows=rows,
            title=f"Close Approach History — {designation} ({len(rows)} events)",
        )

    # ==================================================================
    # GROUP C — Monitoring & Polling
    # ==================================================================

    def _live_pipeline(self) -> None:
        """Option 7: 200-year historical polling via automatic review pipeline."""
        import asyncio
        self.show_info(
            "Launching historical polling pipeline (200-year window).\n"
            "This will run the multi-stage ATLAS review funnel.\n"
            "Press Ctrl-C to interrupt."
        )
        try:
            from aneos_core.integration.pipeline_integration import (
                PipelineIntegration,
                initialize_pipeline_integration,
                run_200_year_poll,
            )

            async def _run():
                await initialize_pipeline_integration()
                result = await run_200_year_poll()
                return result

            result = asyncio.run(_run())

            if result:
                self._display_pipeline_result(result)
            else:
                self.show_info("Pipeline completed but returned no results.")

        except ImportError as exc:
            self.show_error(
                f"Pipeline integration module unavailable: {exc}\n"
                "Check aneos_core/integration/pipeline_integration.py for import errors."
            )
        except KeyboardInterrupt:
            self.show_info("Pipeline interrupted by user.")
        except Exception as exc:
            self.show_error(f"Pipeline error: {exc}")

    def _population_pattern_analysis(self) -> None:
        """Option 8: BC11 network-sigma clustering + harmonics analysis."""
        path_str = self.browse_files(
            search_dirs=[".", "neo_data", "data", "tests"],
            extensions=[".txt", ".csv"],
            prompt="Designation file for population analysis",
        )
        if not path_str:
            self.show_error("No path provided.")
            return

        p = Path(path_str)
        if not p.exists():
            self.show_error(f"File not found: {p}")
            return

        lines = [l.strip() for l in p.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        if not lines:
            self.show_error("File is empty.")
            return

        self.show_info(f"Fetching {len(lines)} NEOs for population analysis…")
        try:
            from aneos_core.data.fetcher import DataFetcher
            from aneos_core.pattern_analysis.session import (
                NetworkAnalysisSession, PatternAnalysisConfig
            )
            fetcher = DataFetcher()
        except Exception as exc:
            self.show_error(f"Failed to initialise pattern analysis: {exc}")
            return

        if self.console:
            with self.console.status(f"[green]Fetching {len(lines)} NEOs…", spinner="dots"):
                neo_map = fetcher.fetch_multiple(lines)
        else:
            neo_map = fetcher.fetch_multiple(lines)

        neo_objects = [v for v in neo_map.values() if v is not None]
        self.show_info(f"Running BC11 pattern analysis on {len(neo_objects)} objects…")

        try:
            cfg = PatternAnalysisConfig(clustering=True, harmonics=True, correlation=False)
            session = NetworkAnalysisSession(config=cfg, fetcher=fetcher)
            if self.console:
                with self.console.status("[green]Running clustering + network sigma…", spinner="dots"):
                    result = session.run(neo_objects)
            else:
                result = session.run(neo_objects)
        except Exception as exc:
            self.show_error(f"Pattern analysis failed: {exc}")
            return

        self._display_pattern_result(result)

    # ==================================================================
    # GROUP D — Results & Reports
    # ==================================================================

    def _results_browser(self) -> None:
        """Option 9: Interactive browser for session detection/impact results + DB."""
        if not self._detection_results and not self._impact_results:
            self.show_info("No results in this session yet. Run options 1–5 first.")
            self._show_db_results()
            return

        # --- Detection results ---
        if self._detection_results:
            det_items = list(self._detection_results.items())
            rows = []
            for i, (desg, result) in enumerate(det_items, 1):
                sigma = getattr(result, "sigma_level", 0.0)
                prob = getattr(result, "artificial_probability", 0.0)
                cls = getattr(result, "classification", "—")
                rows.append([str(i), desg, f"{sigma:.2f}", f"{prob:.4f}", cls])
            self.display_table(
                headers=["#", "Designation", "σ", "P(artificial)", "Classification"],
                rows=rows,
                title=f"Detection Results — {len(det_items)} objects",
            )
            choice = self._ask("View full details for # (Enter to skip)").strip()
            if choice:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(det_items):
                        desg, result = det_items[idx]
                        self._display_detection_result(desg, result, verbose=True)
                except ValueError:
                    pass

        # --- Impact results ---
        if self._impact_results:
            imp_items = list(self._impact_results.items())
            rows = []
            for i, (desg, impact) in enumerate(imp_items, 1):
                prob = getattr(impact, "collision_probability", 0.0)
                energy = getattr(impact, "impact_energy_mt", None)
                method = getattr(impact, "calculation_method", "—")
                rows.append([
                    str(i), desg, f"{prob:.2e}",
                    f"{energy:.1f} MT" if energy is not None else "N/A",
                    method,
                ])
            self.display_table(
                headers=["#", "Designation", "P(impact)", "Energy", "Method"],
                rows=rows,
                title=f"Impact Results — {len(imp_items)} objects",
            )
            choice = self._ask("View full impact details for # (Enter to skip)").strip()
            if choice:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(imp_items):
                        desg, impact = imp_items[idx]
                        self._display_impact_result(desg, impact)
                except ValueError:
                    pass

        self._show_db_results()

    def _export_results(self) -> None:
        """Option 10: Export in-session results to JSON or CSV."""
        if not self._detection_results and not self._impact_results:
            self.show_info("No results to export. Run options 1–5 first.")
            return

        fmt = self._ask("Export format [1=JSON, 2=CSV]").strip()
        if fmt not in ("1", "2"):
            self.show_error("Choose 1 (JSON) or 2 (CSV).")
            return

        out_path = self._ask("Output filename (leave blank for auto)").strip()

        # Build flat records list
        records = []
        for desg, result in self._detection_results.items():
            records.append({
                "designation": desg,
                "type": "detection",
                "sigma_level": getattr(result, "sigma_level", None),
                "artificial_probability": getattr(result, "artificial_probability", None),
                "classification": getattr(result, "classification", None),
                "evidence_count": (getattr(result, "metadata", {}) or {}).get("evidence_count"),
                "combined_p_value": (getattr(result, "metadata", {}) or {}).get("combined_p_value"),
            })
        for desg, impact in self._impact_results.items():
            records.append({
                "designation": desg,
                "type": "impact",
                "collision_probability": getattr(impact, "collision_probability", None),
                "impact_energy_mt": getattr(impact, "impact_energy_mt", None),
                "crater_diameter_km": getattr(impact, "crater_diameter_km", None),
                "calculation_method": getattr(impact, "calculation_method", None),
            })

        try:
            from aneos_core.reporting.exporters import Exporter
            exporter = Exporter()
            if fmt == "1":
                path = exporter.export_to_json(records, output_file=out_path or None)
            else:
                path = exporter.export_to_csv(records, output_file=out_path or None)
            self.show_success(f"Exported {len(records)} records to: {path}")
        except Exception as exc:
            self.show_error(f"Export failed: {exc}")
            # Fallback: write manually
            fallback = Path(out_path or "aneos_results_export.json")
            fallback.write_text(json.dumps(records, indent=2, default=str))
            self.show_success(f"Fallback JSON written to: {fallback}")

    # ==================================================================
    # GROUP E — System
    # ==================================================================

    def _system_health(self) -> None:
        """Option 11: Component health check and API reachability."""
        rows = []
        components = [
            ("DataFetcher",              "aneos_core.data.fetcher",                      "DataFetcher"),
            ("HorizonsSource",           "aneos_core.data.sources.horizons",             "HorizonsSource"),
            ("DetectionManager",         "aneos_core.detection.detection_manager",       "DetectionManager"),
            ("ImpactCalculator",         "aneos_core.analysis.impact_probability",       "ImpactProbabilityCalculator"),
            ("MetricsCollector",         "aneos_core.monitoring.metrics",                "MetricsCollector"),
            ("NetworkAnalysisSession",   "aneos_core.pattern_analysis.session",          "NetworkAnalysisSession"),
            ("PipelineIntegration",      "aneos_core.integration.pipeline_integration",  "PipelineIntegration"),
            ("Exporter",                 "aneos_core.reporting.exporters",               "Exporter"),
        ]
        for label, mod, cls_name in self.track_progress(components, "Checking components"):
            try:
                m = __import__(mod, fromlist=[cls_name])
                getattr(m, cls_name)
                rows.append([label, "✅ OK", "importable"])
            except Exception as exc:
                rows.append([label, "❌ FAIL", str(exc)[:60]])

        # API reachability
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            rows.append(["REST API :8000", "✅ UP", "HTTP 200"])
        except Exception as exc:
            rows.append(["REST API :8000", "⚠ DOWN", str(exc)[:60]])

        self.display_table(
            headers=["Component", "Status", "Detail"],
            rows=rows,
            title="System Health",
        )
        self.display_panel(
            "[yellow]aNEOS is a research prototype.[/yellow]\n"
            "σ < 2 → INCONCLUSIVE, not evidence of natural origin.\n"
            "Bayesian posterior ceiling ≈ 3–5% from orbital+physical data alone.\n"
            "F1=1.000 reported in validation is based on N=4 objects — NOT a generalisation estimate.\n"
            "Real discrimination from propulsion/manoeuvre data is deferred until such a corpus exists.",
            title="Scientific Caveats",
            style="yellow",
        )

    # ==================================================================
    # Shared computation helpers
    # ==================================================================

    def _run_detection(
        self,
        designation: str,
        neo_data,
        manager=None,
        extra_additional: Optional[Dict] = None,
    ):
        """Run the validated detector; return DetectionResult or None on error."""
        oe = neo_data.orbital_elements
        if oe is None:
            self.show_error(f"No orbital elements available for {designation!r}.")
            return None

        orbital = {
            "a": oe.semi_major_axis, "e": oe.eccentricity, "i": oe.inclination,
            "om": oe.ra_of_ascending_node, "w": oe.arg_of_periapsis, "M": oe.mean_anomaly,
        }
        physical = self._build_physical_dict(neo_data)
        additional = self._build_additional_dict(neo_data)
        if extra_additional:
            additional.update(extra_additional)

        try:
            if manager is None:
                from aneos_core.detection.detection_manager import DetectionManager, DetectorType
                manager = DetectionManager(preferred_detector=DetectorType.VALIDATED)
            return manager.analyze_neo(
                orbital_elements=orbital,
                physical_data=physical,
                additional_data=additional,
            )
        except Exception as exc:
            self.show_error(f"Detection failed for {designation!r}: {exc}")
            return None

    def _ask(self, prompt: str) -> str:
        if self.console:
            from rich.prompt import Prompt
            return Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]")
        return input(f"{prompt}: ")

    def _fetch_neo_data(self, designation: str):
        """Fetch NEO data; returns None and prints error on failure."""
        try:
            from aneos_core.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            if self.console:
                with self.console.status(
                    f"[green]Fetching data for [bold]{designation}[/bold]…", spinner="dots"
                ):
                    neo_data = fetcher.fetch_neo_data(designation)
            else:
                neo_data = fetcher.fetch_neo_data(designation)
            if neo_data is None:
                self.show_error(
                    f"No data returned for {designation!r}. "
                    "Check the designation format (e.g. '99942', '2020 SO') and network connectivity."
                )
            return neo_data
        except Exception as exc:
            self.show_error(f"Data fetch failed for {designation!r}: {exc}")
            return None

    def _build_physical_dict(self, neo_data) -> dict:
        physical = {}
        pp = getattr(neo_data, "physical_properties", None)
        if pp is not None:
            if getattr(pp, "diameter_km", None) is not None:
                physical["diameter"] = pp.diameter_km
            if getattr(pp, "albedo", None) is not None:
                physical["albedo"] = pp.albedo
            if getattr(pp, "spectral_type", None) is not None:
                physical["spectral_type"] = pp.spectral_type
            if getattr(pp, "rotation_period_hours", None) is not None:
                physical["rotation_period"] = pp.rotation_period_hours
            if getattr(pp, "absolute_magnitude_h", None) is not None:
                physical["absolute_magnitude"] = pp.absolute_magnitude_h
        sources = getattr(neo_data, "sources_used", [])
        if sources:
            physical["_sources"] = sources
        return physical

    def _build_additional_dict(self, neo_data) -> dict:
        additional = {}
        history = getattr(neo_data, "orbital_history", None)
        if history:
            additional["orbital_history"] = history
        approaches = getattr(neo_data, "close_approaches", None)
        if approaches:
            additional["close_approach_history"] = [
                {
                    "date": str(getattr(ca, "date", "")),
                    "distance_au": getattr(ca, "distance_au", 0.0),
                }
                for ca in approaches
            ]
        return additional

    # ==================================================================
    # Persistence helpers (background threads — non-blocking)
    # ==================================================================

    def _persist_detection_result_async(self, designation: str, result) -> None:
        """Write detection result to DB in a background thread."""
        def _write():
            try:
                from aneos_api.database import SessionLocal, HAS_SQLALCHEMY
                if not HAS_SQLALCHEMY:
                    return
                from aneos_api.database import AnalysisService
                result_data = {
                    "designation": designation,
                    "classification": getattr(result, "classification", None),
                    "overall_score": getattr(result, "artificial_probability", 0.0),
                    "confidence": getattr(result, "artificial_probability", 0.0),
                }
                db = SessionLocal()
                try:
                    service = AnalysisService(db)
                    service.save_analysis_result(result_data)
                finally:
                    db.close()
            except Exception:
                pass  # Non-fatal — in-session result is the source of truth

        threading.Thread(target=_write, daemon=True).start()

    def _persist_impact_result_async(self, designation: str, impact) -> None:
        """Write impact result to DB in a background thread."""
        def _write():
            try:
                from aneos_api.database import SessionLocal, HAS_SQLALCHEMY
                if not HAS_SQLALCHEMY:
                    return
                from aneos_api.database import AnalysisService
                result_data = {
                    "designation": designation,
                    "classification": "impact_assessment",
                    "overall_score": getattr(impact, "collision_probability", 0.0),
                    "confidence": getattr(impact, "calculation_confidence", 0.0),
                }
                db = SessionLocal()
                try:
                    service = AnalysisService(db)
                    service.save_analysis_result(result_data)
                finally:
                    db.close()
            except Exception:
                pass

        threading.Thread(target=_write, daemon=True).start()

    # ==================================================================
    # Display helpers
    # ==================================================================

    def _display_detection_result(self, designation: str, result, verbose: bool = False) -> None:
        sigma = getattr(result, "sigma_level", 0.0)
        prob = getattr(result, "artificial_probability", 0.0)
        cls = getattr(result, "classification", "UNKNOWN")
        risk_factors = getattr(result, "risk_factors", [])
        meta = getattr(result, "metadata", {}) or {}

        lines = [
            f"[bold]Designation:[/bold]          {self.format_designation(designation)}",
            f"[bold]Classification:[/bold]       {cls}",
            f"[bold]σ confidence:[/bold]         {self.format_sigma(sigma)} ({sigma:.3f}σ)",
            f"[bold]P(artificial):[/bold]        {self.format_probability(prob)}",
            f"[bold]Evidence sources:[/bold]     {meta.get('evidence_count', len(risk_factors))}",
            f"[bold]Combined p-value:[/bold]     {meta.get('combined_p_value', 'N/A')}",
            f"[bold]False discovery rate:[/bold] {meta.get('false_discovery_rate', 'N/A')}",
        ]

        if risk_factors:
            lines.append(f"[bold]Evidence types:[/bold]      {', '.join(risk_factors)}")

        if verbose:
            # Show individual evidence source objects if available from metadata
            analysis = getattr(result, "analysis", {}) or {}
            evidence_detail = analysis.get("evidence_breakdown") or analysis.get("evidence_sources")
            if evidence_detail:
                lines.append("\n[bold]Per-source breakdown:[/bold]")
                for ev in (evidence_detail if isinstance(evidence_detail, list) else []):
                    etype = getattr(ev, "evidence_type", None) or ev.get("evidence_type", "?")
                    pval = getattr(ev, "p_value", None) or ev.get("p_value", "?")
                    eff = getattr(ev, "effect_size", None) or ev.get("effect_size", "?")
                    qi = getattr(ev, "quality_score", None) or ev.get("quality_score", "?")
                    lines.append(
                        f"  {str(etype):<30}  p={pval!s:<10}  effect={eff!s:<8}  quality={qi!s}"
                    )

        lines.append(
            "\n[dim]NOTE: σ < 2 = INCONCLUSIVE (not evidence of natural origin). "
            "Bayesian posterior ≈ 3–5% max from orbital+physical data alone.[/dim]"
        )
        self.display_panel("\n".join(lines), title=f"Detection — {designation}", style="cyan")

    def _display_impact_result(self, designation: str, impact) -> None:
        prob = getattr(impact, "collision_probability", 0.0)
        annual = getattr(impact, "collision_probability_per_year", 0.0)
        energy = getattr(impact, "impact_energy_mt", None)
        crater = getattr(impact, "crater_diameter_km", None)
        velocity = getattr(impact, "impact_velocity_km_s", None)
        method = getattr(impact, "calculation_method", "—")
        confidence = getattr(impact, "calculation_confidence", None)
        uncertainty = getattr(impact, "probability_uncertainty", (None, None))
        arc = getattr(impact, "data_arc_years", None)

        lines = [
            f"[bold]Designation:[/bold]           {self.format_designation(designation)}",
            f"[bold]Method:[/bold]                {method}",
            f"[bold]Collision probability:[/bold]  {prob:.3e}",
            f"[bold]Annual rate:[/bold]           {annual:.3e} /yr",
        ]
        if arc is not None:
            lines.append(f"[bold]Observation arc:[/bold]       {arc:.1f} yr")
        if uncertainty and uncertainty[0] is not None:
            lines.append(f"[bold]95% CI:[/bold]                [{uncertainty[0]:.3e}, {uncertainty[1]:.3e}]")
        if confidence is not None:
            lines.append(f"[bold]Calc confidence:[/bold]       {confidence:.2f}")
        if energy is not None:
            lines.append(f"[bold]Impact energy:[/bold]         {energy:.1f} MT TNT")
        if velocity is not None:
            lines.append(f"[bold]Impact velocity:[/bold]       {velocity:.1f} km/s")
        if crater is not None:
            lines.append(f"[bold]Crater diameter:[/bold]       {crater:.2f} km")

        torino = self._rough_torino(prob, energy)
        lines.append(f"[bold]Torino scale (est):[/bold]    {torino}")

        lines.append(
            "\n[dim]NOTE: Impact probability depends critically on observation arc and "
            "orbital uncertainty. Short-arc objects have high uncertainty.[/dim]"
        )
        colour = "red" if prob > 1e-4 else "yellow" if prob > 1e-6 else "green"
        self.display_panel("\n".join(lines), title=f"Impact Assessment — {designation}", style=colour)

    def _display_pipeline_result(self, result: dict) -> None:
        status = result.get("status", "unknown")
        if status == "error":
            self.show_error(f"Pipeline error: {result.get('error_message', 'unknown')}")
            return
        if status == "cancelled":
            self.show_info("Pipeline cancelled by user.")
            return

        rows = [
            ["Status", status],
            ["Total objects processed", str(result.get("total_objects", "N/A"))],
            ["Final candidates", str(result.get("final_candidates", "N/A"))],
            ["Processing time", f"{result.get('processing_time_seconds', 0):.1f} s"],
            ["Compression ratio", f"{result.get('compression_ratio', 0):.1f}:1"],
        ]
        # Surface any pipeline-level stage results if present
        pipeline_result = result.get("pipeline_result")
        if pipeline_result is not None:
            metrics = getattr(pipeline_result, "pipeline_metrics", {}) or {}
            for k, v in list(metrics.items())[:5]:
                rows.append([str(k), str(v)])

        self.display_table(headers=["Metric", "Value"], rows=rows, title="Pipeline Run Summary")

    def _display_pattern_result(self, result: dict) -> None:
        n = result.get("designations_analyzed", 0)
        clusters = result.get("clusters", [])
        network_sigma = result.get("network_sigma", None)
        network_tier = result.get("network_tier", None)

        lines = [
            f"[bold]Objects analysed:[/bold]  {n}",
            f"[bold]Clusters found:[/bold]    {len(clusters)}",
        ]
        if network_sigma is not None:
            lines.append(f"[bold]Network sigma:[/bold]     {network_sigma:.2f}σ")
        if network_tier:
            lines.append(f"[bold]Network tier:[/bold]      {network_tier}")

        if clusters:
            rows = []
            for c in clusters[:10]:
                rows.append([
                    str(c.get("cluster_id", "?")),
                    str(c.get("n_members", "?")),
                    f"{c.get('sigma', 0.0):.2f}",
                    str(c.get("cluster_type", "?")),
                ])
            self.display_table(
                headers=["Cluster ID", "Members", "σ", "Type"],
                rows=rows,
                title=f"Clusters (top {len(rows)} of {len(clusters)})",
            )
        self.display_panel("\n".join(lines), title="Population Pattern Analysis", style="cyan")

    def _show_db_results(self) -> None:
        try:
            from aneos_api.database import SessionLocal, HAS_SQLALCHEMY
            if not HAS_SQLALCHEMY:
                return
            from aneos_api.database import AnalysisService
            db = SessionLocal()
            try:
                service = AnalysisService(db)
                results = service.get_analysis_results(limit=20)
                if results:
                    rows = [
                        [
                            r.get("designation", "?"),
                            r.get("classification", "—"),
                            str(r.get("analysis_date", ""))[:19],
                        ]
                        for r in results
                    ]
                    self.display_table(
                        headers=["Designation", "Classification", "Analysis date"],
                        rows=rows,
                        title=f"Persisted results (last {len(rows)})",
                    )
            finally:
                db.close()
        except Exception:
            pass

    # ==================================================================
    # GROUP F — System extras (options 12–14)
    # ==================================================================

    def _start_api_server(self) -> None:
        """Option 12: Launch the FastAPI REST server via uvicorn."""
        import subprocess
        self.show_info("Starting aNEOS REST API on http://0.0.0.0:8000 …")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "aneos_api.app:app",
                 "--host", "0.0.0.0", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.show_success(f"API server started (PID {proc.pid}).  URL: http://localhost:8000")
            self.wait_for_input("Press Enter to stop the server")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            self.show_info("API server stopped.")
        except Exception as exc:
            self.show_error(f"Failed to start API server: {exc}")

    def _detection_analytics(self) -> None:
        """Option 13: Session detection statistics and JSON export."""
        if not self._detection_results:
            self.show_info("No detection results this session. Run options 1–3 first.")
            return

        tiers = {
            "EXCEPTIONAL (σ≥5)": 0,
            "ANOMALOUS (σ≥4)": 0,
            "SIGNIFICANT (σ≥3)": 0,
            "INTERESTING (σ≥2)": 0,
            "INCONCLUSIVE (σ<2)": 0,
        }
        sigmas = []
        suspicious_rows = []

        for desg, result in self._detection_results.items():
            sigma = getattr(result, "sigma_level", 0.0) or 0.0
            sigmas.append(sigma)
            label = self.format_sigma(sigma)
            if label in tiers:
                tiers[label] += 1
            if sigma >= 3.0:
                cls = getattr(result, "classification", "?")
                suspicious_rows.append([desg, f"{sigma:.2f}", cls])

        mean_sigma = sum(sigmas) / len(sigmas) if sigmas else 0.0
        tier_rows = [[tier, str(count)] for tier, count in tiers.items() if count > 0]

        self.display_table(
            headers=["Tier", "Count"],
            rows=tier_rows,
            title=f"Session Statistics — {len(self._detection_results)} objects, mean σ={mean_sigma:.2f}",
        )
        if suspicious_rows:
            self.display_table(
                headers=["Designation", "σ", "Classification"],
                rows=suspicious_rows,
                title="Suspicious / Significant Objects (σ≥3)",
            )

        export_choice = self._ask("Export summary to JSON? [y/N]").strip().lower()
        if export_choice == "y":
            summary = {
                "session_count": len(self._detection_results),
                "mean_sigma": mean_sigma,
                "tier_counts": tiers,
                "suspicious_objects": [
                    {"designation": r[0], "sigma": float(r[1]), "classification": r[2]}
                    for r in suspicious_rows
                ],
            }
            try:
                from aneos_core.reporting.exporters import Exporter
                path = Exporter().export_to_json([summary], output_file=None)
                self.show_success(f"Summary exported to: {path}")
            except Exception as exc:
                fallback = Path("aneos_session_analytics.json")
                fallback.write_text(json.dumps(summary, indent=2, default=str))
                self.show_success(f"Summary written to: {fallback}")

    def _show_help(self) -> None:
        """Option 14: Display scientific methodology documentation."""
        doc_path = Path(__file__).parent / "docs" / "scientific" / "scientific-documentation.md"
        if not doc_path.exists():
            self.show_error(f"Documentation not found: {doc_path}")
            return
        try:
            text = doc_path.read_text(encoding="utf-8")
            lines = text.splitlines()
            # Extract key sections: Methodology, Sigma, Classification, Detection
            sections: List[str] = []
            capture = False
            for line in lines:
                heading = line.startswith("## ") or line.startswith("# ")
                if heading and any(
                    kw in line for kw in
                    ["Methodology", "Sigma", "Classification", "Detection", "Threshold"]
                ):
                    capture = True
                elif heading and capture:
                    break
                if capture:
                    sections.append(line)
            content = "\n".join(sections[:100]) if sections else "\n".join(lines[:100])
            self.display_panel(content, title="aNEOS Scientific Documentation", style="green")
        except Exception as exc:
            self.show_error(f"Failed to read documentation: {exc}")

    @staticmethod
    def _rough_torino(prob: float, energy_mt: Optional[float]) -> str:
        """Very rough Torino Scale estimate (informational only — not authoritative)."""
        if prob <= 0 or prob < 1e-6:
            return "0 (no hazard)"
        e = energy_mt or 0.0
        if prob >= 0.99:
            if e < 1:
                return "8 (certain regional)"
            if e < 100:
                return "9 (certain regional)"
            return "10 (certain global)"
        if prob > 1e-4:
            return "1–7 (elevated; detailed analysis required)"
        return "1 (normal; merits monitoring)"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    menu = ANEOSMenuV2()
    menu.run()


if __name__ == "__main__":
    main()
