"""Primary grouped menu for aNEOS research workflows.

Navigation is declared in aneos_menu_catalog. Backend presence or an attractive
screen does not establish scientific validity; complete per-option prerequisite,
behavior and rendering QA remains tracked in #41. Legacy coverage is not assumed.
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aneos_menu_base import ANEOSMenuBase
from aneos_version import APPLICATION_VERSION, BUILD_CHANNEL
from aneos_menu_catalog import MENU_OPTIONS


class ANEOSMenuV2(ANEOSMenuBase):
    """Primary grouped research menu; per-feature QA coverage is recorded separately."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self._detection_results: Dict[str, Any] = {}
        self._impact_results: Dict[str, Any] = {}
        self._batch_reports: List[Dict[str, Any]] = []

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
            handlers = {item.choice: getattr(self,item.handler)
                        for item in MENU_OPTIONS if item.handler is not None}
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
        title = f'aNEOS {APPLICATION_VERSION} | {BUILD_CHANNEL}'
        subtitle = 'Rank candidates. Trace evidence. Keep origin unresolved.'
        limits = 'Research tool; not an operational warning service.'
        if self.console:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            table = Table(box=None, padding=(0,1), expand=True, show_header=True)
            table.add_column('Choice',justify='right',width=6,style='cyan',no_wrap=True)
            table.add_column('Feature',ratio=2,overflow='fold')
            table.add_column('What this provides',ratio=3,overflow='fold')
            group = None
            for item in MENU_OPTIONS:
                if item.group != group:
                    if group is not None:table.add_section()
                    table.add_row('',Text(item.group,style='bold cyan'),'')
                    group = item.group
                table.add_row(item.choice,Text(item.label),Text(item.basis))
            self.console.print(Panel(Group(Text(subtitle),Text(limits),Text(''),table),
                                     title=Text(title),border_style='blue',padding=(1,1)))
        else:
            print(f'\n{title}\n{subtitle}\n{limits}')
            group = None
            for item in MENU_OPTIONS:
                if item.group != group:
                    print(f'\n{item.group}');group = item.group
                print(f'{item.choice:>2}  {item.label}\n    {item.basis}')

    def _prompt_choice(self) -> str:
        if self.console:
            from rich.prompt import Prompt
            return Prompt.ask("[bold]Choice[/bold]", default="0").strip()
        return input("Choice [0]: ").strip() or "0"

    # ==================================================================
    # GROUP A — Detection & Classification
    # ==================================================================

    def _detect_single(self) -> None:
        """Option 1: Screen one designation without calibrated origin inference."""
        designation = self._ask("NEO designation (e.g. '99942', '2020 SO', 'tesla')").strip()
        if not designation:
            self.show_error("No designation provided.")
            return

        neo_data = self._fetch_neo_data(designation)
        if neo_data is None:
            return

        self._display_neo_data(designation, neo_data)
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

        self._display_neo_data(designation, neo_data)
        result = self._run_detection(designation, neo_data)
        if result is None:
            return

        self._detection_results[designation.upper()] = result
        self._display_detection_result(designation, result, verbose=True)
        self._persist_detection_result_async(designation, result)

    def _detect_batch(self) -> None:
        """Option 3: shared bounded execution, retaining every requested occurrence."""
        path = self.browse_files(search_dirs=[".", "neo_data", "data", "tests"],
            extensions=[".txt", ".csv"], prompt="Designation file")
        if not path:
            self.show_error("No path provided.")
            return
        try:
            from aneos_core.analysis.batch_reporting import read_input, build_batch_report, render_batch_report
            source = read_input(path)
        except (OSError, ValueError, UnicodeError) as exc:
            self.show_error(f"Batch input unavailable or invalid: {exc}")
            return
        names = list(dict.fromkeys(r['designation'] for r in source['requests']))
        self.show_info(f"Requested {len(source['requests'])} rows; {len(names)} exact designation labels. "
                       "Exact repeats reuse one acquisition; aliases are not merged.")
        try:
            import asyncio
            from aneos_core.analysis.pipeline import AnalysisPipeline
            from aneos_core.analysis.reporting import build_screening_report
            pipeline = AnalysisPipeline()
            if self.console:
                with self.console.status("Screening requested objects; provider timeouts apply…"):
                    results = asyncio.run(pipeline.analyze_batch(names))
            else:
                results = asyncio.run(pipeline.analyze_batch(names))
            if len(results) != len(names) or any(r.designation != name for r, name in zip(results, names)):
                raise ValueError('Batch pipeline returned incomplete or reordered identities')
            reports = {r.designation: build_screening_report(r) for r in results}
            report = build_batch_report(source, reports)
        except Exception as exc:
            self.show_error(f"Batch execution unavailable: {exc}. No completed batch published.")
            return
        self._batch_reports = getattr(self, '_batch_reports', [])
        self._batch_reports.append(report)
        # Latest-object compatibility views do not replace the ordered request history.
        from aneos_core.analysis.object_review import build_assessment_review
        for result in results:
            payload = result.screening_assessment
            if payload is None:
                continue
            assessment = pipeline.screening_service.replay(payload['evidence_snapshot'], payload['evidence_id'])
            view = assessment.to_detection_result()
            view.object_review = build_assessment_review(payload, neo=result.neo_data)
            view.analysis['object_review'] = view.object_review
            self._detection_results[result.designation] = view
            self._persist_detection_result_async(result.designation, view)
        render_batch_report(report, self.console)
        self.show_info("Batch history is session-only until explicitly exported with option 10.")

    def _browse_batch_reports(self) -> None:
        """Browse frozen request history; never fetch or recompute scientific inputs."""
        from aneos_core.analysis.batch_reporting import verify_batch_report, render_batch_report
        from aneos_core.analysis.reporting import render_screening_markdown
        try:
            number = self._ask(f"Batch run # [1..{len(self._batch_reports)}, Enter=latest]").strip()
            index = int(number) - 1 if number else len(self._batch_reports) - 1
            if not 0 <= index < len(self._batch_reports):
                raise ValueError('Batch run number is outside available history')
            report = verify_batch_report(self._batch_reports[index])
            render_batch_report(report, self.console)
            row = self._ask("Request row # for full report (Enter to skip)").strip()
            if not row:
                return
            row_index = int(row) - 1
            if not 0 <= row_index < len(report['requests']):
                raise ValueError('Request row number is outside this batch')
            item = report['requests'][row_index]['report']
            if item['screening_assessment'] is not None:
                from aneos_core.analysis.object_review import build_assessment_review, render_object_review
                from aneos_core.data.models import NEOData
                neo = NEOData.from_dict(item['neo_data']) if item['neo_data'] is not None else None
                review = build_assessment_review(item['screening_assessment'], neo=neo)
                render_object_review(review, console=self.console, verbose=True)
                return
            text = render_screening_markdown(item)
            if self.console:
                from rich.markdown import Markdown
                self.console.print(Markdown(text))
            else:
                print(text)
        except (ValueError, TypeError, KeyError) as exc:
            self.show_error(f"Batch review unavailable: {exc}")

    def _export_batch_report(self) -> None:
        """Existing exporter owns exclusive files; all requests and errors remain."""
        try:
            number = self._ask(f"Batch run # [1..{len(self._batch_reports)}, Enter=latest]").strip()
            index = int(number) - 1 if number else len(self._batch_reports) - 1
            if not 0 <= index < len(self._batch_reports):
                raise ValueError('Batch run number is outside available history')
            fmt = self._ask("Export format [1=JSON, 2=CSV]").strip()
            if fmt not in ('1', '2'):
                raise ValueError('Choose 1 (JSON) or 2 (CSV)')
            path = self._ask("Output filename (leave blank for auto)").strip()
            from aneos_core.reporting.exporters import Exporter
            saved = Exporter().export_batch_report(self._batch_reports[index],
                output_file=path or None, format='json' if fmt == '1' else 'csv')
            self.show_success(f"Exported complete request history to: {saved}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.show_error(f"Batch export refused: {exc}")

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
    # GROUP B — Scoped Impact Evidence
    # ==================================================================

    def _impact_assessment(self) -> None:
        """Option 5: scoped Sentry evidence, not a heuristic impact calculation."""
        designation = self._ask("Primary asteroid number or provisional designation").strip()
        if not designation:
            self.show_error("No designation provided.")
            return
        try:
            from aneos_core.analysis.impact_evidence import get_sentry_impact_service, require_sentry_report
            impact = require_sentry_report(get_sentry_impact_service().assess(designation), designation)
        except Exception:
            self.show_error("Impact source unavailable, invalid or identity unresolved; no probability assessment.")
            return
        self._impact_results[impact['designation'].upper()] = impact
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

        coverage = getattr(approaches, 'query_metadata', {})
        if isinstance(coverage, dict) and coverage:
            self.show_info(f"CAD query coverage: {coverage.get('completeness_status', 'unverified')}; "
                f"returned {coverage.get('count', '?')} of {coverage.get('total', '?')} matching encounters. "
                "Computed encounters are not an observation history; dates retain their native time scale.")
        rows = []
        for ca in approaches:
            dist = getattr(ca, "distance_au", None) or getattr(ca, "dist", None)
            vel = getattr(ca, "relative_velocity_km_s", None) or getattr(ca, "v_rel", None)
            date = getattr(ca, "close_approach_date", None) or getattr(ca, "date", None) or getattr(ca, "t_ca", "?")
            rows.append([
                str(date)[:10] + (f" [{getattr(ca, 'time_scale', None)}]" if getattr(ca, 'time_scale', None) else ""),
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
                self._store_pipeline_candidates(result)
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
        # Check if pipeline candidates are available as a shortcut input source
        pipeline_desgs = [
            d for d, r in self._detection_results.items()
            if getattr(r, "classification", "").startswith("PIPELINE:")
        ]

        lines: Optional[List[str]] = None

        if pipeline_desgs:
            choice = self._ask(
                f"Input source [1=Load from file / 2=Use {len(pipeline_desgs)} pipeline candidates]"
            ).strip()
            if choice == "2":
                lines = pipeline_desgs
            # else: fall through to file browser

        if lines is None:
            # File browser path
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

        # Pre-flight: warn about designations that don't look like known formats
        import re as _re
        _DESG_PAT = _re.compile(
            r'^(\d+[\s\w]*|'             # numbered: "433", "99942", "99942 Apophis"
            r'[12]\d{3}\s+[A-Z][\w]*|'  # provisional: "1998 KY26" / "2004 MN4"
            r'[A-Z][a-z]{2,})$'          # name only: "Apophis", "Ryugu"
        )
        _skipped = [l for l in lines if not _DESG_PAT.match(l.strip())]
        if _skipped:
            self.show_info(
                f"Warning: {len(_skipped)} line(s) don't look like NEO designations "
                f"and may fail: {_skipped[:3]}"
            )
            _filtered = [l for l in lines if _DESG_PAT.match(l.strip())]
            lines = _filtered or lines  # keep all if none pass

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

        from aneos_core.data.models import NEOData
        accepted = {d for d in lines if isinstance(neo_map.get(d), NEOData) and neo_map[d].designation == d}
        neo_objects = [neo_map[d] for d in lines if d in accepted]
        unavailable = [d for d in lines if d not in accepted]
        if unavailable:
            self.show_info(f'Unavailable requested inputs ({len(unavailable)}): {unavailable}')
        if not neo_objects:
            self.show_error('No acquired population inputs; analysis not performed.')
            return
        self.show_info(f"Running BC11 pattern analysis on {len(neo_objects)} objects…")

        try:
            cfg = PatternAnalysisConfig(clustering=True, harmonics=True, correlation=False)
            session = NetworkAnalysisSession(config=cfg, fetcher=fetcher)
            if self.console:
                with self.console.status("[green]Running screening + descriptive population patterns…", spinner="dots"):
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
        if getattr(self, '_batch_reports', []):
            self._browse_batch_reports()
            if not self._detection_results and not self._impact_results:
                return
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
                rows.append([str(i), desg, self._format_score(getattr(result, "screening_score", None)), self.format_probability(prob), cls])
            self.display_table(
                headers=["#", "Designation", "Legacy ranking score", "Origin probability", "Classification"],
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
                        self._show_db_result_for(desg)
                except ValueError:
                    pass

        # --- Impact results ---
        if self._impact_results:
            imp_items = list(self._impact_results.items())
            rows = []
            for i, (desg, impact) in enumerate(imp_items, 1):
                from aneos_core.analysis.impact_evidence import require_sentry_report
                try:
                    report = require_sentry_report(impact, desg)
                    prob = report['collision_probability']
                    rows.append([str(i), desg, f"{prob:.3e}" if prob is not None else "not available",
                                 report['status'], report['source']['method'] or "not available"])
                except ValueError:
                    rows.append([str(i), desg, "not available", "invalid_or_legacy", "not available"])
            self.display_table(
                headers=["#", "Designation", "P(listed events)", "Source status", "Source method"],
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
                        self._show_db_result_for(desg)
                except ValueError:
                    pass

    def _export_results(self) -> None:
        """Option 10: Export in-session results to JSON or CSV."""
        if getattr(self, '_batch_reports', []):
            scope = self._ask("Export scope [1=latest object views, 2=batch request history]").strip()
            if scope == '2':
                self._export_batch_report()
                return
            if scope != '1':
                self.show_error('Choose 1 (latest object views) or 2 (batch request history).')
                return
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
            from aneos_core.analysis.object_review import verify_object_review
            review=getattr(result,'object_review',None)
            if review is not None:
                try:
                    review=verify_object_review(review)
                    if review['designation']!=desg and review['designation'].upper()!=desg:
                        raise ValueError('Object identity mismatch')
                except (ValueError,TypeError,KeyError):
                    self.show_error('Export refused: modified or unbound object review.');return
            assessment=review['assessment'] if review else None
            metadata=(getattr(result,'metadata',{}) or {})
            records.append({
                'object_review':review,
                'object_review_json':json.dumps(review,sort_keys=True,allow_nan=False) if review else None,
                'evidence_basis':'replayed_screening_review' if review else 'legacy_unverified_export',
                'designation':desg,'type':'detection',
                'sigma_level':None,'artificial_probability':None,'combined_p_value':None,
                'screening_score':assessment['screening_score'] if assessment else getattr(result,'screening_score',None),
                'evidence_id':assessment['evidence_id'] if assessment else metadata.get('evidence_id'),
                'classification':assessment['classification'] if assessment else 'LEGACY_UNVERIFIED',
                'evidence_count':len(assessment['evidence_sources']) if assessment else None,
                'non_gravitational_evidence':review['inventory']['non_gravitational'] if review and review['inventory'] else metadata.get('non_gravitational_evidence'),
            })
        for desg, impact in self._impact_results.items():
            from aneos_core.analysis.impact_evidence import require_sentry_report
            try:
                report = require_sentry_report(impact, desg)
            except ValueError:
                self.show_error(f"Export refused: invalid or legacy impact evidence for {desg}.")
                return
            records.append({'designation': desg, 'type': 'impact', 'evidence_basis':'authoritative_external_model_output',
                            'impact_evidence': report,
                            'impact_evidence_json': json.dumps(report, sort_keys=True, allow_nan=False)})

        try:
            from aneos_core.reporting.exporters import Exporter
            exporter = Exporter()
            if fmt == "1":
                path = exporter.export_to_json(records, output_file=out_path or None, exclusive=True)
            else:
                path = exporter.export_to_csv(records, output_file=out_path or None, exclusive=True)
            self.show_success(f"Exported {len(records)} records to: {path}")
        except Exception as exc:
            self.show_error(f"Export failed: {exc}")
            # No fallback write: it could overwrite historical evidence after an
            # exclusive-create refusal or hide a partial export failure.

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
            "Screening ranks are heuristics, not discovery significance.\n"
            "Artificial-origin probability and FDR are not reliably estimated.\n"
            "No independently evaluated calibration is installed.\n"
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
        """Screen the complete snapshot through the shared application service."""
        from aneos_core.analysis.screening import ScreeningAnalysisService
        try:
            service = manager if isinstance(manager, ScreeningAnalysisService) else ScreeningAnalysisService()
            from aneos_core.data.models import NEOData
            from aneos_core.analysis.object_review import build_object_review
            from dataclasses import asdict
            if not isinstance(neo_data,NEOData) or neo_data.designation!=designation:
                raise ValueError('Typed source data/object identity unavailable')
            review=build_object_review(neo_data,policy=service.policy,additional_data=extra_additional)
            result=service.analyze_neo(neo_data,extra_additional).to_detection_result()
            if result.analysis['assessment']!=review['assessment']:
                raise ValueError('Source changed during screening')
            result.object_review=review
            result.screening_policy=asdict(service.policy)
            result.neo_data=NEOData.from_dict(review['inventory']['source_snapshot'])
            result.analysis['object_review']=review
            return result
        except Exception as exc:
            self.show_error(f"Screening failed for {designation!r}: {exc}")
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
                # Detector expects diameter in metres; store both keys for compatibility
                physical["diameter"] = pp.diameter_km * 1000   # metres (detector units)
                physical["diameter_km"] = pp.diameter_km       # km (indicator trigger key)
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
        from aneos_core.data.models import NonGravitationalParameters
        ng = getattr(neo_data, 'nongrav', None)
        if isinstance(ng, NonGravitationalParameters):
            additional['non_gravitational_evidence'] = ng.evidence_report()
        history = getattr(neo_data, "orbital_history", None)
        if history:
            additional["orbital_history"] = history
        approaches = getattr(neo_data, "close_approaches", None)
        if approaches:
            additional["close_approach_history"] = [
                {
                    "date": str(getattr(ca, "close_approach_date", None)
                               or getattr(ca, "date", "") or ""),
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
                    "overall_score": getattr(result, "screening_score", None),
                    "confidence": None,
                    "anomaly_score_data": (getattr(result, "analysis", {}) or {}).get('assessment', {}),
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

    def _persist_impact_result_async(self, designation: str, impact):
        """Compatibility name; synchronous verified archive, never a daemon write.

        No origin-score DB column is used for an impact probability. Failures are
        visible and in-session evidence remains available for a later explicit save.
        """
        try:
            from aneos_core.analysis.impact_evidence import require_sentry_report
            from aneos_core.reporting.evidence_archive import EvidenceArchive
            report = require_sentry_report(impact, designation)
            path = EvidenceArchive().put(report)
            self.show_success(f"Evidence archived and verified: {path}")
            return path
        except Exception:
            self.show_error("Evidence was NOT confirmed saved; the in-session report remains available.")
            return None

    # ==================================================================
    # Display helpers
    # ==================================================================

    def _display_neo_data(self, designation: str, neo_data) -> None:
        """Render a typed, source-bound inventory; no broad completeness claim."""
        from aneos_core.analysis.object_review import build_data_inventory, render_data_inventory
        try:
            if neo_data.designation!=designation:raise ValueError('Source identity mismatch')
            inventory=build_data_inventory(neo_data)
            render_data_inventory(inventory,console=self.console)
        except (ValueError,TypeError,AttributeError):
            self.show_error('Object data cannot be bound to a valid typed snapshot; no completeness claim is available.')

    def _display_detection_result(self, designation: str, result, verbose: bool = False) -> None:
        cls = getattr(result, 'classification', 'INCONCLUSIVE')
        if cls.startswith('PIPELINE:'):
            self._display_pipeline_candidate_detail(designation, result)
            return
        from aneos_core.analysis.object_review import build_assessment_review, verify_object_review, render_object_review
        from aneos_core.analysis.screening import ScreeningPolicy
        try:
            payload=(getattr(result,'analysis',{}) or {}).get('assessment')
            if not isinstance(payload,dict) or str(payload.get('designation','')).casefold()!=designation.casefold():
                raise ValueError('Canonical assessment or object binding unavailable')
            review=getattr(result,'object_review',None)
            if review is None:
                policy=ScreeningPolicy(**(getattr(result,'screening_policy',None) or {}))
                review=build_assessment_review(payload,policy=policy)
            else:
                review=verify_object_review(review)
                if review['assessment']!=payload:raise ValueError('Assessment changed after review')
            if (getattr(result,'classification',None)!=payload['classification'] or
                getattr(result,'screening_score',None)!=payload['screening_score']):
                raise ValueError('Displayed legacy result differs from canonical assessment')
            render_object_review(review,console=self.console,verbose=verbose)
        except (ValueError,TypeError,KeyError,AttributeError):
            self.show_error('Result is legacy, modified or unbound; a verified object explanation is unavailable.')

    @staticmethod
    def _format_score(value):
        from aneos_core.analysis.object_review import number
        return 'unavailable' if value is None else number(value,exact=True)+' (heuristic)'

    def _display_pipeline_candidate_detail(self, designation: str, result) -> None:
        """Detailed ATLAS breakdown for a pipeline candidate (stored proxy object)."""
        analysis = getattr(result, "analysis", {}) or {}
        ms = getattr(result, "multi_stage_validation", {}) or {}
        dist = getattr(result, "miss_distance_au", None)
        vel = getattr(result, "relative_velocity_km_s", None)

        overall = analysis.get("overall_score", 0.0)
        flags = analysis.get("flag_string", "—")
        stage = analysis.get("processing_stage", "—")
        clues = analysis.get("clue_contributions", [])
        cat_scores = analysis.get("category_scores", {})

        lines = [
            f"[bold]Designation:[/bold]      {self.format_designation(designation)}",
            f"[bold]Encounter triage:[/bold] {self._format_score(overall)}  (flags: {flags})",
            f"[bold]Scoring stage:[/bold]    {stage}",
        ]
        if dist is not None:
            lines.append(f"[bold]Miss distance:[/bold]    {dist:.5f} AU  ({dist * 149_597_870.7:.0f} km)")
        if vel is not None:
            lines.append(f"[bold]Rel. velocity:[/bold]    {vel:.2f} km/s")

        # Validation scores
        v_method = ms.get("validation_method", "—")
        lines.append(f"[bold]Required validation:[/bold] {ms.get('validation_status', 'legacy_unverified')}  ({v_method})")
        lines.append(f"[bold]Validation probability:[/bold] unavailable")

        # Category breakdown
        if cat_scores:
            lines.append("\n[bold]Category scores:[/bold]")
            max_cat = max(cat_scores.values()) if cat_scores else 1.0
            for cat, score in cat_scores.items():
                bar = "█" * int((score / max_cat) * 25) if max_cat > 0 else ""
                pct = f"{score*100:.1f}%"
                lines.append(f"  {cat:<28} {pct:>6}  {bar}")

        # Per-clue breakdown
        if clues:
            lines.append("\n[bold]ATLAS clue breakdown:[/bold]")
            for clue in clues:
                name = clue.get("name", "?")
                contrib = clue.get("contribution", 0.0)
                score = clue.get("score", 0.0)
                conf = clue.get("confidence", 0.0)
                flag = clue.get("flag", "")
                explanation = clue.get("explanation", "")
                flag_str = f" [{flag}]" if flag else ""
                lines.append(
                    f"  [bold]{name:<32}[/bold]{flag_str}"
                )
                lines.append(
                    f"    score={score:.3f}  contribution={contrib:.4f}  confidence={conf:.2f}"
                )
                if explanation:
                    lines.append(f"    → {explanation}")

        lines.append(
            "\n[dim]DATA-SOURCE NOTE: CAD API provides orbital geometry only. "
            "Physical, spectral, radar, and thermal clue scores are zero unless "
            "dedicated observational data is available (ADR-053). "
            "Orbit-behaviour scores (approach proximity, velocity anomaly) are proxies "
            "computed from orbital elements — not direct measurements.[/dim]"
        )
        if "Δ" in flags:
            lines.append(
                "\n[dim]Δ FLAG (non-gravitational acceleration): This object shows a velocity "
                "anomaly signal. Dark comets are a known class of NEOs with non-gravitational "
                "acceleration from sublimation without a visible coma or tail. 1998 KY26 is a "
                "candidate dark comet. A Δ flag alone is insufficient to classify an object as "
                "artificial; measured A1/A2/A3 and comparative physical fits are required.[/dim]"
            )
        lines.append(
            "\n[dim]Options 1–3 provide screening ranks, not calibrated sigma significance.[/dim]"
        )
        self.display_panel("\n".join(lines), title=f"ATLAS Detail — {designation}", style="yellow")

    def _display_impact_result(self, designation: str, impact) -> None:
        from aneos_core.analysis.impact_evidence import require_sentry_report, render_sentry_markdown
        try:
            report = require_sentry_report(impact, designation)
            text = render_sentry_markdown(report)
        except ValueError:
            self.show_error("Legacy, modified or unbound impact record: scientific evidence unavailable.")
            return
        self.display_panel(text, title=f"Scoped Sentry Evidence — {report['designation']}", style="cyan")

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

        # Show the actual candidate designations and their anomaly flags
        candidates = getattr(pipeline_result, "validated_candidates", None) if pipeline_result else None
        if candidates:
            cand_rows = []
            for c in candidates:
                desg = c.get("designation", "?")
                ms = c.get("multi_stage_validation", {})
                fs = c.get("first_stage_score", {})
                score = self._format_score(c.get("required_validation_assessment", {}).get("screening_score"))
                method = ms.get("validation_method", "—")
                flags = fs.get("flag_string", "—")
                dist = c.get("miss_distance_au")
                dist_str = f"{dist:.4f} AU" if dist is not None else "—"
                cand_rows.append([desg, score, flags, dist_str, method])
            self.display_table(
                headers=["Designation", "Score", "Flags", "Miss dist", "Method"],
                rows=cand_rows,
                title=f"Final Candidates ({len(cand_rows)})",
            )

    def _store_pipeline_candidates(self, result: dict) -> None:
        """Store pipeline validated candidates in _detection_results for option 9/13."""
        import types
        pipeline_result = result.get("pipeline_result")
        if pipeline_result is None:
            return
        candidates = getattr(pipeline_result, "validated_candidates", None) or []
        for c in candidates:
            desg = c.get("designation", "UNKNOWN")
            ms = c.get("multi_stage_validation", {})
            proxy = types.SimpleNamespace(
                sigma_level=None,
                artificial_probability=None,
                screening_score=c.get("required_validation_assessment", {}).get("screening_score"),
                confidence=ms.get("confidence"),
                classification=f"PIPELINE:{ms.get('validation_method', '?')}",
                analysis=c.get("first_stage_score", {}),
                # Extra fields for rich detail view
                miss_distance_au=c.get("miss_distance_au"),
                relative_velocity_km_s=c.get("relative_velocity_km_s"),
                multi_stage_validation=ms,
            )
            self._detection_results[desg.upper()] = proxy
        if candidates:
            self.show_info(f"Stored {len(candidates)} pipeline candidates — available in options 9 and 13.")

    def _display_pattern_result(self, result: dict) -> None:
        n = result.get("designations_analyzed", 0)
        clusters = result.get("clusters", [])
        lines = [
            f"[bold]Objects analysed:[/bold]  {n}",
            f"[bold]Descriptive groups:[/bold] {len(clusters)}",
            "Statistical significance: not calibrated. Origin: unresolved.",
        ]
        if result.get('schema_version') != 'network-evidence-report-v1':
            lines.append('Unverified legacy/prototype output: no significance or origin claim accepted.')
            clusters = []
        else:
            from aneos_core.pattern_analysis.session import verify_network_report
            try:
                verify_network_report(result)
            except (ValueError, TypeError):
                self.show_error('Population report failed reproducibility validation.')
                return
            for module, info in result.get('modules', {}).items():
                lines.append(f"{module}: {info['status']}")
            candidates = sum(item['assessment']['candidate'] for item in result['object_assessments'])
            lines.append(f'Follow-up candidates (ranking, not origin): {candidates}')
            if result['excluded_by_cap']:
                lines.append(f"Excluded by configured cap: {len(result['excluded_by_cap'])}")
        if clusters:
            rows = [[str(c['cluster_id']), str(c['n_members']), 'Catalogue neighbourhood']
                    for c in clusters[:10]]
            self.display_table(headers=['Group ID', 'Members', 'Basis'], rows=rows,
                title=f'Descriptive groups (first {len(rows)} of {len(clusters)})')
        self.display_panel("\n".join(lines), title="Population Pattern Analysis", style="cyan")

    def _show_db_result_for(self, designation: str) -> None:
        """Show persisted DB results filtered to a specific designation."""
        try:
            from aneos_api.database import SessionLocal, HAS_SQLALCHEMY
            if not HAS_SQLALCHEMY:
                return
            from aneos_api.database import AnalysisService
            db = SessionLocal()
            try:
                service = AnalysisService(db)
                matches = service.get_analysis_results(designation=designation, limit=20)
                if matches:
                    rows = [
                        [
                            r.get("designation", "?"),
                            r.get("classification", "—"),
                            str(r.get("analysis_date", ""))[:19],
                        ]
                        for r in matches
                    ]
                    self.display_table(
                        headers=["Designation", "Classification", "Analysis date"],
                        rows=rows,
                        title=f"Persisted results for {designation}",
                    )
            finally:
                db.close()
        except Exception:
            pass

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
        """Session candidate counts; never pool unavailable sigma as zero."""
        if getattr(self, '_batch_reports', []):
            from aneos_core.analysis.batch_reporting import verify_batch_report
            try:
                batches = [verify_batch_report(r) for r in self._batch_reports]
            except ValueError as exc:
                self.show_error(f'Batch accounting unavailable: {exc}')
                return
            count = sum(b['summary']['request_count'] for b in batches)
            labels = {r['designation'] for b in batches for r in b['requests']}
            self.show_info(f'Batch session: {len(batches)} runs; {count} request occurrences; '
                           f'{len(labels)} distinct exact designation labels. '
                           'Requests are not independent observations. Full history: options 9/10.')
        if not self._detection_results:
            self.show_info('No latest-object screening views this session; batch request history is separate.')
            return
        rows = []
        for designation, result in self._detection_results.items():
            rows.append([designation, self._format_score(getattr(result, 'screening_score', None)),
                         getattr(result, 'classification', 'INCONCLUSIVE')])
        self.display_table(headers=['Designation', 'Screening score', 'Classification'], rows=rows,
                           title='Latest-object screening views — not full request history')
        fmt = self._ask('Export JSON? [y/N]').strip().lower()
        if fmt == 'y':
            from aneos_core.reporting.exporters import Exporter
            records = [(getattr(r, 'analysis', {}) or {}).get('assessment', {})
                       for r in self._detection_results.values()]
            try:
                path = Exporter().export_to_json(records, exclusive=True)
                self.show_success(f'Exported latest-object views to: {path}. Full batch history uses option 10.')
            except Exception as exc:
                self.show_error(f'Export refused: {exc}')
        else:
            self.show_info("Export skipped.")

    def _show_help(self) -> None:
        """Option 14: Display scientific methodology documentation."""
        doc_path = Path(__file__).parent / 'docs' / 'scientific' / 'screening-guide.md'
        if not doc_path.exists():
            self.show_error(f'Current scientific guide not found: {doc_path}')
            return
        try:
            text = doc_path.read_text(encoding='utf-8')
            if self.console:
                from rich.markdown import Markdown
                from rich.panel import Panel
                self.console.print(Panel(Markdown(text),title='Scientific guide — evidence and limits',
                                         border_style='blue',padding=(1,1)))
            else:
                print(text)
        except Exception as exc:
            self.show_error(f'Failed to read current scientific guide: {exc}')

    def _rendezvous_scan(self) -> None:
        """Option 15: PA-6 MOID-based pairwise rendezvous scan of all PHAs."""
        import asyncio
        self.show_info(
            "Scanning all ~2,400 Potentially Hazardous Asteroids for anomalous\n"
            "pairwise orbital proximity (no Earth close-approach required).\n"
            "This checks: orbital similarity (Drummond D), period resonance, A2 correlation.\n"
            "Expect 30–60 s for SBDB fetch + pairwise computation."
        )
        try:
            from aneos_core.pattern_analysis.rendezvous import PHAMoidScanner
            scanner = PHAMoidScanner()
            pairs = asyncio.run(scanner.run(max_pairs=30))
        except Exception as exc:
            self.show_error(f"Rendezvous scan failed: {exc}")
            return

        if not pairs:
            self.show_info("No anomalous pairs found above threshold.")
            return

        rows = []
        for p in pairs:
            flags = ", ".join(p.flags)
            resonance = p.resonance or "—"
            a2 = "yes" if p.a2_correlated else "—"
            rows.append([
                p.designation_a, p.designation_b,
                f"{p.u_d:.4f}", resonance, a2,
                f"{p.anomaly_score:.3f}", flags,
            ])
        self.display_table(
            headers=["Object A", "Object B", "U_D", "Resonance", "A2 corr", "Score", "Flags"],
            rows=rows,
            title=f"Rendezvous Scan — Top {len(rows)} Anomalous Pairs",
        )
        self.show_info(
            "U_D = Drummond orbital similarity (lower = more similar).\n"
            "Resonance = period ratio near a simple fraction.\n"
            "A2 corr = both objects have significant same-sign non-gravitational acceleration."
        )

    @staticmethod
    def _rough_torino(prob: float, energy_mt: Optional[float]) -> str:
        """Controlled deprecation: no local Torino class from heuristic inputs."""
        raise NotImplementedError('Use a source-reported Torino value within its documented scope')


def main():
    """Start the interactive menu when this module is executed as a script."""
    ANEOSMenuV2().run()


if __name__ == '__main__':
    main()
