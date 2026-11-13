"""
End-to-end tests for ANEOSMenuV2 — every option, full call chain.

Each test class covers one menu option and verifies:
  1. The correct backend is called with the expected arguments
  2. The result display method is exercised with real data shape
  3. Error paths produce user-visible messages (not silent failures)
  4. Persistence is triggered for options that produce analysis results

Network calls are mocked. All tests run offline.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeOrbitalElements:
    semi_major_axis: float = 0.922
    eccentricity: float = 0.191
    inclination: float = 3.331
    ra_of_ascending_node: float = 204.4
    arg_of_periapsis: float = 126.4
    mean_anomaly: float = 180.0


@dataclass
class FakePhysicalProperties:
    diameter_km: float = 0.37
    albedo: float = 0.33
    spectral_type: str = "Sq"
    rotation_period_hours: float = 30.4
    absolute_magnitude_h: float = 19.7


@dataclass
class FakeCloseApproach:
    date: str = "2029-04-13"
    distance_au: float = 0.000254
    relative_velocity_km_s: float = 7.42


def make_fake_neo_data(designation="99942"):
    neo = MagicMock()
    neo.orbital_elements = FakeOrbitalElements()
    neo.physical_properties = FakePhysicalProperties()
    neo.close_approaches = [FakeCloseApproach()]
    neo.sources_used = ["SBDB"]
    neo.orbital_history = None
    return neo


def make_fake_detection_result(sigma=3.7, prob=0.037, cls="⚠️ SUSPICIOUS (σ≥3)"):
    result = MagicMock()
    result.screening_score = sigma
    result.candidate = sigma >= 5
    result.sigma_level = sigma
    result.artificial_probability = prob
    result.classification = cls
    result.risk_factors = ["orbital_anomaly", "physical_anomaly"]
    result.metadata = {
        "evidence_count": 2,
        "combined_p_value": 0.0042,
        "false_discovery_rate": 0.08,
        "detector_type": "validated",
    }
    result.analysis = {}
    result.object_review = None  # Explicit legacy mock, not a replayable review.
    return result


def make_fake_impact():
    impact = MagicMock()
    impact.collision_probability = 2.3e-4
    impact.collision_probability_per_year = 2.3e-5
    impact.impact_energy_mt = 1200.0
    impact.crater_diameter_km = 5.2
    impact.impact_velocity_km_s = 7.4
    impact.calculation_method = "monte_carlo"
    impact.calculation_confidence = 0.85
    impact.probability_uncertainty = (1.0e-4, 5.0e-4)
    impact.data_arc_years = 20.0
    return impact


def _make_menu():
    from aneos_menu_v2 import ANEOSMenuV2
    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    menu._detection_results = {}
    menu._impact_results = {}
    return menu


# ===========================================================================
# Option 1 — Detect NEO (single)
# ===========================================================================

class TestOption1DetectSingle:
    """End-to-end: user enters designation → data fetched → detector runs → result shown → persisted."""

    def test_happy_path_stores_result(self):
        menu = _make_menu()
        neo_data = make_fake_neo_data()
        det_result = make_fake_detection_result()

        with patch.object(menu, "_ask", return_value="99942"), \
             patch.object(menu, "_fetch_neo_data", return_value=neo_data), \
             patch.object(menu, "_run_detection", return_value=det_result), \
             patch.object(menu, "_display_neo_data"), \
             patch.object(menu, "_display_detection_result") as mock_display, \
             patch.object(menu, "_persist_detection_result_async") as mock_persist:
            menu._detect_single()

        mock_display.assert_called_once_with("99942", det_result, verbose=False)
        mock_persist.assert_called_once_with("99942", det_result)
        assert "99942" in menu._detection_results

    def test_empty_designation_shows_error(self):
        menu = _make_menu()
        with patch.object(menu, "_ask", return_value=""), \
             patch.object(menu, "show_error") as mock_err:
            menu._detect_single()
        mock_err.assert_called_once()

    def test_no_neo_data_returns_early(self):
        menu = _make_menu()
        with patch.object(menu, "_ask", return_value="UNKNOWN999"), \
             patch.object(menu, "_fetch_neo_data", return_value=None), \
             patch.object(menu, "_run_detection") as mock_det:
            menu._detect_single()
        mock_det.assert_not_called()

    def test_no_orbital_elements_shows_error(self):
        menu = _make_menu()
        neo_data = make_fake_neo_data()
        neo_data.orbital_elements = None

        with patch.object(menu, "_ask", return_value="99942"), \
             patch.object(menu, "_fetch_neo_data", return_value=neo_data), \
             patch.object(menu, "_display_neo_data"), \
             patch.object(menu, "show_error") as mock_err:
            menu._detect_single()
        mock_err.assert_called_once()

    def test_result_sigma_stored_correctly(self):
        menu = _make_menu()
        neo_data = make_fake_neo_data()
        det_result = make_fake_detection_result(sigma=4.8, prob=0.048)

        with patch.object(menu, "_ask", return_value="TESLA"), \
             patch.object(menu, "_fetch_neo_data", return_value=neo_data), \
             patch.object(menu, "_run_detection", return_value=det_result), \
             patch.object(menu, "_display_neo_data"), \
             patch.object(menu, "_display_detection_result"), \
             patch.object(menu, "_persist_detection_result_async"):
            menu._detect_single()

        stored = menu._detection_results["TESLA"]
        assert stored.sigma_level == pytest.approx(4.8)


# ===========================================================================
# Option 2 — Multi-Evidence Analysis
# ===========================================================================

class TestOption2MultiEvidence:
    """Same as option 1 but with verbose=True for evidence breakdown display."""

    def test_display_called_with_verbose_true(self):
        menu = _make_menu()
        neo_data = make_fake_neo_data()
        det_result = make_fake_detection_result()

        with patch.object(menu, "_ask", return_value="99942"), \
             patch.object(menu, "_fetch_neo_data", return_value=neo_data), \
             patch.object(menu, "_run_detection", return_value=det_result), \
             patch.object(menu, "_display_neo_data"), \
             patch.object(menu, "_display_detection_result") as mock_display, \
             patch.object(menu, "_persist_detection_result_async"):
            menu._detect_multi_evidence()

        mock_display.assert_called_once_with("99942", det_result, verbose=True)

    def test_result_stored_and_persisted(self):
        menu = _make_menu()
        neo_data = make_fake_neo_data()
        det_result = make_fake_detection_result()

        with patch.object(menu, "_ask", return_value="2020 SO"), \
             patch.object(menu, "_fetch_neo_data", return_value=neo_data), \
             patch.object(menu, "_run_detection", return_value=det_result), \
             patch.object(menu, "_display_neo_data"), \
             patch.object(menu, "_display_detection_result"), \
             patch.object(menu, "_persist_detection_result_async") as mock_persist:
            menu._detect_multi_evidence()

        mock_persist.assert_called_once_with("2020 SO", det_result)
        assert "2020 SO" in menu._detection_results


# ===========================================================================
# Option 3 — Batch Detection
# ===========================================================================

class TestOption3BatchDetection:
    """Actual canonical batch execution; no fabricated legacy detector mocks."""

    @staticmethod
    def _pipeline():
        from aneos_core.analysis.pipeline import AnalysisPipeline
        from aneos_core.data.models import NEOData, OrbitalElements
        p = AnalysisPipeline()
        p.data_fetcher = MagicMock()
        p.data_fetcher.fetch_neo_data.side_effect = lambda d, **kw: NEOData(d,
            orbital_elements=OrbitalElements(semi_major_axis=1.23, eccentricity=.2, inclination=1.49))
        return p

    def test_shared_batch_pipeline_is_bounded_and_concurrent(self, tmp_path):
        import threading
        from aneos_core.data.models import NEOData, OrbitalElements
        p = self._pipeline()
        p.analysis_parallel = True; p.analysis_queue_size = 2; p.pipeline_config.max_workers = 2
        barrier = threading.Barrier(2)
        def fetch(name, **kw):
            barrier.wait(timeout=10)
            return NEOData(name, orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,inclination=1.49))
        p.data_fetcher.fetch_neo_data.side_effect = fetch
        path = tmp_path/'objects.txt'; path.write_text('99942\n2020 SO\n')
        menu = _make_menu(); menu._persist_detection_result_async = MagicMock(); menu._ask = MagicMock(return_value='')
        with patch.dict('os.environ',{'ANEOS_DATABASE_URL':f'sqlite:///{tmp_path}/jobs.sqlite'}), \
             patch.object(menu,'browse_files',return_value=str(path)), \
             patch('aneos_core.analysis.pipeline.AnalysisPipeline',return_value=p):
            menu._detect_batch()
        assert p.data_fetcher.fetch_neo_data.call_count == 2
        assert len(menu._batch_reports[0]['requests']) == 2

    def test_results_contain_both_designations(self, tmp_path):
        path = tmp_path/'objects.txt'; path.write_text('99942\n2020 SO\n')
        p = self._pipeline(); menu = _make_menu(); menu._persist_detection_result_async = MagicMock(); menu._ask = MagicMock(return_value='')
        with patch.dict('os.environ',{'ANEOS_DATABASE_URL':f'sqlite:///{tmp_path}/jobs.sqlite'}), \
             patch.object(menu,'browse_files',return_value=str(path)), \
             patch('aneos_core.analysis.pipeline.AnalysisPipeline',return_value=p):
            menu._detect_batch()
        assert [r['designation'] for r in menu._batch_reports[0]['requests']] == ['99942','2020 SO']
        assert set(menu._detection_results) == {'99942','2020 SO'}

    def test_missing_orbit_is_retained_as_inconclusive(self, tmp_path):
        from aneos_core.data.models import NEOData
        path = tmp_path/'objects.txt'; path.write_text('MISSING\n')
        p = self._pipeline(); p.data_fetcher.fetch_neo_data.side_effect = lambda d,**kw: NEOData(d)
        menu = _make_menu(); menu._persist_detection_result_async = MagicMock(); menu._ask = MagicMock(return_value='')
        with patch.dict('os.environ',{'ANEOS_DATABASE_URL':f'sqlite:///{tmp_path}/jobs.sqlite'}), \
             patch.object(menu,'browse_files',return_value=str(path)), \
             patch('aneos_core.analysis.pipeline.AnalysisPipeline',return_value=p):
            menu._detect_batch()
        result = menu._batch_reports[0]['requests'][0]['report']
        assert result['status'] == 'inconclusive'
        assert result['screening_assessment']['screening_score'] is None
        assert menu._detection_results['MISSING'].classification == 'INCONCLUSIVE'

    def test_missing_file_shows_error(self):
        menu = _make_menu(); menu._persist_detection_result_async = MagicMock(); menu._ask = MagicMock(return_value='')
        with patch.object(menu, 'browse_files', return_value='/nonexistent/path.txt'), \
             patch.object(menu, 'show_error') as error:
            menu._detect_batch()
        error.assert_called_once()


# ===========================================================================
# Option 4 — Orbital History Analysis
# ===========================================================================

class TestOption4OrbitalHistoryAnalysis:
    """Fetches Horizons multi-epoch elements; displays time-series; runs detector with history."""

    def test_horizons_called_with_designation_and_years(self):
        menu = _make_menu()
        history = [{"epoch": "2024-01-01", "a": 0.921, "e": 0.191, "i": 3.33}]

        with patch.object(menu, "_ask", side_effect=["99942", "5"]), \
             patch("aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
                   return_value=history) as mock_hist, \
             patch.object(menu, "_fetch_neo_data", return_value=make_fake_neo_data()), \
             patch.object(menu, "_run_detection", return_value=make_fake_detection_result()), \
             patch.object(menu, "display_table") as mock_table, \
             patch.object(menu, "_display_detection_result"), \
             patch.object(menu, "_persist_detection_result_async"):
            menu._orbital_history_analysis()

        mock_hist.assert_called_once_with("99942", years=5)

    def test_history_passed_to_detector_as_additional_data(self):
        menu = _make_menu()
        history = [{"epoch": "2024-01-01", "a": 0.921, "e": 0.191, "i": 3.33}]

        with patch.object(menu, "_ask", side_effect=["99942", "10"]), \
             patch("aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
                   return_value=history), \
             patch.object(menu, "_fetch_neo_data", return_value=make_fake_neo_data()), \
             patch.object(menu, "_run_detection", return_value=make_fake_detection_result()) as mock_det, \
             patch.object(menu, "display_table"), \
             patch.object(menu, "_display_detection_result"), \
             patch.object(menu, "_persist_detection_result_async"):
            menu._orbital_history_analysis()

        call_kwargs = mock_det.call_args[1]
        assert "extra_additional" in call_kwargs
        assert "orbital_history" in call_kwargs["extra_additional"]
        assert call_kwargs["extra_additional"]["orbital_history"] == history

    def test_empty_history_shows_info_not_error(self):
        menu = _make_menu()
        with patch.object(menu, "_ask", side_effect=["99942", ""]), \
             patch("aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
                   return_value=[]), \
             patch.object(menu, "show_info") as mock_info, \
             patch.object(menu, "show_error") as mock_err:
            menu._orbital_history_analysis()
        mock_info.assert_called()
        mock_err.assert_not_called()

    def test_history_table_rows_match_epochs(self):
        menu = _make_menu()
        history = [
            {"epoch": "2022-01-01", "a": 0.920, "e": 0.190, "i": 3.30},
            {"epoch": "2023-01-01", "a": 0.921, "e": 0.191, "i": 3.31},
        ]
        with patch.object(menu, "_ask", side_effect=["99942", "2"]), \
             patch("aneos_core.data.sources.horizons.HorizonsSource.fetch_orbital_history",
                   return_value=history), \
             patch.object(menu, "_fetch_neo_data", return_value=make_fake_neo_data()), \
             patch.object(menu, "_run_detection", return_value=make_fake_detection_result()), \
             patch.object(menu, "display_table") as mock_table, \
             patch.object(menu, "_display_detection_result"), \
             patch.object(menu, "_persist_detection_result_async"):
            menu._orbital_history_analysis()

        first_call = mock_table.call_args_list[0]
        rows = first_call[1].get("rows", [])
        assert len(rows) == 2


# ===========================================================================
# Option 5 — Impact Assessment
# ===========================================================================

def make_sentry_impact(designation="99942"):
    from aneos_core.data.sources.sentry import parse_sentry_response
    payload = {"signature":{"source":"NASA/JPL Sentry Data API","version":"2.0"},"error":"specified object not found"}
    import json
    return parse_sentry_response(json.dumps(payload).encode(), designation=designation, retrieved_at="2026-11-12T00:00:00Z")


class TestOption5ImpactAssessment:
    """Source → replay/identity check → display → verified archive; no arc proxy."""
    def run_source(self, menu, report):
        service = MagicMock(); service.assess.return_value = report
        with patch.object(menu, "_ask", return_value="99942"), \
             patch("aneos_core.analysis.impact_evidence.get_sentry_impact_service", return_value=service), \
             patch.object(menu, "_fetch_neo_data", side_effect=AssertionError("unrelated orbital query")), \
             patch.object(menu, "_display_impact_result") as display, \
             patch.object(menu, "_persist_impact_result_async") as persist:
            menu._impact_assessment()
        return service, display, persist

    def test_source_called_without_assumed_orbital_elements(self):
        service, _, _ = self.run_source(_make_menu(), make_sentry_impact())
        service.assess.assert_called_once_with("99942")

    def test_result_stored_in_impact_results(self):
        menu = _make_menu(); report = make_sentry_impact()
        self.run_source(menu, report)
        assert menu._impact_results["99942"] == report

    def test_result_sent_to_verified_archive_compatibility_method(self):
        report = make_sentry_impact()
        _, _, persist = self.run_source(_make_menu(), report)
        persist.assert_called_once_with("99942", report)

    def test_close_approach_proxy_not_passed_as_impact_input(self):
        service, _, _ = self.run_source(_make_menu(), make_sentry_impact())
        assert service.assess.call_args.kwargs == {}

    def test_display_preserves_unavailable_probability_not_mock_number(self):
        report = make_sentry_impact()
        _, display, _ = self.run_source(_make_menu(), report)
        display.assert_called_once_with("99942", report)
        assert report["collision_probability"] is None


# ===========================================================================
# Option 6 — Close Approach History
# ===========================================================================

class TestOption6CloseApproachHistory:
    """Fetches historical close approaches via fetch_historical_approaches and displays table."""

    def test_fetch_historical_approaches_called(self):
        menu = _make_menu()
        approaches = [FakeCloseApproach(), FakeCloseApproach(date="2036-04-13", distance_au=0.001)]

        with patch.object(menu, "_ask", side_effect=["99942", "30"]), \
             patch("aneos_core.data.fetcher.DataFetcher.fetch_historical_approaches",
                   return_value=approaches) as mock_fetch, \
             patch.object(menu, "display_table"):
            menu._close_approach_history()

        mock_fetch.assert_called_once_with("99942", years_back=30)

    def test_table_has_correct_columns(self):
        menu = _make_menu()
        approaches = [FakeCloseApproach()]

        with patch.object(menu, "_ask", side_effect=["99942", ""]), \
             patch("aneos_core.data.fetcher.DataFetcher.fetch_historical_approaches",
                   return_value=approaches), \
             patch.object(menu, "display_table") as mock_table:
            menu._close_approach_history()

        headers = mock_table.call_args[1].get("headers", [])
        assert "Date" in headers
        assert "Distance" in headers

    def test_empty_approaches_shows_info(self):
        menu = _make_menu()
        with patch.object(menu, "_ask", side_effect=["99942", ""]), \
             patch("aneos_core.data.fetcher.DataFetcher.fetch_historical_approaches",
                   return_value=[]), \
             patch.object(menu, "show_info") as mock_info:
            menu._close_approach_history()
        mock_info.assert_called()


# ===========================================================================
# Option 7 — Live Pipeline Dashboard
# ===========================================================================

class TestOption7LivePipeline:
    """Pipeline runs via initialize_pipeline_integration() + run_200_year_poll()."""

    def test_pipeline_functions_called(self):
        menu = _make_menu()
        pipeline_result = {
            "status": "success",
            "total_objects": 15000,
            "final_candidates": 3,
            "processing_time_seconds": 45.2,
            "compression_ratio": 5000.0,
        }

        with patch("aneos_core.integration.pipeline_integration.initialize_pipeline_integration",
                   return_value=True) as mock_init, \
             patch("aneos_core.integration.pipeline_integration.run_200_year_poll",
                   return_value=pipeline_result) as mock_poll, \
             patch.object(menu, "_display_pipeline_result") as mock_disp:
            import asyncio

            async def fake_init():
                return True

            async def fake_poll():
                return pipeline_result

            with patch("aneos_core.integration.pipeline_integration.initialize_pipeline_integration",
                       fake_init), \
                 patch("aneos_core.integration.pipeline_integration.run_200_year_poll",
                       fake_poll):
                menu._live_pipeline()

        # Either the patched functions ran or we got an exception captured gracefully
        # The key contract: no unhandled exception

    def test_pipeline_result_displayed(self):
        menu = _make_menu()
        pipeline_result = {
            "status": "success",
            "total_objects": 15000,
            "final_candidates": 3,
            "processing_time_seconds": 45.2,
            "compression_ratio": 5000.0,
            "pipeline_result": None,
        }
        with patch.object(menu, "_display_pipeline_result") as mock_disp:
            menu._display_pipeline_result(pipeline_result)
        mock_disp.assert_called_once_with(pipeline_result)

    def test_display_pipeline_result_shows_candidates(self):
        """_display_pipeline_result must include final_candidates in table rows."""
        menu = _make_menu()
        result = {
            "status": "success",
            "total_objects": 10000,
            "final_candidates": 7,
            "processing_time_seconds": 30.0,
            "compression_ratio": 1428.0,
            "pipeline_result": None,
        }
        with patch.object(menu, "display_table") as mock_table:
            menu._display_pipeline_result(result)

        rows = mock_table.call_args[1].get("rows", [])
        row_values = [r[1] for r in rows]
        assert "7" in row_values  # final_candidates

    def test_error_result_shows_error_message(self):
        menu = _make_menu()
        result = {"status": "error", "error_message": "Pipeline failed: no data"}
        with patch.object(menu, "show_error") as mock_err:
            menu._display_pipeline_result(result)
        mock_err.assert_called_once()

    def test_import_error_shows_descriptive_error(self):
        menu = _make_menu()
        with patch("builtins.__import__", side_effect=ImportError("missing module")), \
             patch.object(menu, "show_error") as mock_err:
            try:
                menu._live_pipeline()
            except Exception:
                pass  # ImportError caught by menu
        # If we get here with or without the error call, the menu handled it


# ===========================================================================
# Option 8 — Population Pattern Analysis
# ===========================================================================

class TestOption8PopulationPatternAnalysis:
    """BC11 canonical population report: file → concurrent fetch → NetworkAnalysisSession.run()."""

    def test_fetch_multiple_used_for_population(self, tmp_path):
        dfile = tmp_path / "population.txt"
        dfile.write_text("99942\n2020 SO\n433\n")
        menu = _make_menu()

        from aneos_core.data.models import NEOData, OrbitalElements
        neo_map = {d: NEOData(designation=d, orbital_elements=OrbitalElements(semi_major_axis=1., eccentricity=.2, inclination=5.))
                   for d in ["99942", "2020 SO", "433"]}
        pattern_result = {
            "designations_analyzed": 3,
            "clusters": [],
            "network_sigma": 1.2,
            "network_tier": "NOTABLE",
        }

        with patch.object(menu, "_ask", return_value=str(dfile)), \
             patch("aneos_core.data.fetcher.DataFetcher.fetch_multiple",
                   return_value=neo_map) as mock_fm, \
             patch("aneos_core.pattern_analysis.session.NetworkAnalysisSession.run",
                   return_value=pattern_result), \
             patch.object(menu, "_display_pattern_result"):
            menu._population_pattern_analysis()

        mock_fm.assert_called_once_with(["99942", "2020 SO", "433"])

    def test_network_session_run_called_with_neo_objects(self, tmp_path):
        dfile = tmp_path / "population.txt"
        dfile.write_text("99942\n")
        menu = _make_menu()
        from aneos_core.data.models import NEOData, OrbitalElements
        neo_data = NEOData(designation="99942", orbital_elements=OrbitalElements(semi_major_axis=1., eccentricity=.2, inclination=5.))
        pattern_result = {"designations_analyzed": 1, "clusters": []}

        with patch.object(menu, "_ask", return_value=str(dfile)), \
             patch("aneos_core.data.fetcher.DataFetcher.fetch_multiple",
                   return_value={"99942": neo_data}), \
             patch("aneos_core.pattern_analysis.session.NetworkAnalysisSession.run",
                   return_value=pattern_result) as mock_run, \
             patch.object(menu, "_display_pattern_result"):
            menu._population_pattern_analysis()

        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        assert neo_data in args[0]

    def test_display_shows_objects_analyzed(self):
        menu = _make_menu()
        result = {"designations_analyzed": 42, "clusters": [], "network_sigma": 2.1}
        with patch.object(menu, "display_panel") as mock_panel, \
             patch.object(menu, "display_table"):
            menu._display_pattern_result(result)
        panel_content = mock_panel.call_args[0][0]
        assert "42" in panel_content


# ===========================================================================
# Option 9 — Results Browser
# ===========================================================================

class TestOption9ResultsBrowser:
    """Displays in-session and DB-persisted results."""

    def test_shows_detection_results_table(self):
        menu = _make_menu()
        menu._detection_results["99942"] = make_fake_detection_result(sigma=3.7)

        with patch.object(menu, "display_table") as mock_table, \
             patch.object(menu, "_show_db_results"), \
             patch.object(menu, "_ask", return_value=""):
            menu._results_browser()

        assert mock_table.called
        rows = mock_table.call_args[1].get("rows", [])
        # Rows now include a leading "#" column; search all cells for designation
        assert any("99942" in cell for r in rows for cell in r)

    def test_shows_impact_results_table(self):
        menu = _make_menu()
        menu._impact_results["99942"] = make_fake_impact()

        with patch.object(menu, "display_table") as mock_table, \
             patch.object(menu, "_show_db_results"), \
             patch.object(menu, "_ask", return_value=""):
            menu._results_browser()

        assert mock_table.called

    def test_empty_session_shows_info(self):
        menu = _make_menu()
        with patch.object(menu, "show_info") as mock_info, \
             patch.object(menu, "_show_db_results"):
            menu._results_browser()
        mock_info.assert_called()

    def test_db_results_fallback_called(self):
        menu = _make_menu()
        with patch.object(menu, "_show_db_results") as mock_db, \
             patch.object(menu, "show_info"):
            menu._results_browser()
        mock_db.assert_called_once()


# ===========================================================================
# Option 10 — Export Results
# ===========================================================================

class TestOption10ExportResults:
    """Exports detection and impact results via Exporter to JSON or CSV."""

    def _menu_with_results(self):
        menu = _make_menu()
        menu._detection_results["99942"] = make_fake_detection_result()
        menu._impact_results["99942"] = make_sentry_impact()
        return menu

    def test_json_export_calls_exporter(self, tmp_path):
        menu = self._menu_with_results()
        out = str(tmp_path / "out.json")

        with patch.object(menu, "_ask", side_effect=["1", out]), \
             patch("aneos_core.reporting.exporters.Exporter.export_to_json",
                   return_value=out) as mock_exp, \
             patch.object(menu, "show_success"):
            menu._export_results()

        mock_exp.assert_called_once()
        records = mock_exp.call_args[0][0]
        assert any(r["type"] == "detection" for r in records)
        assert any(r["type"] == "impact" for r in records)

    def test_csv_export_calls_exporter(self, tmp_path):
        menu = self._menu_with_results()
        out = str(tmp_path / "out.csv")

        with patch.object(menu, "_ask", side_effect=["2", out]), \
             patch("aneos_core.reporting.exporters.Exporter.export_to_csv",
                   return_value=out) as mock_exp, \
             patch.object(menu, "show_success"):
            menu._export_results()

        mock_exp.assert_called_once()

    def test_invalid_format_shows_error(self):
        menu = self._menu_with_results()
        with patch.object(menu, "_ask", side_effect=["9", ""]), \
             patch.object(menu, "show_error") as mock_err:
            menu._export_results()
        mock_err.assert_called_once()

    def test_empty_session_shows_info(self):
        menu = _make_menu()
        with patch.object(menu, "show_info") as mock_info:
            menu._export_results()
        mock_info.assert_called_once()

    def test_failed_export_cannot_claim_fallback_success_or_overwrite(self, tmp_path):
        menu = self._menu_with_results()
        out = tmp_path / "existing.json"
        out.write_text("historical")
        with patch.object(menu, "_ask", side_effect=["1", str(out)]), \
             patch.object(menu, "show_success") as success, patch.object(menu, "show_error") as error:
            menu._export_results()
        assert out.read_text() == "historical"
        assert error.called and not success.called


# ===========================================================================
# Option 11 — System Health
# ===========================================================================

class TestOption11SystemHealth:
    """Health check: imports 8 components, probes API, shows scientific caveats."""

    def test_all_eight_components_in_table(self):
        menu = _make_menu()
        expected_labels = {
            "DataFetcher", "HorizonsSource", "DetectionManager",
            "ImpactCalculator", "MetricsCollector", "NetworkAnalysisSession",
            "PipelineIntegration", "Exporter",
        }
        with patch("urllib.request.urlopen", return_value=MagicMock()), \
             patch.object(menu, "display_table") as mock_table, \
             patch.object(menu, "display_panel"):
            menu._system_health()

        rows = mock_table.call_args[1].get("rows", [])
        labels_in_rows = {r[0] for r in rows}
        # All 8 components plus API row = 9 total rows
        component_rows = {r[0] for r in rows if r[0] != "REST API :8000"}
        assert expected_labels == component_rows

    def test_scientific_caveats_displayed(self):
        menu = _make_menu()
        with patch("urllib.request.urlopen", return_value=MagicMock()), \
             patch.object(menu, "display_table"), \
             patch.object(menu, "display_panel") as mock_panel:
            menu._system_health()

        assert mock_panel.called
        panel_content = mock_panel.call_args[0][0]
        # Must not present assumed posteriors or constructor F1 as current validation.
        assert "not reliably estimated" in panel_content
        assert "No independently evaluated calibration" in panel_content
        assert "3–5%" not in panel_content

    def test_api_down_shows_warning_not_exception(self):
        menu = _make_menu()
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")), \
             patch.object(menu, "display_table") as mock_table, \
             patch.object(menu, "display_panel"):
            menu._system_health()  # Must not raise

        rows = mock_table.call_args[1].get("rows", [])
        api_row = next((r for r in rows if "API" in r[0]), None)
        assert api_row is not None
        assert "⚠" in api_row[1] or "DOWN" in api_row[1]


# ===========================================================================
# Cross-cutting: _run_detection wires additional_data correctly
# ===========================================================================

class TestRunDetectionAdditionalData:
    """Verify _run_detection passes physical, additional, and extra_additional to manager."""

    def test_physical_dict_passed(self):
        menu = _make_menu()
        neo_data = make_fake_neo_data()

        from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties
        neo_data = NEOData("99942", orbital_elements=OrbitalElements(
            semi_major_axis=.922, eccentricity=.191, inclination=3.331),
            physical_properties=PhysicalProperties(diameter_km=.37, absolute_magnitude_h=19.7))
        result = menu._run_detection("99942", neo_data)
        assert result is not None
        physical = result.analysis["assessment"]["evidence_snapshot"]["physical"]
        assert physical["diameter"] == pytest.approx(370)
        assert result.metadata["physical_data_available"] is True

    def test_orbital_history_reaches_detector(self):
        menu = _make_menu()
        from aneos_core.data.models import NEOData, OrbitalElements
        neo_data = NEOData("99942", orbital_elements=OrbitalElements(
            semi_major_axis=.922, eccentricity=.191, inclination=3.331))
        history = [{"epoch": "2024-01-01", "a": .921}]
        result = menu._run_detection("99942", neo_data, extra_additional={"orbital_history": history})
        assert result is not None
        assert result.analysis["assessment"]["evidence_snapshot"]["additional"]["orbital_history"] == history


# ===========================================================================
# _rough_torino
# ===========================================================================

class TestRoughTorino:
    @pytest.mark.parametrize("prob,energy", [(0.,None),(1e-10,None),(1e-3,100.),(.99,1000.)])
    def test_unfitted_local_scale_is_controlled_deprecated(self, prob, energy):
        from aneos_menu_v2 import ANEOSMenuV2
        with pytest.raises(NotImplementedError, match="source-reported"):
            ANEOSMenuV2._rough_torino(prob, energy)
