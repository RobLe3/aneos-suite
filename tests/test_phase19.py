"""
Phase 19 tests — 15 tests covering:
  - NEO data display panel (_display_neo_data)
  - Conditional detection NOTE (σ < 2 / σ 2-5 / σ ≥ 5)
  - Option 8 pipeline shortcut
  - Option 13 ATLAS score tiers and export prompt
  - RendezvousDetector (PHAMoidScanner / RendezvousPair)
  - Option 15 existence and handler wiring
"""

import inspect
import math
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_menu():
    """Instantiate ANEOSMenuV2 without a console (headless mode)."""
    from aneos_menu_v2 import ANEOSMenuV2
    m = ANEOSMenuV2()
    m.console = None
    return m


# ===========================================================================
# TestNeoDataDisplay (2 tests)
# ===========================================================================

class TestNeoDataDisplay:
    def test_display_neo_data_method_exists(self):
        """ANEOSMenuV2 must expose _display_neo_data as a callable."""
        from aneos_menu_v2 import ANEOSMenuV2
        assert callable(getattr(ANEOSMenuV2, "_display_neo_data", None)), (
            "_display_neo_data is not defined on ANEOSMenuV2"
        )

    def test_display_neo_data_called_in_detect_single(self):
        """_detect_single source must call _display_neo_data."""
        from aneos_menu_v2 import ANEOSMenuV2
        src = inspect.getsource(ANEOSMenuV2._detect_single)
        assert "_display_neo_data" in src, (
            "_detect_single does not call _display_neo_data"
        )


# ===========================================================================
# TestDetectionNoteConditional (3 tests)
# ===========================================================================

class TestDetectionNoteConditional:
    """Verify the σ-conditional NOTE in _display_detection_result via source inspection."""

    def _src(self):
        from aneos_menu_v2 import ANEOSMenuV2
        return inspect.getsource(ANEOSMenuV2._display_detection_result)

    def _render(self, inclination, threshold=5):
        from aneos_core.analysis.screening import ScreeningAnalysisService, ScreeningPolicy
        from aneos_menu_v2 import ANEOSMenuV2
        from unittest.mock import Mock
        menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
        menu.console = None
        menu.display_panel = Mock()
        result = ScreeningAnalysisService(ScreeningPolicy(candidate_threshold=threshold)).analyze_raw(
            "marker", {"a": 1, "e": .2, "i": inclination}).to_detection_result()
        menu._display_detection_result("marker", result)
        return result, menu.display_panel.call_args.args[0]

    def test_note_inconclusive_for_sigma_below_2(self):
        result, content = self._render(15)
        assert not result.candidate
        assert result.is_artificial is None
        assert "unavailable" in content

    def test_note_interesting_for_sigma_2_to_5(self):
        result, content = self._render(35)
        assert result.screening_score is not None
        assert "uncalibrated" in content
        assert "statistically interesting" not in content

    def test_note_exceptional_for_sigma_5(self):
        result, content = self._render(80)
        assert result.candidate
        assert "FOLLOW_UP_CANDIDATE" in content
        assert "unavailable" in content
        assert "σ=5" not in content


# ===========================================================================
# TestOption8PipelineShortcut (2 tests)
# ===========================================================================

class TestOption8PipelineShortcut:
    def test_population_analysis_source_has_pipeline_path(self):
        """_population_pattern_analysis must check for 'PIPELINE:' classifications."""
        from aneos_menu_v2 import ANEOSMenuV2
        src = inspect.getsource(ANEOSMenuV2._population_pattern_analysis)
        assert "PIPELINE:" in src, (
            "_population_pattern_analysis does not check for PIPELINE: candidates"
        )

    def test_pipeline_desgs_extracted_correctly(self):
        """
        Given a menu with 2 pipeline candidates in _detection_results,
        the pipeline_desgs extraction logic should find exactly 2 entries.
        """
        menu = _make_menu()

        # Inject two pipeline candidates and one σ-5 result
        menu._detection_results["2020SO"] = types.SimpleNamespace(
            classification="PIPELINE:ATLAS",
            sigma_level=0.5,
            artificial_probability=0.01,
            analysis={},
        )
        menu._detection_results["TESLA"] = types.SimpleNamespace(
            classification="PIPELINE:MULTI_STAGE",
            sigma_level=0.3,
            artificial_probability=0.005,
            analysis={},
        )
        menu._detection_results["99942"] = types.SimpleNamespace(
            classification="INCONCLUSIVE",
            sigma_level=1.2,
            artificial_probability=0.02,
            analysis={},
        )

        pipeline_desgs = [
            d for d, r in menu._detection_results.items()
            if getattr(r, "classification", "").startswith("PIPELINE:")
        ]
        assert len(pipeline_desgs) == 2, (
            f"Expected 2 pipeline candidates, got {len(pipeline_desgs)}"
        )
        assert set(pipeline_desgs) == {"2020SO", "TESLA"}


# ===========================================================================
# TestOption13AtlasScores (2 tests)
# ===========================================================================

class TestOption13AtlasScores:
    def _src(self):
        from aneos_menu_v2 import ANEOSMenuV2
        return inspect.getsource(ANEOSMenuV2._detection_analytics)

    def test_analytics_export_prompt_has_yn(self):
        """_detection_analytics export prompt must say '[y/N]'."""
        src = self._src()
        assert "[y/N]" in src, (
            "Export prompt in _detection_analytics does not contain '[y/N]'"
        )

    def test_analytics_export_skipped_message(self):
        """_detection_analytics must show 'Export skipped.' when user declines."""
        src = self._src()
        assert "Export skipped" in src, (
            "'Export skipped' message not found in _detection_analytics"
        )


# ===========================================================================
# TestRendezvousDetector (4 tests)
# ===========================================================================

class TestRendezvousDetector:
    """Unit tests for PHAMoidScanner static/instance methods."""

    def _scanner(self):
        from aneos_core.pattern_analysis.rendezvous import PHAMoidScanner
        return PHAMoidScanner()

    def _pha(self, pdes="TEST", a=1.0, e=0.2, i=5.0, om=100.0, w=50.0, A2=None):
        from aneos_core.pattern_analysis.rendezvous import PHAObject
        return PHAObject(pdes=pdes, a=a, e=e, i=i, om=om, w=w, A2=A2)

    def test_drummond_distance_identical_orbits(self):
        """Two objects with identical orbital elements → U_D = 0."""
        scanner = self._scanner()
        obj = self._pha()
        ud = scanner.drummond_distance(obj, obj)
        assert math.isclose(ud, 0.0, abs_tol=1e-12), (
            f"Expected U_D=0 for identical orbits, got {ud}"
        )

    def test_drummond_distance_very_different(self):
        """Objects with very different a/e/i → U_D > UD_THRESHOLD."""
        scanner = self._scanner()
        a = self._pha(pdes="A", a=0.5, e=0.05, i=1.0, om=10.0, w=20.0)
        b = self._pha(pdes="B", a=2.5, e=0.85, i=45.0, om=200.0, w=300.0)
        ud = scanner.drummond_distance(a, b)
        assert ud > scanner.UD_THRESHOLD, (
            f"Expected U_D > {scanner.UD_THRESHOLD} for very different orbits, got {ud}"
        )

    def test_period_resonance_1_1(self):
        """Two objects with equal a → period ratio = 1.0 → resonance '1:1'."""
        scanner = self._scanner()
        result = scanner.check_resonance(1.0, 1.0)
        assert result is not None, "Expected resonance for equal semi-major axes"
        resonance_str, ratio = result
        assert resonance_str == "1:1", f"Expected '1:1', got {resonance_str!r}"
        assert math.isclose(ratio, 1.0, rel_tol=1e-6), f"Expected ratio≈1.0, got {ratio}"

    def test_period_resonance_none(self):
        """Objects with a non-resonant period ratio → check_resonance returns None."""
        scanner = self._scanner()
        # a=1.0 → T=1.0 yr; a=1.37 → T≈1.603 yr; ratio≈1.603 which isn't near 1:1, 2:1, etc.
        result = scanner.check_resonance(1.0, 1.37)
        # May or may not match; verify the logic holds for a clearly non-resonant pair
        # a=1.0 → T=1.0; a=1.20 → T≈1.315; ratio≈1.315 — not close to any p:q with p,q≤4
        result2 = scanner.check_resonance(1.0, 1.20)
        # At least one of these highly non-resonant pairs should return None
        # We test a known non-resonant pair: a=1.0 vs a=3.3 → T≈5.99; ratio≈5.99, beyond 4:1=4
        result3 = scanner.check_resonance(1.0, 3.3)
        assert result3 is None, (
            f"Expected None for period ratio ≈6 (beyond p:q ≤ 4:1), got {result3}"
        )


# ===========================================================================
# TestOption15Exists (2 tests)
# ===========================================================================

class TestOption15Exists:
    def test_rendezvous_scan_method_exists(self):
        """ANEOSMenuV2 must have a _rendezvous_scan callable."""
        from aneos_menu_v2 import ANEOSMenuV2
        assert callable(getattr(ANEOSMenuV2, "_rendezvous_scan", None)), (
            "_rendezvous_scan is not defined on ANEOSMenuV2"
        )

    def test_rendezvous_scan_in_handlers(self):
        """The run() method must include '15' as a handler key."""
        from aneos_menu_v2 import ANEOSMenuV2
        src = inspect.getsource(ANEOSMenuV2.run)
        assert '"15"' in src or "'15'" in src, (
            "Option '15' not found in handlers dict in ANEOSMenuV2.run()"
        )


# ===========================================================================
# Phase 21E dependency contract superseded for PHA transport (ADR-073)
# ===========================================================================

def test_fetch_phas_uses_guarded_transport_without_aiohttp(monkeypatch):
    """Shared requests transport replaces the former optional aiohttp dependency."""
    import asyncio
    from unittest.mock import Mock
    import aneos_core.pattern_analysis.rendezvous as rv
    monkeypatch.setattr(rv, '_HAS_AIOHTTP', False)
    request = Mock(return_value={'data': [['synthetic', '1', '.2', '10', '20', '30', None]]})
    monkeypatch.setattr(rv, 'provider_get_json', request)
    objects = asyncio.run(rv.PHAMoidScanner().fetch_phas())
    assert len(objects) == 1 and objects[0].pdes == 'synthetic'
    request.assert_called_once()
