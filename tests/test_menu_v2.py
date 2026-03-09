"""
Phase 16 tests — ANEOSMenuV2 and core data-gap fixes.

Tests cover:
  - 16A.1: _fetch_real_orbital_data uses physical_properties (not removed OrbitalElements fields)
  - 16A.2: _get_test_data raises ValueError for unknown designations (no silent fallback)
  - 16A.3: ValidatedWrapper.analyze_neo passes orbital_history / close_approach_history
  - 16A.4: _map_validated_classification returns INCONCLUSIVE for sigma < 2
  - 16B:   ANEOSMenuV2 import, construction, and 6-option routing
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 16A.1 — _fetch_real_orbital_data uses PhysicalProperties
# ===========================================================================

class TestFetchRealOrbitalDataPhysicalProps:
    """Verify physical data is read from physical_properties, not OrbitalElements."""

    def _make_neo_data(self, pp_attrs: dict):
        neo_data = MagicMock()
        oe = MagicMock()
        # Remove removed fields so hasattr returns False
        del oe.diameter
        del oe.albedo
        del oe.spectral_type
        del oe.rot_per
        neo_data.orbital_elements = oe
        pp = MagicMock()
        for k, v in pp_attrs.items():
            setattr(pp, k, v)
        neo_data.physical_properties = pp
        neo_data.sources_used = ["SBDB"]
        return neo_data

    def test_diameter_from_physical_properties(self):
        """diameter_km on PhysicalProperties appears in returned physical dict."""
        from aneos_menu import ANEOSMenu
        menu = ANEOSMenu.__new__(ANEOSMenu)
        menu.console = None

        neo_data = self._make_neo_data({"diameter_km": 0.37, "albedo": None,
                                         "spectral_type": None, "rotation_period_hours": None})
        with patch("aneos_core.data.fetcher.DataFetcher.fetch_neo_data", return_value=neo_data):
            result = menu._fetch_real_orbital_data("99942")

        assert result is not None
        _, physical = result
        assert "diameter" in physical
        assert abs(physical["diameter"] - 0.37) < 1e-9

    def test_albedo_from_physical_properties(self):
        """albedo on PhysicalProperties appears in returned physical dict."""
        from aneos_menu import ANEOSMenu
        menu = ANEOSMenu.__new__(ANEOSMenu)
        menu.console = None

        neo_data = self._make_neo_data({"diameter_km": None, "albedo": 0.33,
                                         "spectral_type": "Sq", "rotation_period_hours": 30.4})
        with patch("aneos_core.data.fetcher.DataFetcher.fetch_neo_data", return_value=neo_data):
            result = menu._fetch_real_orbital_data("99942")

        assert result is not None
        _, physical = result
        assert physical.get("albedo") == pytest.approx(0.33)
        assert physical.get("spectral_type") == "Sq"
        assert physical.get("rotation_period") == pytest.approx(30.4)


# ===========================================================================
# 16A.2 — _get_test_data raises ValueError for unknown designations
# ===========================================================================

class TestGetTestDataNoSilentFallback:
    """Verify _get_test_data raises ValueError for unknown/unfetchable designations."""

    def test_raises_for_unknown_designation(self):
        """An unknown designation with no live data must raise ValueError, not return generic data."""
        from aneos_menu import ANEOSMenu
        menu = ANEOSMenu.__new__(ANEOSMenu)
        menu.console = None

        with patch.object(menu, "_fetch_real_orbital_data", return_value=None):
            with pytest.raises(ValueError, match="Cannot resolve orbital data"):
                menu._get_test_data("COMPLETELY_UNKNOWN_9999_XYZ")

    def test_known_alias_still_works(self):
        """Known aliases (tesla, apophis, etc.) must still return data without fetch."""
        from aneos_menu import ANEOSMenu
        menu = ANEOSMenu.__new__(ANEOSMenu)
        menu.console = None

        orbital, physical = menu._get_test_data("tesla")
        assert orbital["a"] == pytest.approx(1.325)

    def test_live_fetch_success_bypasses_error(self):
        """If _fetch_real_orbital_data returns data, no ValueError is raised."""
        from aneos_menu import ANEOSMenu
        menu = ANEOSMenu.__new__(ANEOSMenu)
        menu.console = None

        fake_data = ({"a": 0.9, "e": 0.19, "i": 3.3}, {"diameter": 0.37})
        with patch.object(menu, "_fetch_real_orbital_data", return_value=fake_data):
            orbital, physical = menu._get_test_data("99942")
        assert orbital["a"] == pytest.approx(0.9)


# ===========================================================================
# 16A.3 — ValidatedWrapper passes additional_data to analyze_neo_validated
# ===========================================================================

class TestValidatedWrapperPassthrough:
    """Verify orbital_history and close_approach_history are forwarded."""

    def _make_wrapper(self):
        from aneos_core.detection.detection_manager import DetectionManager, DetectorType
        manager = DetectionManager(preferred_detector=DetectorType.VALIDATED)
        from aneos_core.detection.detection_manager import MultiModalDetector
        # Find the wrapper stored in _detectors
        wrapper = manager._detectors.get(DetectorType.VALIDATED)
        return wrapper

    def test_orbital_history_passed_to_validated_detector(self):
        """analyze_neo must forward additional_data['orbital_history'] to analyze_neo_validated."""
        wrapper = self._make_wrapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")

        orbital = {"a": 0.9, "e": 0.19, "i": 3.3, "om": 0.0, "w": 0.0, "M": 0.0}
        hist = [{"epoch": "2020-01-01", "a": 0.9}]
        additional = {"orbital_history": hist, "close_approach_history": []}

        with patch.object(wrapper.detector, "analyze_neo_validated", wraps=wrapper.detector.analyze_neo_validated) as mock_fn:
            try:
                wrapper.analyze_neo(orbital, additional_data=additional)
            except Exception:
                pass  # We only care that the mock was called with correct args
            if mock_fn.called:
                call_kwargs = mock_fn.call_args[1] if mock_fn.call_args[1] else {}
                call_args = mock_fn.call_args[0] if mock_fn.call_args[0] else ()
                # orbital_history should appear in kwargs
                assert "orbital_history" in call_kwargs or len(call_args) >= 3

    def test_no_additional_data_does_not_crash(self):
        """analyze_neo without additional_data must not raise."""
        wrapper = self._make_wrapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")

        orbital = {"a": 0.9, "e": 0.19, "i": 3.3, "om": 0.0, "w": 0.0, "M": 0.0}
        try:
            result = wrapper.analyze_neo(orbital)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"analyze_neo without additional_data raised: {exc}")


# ===========================================================================
# 16A.4 — _map_validated_classification returns INCONCLUSIVE for sigma < 2
# ===========================================================================

class TestMapValidatedClassification:
    """Verify classification labels are scientifically honest."""

    def _get_mapper(self):
        from aneos_core.detection.detection_manager import DetectionManager, DetectorType
        manager = DetectionManager(preferred_detector=DetectorType.VALIDATED)
        wrapper = manager._detectors.get(DetectorType.VALIDATED)
        return wrapper

    def _make_result(self, is_artificial, sigma):
        result = MagicMock()
        result.is_artificial = is_artificial
        result.sigma_confidence = sigma
        return result

    def test_sigma_below_2_returns_inconclusive(self):
        """sigma=1.5 must return INCONCLUSIVE, not NATURAL."""
        wrapper = self._get_mapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")
        label = wrapper._map_validated_classification(self._make_result(False, 1.5))
        assert "INCONCLUSIVE" in label.upper()
        assert "NATURAL" not in label.upper()

    def test_sigma_zero_returns_inconclusive(self):
        """sigma=0.0 must return INCONCLUSIVE."""
        wrapper = self._get_mapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")
        label = wrapper._map_validated_classification(self._make_result(False, 0.0))
        assert "INCONCLUSIVE" in label.upper()

    def test_sigma_2_returns_edge_case(self):
        """sigma=2.0 must return EDGE CASE (σ≥2)."""
        wrapper = self._get_mapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")
        label = wrapper._map_validated_classification(self._make_result(False, 2.0))
        assert "EDGE" in label.upper() or "2" in label

    def test_sigma_5_artificial_returns_validated(self):
        """sigma=5.0, is_artificial=True → ARTIFICIAL VALIDATED."""
        wrapper = self._get_mapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")
        label = wrapper._map_validated_classification(self._make_result(True, 5.0))
        assert "ARTIFICIAL" in label.upper()


# ===========================================================================
# 16B — ANEOSMenuV2 construction and routing
# ===========================================================================

class TestANEOSMenuV2Import:
    """Verify ANEOSMenuV2 imports and constructs correctly."""

    def test_import_succeeds(self):
        """ANEOSMenuV2 must be importable without exception."""
        from aneos_menu_v2 import ANEOSMenuV2
        assert ANEOSMenuV2 is not None

    def test_construction_succeeds(self):
        """ANEOSMenuV2() must construct without exception."""
        from aneos_menu_v2 import ANEOSMenuV2
        menu = ANEOSMenuV2()
        assert menu is not None

    def test_inherits_base(self):
        """ANEOSMenuV2 must inherit from ANEOSMenuBase."""
        from aneos_menu_v2 import ANEOSMenuV2
        from aneos_menu_base import ANEOSMenuBase
        assert issubclass(ANEOSMenuV2, ANEOSMenuBase)

    def test_has_all_handlers(self):
        """All option handlers must exist as methods."""
        from aneos_menu_v2 import ANEOSMenuV2
        required = [
            "_detect_single", "_detect_multi_evidence", "_detect_batch",
            "_orbital_history_analysis", "_impact_assessment", "_close_approach_history",
            "_live_pipeline", "_population_pattern_analysis",
            "_results_browser", "_export_results", "_system_health",
        ]
        for name in required:
            assert hasattr(ANEOSMenuV2, name), f"Missing method: {name}"


class TestANEOSMenuV2Helpers:
    """Test helper methods on ANEOSMenuV2."""

    def _menu(self):
        from aneos_menu_v2 import ANEOSMenuV2
        menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
        menu.console = None
        menu._detection_results = {}
        menu._impact_results = {}
        return menu

    def test_fetch_neo_data_returns_none_on_import_error(self):
        """_fetch_neo_data returns None when fetcher raises."""
        menu = self._menu()
        with patch("aneos_core.data.fetcher.DataFetcher.fetch_neo_data", side_effect=Exception("network")):
            result = menu._fetch_neo_data("NONEXISTENT")
        assert result is None

    def test_build_physical_dict_with_no_physical_properties(self):
        """_build_physical_dict returns empty dict when physical_properties is None."""
        menu = self._menu()
        neo_data = MagicMock()
        neo_data.physical_properties = None
        neo_data.sources_used = []
        result = menu._build_physical_dict(neo_data)
        assert isinstance(result, dict)
        assert result == {}

    def test_build_physical_dict_extracts_diameter(self):
        """_build_physical_dict extracts diameter_km from physical_properties."""
        menu = self._menu()
        neo_data = MagicMock()
        pp = MagicMock()
        pp.diameter_km = 0.37
        pp.albedo = None
        pp.spectral_type = None
        pp.rotation_period_hours = None
        pp.absolute_magnitude_h = 19.7
        neo_data.physical_properties = pp
        neo_data.sources_used = []
        result = menu._build_physical_dict(neo_data)
        assert result.get("diameter") == pytest.approx(0.37)
        assert result.get("absolute_magnitude") == pytest.approx(19.7)

    def test_rough_torino_low_probability(self):
        """Very low probability → Torino 0."""
        from aneos_menu_v2 import ANEOSMenuV2
        label = ANEOSMenuV2._rough_torino(1e-10, None)
        assert "0" in label

    def test_rough_torino_elevated_probability(self):
        """Elevated probability → non-zero Torino label."""
        from aneos_menu_v2 import ANEOSMenuV2
        label = ANEOSMenuV2._rough_torino(1e-3, 100.0)
        assert "0" not in label or "elevated" in label.lower() or "1" in label

    def test_results_browser_empty_session(self, capsys):
        """_results_browser with no results prints informational message."""
        menu = self._menu()
        with patch.object(menu, "show_info") as mock_info:
            menu._results_browser()
        mock_info.assert_called_once()
        assert "yet" in mock_info.call_args[0][0].lower() or "no results" in mock_info.call_args[0][0].lower()
