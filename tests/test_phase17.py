"""
Phase 17 tests — SBDB nongrav fix, observation-arc wiring, spurious risk-factor fix.

Tests cover:
  - 17A: SBDB no longer sends nongrav=1 by default; health check uses Apophis
  - 17B: SBDB extracts first_obs / last_obs into orbital_data
  - 17C: DataFetcher wires observation dates into NEOData fields
  - 17D: _impact_assessment derives arc from real dates, defaults to 30 days
  - 17E: ValidatedWrapper suppresses 'physical_properties' risk factor when
         physical_data has no real keys
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 17A — SBDB default params no longer include nongrav=1
# ===========================================================================

class TestSBDBNongravFallback:
    """SBDB primary fetch must not include nongrav=1 in default params."""

    def _make_source(self):
        from aneos_core.data.sources.sbdb import SBDBSource
        from aneos_core.config.settings import APIConfig
        return SBDBSource(APIConfig())

    def test_primary_params_have_no_nongrav(self):
        """fetch_orbital_elements must call _http_get WITHOUT nongrav=1 in first call."""
        source = self._make_source()

        captured_params = []

        async def fake_http_get(path, params):
            captured_params.append(dict(params))
            # Return a minimal valid SBDB response
            return {
                "orbit": {
                    "elements": [{"name": "a", "value": "1.0"},
                                  {"name": "e", "value": "0.2"},
                                  {"name": "i", "value": "5.0"}],
                }
            }

        import asyncio
        with patch.object(source, "_http_get", side_effect=fake_http_get):
            asyncio.run(source.fetch_orbital_elements("99942"))

        assert captured_params, "No _http_get calls made"
        first_call = captured_params[0]
        assert "nongrav" not in first_call, (
            f"nongrav key found in primary SBDB params: {first_call}"
        )

    def test_health_check_uses_apophis(self):
        """health_check must query '99942', not '1 Ceres'."""
        source = self._make_source()

        queried = []

        async def fake_fetch(designation):
            queried.append(designation)
            r = MagicMock()
            r.success = True
            return r

        import asyncio
        with patch.object(source, "fetch_orbital_elements", side_effect=fake_fetch):
            asyncio.run(source.health_check())

        assert queried, "fetch_orbital_elements never called in health_check"
        assert queried[0] == "99942", (
            f"health_check queried {queried[0]!r} instead of '99942'"
        )


# ===========================================================================
# 17B — SBDB extracts first_obs / last_obs
# ===========================================================================

class TestSBDBObservationDateExtraction:
    """_parse_sbdb_response must extract first_obs / last_obs from orbit section."""

    def _make_source(self):
        from aneos_core.data.sources.sbdb import SBDBSource
        from aneos_core.config.settings import APIConfig
        return SBDBSource(APIConfig())

    def test_first_obs_extracted(self):
        source = self._make_source()
        data = {
            "orbit": {
                "elements": [],
                "first_obs": "1990-Jan-01",
                "last_obs":  "2024-Mar-15",
            }
        }
        result = source._parse_sbdb_response(data, "99942")
        assert "first_observation_date" in result, "first_observation_date missing"
        assert "1990" in result["first_observation_date"]

    def test_last_obs_extracted(self):
        source = self._make_source()
        data = {
            "orbit": {
                "elements": [],
                "first_obs": "1990-Jan-01",
                "last_obs":  "2024-Mar-15",
            }
        }
        result = source._parse_sbdb_response(data, "99942")
        assert "last_observation_date" in result, "last_observation_date missing"
        assert "2024" in result["last_observation_date"]

    def test_missing_obs_dates_not_present(self):
        """When orbit has no first_obs/last_obs, keys must be absent from result."""
        source = self._make_source()
        data = {"orbit": {"elements": []}}
        result = source._parse_sbdb_response(data, "TEST")
        assert "first_observation_date" not in result
        assert "last_observation_date" not in result


# ===========================================================================
# 17C — DataFetcher wires observation dates into NEOData
# ===========================================================================

class TestFetcherObservationDateWiring:
    """_fetch_from_source must set neo_data.first_observation / last_observation."""

    def test_first_observation_wired(self):
        from aneos_core.data.fetcher import DataFetcher

        fetcher = DataFetcher.__new__(DataFetcher)
        fetcher.logger = MagicMock()

        # Build a fake source that returns orbital_elements_data with observation dates
        fake_source = MagicMock()
        fake_source.health_check = MagicMock(return_value=True)
        fetch_result = MagicMock()
        fetch_result.success = True
        fetch_result.data = {
            "semi_major_axis": 1.0,
            "eccentricity": 0.2,
            "inclination": 5.0,
            "first_observation_date": "1990-Jan-01",
            "last_observation_date":  "2024-Mar-15",
            "_physical": {},
        }
        fake_source.fetch_orbital_elements = MagicMock(return_value=fetch_result)

        result = fetcher._fetch_from_source("TestSource", fake_source, "99942")
        assert result is not None, "_fetch_from_source returned None"
        assert result.first_observation is not None, "first_observation not wired"
        assert result.first_observation.year == 1990

    def test_last_observation_wired(self):
        from aneos_core.data.fetcher import DataFetcher

        fetcher = DataFetcher.__new__(DataFetcher)
        fetcher.logger = MagicMock()

        fake_source = MagicMock()
        fake_source.health_check = MagicMock(return_value=True)
        fetch_result = MagicMock()
        fetch_result.success = True
        fetch_result.data = {
            "semi_major_axis": 1.0,
            "eccentricity": 0.2,
            "inclination": 5.0,
            "first_observation_date": "1990-Jan-01",
            "last_observation_date":  "2024-Mar-15",
            "_physical": {},
        }
        fake_source.fetch_orbital_elements = MagicMock(return_value=fetch_result)

        result = fetcher._fetch_from_source("TestSource", fake_source, "99942")
        assert result is not None
        assert result.last_observation is not None, "last_observation not wired"
        assert result.last_observation.year == 2024


# ===========================================================================
# 17D — _impact_assessment uses real arc when dates available
# ===========================================================================

class TestImpactArcHandling:
    """_impact_assessment must pass real arc_days when first/last_observation set."""

    def _make_menu(self):
        from aneos_menu_v2 import ANEOSMenuV2
        menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
        menu.console = None
        menu._detection_results = {}
        menu._impact_results = {}
        return menu

    def _make_neo_data(self, first_obs=None, last_obs=None):
        neo_data = MagicMock()
        oe = MagicMock()
        oe.semi_major_axis = 0.922
        oe.eccentricity = 0.191
        oe.inclination = 3.33
        oe.is_complete = MagicMock(return_value=True)
        neo_data.orbital_elements = oe
        neo_data.close_approaches = []
        neo_data.physical_properties = None
        neo_data.first_observation = first_obs
        neo_data.last_observation = last_obs
        return neo_data

    def test_arc_derived_from_dates_when_available(self):
        """When first/last observation are set, arc_days must exceed 30-day default."""
        menu = self._make_menu()

        first = datetime(1990, 1, 1, tzinfo=timezone.utc)
        last  = datetime(2024, 3, 15, tzinfo=timezone.utc)
        neo_data = self._make_neo_data(first, last)

        captured_arc = []

        def fake_calc(**kwargs):
            captured_arc.append(kwargs.get("observation_arc_days", -1))
            result = MagicMock()
            result.collision_probability = 1e-10
            result.probability_uncertainty = (0.0, 0.0)
            result.torino_scale = 0
            result.palermo_scale_technical = -10
            result.most_probable_impact_time = None
            result.impact_energy_mt = None
            result.crater_diameter_km = None
            result.moon_impact_probability = {}
            result.data_arc_years = 30.0
            result.calculation_confidence = 0.9
            result.risk_factors = []
            result.assumptions = []
            result.limitations = []
            return result

        with patch.object(menu, "_fetch_neo_data", return_value=neo_data), \
             patch.object(menu, "_ask", return_value="99942"), \
             patch.object(menu, "_display_impact_result"), \
             patch.object(menu, "_persist_impact_result_async"), \
             patch("aneos_core.analysis.impact_probability.ImpactProbabilityCalculator"
                   ".calculate_comprehensive_impact_probability",
                   side_effect=fake_calc):
            menu._impact_assessment()

        assert captured_arc, "calculate_comprehensive_impact_probability never called"
        expected_days = (last - first).days
        assert captured_arc[0] == pytest.approx(expected_days, abs=1), (
            f"Expected arc ~{expected_days} days, got {captured_arc[0]}"
        )

    def test_arc_defaults_to_30_when_dates_absent(self):
        """When observation dates are None, arc_days defaults to 30.0."""
        menu = self._make_menu()
        neo_data = self._make_neo_data()  # no dates

        captured_arc = []

        def fake_calc(**kwargs):
            captured_arc.append(kwargs.get("observation_arc_days", -1))
            result = MagicMock()
            result.collision_probability = 1e-10
            result.probability_uncertainty = (0.0, 0.0)
            result.torino_scale = 0
            result.palermo_scale_technical = -10
            result.most_probable_impact_time = None
            result.impact_energy_mt = None
            result.crater_diameter_km = None
            result.moon_impact_probability = {}
            result.data_arc_years = 30.0
            result.calculation_confidence = 0.9
            result.risk_factors = []
            result.assumptions = []
            result.limitations = []
            return result

        with patch.object(menu, "_fetch_neo_data", return_value=neo_data), \
             patch.object(menu, "_ask", return_value="TEST1"), \
             patch.object(menu, "_display_impact_result"), \
             patch.object(menu, "_persist_impact_result_async"), \
             patch("aneos_core.analysis.impact_probability.ImpactProbabilityCalculator"
                   ".calculate_comprehensive_impact_probability",
                   side_effect=fake_calc):
            menu._impact_assessment()

        assert captured_arc, "calculate_comprehensive_impact_probability never called"
        assert captured_arc[0] == pytest.approx(30.0, abs=0.1), (
            f"Expected default 30.0, got {captured_arc[0]}"
        )


# ===========================================================================
# 17E — ValidatedWrapper suppresses spurious physical_properties risk factor
# ===========================================================================

class TestPhysicalPropertiesRiskFactor:
    """physical_properties must not appear in risk_factors when no real physical data."""

    def _get_wrapper(self):
        from aneos_core.detection.detection_manager import DetectionManager, DetectorType
        manager = DetectionManager(preferred_detector=DetectorType.VALIDATED)
        return manager._detectors.get(DetectorType.VALIDATED)

    def _make_orbital(self):
        return {"a": 0.922, "e": 0.191, "i": 3.33, "om": 0.0, "w": 0.0, "M": 0.0}

    def test_no_physical_data_excludes_physical_risk_factor(self):
        """With physical_data={}, risk_factors must not include 'physical_properties'."""
        wrapper = self._get_wrapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")

        result = wrapper.analyze_neo(self._make_orbital(), physical_data={})
        assert "physical_properties" not in result.risk_factors, (
            f"Spurious 'physical_properties' in risk_factors: {result.risk_factors}"
        )

    def test_no_physical_data_metadata_flag(self):
        """With physical_data={}, metadata['physical_data_available'] must be False."""
        wrapper = self._get_wrapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")

        result = wrapper.analyze_neo(self._make_orbital(), physical_data={})
        assert result.metadata.get("physical_data_available") is False, (
            f"physical_data_available should be False, got: {result.metadata.get('physical_data_available')}"
        )

    def test_sources_only_physical_data_excluded(self):
        """With physical_data={'_sources': [...]}, risk_factors must not include 'physical_properties'."""
        wrapper = self._get_wrapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")

        result = wrapper.analyze_neo(
            self._make_orbital(),
            physical_data={"_sources": ["Horizons", "MPC"]},
        )
        assert "physical_properties" not in result.risk_factors, (
            f"Spurious 'physical_properties' in risk_factors: {result.risk_factors}"
        )

    def test_real_physical_data_includes_physical_risk_factor(self):
        """With diameter in physical_data, 'physical_properties' may appear in risk_factors."""
        wrapper = self._get_wrapper()
        if wrapper is None:
            pytest.skip("ValidatedWrapper not loaded")

        result = wrapper.analyze_neo(
            self._make_orbital(),
            physical_data={"diameter": 0.37, "albedo": 0.23},
        )
        assert result.metadata.get("physical_data_available") is True, (
            "physical_data_available should be True when real data supplied"
        )
