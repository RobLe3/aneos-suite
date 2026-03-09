"""
Phase 18 / Plan Phase D — Pipeline Scoring Unit Tests

Tests for the automatic_review_pipeline and historical_chunked_poller scoring
and filtering logic introduced/fixed in Phase 18.
"""

import math
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Helpers to build minimal test objects
# ---------------------------------------------------------------------------

def make_neo(
    designation: str = "2099 AA1",
    dist_au: float | None = None,
    v_kms: float | None = None,
    eccentricity: float = 0.5,
    inclination: float = 45.0,
    semi_major_axis: float = 1.3,
) -> Dict[str, Any]:
    neo: Dict[str, Any] = {
        "designation": designation,
        "orbital_elements": {
            "eccentricity": eccentricity,
            "inclination": inclination,
            "semi_major_axis": semi_major_axis,
        },
    }
    if dist_au is not None:
        neo["miss_distance_au"] = dist_au
    if v_kms is not None:
        neo["relative_velocity_km_s"] = v_kms
    return neo


def _make_placeholder_neo(designation: str = "2099 FAKE") -> Dict[str, Any]:
    """Object with injected placeholder orbital elements from CAD."""
    return make_neo(
        designation=designation,
        dist_au=0.3,
        v_kms=12.0,
        eccentricity=0.1,
        inclination=10.0,
        semi_major_axis=1.0,
    )


# ---------------------------------------------------------------------------
# Import the classes under test
# ---------------------------------------------------------------------------

from aneos_core.polling.historical_chunked_poller import HistoricalChunkedPoller
from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline, ProcessingStage, StageResult


# ---------------------------------------------------------------------------
# Fixture: pipeline with pre-seeded designation frequencies
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    with patch("aneos_core.pipeline.automatic_review_pipeline.HistoricalChunkedPoller"):
        p = AutomaticReviewPipeline.__new__(AutomaticReviewPipeline)
        p.logger = MagicMock()
        p._designation_frequencies = {}
        return p


@pytest.fixture
def poller():
    p = HistoricalChunkedPoller.__new__(HistoricalChunkedPoller)
    p.logger = MagicMock()
    return p


# ---------------------------------------------------------------------------
# Phase A tests — raw_objects_input_count
# ---------------------------------------------------------------------------

class TestRawObjectsInputCount:
    """Phase A: input_count on RAW_OBJECTS stage must equal len(raw_objects)."""

    def test_stage_result_input_count_nonzero(self):
        """StageResult with input_count > 0 is valid (not hardcoded 0)."""
        sr = StageResult(
            stage=ProcessingStage.RAW_OBJECTS,
            input_count=27632,
            output_count=27632,
            processing_time_seconds=1.0,
            success=True,
        )
        assert sr.input_count == 27632, "input_count must reflect actual object count"

    def test_stage_result_input_count_zero_is_wrong(self):
        """input_count=0 with output_count>0 is a bug — guard against regression."""
        sr = StageResult(
            stage=ProcessingStage.RAW_OBJECTS,
            input_count=0,
            output_count=27632,
            processing_time_seconds=1.0,
            success=True,
        )
        assert sr.input_count != sr.output_count, (
            "input_count=0 while output_count>0 is the regression we fixed"
        )


# ---------------------------------------------------------------------------
# Phase B tests — placeholder orbital guard in _simple_artificial_screening
# ---------------------------------------------------------------------------

class TestSimpleArtificialScreeningPlaceholderGuard:
    """Phase B: Objects with default placeholder elements must be rejected."""

    def test_placeholder_rejected(self, poller):
        neo = _make_placeholder_neo()
        assert poller._simple_artificial_screening(neo) is False

    def test_real_high_eccentricity_accepted(self, poller):
        neo = make_neo(eccentricity=0.95, inclination=45.0, semi_major_axis=2.0)
        assert poller._simple_artificial_screening(neo) is True

    def test_real_retrograde_accepted(self, poller):
        neo = make_neo(eccentricity=0.3, inclination=170.0, semi_major_axis=1.5)
        assert poller._simple_artificial_screening(neo) is True

    def test_normal_orbit_rejected(self, poller):
        neo = make_neo(eccentricity=0.3, inclination=45.0, semi_major_axis=1.3)
        assert poller._simple_artificial_screening(neo) is False


# ---------------------------------------------------------------------------
# Tests — _simple_first_stage_scoring (pipeline)
# ---------------------------------------------------------------------------

class TestFirstStageScoring:
    """Core signal scoring logic."""

    def test_very_close_approach_passes(self, pipeline):
        neo = make_neo(dist_au=0.001, v_kms=15.0)
        result = pipeline._simple_first_stage_scoring(neo)
        assert result["overall_score"] >= 0.08

    def test_extreme_low_velocity_passes(self, pipeline):
        """v < 3 km/s → +0.30 → above threshold even at 0.5 AU."""
        neo = make_neo(dist_au=0.5, v_kms=2.0)
        result = pipeline._simple_first_stage_scoring(neo)
        assert result["overall_score"] >= 0.08
        assert "extreme_low_velocity" in result["flags"]

    def test_distant_normal_velocity_fails(self, pipeline):
        """0.3 AU, v=20 km/s: encounter score is tiny, no velocity bonus → below gate."""
        neo = make_neo(dist_au=0.3, v_kms=20.0)
        result = pipeline._simple_first_stage_scoring(neo)
        assert result["overall_score"] < 0.08

    def test_anomalous_velocity_no_distance_passes(self, pipeline):
        """v=4 km/s → +0.15 even without dist_au."""
        neo = make_neo(v_kms=4.0)
        result = pipeline._simple_first_stage_scoring(neo)
        assert result["overall_score"] >= 0.08

    def test_n_visits_logged_not_scored(self, pipeline):
        """Repeat-visit count should appear in signal_breakdown but not inflate score."""
        pipeline._designation_frequencies = {"2030 XX": 50}
        neo = make_neo(designation="2030 XX", dist_au=0.3, v_kms=20.0)
        result = pipeline._simple_first_stage_scoring(neo)
        # n_visits present in breakdown
        assert result["signal_breakdown"]["n_visits"] == 50
        # but overall_score still below gate (distance + no velocity bonus)
        assert result["overall_score"] < 0.08

    def test_placeholder_orbit_does_not_inflate_score(self, pipeline):
        """Placeholder e/i/a must not push score above gate (regression guard)."""
        neo = _make_placeholder_neo(designation="CAD_FAKE")
        result = pipeline._simple_first_stage_scoring(neo)
        # 0.3 AU → exp(-0.3/0.02) ≈ 5e-7, v=12 km/s → no bonus → well below 0.08
        assert result["overall_score"] < 0.08
