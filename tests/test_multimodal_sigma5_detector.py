"""Regression tests for the multi-modal sigma-5 artificial NEO detector."""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from aneos_core.detection.multimodal_sigma5_artificial_neo_detector import (
    MultiModalSigma5ArtificialNEODetector,
)


@pytest.fixture()
def detector() -> MultiModalSigma5ArtificialNEODetector:
    """Provide a fresh detector instance for each test."""

    return MultiModalSigma5ArtificialNEODetector()


def test_multimodal_detector_flags_known_artificial_sigma5(detector, caplog):
    """The canonical artificial case should comfortably exceed the sigma-5 bar."""

    orbital_elements = {
        "a": 1.32,
        "e": 0.256,
        "i": 1.08,
    }
    physical_signature = {
        "diameter": 12.0,
        "mass_estimate": 4000,
        "absolute_magnitude": 28.0,
        "radar_signature": {
            "radar_cross_section": 15.0,
            "polarization_ratio": 0.4,
        },
    }

    with caplog.at_level(logging.WARNING):
        result = detector.analyze_neo_multimodal(
            orbital_elements,
            physical_signature,
            datetime(2018, 2, 20),
        )

    assert bool(result.is_artificial) is True
    assert result.sigma_level >= detector.SIGMA_5_THRESHOLD
    assert result.statistical_certainty == detector.SIGMA_5_CERTAINTY
    assert result.false_positive_rate == detector.SIGMA_5_FALSE_POSITIVE_RATE

    fusion = result.analysis["evidence_fusion"]
    assert set(fusion["active_sources"]) == {"orbital", "physical", "temporal"}
    assert float(fusion["multimodal_bonus"]) >= 1.0
    assert any(
        "MULTIMODAL SIGMA 5 ARTIFICIAL DETECTION" in message
        for message in caplog.messages
    )


def test_multimodal_detector_rejects_natural_control_case(detector, caplog):
    """A natural control object should stay below the sigma-5 decision threshold."""

    natural_orbit = {
        "a": 1.68,
        "e": 0.42,
        "i": 11.2,
    }
    natural_properties = {
        "diameter": 240,
        "mass_estimate": 3.5e7,
        "absolute_magnitude": 21.3,
    }

    with caplog.at_level(logging.WARNING):
        result = detector.analyze_neo_multimodal(
            natural_orbit,
            natural_properties,
            datetime(2023, 3, 15),
        )

    assert bool(result.is_artificial) is False
    assert result.sigma_level < detector.SIGMA_5_THRESHOLD
    assert result.false_positive_rate > detector.SIGMA_5_FALSE_POSITIVE_RATE

    overall = result.analysis["overall"]
    assert overall["decision"] == "NATURAL"
    assert bool(overall["sigma_5_threshold_met"]) is False
    assert all(
        "MULTIMODAL SIGMA 5 ARTIFICIAL DETECTION" not in message
        for message in caplog.messages
    )
