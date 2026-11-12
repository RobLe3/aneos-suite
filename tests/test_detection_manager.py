"""Tests for DetectionManager unified detector selection and analysis."""

import pytest
from aneos_core.detection.detection_manager import DetectionManager, DetectorType
from aneos_core.interfaces.detection import DetectionResult


def test_auto_selects_validated():
    """AUTO mode must select the VALIDATED detector when it is available."""
    mgr = DetectionManager(preferred_detector=DetectorType.AUTO)
    # If VALIDATED loaded successfully it must be present and chosen first
    assert DetectorType.VALIDATED in mgr._detectors, (
        "ValidatedSigma5ArtificialNEODetector failed to load"
    )
    selected = mgr._select_best_detector({"a": 1.0, "e": 0.1, "i": 5.0}, None)
    assert selected == DetectorType.VALIDATED


def test_analyze_neo_returns_detection_result():
    """analyze_neo() must return a valid DetectionResult."""
    mgr = DetectionManager(preferred_detector=DetectorType.AUTO)
    result = mgr.analyze_neo({"a": 1.0, "e": 0.1, "i": 5.0})
    assert isinstance(result, DetectionResult)
    assert result.is_artificial is None
    assert result.confidence is None
    assert result.artificial_probability is None
    assert result.screening_score is not None
    assert result.metadata["five_sigma_claim_permitted"] is False
