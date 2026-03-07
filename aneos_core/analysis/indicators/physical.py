"""
Physical property anomaly indicators for aNEOS.

Detects anomalies in diameter, albedo, and spectral type that may indicate
an artificial object in heliocentric orbit.
"""

from typing import List
import logging

from .base import (
    AnomalyIndicator, IndicatorResult, IndicatorConfig,
    NumericRangeIndicator,
)
from ...data.models import NEOData

logger = logging.getLogger(__name__)


class DiameterAnomalyIndicator(NumericRangeIndicator):
    """
    Detects anomalous object sizes consistent with spacecraft scale.

    Population context: natural NEOs detected by surveys typically have
    diameters 0.01–50 km.  Objects smaller than ~0.01 km in tracked
    heliocentric orbits are unusual.  Artificials can be effectively
    point-sized relative to survey resolution.
    """

    def __init__(self, config: IndicatorConfig):
        super().__init__(
            name="diameter_anomalies",
            config=config,
            normal_range=(0.01, 50.0),
            extreme_threshold=100.0,
        )

    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        if not neo_data.orbital_elements:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_orbital_elements': True},
            )

        # Prefer physical_properties; fall back to orbital_elements for backward compat
        pp = neo_data.physical_properties
        diameter = (pp.diameter_km if pp and pp.diameter_km is not None
                    else None)

        if diameter is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_diameter_data': True},
            )

        score = 0.0
        factors: List[str] = []
        confidence = 0.8

        if diameter < 0.001:  # < 1 m — spacecraft scale
            score = 1.0
            factors.append("Sub-metre diameter: consistent with spacecraft debris")
        elif diameter < 0.01:  # < 10 m
            score = 0.7
            factors.append("Very small diameter: unusual for tracked heliocentric object")
        elif diameter > 50.0:
            score = 0.4
            factors.append(f"Very large diameter ({diameter:.1f} km): Pluto-scale in NEO space")

        weighted_score = self.calculate_weighted_score(score, confidence)
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata={'diameter_km': diameter},
            contributing_factors=factors,
        )

    def get_description(self) -> str:
        return "Detects anomalous object size (spacecraft scale) based on diameter measurements"


class AlbedoAnomalyIndicator(NumericRangeIndicator):
    """
    Detects anomalously high albedo inconsistent with natural asteroid surfaces.

    Population context (NEOWISE survey, Mainzer et al. 2011): 99% of natural
    NEOs have geometric albedo 0.01–0.45.  Artificial surfaces (TiO₂ paint,
    MLI blankets) have albedo 0.5–0.9.  2020 SO (Centaur stage) was confirmed
    partly via anomalous radiation pressure consistent with high-albedo structure.
    """

    NATURAL_MIN = 0.01
    NATURAL_MAX = 0.45
    HIGH_ALBEDO_THRESHOLD = 0.50

    def __init__(self, config: IndicatorConfig):
        super().__init__(
            name="albedo_anomalies",
            config=config,
            normal_range=(0.01, 0.45),
            extreme_threshold=0.8,
        )

    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        if not neo_data.orbital_elements:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_orbital_elements': True},
            )

        pp = neo_data.physical_properties
        albedo = (pp.albedo if pp and pp.albedo is not None
                  else None)

        if albedo is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_albedo_data': True},
            )

        score = 0.0
        factors: List[str] = []
        confidence = 0.8

        if self.NATURAL_MIN <= albedo <= self.NATURAL_MAX:
            score = 0.0
        elif albedo > self.HIGH_ALBEDO_THRESHOLD:
            score = min((albedo - self.NATURAL_MAX) / self.HIGH_ALBEDO_THRESHOLD, 1.0)
            factors.append(f"High albedo {albedo:.2f} — consistent with spacecraft surface")
        elif albedo < self.NATURAL_MIN:
            score = 0.3
            factors.append(f"Very dark albedo {albedo:.2f} — possible carbon-rich coating")

        weighted_score = self.calculate_weighted_score(score, confidence)
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata={'albedo_value': albedo},
            contributing_factors=factors,
        )

    def get_description(self) -> str:
        return "Detects anomalously high albedo inconsistent with natural asteroid surfaces (NEOWISE survey bounds)"


class SpectralAnomalyIndicator(AnomalyIndicator):
    """
    Detects spectral type anomalies inconsistent with orbital position.

    Population context: inner NEOs (a < 1.0 AU) are predominantly S/Q-types.
    D/T/P-types are outer-belt/comet compositions — finding one in a
    near-circular inner orbit is unusual.  Artificial objects lack
    mineralogical absorption bands and may show flat or anomalously blue spectra.
    """

    _OUTER_TYPES = {'D', 'T', 'P'}

    def __init__(self, config: IndicatorConfig):
        super().__init__(name="spectral_anomalies", config=config)

    def evaluate(self, neo_data: NEOData) -> IndicatorResult:
        if not neo_data.orbital_elements:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_orbital_elements': True},
            )

        pp = neo_data.physical_properties
        spectral_type = (pp.spectral_type if pp and pp.spectral_type is not None
                         else None)
        a = neo_data.orbital_elements.semi_major_axis

        if spectral_type is None:
            return IndicatorResult(
                indicator_name=self.name,
                raw_score=0.0,
                weighted_score=0.0,
                confidence=0.0,
                metadata={'no_spectral_data': True},
            )

        score = 0.0
        factors: List[str] = []
        confidence = 0.7 if (spectral_type and a is not None) else 0.3

        spec_upper = spectral_type.strip().upper()

        if a is not None and a < 1.0 and spec_upper in self._OUTER_TYPES:
            score = 0.6
            factors.append(
                f"Outer-belt spectral type {spectral_type} in inner-orbit NEO (a={a:.3f} AU)"
            )
        elif spec_upper in {'U', 'UNKNOWN', ''}:
            score = 0.15
            factors.append("No spectral classification — featureless spectrum possible")

        weighted_score = self.calculate_weighted_score(score, confidence)
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=weighted_score,
            confidence=confidence,
            metadata={'spectral_type': spectral_type, 'semi_major_axis': a},
            contributing_factors=factors,
        )

    def get_description(self) -> str:
        return "Detects spectral type anomalies inconsistent with the object's orbital position"
