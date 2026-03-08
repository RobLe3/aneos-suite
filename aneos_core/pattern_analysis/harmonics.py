"""Synodic Period Harmonic Analysis — PA-3 (ADR-044)."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aneos_core.data.models import NEOData

logger = logging.getLogger(__name__)

# Target periods (days): major Earth-synodic resonances
TARGET_PERIODS = [10, 30, 90, 182, 365, 730, 1825, 3650, 36525]
MIN_EPOCHS = 5


@dataclass
class HarmonicSignal:
    designation: str
    dominant_period_days: float
    power_excess_sigma: float
    target_periods_tested: List[float]
    p_value: float


class SynodicHarmonicAnalyzer:
    """
    Detect periodic Earth close-approach patterns (PA-3).
    Requires historical close-approach data (ADR-041 — fetched via DataFetcher).
    Objects with < MIN_EPOCHS historical approaches are silently skipped.
    """

    def __init__(self, fetcher=None, years_back: int = 30):
        self.fetcher = fetcher
        self.years_back = years_back

    def run(self, neo_objects: List["NEOData"]) -> List[HarmonicSignal]:
        signals = []
        for neo in neo_objects:
            try:
                signal = self._analyze_object(neo)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                logger.debug(f"Harmonic analysis skipped for {neo.designation}: {exc}")
        return signals

    def _analyze_object(self, neo: "NEOData") -> Optional[HarmonicSignal]:
        # Gather approach epochs from existing close_approaches OR fetch historical
        approaches = list(neo.close_approaches)
        if self.fetcher is not None:
            try:
                historical = self.fetcher.fetch_historical_approaches(
                    neo.designation, years_back=self.years_back
                )
                # Merge — deduplicate by date
                existing_dates = {a.close_approach_date for a in approaches if a.close_approach_date}
                for h in historical:
                    if h.close_approach_date and h.close_approach_date not in existing_dates:
                        approaches.append(h)
            except Exception:
                pass

        epochs = sorted(
            a.close_approach_date for a in approaches if a.close_approach_date
        )
        if len(epochs) < MIN_EPOCHS:
            return None

        # Convert to days from earliest epoch
        t0 = epochs[0]
        t_days = np.array([(e - t0).total_seconds() / 86400.0 for e in epochs])

        t_max = t_days[-1]
        if t_max <= 0:
            return None

        # Build a dense binary signal (1 at approach times, 0 otherwise).
        # This allows LombScargle to operate on event timing rather than a
        # constant signal (which would produce NaN power with zero variance).
        t_dense = np.arange(0.0, t_max + 1.0, 1.0)
        y_dense = np.zeros(len(t_dense))
        for t in t_days:
            idx = int(round(t))
            if 0 <= idx < len(t_dense):
                y_dense[idx] = 1.0

        try:
            from astropy.timeseries import LombScargle
            freq_grid = np.array([1.0 / p for p in TARGET_PERIODS])
            ls = LombScargle(t_dense, y_dense)
            power = ls.power(freq_grid)
        except ImportError:
            from scipy.signal import lombscargle
            freq_grid_rad = np.array([2 * np.pi / p for p in TARGET_PERIODS])
            power = lombscargle(t_dense, y_dense, freq_grid_rad, normalize=True)

        # Background power: mean of all tested frequencies
        bg_mean = power.mean()
        bg_std = power.std() + 1e-9
        best_idx = int(np.argmax(power))
        best_period = TARGET_PERIODS[best_idx]
        best_power = power[best_idx]
        power_excess_sigma = (best_power - bg_mean) / bg_std

        # Normal null p-value
        from scipy import stats
        p_value = 1 - stats.norm.cdf(power_excess_sigma)

        if power_excess_sigma < 1.0:
            return None  # below detection threshold — skip

        return HarmonicSignal(
            designation=neo.designation,
            dominant_period_days=float(best_period),
            power_excess_sigma=float(power_excess_sigma),
            target_periods_tested=list(TARGET_PERIODS),
            p_value=float(p_value),
        )
