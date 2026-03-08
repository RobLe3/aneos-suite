"""NetworkSigmaCombiner — Fisher's method across population sub-modules (ADR-047)."""
from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
from scipy import stats
from aneos_core.utils.statistical_utils import p_value_to_sigma

logger = logging.getLogger(__name__)

NETWORK_TIERS = [
    (5.0, "NETWORK_EXCEPTIONAL"),
    (3.0, "NETWORK_ANOMALY"),
    (2.0, "NETWORK_INTERESTING"),
    (1.0, "NETWORK_NOTABLE"),
    (0.0, "NETWORK_ROUTINE"),
]


class NetworkSigmaCombiner:
    """
    Combine p-values from enabled pattern analysis sub-modules using Fisher's method.
    χ² = -2 Σ ln(p_i)   under H₀: χ²(2k) where k = number of valid p-values.
    """

    def combine(
        self,
        sub_module_p_values: Dict[str, Optional[float]],
        n_objects: int = 1,
    ) -> Dict:
        """
        Returns dict with keys: network_sigma, network_tier, combined_p_value,
        sub_module_p_values (Bonferroni-corrected), chi2_stat, degrees_of_freedom.
        """
        # Bonferroni correction across N objects
        corrected = {}
        for key, p in sub_module_p_values.items():
            if p is not None and 0 < p <= 1:
                corrected[key] = min(p * max(n_objects, 1), 1.0)
            else:
                corrected[key] = p

        valid_p = [p for p in corrected.values() if p is not None and 0 < p <= 1]
        if not valid_p:
            return {
                "network_sigma": 0.0,
                "network_tier": "NETWORK_ROUTINE",
                "combined_p_value": 1.0,
                "sub_module_p_values": corrected,
                "chi2_stat": 0.0,
                "degrees_of_freedom": 0,
            }

        # Clamp to avoid log(0)
        clamped = [max(p, 1e-300) for p in valid_p]
        chi2_stat = -2.0 * sum(np.log(p) for p in clamped)
        df = 2 * len(clamped)
        combined_p = 1.0 - stats.chi2.cdf(chi2_stat, df)

        try:
            sigma = p_value_to_sigma(combined_p)
        except Exception:
            sigma = 0.0

        tier = "NETWORK_ROUTINE"
        for threshold, name in NETWORK_TIERS:
            if sigma >= threshold:
                tier = name
                break

        return {
            "network_sigma": round(sigma, 3),
            "network_tier": tier,
            "combined_p_value": combined_p,
            "sub_module_p_values": corrected,
            "chi2_stat": chi2_stat,
            "degrees_of_freedom": df,
        }
