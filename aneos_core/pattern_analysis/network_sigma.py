"""NetworkSigmaCombiner — Fisher's and Stouffer's methods across population sub-modules (ADR-047)."""
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
        method: str = 'fisher',
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Combine p-values using Fisher's chi² method (default) or Stouffer's
        weighted z-score method.

        Returns dict with keys: network_sigma, network_tier, combined_p_value,
        sub_module_p_values (Bonferroni-corrected), method, and method-specific
        keys (chi2_stat/degrees_of_freedom for Fisher; z_combined for Stouffer).

        Args:
            sub_module_p_values: Mapping of sub-module name → p-value (None = absent).
            n_objects: Number of objects analysed (Bonferroni multiplier).
            method: 'fisher' (default) or 'stouffer'.
            weights: Optional per-key weights for Stouffer's method (defaults to 1.0).
        """
        if method not in ('fisher', 'stouffer'):
            raise ValueError(f"Unknown method '{method}'. Use 'fisher' or 'stouffer'.")

        # Bonferroni correction across N objects
        corrected = {}
        for key, p in sub_module_p_values.items():
            if p is not None and 0 < p <= 1:
                corrected[key] = min(p * max(n_objects, 1), 1.0)
            else:
                corrected[key] = p

        _empty_base = {
            "network_sigma": 0.0,
            "network_tier": "NETWORK_ROUTINE",
            "combined_p_value": 1.0,
            "sub_module_p_values": corrected,
            "method": method,
        }

        if method == 'stouffer':
            z_scores, w_list, keys_used = [], [], []
            for key, p in corrected.items():
                if p is not None and 0 < p < 1:
                    z_scores.append(float(stats.norm.ppf(1.0 - p)))
                    w_list.append(float((weights or {}).get(key, 1.0)))
                    keys_used.append(key)
            if not z_scores:
                return {**_empty_base, "z_combined": 0.0}
            w_arr = np.array(w_list)
            z_arr = np.array(z_scores)
            w_norm = np.sqrt(np.dot(w_arr, w_arr))
            z_combined = float(np.dot(w_arr, z_arr) / w_norm) if w_norm > 0 else 0.0
            combined_p = float(1.0 - stats.norm.cdf(z_combined))
            try:
                sigma = p_value_to_sigma(combined_p) if combined_p > 0 else 0.0
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
                "method": "stouffer",
                "z_combined": z_combined,
            }

        # Fisher's method (default)
        valid_p = [p for p in corrected.values() if p is not None and 0 < p <= 1]
        if not valid_p:
            return {**_empty_base, "chi2_stat": 0.0, "degrees_of_freedom": 0}

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
            "method": "fisher",
            "chi2_stat": chi2_stat,
            "degrees_of_freedom": df,
        }
