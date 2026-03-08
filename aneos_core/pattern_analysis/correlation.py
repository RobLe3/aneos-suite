"""Cross-Object Non-Gravitational Correlation — PA-5 (ADR-446)."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from aneos_core.data.models import NEOData

logger = logging.getLogger(__name__)

MIN_A2_OBJECTS = 5   # minimum objects with A2 per cluster


@dataclass
class CorrelationMatrix:
    cluster_id: str
    designations: List[str]
    matrix: List[List[float]]           # Pearson r values
    flagged_pairs: List[Tuple[str, str, float]]   # (desig_a, desig_b, r)
    p_values: List[List[float]]
    bonferroni_threshold: float
    min_p_value: float = 1.0            # minimum p-value across all pairs


class NonGravCorrelator:
    """
    Detect correlated non-gravitational accelerations within orbital clusters (PA-5).
    Requires NonGravitationalParameters.a2 on NEOData objects (ADR-040).
    Clusters with < MIN_A2_OBJECTS objects having A2 measurements are silently skipped.

    Note: pearsonr with single-element vectors produces NaN — expected behaviour
    guarded by np.nan_to_num. Full covariance treatment (multi-epoch A2 time series)
    is a v1.2 extension.
    """

    R_THRESHOLD = 0.7
    P_THRESHOLD = 0.01

    def run(
        self,
        neo_objects: List["NEOData"],
        clusters_dicts: List[Dict],
    ) -> Optional[CorrelationMatrix]:
        """Return the most significant CorrelationMatrix across all clusters, or None."""
        neo_map = {neo.designation: neo for neo in neo_objects}
        best: Optional[CorrelationMatrix] = None

        for cluster_dict in clusters_dicts:
            members = cluster_dict.get("members", [])
            cluster_id = cluster_dict.get("cluster_id", "unknown")
            # Filter to objects with A2
            a2_objects = [
                (d, neo_map[d].nongrav.a2)
                for d in members
                if d in neo_map
                and neo_map[d].nongrav is not None
                and neo_map[d].nongrav.a2 is not None
            ]
            if len(a2_objects) < MIN_A2_OBJECTS:
                logger.debug(
                    f"Cluster {cluster_id}: {len(a2_objects)} objects with A2 "
                    f"(need {MIN_A2_OBJECTS}), skipping correlation"
                )
                continue

            designations = [d for d, _ in a2_objects]
            a2_values = np.array([v for _, v in a2_objects])
            n = len(a2_values)

            # Bonferroni correction for n*(n-1)/2 pairs
            n_pairs = n * (n - 1) // 2
            bonferroni_threshold = self.P_THRESHOLD / max(n_pairs, 1)

            matrix = [[0.0] * n for _ in range(n)]
            p_mat = [[1.0] * n for _ in range(n)]
            flagged: List[Tuple[str, str, float]] = []
            min_p = 1.0

            # PA-5 NOTE: Each a2_value is a single scalar measurement (one epoch per object).
            # pearsonr on 1-element arrays always yields NaN → masked to r=0.0, p=1.0.
            # This means no pairs will be flagged in v1.x. Full multi-epoch A2 covariance
            # analysis is deferred to v1.2. Informational log to make this explicit.
            logger.info(
                f"Cluster {cluster_id}: PA-5 correlation with {n} single-epoch A2 scalars. "
                "Individual-pair pearsonr is undefined; no pairs will be flagged (v1.2 extension)."
            )
            for i in range(n):
                for j in range(i + 1, n):
                    r, p = stats.pearsonr([a2_values[i]], [a2_values[j]])
                    r_val = float(np.nan_to_num(r))
                    p_val = float(np.nan_to_num(p, nan=1.0))
                    matrix[i][j] = matrix[j][i] = r_val
                    p_mat[i][j] = p_mat[j][i] = p_val
                    min_p = min(min_p, p_val)
                    if abs(r_val) > self.R_THRESHOLD and p_val < bonferroni_threshold:
                        flagged.append((designations[i], designations[j], r_val))

            cm = CorrelationMatrix(
                cluster_id=cluster_id,
                designations=designations,
                matrix=matrix,
                flagged_pairs=flagged,
                p_values=p_mat,
                bonferroni_threshold=bonferroni_threshold,
                min_p_value=min_p,
            )
            if best is None or cm.min_p_value < best.min_p_value:
                best = cm

        return best
