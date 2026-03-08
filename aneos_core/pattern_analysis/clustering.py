"""Orbital Element Clustering — PA-1 (ADR-043)."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from scipy import stats
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from aneos_core.data.models import NEOData

logger = logging.getLogger(__name__)

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# Granvik et al. 2018 debiased NEO population reference bounds (Table 1)
# Used for [0,1] min-max normalisation before clustering
_GRANVIK_BOUNDS = {
    "a":   (0.7, 4.0),    # AU
    "e":   (0.0, 0.99),
    "i":   (0.0, 180.0),  # degrees
}
# Known NEO family signatures (excluded from anomaly scoring)
_KNOWN_FAMILIES = [
    {"name": "Hungaria",   "a_center": 1.88, "a_tol": 0.05, "i_min": 16.0},
    {"name": "Alinda",     "a_center": 2.50, "a_tol": 0.05, "i_min": 0.0},
    {"name": "Phocaea",    "a_center": 2.36, "a_tol": 0.10, "i_min": 18.0},
]


@dataclass
class OrbitalCluster:
    cluster_id: str
    members: List[str]                     # designations
    centroid: Dict[str, float]             # a, e, i, omega, node (mean)
    density_sigma: float                   # sigma above Granvik background
    p_value: float                         # probability of N+ objects in this volume by chance
    known_family: Optional[str] = None     # matched family name, if any
    n_members: int = 0

    def __post_init__(self):
        self.n_members = len(self.members)


class OrbitalElementClusterer:
    """
    Detect non-random orbital element clusters in a NEO population (PA-1).
    Quality gate: N >= 10 members AND density > 3σ above Granvik 2018 background.
    """

    MIN_CLUSTER_SIZE = 10      # minimum members for reporting
    DENSITY_SIGMA_THRESHOLD = 3.0

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def run(self, neo_objects: List["NEOData"]) -> List[OrbitalCluster]:
        """Cluster and return anomalous orbital groups."""
        vectors, labels = self._build_feature_matrix(neo_objects)
        if len(vectors) < self.min_cluster_size:
            logger.debug("Clustering: too few objects with complete elements")
            return []

        cluster_labels = self._cluster(vectors)
        return self._evaluate_clusters(cluster_labels, labels, neo_objects, vectors)

    def _build_feature_matrix(self, neo_objects):
        rows, designations = [], []
        for neo in neo_objects:
            oe = neo.orbital_elements
            if oe is None:
                continue
            a = oe.semi_major_axis
            e = oe.eccentricity
            i = oe.inclination
            om = oe.ra_of_ascending_node or oe.ascending_node
            w = oe.arg_of_periapsis or oe.argument_of_perihelion
            if any(v is None for v in [a, e, i]):
                continue
            # Normalise a, e, i by Granvik bounds
            a_n = np.clip((a - _GRANVIK_BOUNDS["a"][0]) /
                          (_GRANVIK_BOUNDS["a"][1] - _GRANVIK_BOUNDS["a"][0]), 0, 1)
            e_n = np.clip(e / _GRANVIK_BOUNDS["e"][1], 0, 1)
            i_n = np.clip(i / _GRANVIK_BOUNDS["i"][1], 0, 1)
            # Missing angular elements default to 0° (ω=0, Ω=0). For SBDB-fetched data
            # these are almost always present. If both are missing, the feature sits at
            # (sin=0.5, cos=1.0) and may artificially cluster with other missing-data objects.
            om_rad = np.radians(om) if om is not None else 0.0
            w_rad  = np.radians(w)  if w  is not None else 0.0
            rows.append([
                a_n, e_n, i_n,
                np.sin(om_rad) * 0.5 + 0.5, np.cos(om_rad) * 0.5 + 0.5,
                np.sin(w_rad)  * 0.5 + 0.5, np.cos(w_rad)  * 0.5 + 0.5,
            ])
            designations.append(neo.designation)
        return np.array(rows) if rows else np.empty((0, 7)), designations

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        if HAS_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
            )
        else:
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=0.1, min_samples=self.min_samples, metric="euclidean")
        return clusterer.fit_predict(X)

    def _evaluate_clusters(self, labels, designations, neo_objects, X):
        unique_labels = set(labels) - {-1}
        n_evaluated = len(unique_labels)  # clusters DBSCAN/HDBSCAN found (before quality gate)
        results = []

        for cid in unique_labels:
            mask = labels == cid
            members = [designations[i] for i, m in enumerate(mask) if m]
            if len(members) < self.MIN_CLUSTER_SIZE:
                continue
            cluster_X = X[mask]
            centroid_raw = cluster_X.mean(axis=0)
            # Convert normalised a,e,i back to physical for reporting
            a_c = centroid_raw[0] * (_GRANVIK_BOUNDS["a"][1] - _GRANVIK_BOUNDS["a"][0]) + _GRANVIK_BOUNDS["a"][0]
            e_c = centroid_raw[1] * _GRANVIK_BOUNDS["e"][1]
            i_c = centroid_raw[2] * _GRANVIK_BOUNDS["i"][1]
            centroid = {"a": a_c, "e": e_c, "i": i_c}

            # Background density: exclude cluster members to avoid self-inflation
            non_cluster_X = X[~mask]
            background_density = self._estimate_background_density(non_cluster_X)

            # Local density: count neighbours of cluster members in the full dataset
            local_density = self._local_density(cluster_X, X)
            density_sigma = (local_density - background_density[0]) / max(background_density[1], 1e-9)

            if density_sigma < self.DENSITY_SIGMA_THRESHOLD:
                continue  # Does not exceed quality gate

            # Poisson p-value: probability of N+ members in this volume by chance
            expected = max(background_density[0] * len(members), 1e-9)
            p_value = 1 - stats.poisson.cdf(len(members) - 1, expected)

            # Check known family exclusion
            known_family = self._match_known_family(a_c, i_c)

            results.append(OrbitalCluster(
                cluster_id=f"cluster_{cid:03d}",
                members=members,
                centroid=centroid,
                density_sigma=density_sigma,
                p_value=p_value,
                known_family=known_family,
            ))

        # Bonferroni correction: multiply each cluster's p_value by number of candidates tested
        if n_evaluated > 1:
            for cluster in results:
                cluster.p_value = min(cluster.p_value * n_evaluated, 1.0)
        return results

    def _estimate_background_density(self, X):
        """Return (mean, std) of per-object neighbourhood density over whole population."""
        if len(X) < 2:
            return (1.0, 1.0)
        tree = KDTree(X)
        counts = np.array([len(tree.query_ball_point(row, r=0.2)) - 1 for row in X], dtype=float)
        return (counts.mean(), counts.std() + 1e-9)

    def _local_density(self, cluster_X, all_X):
        tree = KDTree(all_X)
        counts = [len(tree.query_ball_point(row, r=0.2)) for row in cluster_X]
        return float(np.mean(counts))

    def _match_known_family(self, a: float, i: float) -> Optional[str]:
        for fam in _KNOWN_FAMILIES:
            if abs(a - fam["a_center"]) <= fam["a_tol"] and i >= fam["i_min"]:
                return fam["name"]
        return None
