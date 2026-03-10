"""NetworkAnalysisSession — aggregate root for BC11 (ADR-042)."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc

if TYPE_CHECKING:
    from aneos_core.data.models import NEOData

logger = logging.getLogger(__name__)


@dataclass
class PatternAnalysisConfig:
    clustering: bool = True
    harmonics: bool = True
    correlation: bool = False   # requires ADR-040 nongrav data
    rendezvous: bool = False    # deferred ADR-045
    historical_years: int = 30
    max_objects: int = 500
    min_sigma_filter: float = 1.0   # only objects with sigma >= this enter analysis


class NetworkAnalysisSession:
    """
    Orchestrates all enabled pattern analysis sub-modules over a batch of NEOData.
    Reads NEOData from BC1; outputs a result dict consumed by BC7/BC8.
    Never modifies individual NEOData records.
    """

    def __init__(self, config: Optional[PatternAnalysisConfig] = None,
                 fetcher=None):
        self.config = config or PatternAnalysisConfig()
        self.fetcher = fetcher  # DataFetcher, used for historical_approaches if harmonics=True

    def run(self, neo_objects: List["NEOData"]) -> Dict:
        """Run all enabled sub-modules and return a combined result dict."""
        # Apply object cap
        objects = neo_objects[:self.config.max_objects]

        # Apply sigma filter using BC5 per-object scores if available
        scored = [o for o in objects if o.dynamic_anomaly_score is not None]
        if scored and self.config.min_sigma_filter > 0.0:
            objects = [
                o for o in objects
                if o.dynamic_anomaly_score is None
                or o.dynamic_anomaly_score >= self.config.min_sigma_filter
            ]
            n_filtered = len(neo_objects[:self.config.max_objects]) - len(objects)
            if n_filtered:
                logger.info(
                    f"min_sigma_filter={self.config.min_sigma_filter}: "
                    f"excluded {n_filtered} objects below threshold"
                )

        n = len(objects)
        logger.info(f"NetworkAnalysisSession: analysing {n} objects")

        sub_module_p_values: Dict[str, Optional[float]] = {}
        result: Dict = {
            "designations_analyzed": n,
            "clusters": [],
            "harmonic_signals": [],
            "correlation_matrix": None,
            "analysis_metadata": {
                "session_start": datetime.now(UTC).isoformat(),
                "config": {
                    "clustering": self.config.clustering,
                    "harmonics": self.config.harmonics,
                    "correlation": self.config.correlation,
                    "historical_years": self.config.historical_years,
                    "max_objects": self.config.max_objects,
                },
            },
        }

        # PA-1: Clustering
        if self.config.clustering:
            from .clustering import OrbitalElementClusterer
            clusters = OrbitalElementClusterer().run(objects)
            result["analysis_metadata"]["n_clusters_evaluated"] = len(clusters)
            result["clusters"] = [self._cluster_to_dict(c) for c in clusters]
            # Combined clustering p-value: minimum p-value of non-family clusters
            anomalous = [c for c in clusters if c.known_family is None]
            sub_module_p_values["clustering"] = min(
                (c.p_value for c in anomalous), default=None
            )

        # PA-3: Harmonics (requires fetcher for historical approaches)
        if self.config.harmonics:
            from .harmonics import SynodicHarmonicAnalyzer
            analyzer = SynodicHarmonicAnalyzer(
                fetcher=self.fetcher,
                years_back=self.config.historical_years,
            )
            signals = analyzer.run(objects)
            n_skipped_harmonics = len(objects) - len(signals)
            result["analysis_metadata"]["objects_skipped_insufficient_harmonics"] = n_skipped_harmonics
            if n_skipped_harmonics > len(objects) // 2:
                logger.info(
                    f"Harmonics: {n_skipped_harmonics}/{len(objects)} objects skipped "
                    "(insufficient historical close-approach epochs, need ≥5)"
                )
            result["harmonic_signals"] = [self._signal_to_dict(s) for s in signals]
            sub_module_p_values["harmonics"] = min(
                (s.p_value for s in signals), default=None
            )

        # PA-5: Non-Gravitational Correlation
        if self.config.correlation:
            from .correlation import NonGravCorrelator
            corr = NonGravCorrelator()
            matrix = corr.run(objects, result.get("clusters", []))
            result["correlation_matrix"] = self._matrix_to_dict(matrix) if matrix else None
            if matrix:
                sub_module_p_values["correlation"] = matrix.min_p_value

        # PA-6: RendezvousDetector — Stage 1 (MOID-based Drummond pre-filter) is implemented
        # in aneos_core/pattern_analysis/rendezvous.py (PHAMoidScanner).
        # Stage 2 (REBOUND orbit propagation) remains deferred per ADR-045.
        # The session rendezvous flag still triggers a warning; use PHAMoidScanner directly
        # (option 15 in the menu) for the Stage 1 scan.
        if self.config.rendezvous:
            logger.warning(
                "rendezvous=True requested in session config. PA-6 Stage 1 (MOID pre-filter) "
                "is available via PHAMoidScanner. Stage 2 (REBOUND propagation) is deferred "
                "(ADR-045). No in-session rendezvous analysis will be performed here."
            )

        # Combine
        from .network_sigma import NetworkSigmaCombiner
        combined = NetworkSigmaCombiner().combine(sub_module_p_values, n_objects=n)
        result.update(combined)
        return result

    # ---- Serialisation helpers ----

    def _cluster_to_dict(self, c) -> Dict:
        return {
            "cluster_id": c.cluster_id,
            "members": c.members,
            "n_members": c.n_members,
            "centroid": c.centroid,
            "density_sigma": c.density_sigma,
            "p_value": c.p_value,
            "known_family": c.known_family,
        }

    def _signal_to_dict(self, s) -> Dict:
        return {
            "designation": s.designation,
            "dominant_period_days": s.dominant_period_days,
            "power_excess_sigma": s.power_excess_sigma,
            "target_periods_tested": s.target_periods_tested,
            "p_value": s.p_value,
        }

    def _matrix_to_dict(self, m) -> Optional[Dict]:
        if m is None:
            return None
        return {
            "cluster_id": m.cluster_id,
            "designations": m.designations,
            "matrix": m.matrix,
            "flagged_pairs": m.flagged_pairs,
            "bonferroni_threshold": m.bonferroni_threshold,
        }
