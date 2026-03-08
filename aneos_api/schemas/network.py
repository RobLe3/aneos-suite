"""Population Pattern Analysis — API response schemas (ADR-448)."""
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class OrbitalClusterResult(BaseModel):
    cluster_id: str
    members: List[str]
    n_members: int
    centroid: Dict[str, float]
    density_sigma: float
    p_value: float
    known_family: Optional[str] = None


class HarmonicSignalResult(BaseModel):
    designation: str
    dominant_period_days: float
    power_excess_sigma: float
    target_periods_tested: List[float]
    p_value: float


class CorrelationMatrixResult(BaseModel):
    cluster_id: str
    designations: List[str]
    flagged_pairs: List[List]      # [[desig_a, desig_b, r], ...]
    bonferroni_threshold: float
    min_p_value: float


class NetworkRequest(BaseModel):
    designations: List[str] = Field(
        ..., min_length=1, max_length=500,
        description="1–500 NEO designations to analyse as a population"
    )
    historical_years: int = Field(
        30, ge=1, le=100,
        description="Years of historical close-approach data to fetch (1–100)"
    )
    clustering: bool = True
    harmonics: bool = True
    correlation: bool = False
    rendezvous: bool = False


class NetworkStatusResponse(BaseModel):
    job_id: str
    status: str                    # "queued", "processing", "complete", "error"
    designations_analyzed: int = 0
    clusters: List[OrbitalClusterResult] = []
    harmonic_signals: List[HarmonicSignalResult] = []
    correlation_matrix: Optional[CorrelationMatrixResult] = None
    network_sigma: float = 0.0
    network_tier: str = "NETWORK_ROUTINE"
    combined_p_value: float = 1.0
    sub_module_p_values: Dict[str, Optional[float]] = {}
    analysis_metadata: Dict[str, Any] = {}
    error: Optional[str] = None
