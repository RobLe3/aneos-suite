"""Population Pattern Analysis — API response schemas (ADR-448)."""
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator, StrictBool, StrictInt


class OrbitalClusterResult(BaseModel):
    cluster_id: str
    members: List[str]
    n_members: int
    centroid: Dict[str, float]
    density_sigma: Optional[float] = None
    p_value: Optional[float] = None
    evidence_basis: Optional[str] = None
    interpretation: Optional[str] = None
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
    model_config = ConfigDict(extra="forbid")

    @field_validator("designations")
    @classmethod
    def valid_designations(cls,values):
        if any(not d or d!=d.strip() or len(d)>80 or any(ord(c)<32 for c in d) for d in values):
            raise ValueError("Invalid designation")
        return values

    designations: List[str] = Field(
        ..., min_length=1, max_length=500,
        description="1–500 NEO designations to analyse as a population"
    )
    historical_years: StrictInt = Field(
        30, ge=1, le=100,
        description="Fixed years-times-365-day CAD window anchored at first submission; bounded row coverage remains explicit"
    )
    clustering: StrictBool = True
    harmonics: StrictBool = True
    correlation: StrictBool = False
    rendezvous: StrictBool = False


class NetworkStatusResponse(BaseModel):
    job_id: str
    status: str                    # "queued", "processing", "complete", "error"
    designations_analyzed: int = 0
    clusters: List[OrbitalClusterResult] = []
    harmonic_signals: List[HarmonicSignalResult] = []
    correlation_matrix: Optional[CorrelationMatrixResult] = None
    network_sigma: Optional[float] = None
    network_tier: str = "UNAVAILABLE"
    combined_p_value: Optional[float] = None
    sub_module_p_values: Dict[str, Optional[float]] = {}
    analysis_metadata: Dict[str, Any] = {}
    scientific_report: Optional[Dict[str, Any]] = None
    acquisition_outcomes: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    execution_status: str = "queued"
    execution_result: Optional[Dict[str, Any]] = None
    attempts: int = 0
    checkpoint_count: int = 0
    revision: int = 0
