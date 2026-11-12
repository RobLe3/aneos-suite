from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, ConfigDict, model_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    Field = lambda *a, **kw: None  # type: ignore


class OrbitalInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    a: float = Field(..., ge=-1000.0, le=1000.0,
                     description="Semi-major axis (AU)")
    e: float = Field(..., ge=0.0, le=2.0,
                     description="Eccentricity (0–1 bound, up to 2 for hyperbolic)")
    i: float = Field(..., ge=0.0, le=180.0,
                     description="Inclination (degrees)")
    designation: Optional[str] = Field(None, min_length=1, max_length=50)
    diameter_km: Optional[float] = Field(None, gt=0.0, le=10000.0)
    albedo: Optional[float] = Field(None, ge=0.0, le=1.0)
    absolute_magnitude: Optional[float] = None
    spectral_type: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None
    evidence_metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_orbit(self):
        if self.a == 0 or (self.a < 0 and self.e < 1) or (self.a > 0 and self.e > 1):
            raise ValueError('Inconsistent semi-major axis and eccentricity')
        return self

    orbital_history: Optional[List[Dict[str, Any]]] = None  # time-series from GET /history


class EvidenceSummary(BaseModel):
    type: str
    anomaly_score: Optional[float]
    p_value: Optional[float]
    quality_score: Optional[float]
    effect_size: Optional[float]
    confidence_interval: List[float] = []   # [lower_95, upper_95]
    sample_size: int = 0
    analyzed: bool = True
    data_available: bool = True
    unavailable_reason: Optional[str] = None
    score_semantics: str = "uncalibrated_ranking_heuristic"
    evidence_basis: str = "heuristic_proxy"
    dependency_group: Optional[str] = None


class DetectionResponse(BaseModel):
    designation: str
    is_artificial: Optional[bool] = None
    artificial_probability: Optional[float] = None
    sigma_confidence: Optional[float] = None
    screening_score: Optional[float] = None
    candidate: bool = False
    significance_status: str = 'unavailable_uncalibrated_model'
    origin_status: str = 'inconclusive'
    human_origin_status: str = 'not_assessed'
    evidence_id: str = ''
    policy_id: str = ''
    evidence_snapshot: Dict[str, Any] = Field(default_factory=dict)
    non_gravitational_evidence: Optional[Dict[str, Any]] = None
    limitations: List[str] = Field(default_factory=list)
    schema_version: str = 'screening-assessment-v1'
    engine_version: str = 'catalogue-ranking-v1'
    sigma_tier: str = "UNAVAILABLE"
    classification: str
    confidence: Optional[float] = None
    evidence_count: int = 0
    evidence_sources: List[EvidenceSummary] = []
    spacecraft_veto: bool = False
    veto_reason: Optional[str] = None
    data_source: Optional[str] = None
    data_freshness: Optional[str] = None
    interpretation: str = ""
    combined_p_value: Optional[float] = None       # Fisher's combined probability across evidence
    false_discovery_rate: Optional[float] = None   # Expected FDR at current sigma threshold
    analysis_metadata: Dict[str, Any] = {}         # Detector version, method, population refs
