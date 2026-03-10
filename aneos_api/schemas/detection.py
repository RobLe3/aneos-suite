from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    Field = lambda *a, **kw: None  # type: ignore


class OrbitalInput(BaseModel):
    a: float = Field(..., ge=0.1, le=1000.0,
                     description="Semi-major axis (AU)")
    e: float = Field(..., ge=0.0, le=2.0,
                     description="Eccentricity (0–1 bound, up to 2 for hyperbolic)")
    i: float = Field(..., ge=0.0, le=180.0,
                     description="Inclination (degrees)")
    designation: Optional[str] = Field(None, min_length=1, max_length=50)
    diameter_km: Optional[float] = Field(None, gt=0.0, le=10000.0)
    albedo: Optional[float] = Field(None, ge=0.0, le=1.0)
    orbital_history: Optional[List[Dict[str, Any]]] = None  # time-series from GET /history


class EvidenceSummary(BaseModel):
    type: str
    anomaly_score: float
    p_value: float
    quality_score: float
    effect_size: float
    confidence_interval: List[float] = []   # [lower_95, upper_95]
    sample_size: int = 0
    analyzed: bool = True
    data_available: bool = True


class DetectionResponse(BaseModel):
    designation: str
    is_artificial: bool
    artificial_probability: float
    sigma_confidence: float
    sigma_tier: str = "ROUTINE"
    classification: str
    confidence: float
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
