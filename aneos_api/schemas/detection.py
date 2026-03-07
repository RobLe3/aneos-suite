from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


class OrbitalInput(BaseModel):
    a: float                          # semi-major axis (AU)
    e: float                          # eccentricity
    i: float                          # inclination (degrees)
    designation: Optional[str] = None
    diameter_km: Optional[float] = None
    albedo: Optional[float] = None
    orbital_history: Optional[List[Dict[str, Any]]] = None  # time-series from GET /history


class EvidenceSummary(BaseModel):
    type: str
    anomaly_score: float
    p_value: float
    quality_score: float
    effect_size: float
    confidence_interval: List[float] = []   # [lower_95, upper_95]
    sample_size: int = 0


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
