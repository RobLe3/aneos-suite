from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImpactResponse(BaseModel):
    """Scoped external model evidence, not a local probability calculation.

    Legacy numeric keys remain nullable; absent evidence is never zero risk.
    Complete source snapshots and hashes permit deterministic report replay.
    """
    schema_version: str
    designation: str
    status: str
    assessment_basis: str
    identity: Dict[str, Any]
    collision_probability: Optional[float] = None
    collision_probability_per_year: Optional[float] = None
    probability_uncertainty: Optional[List[float]] = None
    calculation_confidence: Optional[float] = None
    moon_collision_probability: Optional[float] = None
    moon_earth_ratio: Optional[float] = None
    impact_energy_mt: Optional[float] = None
    crater_diameter_km: Optional[float] = None
    damage_radius_km: Optional[float] = None
    risk_level: str
    comparative_risk: Optional[str] = None
    time_to_impact_years: Optional[float] = None
    peak_risk_period: Optional[List[int]] = None
    keyhole_passages: Optional[List[Dict[str, Any]]] = None
    primary_risk_factors: Optional[List[str]] = None
    impact_probability_by_decade: Optional[Dict[str, float]] = None
    source_torino_max: Optional[float] = None
    source_palermo_cumulative: Optional[float] = None
    source: Dict[str, Any]
    source_snapshot: Dict[str, Any]
    report_sha256: str
    time_scope: Optional[Dict[str, Any]] = None
    virtual_impactors: List[Dict[str, Any]] = Field(default_factory=list)
    physical_estimate_basis: Optional[Dict[str, Any]] = None
    limitations: List[str]
