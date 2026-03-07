from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


class ImpactResponse(BaseModel):
    designation: str
    collision_probability: float
    probability_uncertainty: List[float] = []      # [lower_bound, upper_bound]
    calculation_confidence: Optional[float] = None
    moon_collision_probability: Optional[float] = None
    moon_earth_ratio: Optional[float] = None
    impact_energy_mt: Optional[float] = None
    crater_diameter_km: Optional[float] = None
    damage_radius_km: Optional[float] = None
    risk_level: str
    comparative_risk: Optional[str] = None
    time_to_impact_years: Optional[float] = None
    peak_risk_period: Optional[List[int]] = None   # [start_year, end_year]
    keyhole_passages: List[Dict[str, Any]] = []
    primary_risk_factors: List[str] = []
    impact_probability_by_decade: Dict[str, float] = {}
