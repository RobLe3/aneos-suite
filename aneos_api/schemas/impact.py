from __future__ import annotations

from typing import Optional

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


class ImpactResponse(BaseModel):
    designation: str
    collision_probability: float
    moon_collision_probability: Optional[float] = None
    moon_earth_ratio: Optional[float] = None
    impact_energy_mt: Optional[float] = None
    crater_diameter_km: Optional[float] = None
    risk_level: str
    time_to_impact_years: Optional[float] = None
