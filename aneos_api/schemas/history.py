from __future__ import annotations
from typing import List, Optional

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


class OrbitalElementPoint(BaseModel):
    epoch: Optional[str] = None
    a: Optional[float] = None      # semi-major axis (AU)
    e: Optional[float] = None      # eccentricity
    i: Optional[float] = None      # inclination (deg)
    node: Optional[float] = None   # ascending node (deg)
    peri: Optional[float] = None   # argument of perihelion (deg)
    M: Optional[float] = None      # mean anomaly (deg)


class OrbitalHistoryResponse(BaseModel):
    designation: str
    years: int
    points: List[OrbitalElementPoint]
    data_source: str = "JPL Horizons"
