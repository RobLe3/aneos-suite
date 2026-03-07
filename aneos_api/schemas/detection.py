from __future__ import annotations

from typing import Optional

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


class DetectionResponse(BaseModel):
    designation: str
    is_artificial: bool
    artificial_probability: float
    sigma_confidence: float
    classification: str
    confidence: float
    evidence_count: int = 0
    interpretation: str = (
        "sigma_confidence = rarity under natural NEO null hypothesis. "
        "artificial_probability incorporates 0.1% base rate prior."
    )
