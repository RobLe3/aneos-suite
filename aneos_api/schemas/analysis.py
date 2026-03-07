"""Analysis schemas — AnalysisResponse is canonical in aneos_api.models."""
from aneos_api.models import AnalysisResponse  # noqa: F401

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


class IndicatorScores(BaseModel):
    orbital: float = 0.0
    velocity: float = 0.0
    temporal: float = 0.0
    geographic: float = 0.0
    physical: float = 0.0
    behavioral: float = 0.0


__all__ = ["AnalysisResponse", "IndicatorScores"]
