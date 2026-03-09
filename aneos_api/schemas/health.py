from __future__ import annotations

from datetime import datetime, UTC
from typing import Dict

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    def Field(default=None, **kwargs):  # type: ignore
        return default


class CheckResult(BaseModel):
    status: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, CheckResult] = Field(default_factory=dict)
    version: str = "0.7.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
