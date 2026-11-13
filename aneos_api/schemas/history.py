"""Computed orbital history transport, preserving native time evidence."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from aneos_core.data.epoch_evidence import NativeEpoch


class NativeEpochResponse(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False, extra="forbid")
    schema_version: Literal["native-epoch-v1"]
    native_value: str
    format: Literal["JD", "MJD", "calendar"]
    time_scale: Literal["TDB", "TT", "TDT", "UTC", "UT1", "UT", "UNKNOWN"]
    source: str
    role: str
    label: Optional[str]
    julian_day_same_scale: Optional[str]
    conversion: Literal["representation_only_no_time_scale_conversion"]
    time_scale_status: Literal["not_provided", "declared_not_independently_verified"]

    @model_validator(mode="after")
    def replay_native_epoch(self):
        NativeEpoch.from_dict(self.model_dump())
        return self


class OrbitalElementPoint(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False, extra="forbid")
    epoch: Optional[str] = Field(
        None, description="Native header label, not a UTC observation timestamp"
    )
    native_epoch: Optional[NativeEpochResponse] = None
    a: Optional[float] = None  # AU; hyperbolic values may be negative.
    e: Optional[float] = Field(None, ge=0.0)
    i: Optional[float] = Field(None, ge=0.0, le=180.0)
    node: Optional[float] = None
    peri: Optional[float] = None
    M: Optional[float] = None

    @model_validator(mode="after")
    def bind_header(self):
        if self.native_epoch is not None:
            if (
                self.native_epoch.role != "computed_elements"
                or self.native_epoch.source != "Horizons"
                or self.epoch is None
                or self.native_epoch.label != self.epoch
            ):
                raise ValueError("Computed point/header/native epoch binding mismatch")
        return self


class OrbitalHistoryResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    designation: str
    years: int
    points: List[OrbitalElementPoint]
    data_source: Literal["JPL Horizons"] = "JPL Horizons"
    schema_version: Literal["orbital-history-v2"] = "orbital-history-v2"
    evidence_basis: Literal["computed_ephemeris_not_observations"] = (
        "computed_ephemeris_not_observations"
    )
    scientific_status: Literal["not_independently_validated"] = (
        "not_independently_validated"
    )
    returned_points: int = 0
    epoch_status: str = "legacy_unbound"
    time_axis_status: str = "unavailable_unbound_or_incompatible_epochs"
    coverage_status: Literal["bounded_computed_samples_not_observation_coverage"] = (
        "bounded_computed_samples_not_observation_coverage"
    )
    query_evidence: Optional[Dict[str, Any]] = None
    limitations: List[str] = Field(
        default_factory=lambda: [
            "Computed osculating elements are not observations, maneuvers or independent acceleration measurements.",
            "Native scales/labels are retained; no UTC conversion, covariance propagation or scientific calibration is supplied.",
            "Requested years and returned sample count do not establish complete observational coverage.",
        ]
    )

    @model_validator(mode="after")
    def describe_actual_points(self):
        self.returned_points = len(self.points)
        self.epoch_status = (
            "native_epochs_preserved"
            if self.points and all(p.native_epoch is not None for p in self.points)
            else "legacy_unbound"
        )
        epochs = [p.native_epoch for p in self.points if p.native_epoch is not None]
        scales = {p.time_scale for p in epochs}
        self.time_axis_status = (
            "native_single_scale_numeric_axis"
            if self.points
            and len(epochs) == len(self.points)
            and len(scales) == 1
            and "UNKNOWN" not in scales
            and all(p.julian_day_same_scale is not None for p in epochs)
            else "unavailable_unbound_or_incompatible_epochs"
        )
        return self
