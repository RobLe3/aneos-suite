"""Native astronomical epochs without implicit UTC or time-scale conversion."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import re

SCALES = {"TDB", "TT", "TDT", "UTC", "UT1", "UT", "UNKNOWN"}


def decimal_value(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Epoch must be a finite numeric literal")
    if len(str(value)) > 200:
        raise ValueError("Epoch literal too long")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or abs(number.adjusted()) > 100:
            raise ValueError("Invalid finite epoch")
        return number
    except InvalidOperation as exc:
        raise ValueError("Invalid epoch literal") from exc


@dataclass(frozen=True)
class NativeEpoch:
    native_value: str
    format: str
    time_scale: str
    source: str
    role: str = "osculating_elements"
    label: str = None

    def __post_init__(self):
        if not isinstance(self.native_value, str) or not self.native_value.strip():
            raise ValueError("Native epoch must retain a nonempty literal")
        if (
            not isinstance(self.format, str)
            or not isinstance(self.time_scale, str)
            or self.format not in ("JD", "MJD", "calendar")
            or self.time_scale not in SCALES
        ):
            raise ValueError("Unsupported epoch format/time scale")
        if (
            not isinstance(self.source, str)
            or not self.source
            or not isinstance(self.role, str)
            or not self.role
        ):
            raise ValueError("Native epoch source/role required")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("Invalid epoch label")
        if self.format in ("JD", "MJD"):
            decimal_value(self.native_value)
        elif len(self.native_value) > 300 or not re.match(
            r"^(?:A\.D\.|B\.C\.)?\s*\d{1,6}-(?:[A-Za-z]{3}|\d{2})-\d{1,2}(?:[T ]|$)",
            self.native_value,
        ):
            raise ValueError("Invalid/unsupported native calendar label")

    def julian_day(self):
        if self.format == "calendar":
            return None  # Calendar convention/era is not guessed.
        value = decimal_value(self.native_value)
        with localcontext() as context:
            context.prec = max(50, len(self.native_value) + 20)
            return value + Decimal("2400000.5") if self.format == "MJD" else value

    def to_dict(self):
        jd = self.julian_day()
        return {
            "schema_version": "native-epoch-v1",
            "native_value": self.native_value,
            "format": self.format,
            "time_scale": self.time_scale,
            "source": self.source,
            "role": self.role,
            "label": self.label,
            "julian_day_same_scale": format(jd, "f") if jd is not None else None,
            "conversion": "representation_only_no_time_scale_conversion",
            "time_scale_status": (
                "not_provided"
                if self.time_scale == "UNKNOWN"
                else "declared_not_independently_verified"
            ),
        }

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict):
            raise ValueError("Invalid native epoch record")
        try:
            epoch = cls(
                **{
                    key: record[key]
                    for key in (
                        "native_value",
                        "format",
                        "time_scale",
                        "source",
                        "role",
                        "label",
                    )
                }
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete native epoch") from exc
        if epoch.to_dict() != record:
            raise ValueError("Native epoch does not reproduce")
        return epoch

    def same_julian_instant(self, other):
        """Numeric epoch equality only, NOT covariance/frame/solution compatibility."""
        if not isinstance(other, NativeEpoch):
            raise TypeError("Expected NativeEpoch")
        if self.time_scale == "UNKNOWN" or self.time_scale != other.time_scale:
            return None
        a, b = self.julian_day(), other.julian_day()
        return None if a is None or b is None else a == b


def horizons_header_epoch(line, role="computed_elements"):
    """Parse explicit Horizons JD=calendar or calendar headers, retaining labels."""
    stripped = line.strip()
    scale_match = re.search(r"\b(TDB|TDT|TT|UTC|UT1|UT)\s*\)?\s*$", stripped)
    scale = scale_match.group(1) if scale_match else "UNKNOWN"
    jd = re.match(
        r"^([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*=\s*(?:A\.D\.|B\.C\.)?\s*\d",
        stripped,
    )
    if jd:
        return NativeEpoch(jd.group(1), "JD", scale, "Horizons", role, stripped)
    if re.match(r"^(?:A\.D\.|B\.C\.)?\s*\d{1,6}-[A-Za-z]{3}-\d{1,2}\s+\d\d:", stripped):
        return NativeEpoch(stripped, "calendar", scale, "Horizons", role, stripped)
    return None


def verify_neo_epoch(neo):
    """Bind new SBDB epochs to their retained orbit solution, not a UTC guess."""
    epoch = neo.orbital_elements.native_epoch if neo.orbital_elements else None
    if epoch is None:
        return True  # Historical snapshots are not silently upgraded.
    meta = neo.evidence_metadata
    source = (meta.get("field_sources") or {}).get("orbital_elements")
    solution = (
        meta.get("source_solutions", {}).get(source, {}) if source else meta
    ).get("solution", {})
    if (
        solution.get("source") == "JPL SBDB"
        and solution.get("epoch_jd_tdb") is not None
    ):
        expected = NativeEpoch(str(solution["epoch_jd_tdb"]), "JD", "TDB", "SBDB")
        if (
            epoch.source != "SBDB"
            or epoch.role != "osculating_elements"
            or epoch.same_julian_instant(expected) is not True
        ):
            raise ValueError("Native orbital epoch differs from source solution")
    if (
        solution.get("source") == "JPL Horizons"
        and solution.get("native_epoch") is not None
    ):
        expected = NativeEpoch.from_dict(solution["native_epoch"])
        if epoch.to_dict() != expected.to_dict():
            raise ValueError(
                "Native orbital epoch differs from Horizons source solution"
            )
    return True
