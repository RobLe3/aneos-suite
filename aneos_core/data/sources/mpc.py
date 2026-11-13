"""
MPC (Minor Planet Center) data source implementation.

Uses the astroquery.mpc module to query the IAU Minor Planet Center for
orbital elements and observational data of Near-Earth Objects.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base import DataSourceBase
from ...config.settings import APIConfig
from ..cache import CacheManager
from ..provider_http import APPLICATION_HEADERS

logger = logging.getLogger(__name__)


def _safe_float(v: Any) -> Optional[float]:
    """Convert value to float, returning None on failure."""
    if isinstance(v, bool) or v is None or v == "" or v != v:  # last check: NaN
        return None
    try:
        import math

        number = float(v)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


class _MPCIdentityRow(dict):
    """Native row with separate lookup-binding metadata; no native fields invented."""


class MPCSource(DataSourceBase):
    """
    Minor Planet Center (MPC) data source via astroquery.

    Provides orbital elements, absolute magnitude, MOID, and arc data
    from the IAU Minor Planet Center database.
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        if config is None:
            config = APIConfig()
        super().__init__(name="MPC", config=config, cache_manager=cache_manager)
        self.base_url = config.mpc_url

    def get_base_url(self) -> str:
        return self.base_url

    async def health_check(self) -> bool:
        """Connectivity check — try fetching a known object."""
        import aiohttp

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8)
            ) as session:
                async with session.get(self.base_url) as response:
                    return response.status in (200, 301, 302)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Synchronous implementation (overrides the abstract async stub)      #
    # ------------------------------------------------------------------ #

    def fetch_acquisition(self, designation):
        """One bounded catalogue query, including its native outcome and all fields."""
        from .base import FetchResult
        from ...utils.errors import AmbiguousObjectError, SourceIdentityError

        row = None
        try:
            row = self._query_mpc(designation)
            if row is None:
                return FetchResult(False, source=self.name, outcome="missing")
            result = {
                "semi_major_axis": _safe_float(row.get("semimajor_axis")),
                "eccentricity": _safe_float(row.get("eccentricity")),
                "inclination": _safe_float(row.get("inclination")),
                "ra_of_ascending_node": _safe_float(row.get("ascending_node")),
                "arg_of_periapsis": _safe_float(row.get("argument_of_perihelion")),
                "mean_anomaly": _safe_float(row.get("mean_anomaly")),
            }
            physical = {
                "albedo": _safe_float(row.get("albedo")),
                "absolute_magnitude_h": _safe_float(row.get("absolute_magnitude")),
            }
            result["_physical"] = {k: v for k, v in physical.items() if v is not None}
            numeric = (
                "semimajor_axis",
                "eccentricity",
                "inclination",
                "ascending_node",
                "argument_of_perihelion",
                "mean_anomaly",
                "albedo",
                "absolute_magnitude",
            )
            result["_evidence_metadata"] = {
                "native_catalogue_row": dict(row),
                "identity_assessment": getattr(
                    row,
                    "identity_assessment",
                    {
                        "status": "not_established",
                        "basis": "caller_supplied_catalogue_row",
                    },
                ),
                "unparsed_numeric_fields": [
                    k
                    for k in numeric
                    if row.get(k) not in (None, "") and _safe_float(row[k]) is None
                ],
                "source_documentation": "https://astroquery.readthedocs.io/en/stable/api/astroquery.mpc.MPCClass.html",
                "normalization_status": "named_catalogue_fields_not_independently_verified",
                "additional_native_fields": "MOID, delta_v, observation arc and other fields remain native; delta_v is a transfer estimate, not a measured acceleration",
            }
            return FetchResult(
                True,
                data=result,
                source=self.name,
                outcome="available",
                raw_response=dict(row),
            )
        except AmbiguousObjectError as exc:
            return FetchResult(
                False,
                source=self.name,
                outcome="ambiguous",
                raw_response=getattr(exc, "native_response", None),
                error_code="MPC_AMBIGUOUS_IDENTITY",
            )
        except SourceIdentityError as exc:
            return FetchResult(
                False,
                source=self.name,
                outcome="invalid",
                raw_response=getattr(exc, "native_response", None),
                error_code="MPC_NATIVE_IDENTITY_MISMATCH",
            )
        except (ValueError, TypeError):
            return FetchResult(
                False, source=self.name, outcome="invalid", raw_response=row
            )
        except ImportError:
            return FetchResult(
                False,
                source=self.name,
                outcome="unavailable",
                error_code="MPC_DEPENDENCY_UNAVAILABLE",
            )
        except Exception:
            return FetchResult(
                False,
                source=self.name,
                outcome="unavailable",
                error_code="MPC_QUERY_UNAVAILABLE",
            )

    def fetch_orbital_elements(self, designation: str) -> Optional[Dict[str, Any]]:
        """Compatibility projection; the acquisition packet retains detailed outcomes."""
        result = self.fetch_acquisition(designation)
        return result.data if result.success else None

    def fetch_physical_properties(self, designation: str) -> Optional[Dict[str, Any]]:
        """Fetch absolute magnitude and MOID from MPC."""
        try:
            row = self._query_mpc(designation)
            if row is None:
                return None

            return {
                "absolute_magnitude": _safe_float(row.get("absolute_magnitude")),
                "earth_moid": _safe_float(row.get("earth_moid")),
                "delta_v": _safe_float(row.get("delta_v")),
            }
        except Exception as e:
            self.logger.error(
                f"MPC fetch_physical_properties error for {designation}: {e}"
            )
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _query_mpc(self, designation: str) -> Optional[Dict[str, Any]]:
        """Use documented exact lookup fields and at most two rows to detect ambiguity.

        A provisional year is never a numbered asteroid. Native identity is checked
        before returning a row; unknown alias equivalence is not guessed.
        """
        import re
        from urllib.parse import unquote_plus
        from astroquery.mpc import MPC as _MPC
        from ...utils.errors import AmbiguousObjectError, SourceIdentityError
        from ..enrichment import identifier

        stripped = identifier(designation, "designation")
        provisional = re.fullmatch(
            r"(?:1[89]\d\d|2[012]\d\d)\s+[A-Z]{2}\d*", stripped, flags=re.I
        )
        numbered = re.fullmatch(r"(\d+)(?:\s+([A-Za-z]+))?", stripped)
        if provisional:
            lookup = {"designation": "+".join(stripped.upper().split())}
            basis = "exact_native_provisional_designation"
        elif numbered:
            lookup = {"number": int(numbered.group(1))}
            basis = "exact_native_number"
        else:
            lookup = {"name": stripped}
            basis = "exact_native_name"
        rows = _MPC.query_objects("asteroid", limit=2, **lookup)
        return self._bind_mpc_rows(designation, rows)

    @staticmethod
    def _bind_mpc_rows(designation, rows):
        import re
        from urllib.parse import unquote_plus
        from ...utils.errors import AmbiguousObjectError, SourceIdentityError
        from ..enrichment import identifier

        stripped = identifier(designation, "designation")
        provisional = re.fullmatch(
            r"(?:1[89]\d\d|2[012]\d\d)\s+[A-Z]{2}\d*", stripped, flags=re.I
        )
        numbered = re.fullmatch(r"(\d+)(?:\s+([A-Za-z]+))?", stripped)
        lookup = (
            {"designation": "+".join(stripped.upper().split())}
            if provisional
            else {"number": int(numbered.group(1))} if numbered else {"name": stripped}
        )
        basis = (
            "exact_native_provisional_designation"
            if provisional
            else "exact_native_number" if numbered else "exact_native_name"
        )
        if rows is None:
            raise ValueError("MPC returned no structured response")
        if len(rows) == 0:
            return None
        if len(rows) != 1:
            error = AmbiguousObjectError("MPC returned multiple candidate identities")
            error.native_response = rows
            raise error
        row = rows[0]
        if not isinstance(row, dict):
            raise SourceIdentityError("Unsupported MPC row structure")
        if provisional:
            native = row.get("designation")
            match = (
                isinstance(native, str)
                and "".join(unquote_plus(native).split()).casefold()
                == "".join(stripped.split()).casefold()
            )
        elif numbered:
            native = row.get("number")
            if (
                native is None
                and isinstance(row.get("designation"), str)
                and row["designation"].isdigit()
            ):
                native = row["designation"]
            match = (
                type(native) is int or isinstance(native, str) and native.isdigit()
            ) and int(native) == lookup["number"]
            if numbered.group(2):
                match = (
                    match
                    and isinstance(row.get("name"), str)
                    and row["name"].strip().casefold() == numbered.group(2).casefold()
                )
        else:
            match = (
                isinstance(row.get("name"), str)
                and row["name"].strip().casefold() == stripped.casefold()
            )
        if not match:
            error = SourceIdentityError(
                "MPC native identifiers do not match requested identity"
            )
            error.native_response = row
            raise error
        result = _MPCIdentityRow(row)
        result.identity_assessment = {
            "status": "matched_native_identifier_not_authenticated",
            "basis": basis,
            "lookup": lookup,
            "requested_designation": stripped,
        }
        return result

    def _make_request(self, designation: str) -> Optional[Dict]:
        """Query the MPC web service for a designation.

        Uses the MPC search_orbits endpoint as a direct HTTP fallback
        independent of astroquery. Returns a Keplerian element dict or None.
        """
        import requests as _req

        try:
            resp = _req.get(
                "https://minorplanetcenter.net/web_service/search_orbits",
                params={"object_designation": designation},
                timeout=10,
                headers=APPLICATION_HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return None
            obj = self._bind_mpc_rows(
                designation, data if isinstance(data, list) else [data]
            )
            if obj is None:
                return None
            return {
                "a": _safe_float(obj.get("a", obj.get("semimajor_axis"))),
                "e": _safe_float(obj.get("e", obj.get("eccentricity"))),
                "i": _safe_float(obj.get("i", obj.get("inclination"))),
                "omega": _safe_float(obj.get("Node", obj.get("ascending_node"))),
                "w": _safe_float(obj.get("Peri", obj.get("argument_of_perihelion"))),
                "M": _safe_float(obj.get("M", obj.get("mean_anomaly"))),
                "H": _safe_float(obj.get("H", obj.get("absolute_magnitude"))),
                "source": "MPC",
            }
        except Exception as exc:
            self.logger.debug(f"MPC _make_request failed for {designation}: {exc}")
            return None

    def get_object_summary(self, designation: str) -> Optional[Dict[str, Any]]:
        """Return a combined orbital + physical summary dict."""
        try:
            orbital = self.fetch_orbital_elements(designation)
            physical = self.fetch_physical_properties(designation)
            return {
                "designation": designation,
                "orbital_elements": orbital,
                "physical_properties": physical,
                "data_source": "MPC",
                "retrieved_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"MPC summary error for {designation}: {e}")
            return None
