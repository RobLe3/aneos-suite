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

logger = logging.getLogger(__name__)


def _safe_float(v: Any) -> Optional[float]:
    """Convert value to float, returning None on failure."""
    if v is None or v == "" or v != v:  # last check: NaN
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


class MPCSource(DataSourceBase):
    """
    Minor Planet Center (MPC) data source via astroquery.

    Provides orbital elements, absolute magnitude, MOID, and arc data
    from the IAU Minor Planet Center database.
    """

    def __init__(self, config: Optional[APIConfig] = None, cache_manager: Optional[CacheManager] = None):
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
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                async with session.get(self.base_url) as response:
                    return response.status in (200, 301, 302)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Synchronous implementation (overrides the abstract async stub)      #
    # ------------------------------------------------------------------ #

    def fetch_orbital_elements(self, designation: str) -> Optional[Dict[str, Any]]:
        """Fetch orbital elements from MPC via astroquery."""
        try:
            row = self._query_mpc(designation)
            if row is None:
                return None

            return {
                "semi_major_axis":     _safe_float(row.get("semimajor_axis")),
                "eccentricity":        _safe_float(row.get("eccentricity")),
                "inclination":         _safe_float(row.get("inclination")),
                "ra_of_ascending_node": _safe_float(row.get("ascending_node")),
                "arg_of_periapsis":    _safe_float(row.get("argument_of_perihelion")),
                "mean_anomaly":        _safe_float(row.get("mean_anomaly")),
                # Physical fields that OrbitalElements also carries
                "albedo":              _safe_float(row.get("albedo")),
            }
        except Exception as e:
            self.logger.error(f"MPC fetch_orbital_elements error for {designation}: {e}")
            return None

    def fetch_physical_properties(self, designation: str) -> Optional[Dict[str, Any]]:
        """Fetch absolute magnitude and MOID from MPC."""
        try:
            row = self._query_mpc(designation)
            if row is None:
                return None

            return {
                "absolute_magnitude": _safe_float(row.get("absolute_magnitude")),
                "earth_moid":         _safe_float(row.get("earth_moid")),
                "delta_v":            _safe_float(row.get("delta_v")),
            }
        except Exception as e:
            self.logger.error(f"MPC fetch_physical_properties error for {designation}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _query_mpc(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Query MPC via astroquery for a single object.

        Tries by name first; falls back to parsing the designation as a
        catalogue number (e.g. '99942' or '99942 Apophis').
        """
        from astroquery.mpc import MPC as _MPC  # lazy import

        result_list = None

        # Try as a name string
        try:
            result_list = _MPC.query_object("asteroid", name=designation)
        except Exception:
            pass

        # Try as a number (handles '99942', '99942 Apophis', etc.)
        if not result_list:
            try:
                number = int(designation.strip().split()[0])
                result_list = _MPC.query_object("asteroid", number=number)
            except Exception:
                pass

        if not result_list:
            self.logger.warning(f"MPC: no result for {designation!r}")
            return None

        return result_list[0]  # astroquery returns a list of dicts

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
