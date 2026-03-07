"""
NEODyS data source implementation.

Fetches osculating orbital elements from the University of Pisa NEODyS-2
catalogue.  Objects are stored as equinoctial-element files (.eq0) at:
  https://newton.spacedys.com/~neodys2/epoch/{number}.eq0

Equinoctial elements are converted to classical Keplerian elements
(a, e, i, Omega, omega, M) for use with the rest of the pipeline.
"""

import math
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime

import requests

from .base import DataSourceBase
from ...config.settings import APIConfig
from ..cache import CacheManager

logger = logging.getLogger(__name__)

_NEODYS_EPOCH_BASE = "https://newton.spacedys.com/~neodys2/epoch/"


class NEODySSource(DataSourceBase):
    """
    NEODyS-2 (Near Earth Objects Dynamic Site) data source.

    Downloads the osculating .eq0 element file for each object and converts
    the equinoctial elements to classical orbital elements.
    """

    def __init__(self, config: Optional[APIConfig] = None, cache_manager: Optional[CacheManager] = None):
        if config is None:
            config = APIConfig()
        super().__init__(name="NEODyS", config=config, cache_manager=cache_manager)
        self.base_url = _NEODYS_EPOCH_BASE

    def get_base_url(self) -> str:
        return self.base_url

    async def health_check(self) -> bool:
        """Check that the NEODyS epoch directory is accessible."""
        import aiohttp
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                async with session.get(self.base_url) as response:
                    return response.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Synchronous implementation (overrides the abstract async stub)      #
    # ------------------------------------------------------------------ #

    def fetch_orbital_elements(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and convert equinoctial elements from NEODyS.

        The catalogue number is resolved via astroquery.mpc when the
        designation is a name rather than a plain integer.
        """
        try:
            number = self._resolve_number(designation)
            if number is None:
                self.logger.debug(f"NEODyS: could not resolve number for {designation!r}")
                return None

            raw = self._fetch_eq0(number)
            if raw is None:
                return None

            return raw  # _fetch_eq0 → _parse_eq0 already converts to Keplerian elements
        except Exception as e:
            self.logger.error(f"NEODyS fetch_orbital_elements error for {designation}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_number(self, designation: str) -> Optional[int]:
        """
        Convert a designation string to an asteroid number.

        Handles:
        - Plain integers:  '99942'
        - Number + name:   '99942 Apophis'
        - Name only:       'Apophis'  (queries MPC)
        - Provisional:     '2004 MN4' (queries MPC)
        """
        stripped = designation.strip()

        # Try leading integer
        m = re.match(r'^(\d+)', stripped)
        if m:
            return int(m.group(1))

        # Fall back to MPC name lookup
        try:
            from astroquery.mpc import MPC as _MPC
            results = _MPC.query_object("asteroid", name=stripped)
            if results:
                num = results[0].get("number")
                if num is not None:
                    return int(num)
        except Exception:
            pass

        return None

    def _fetch_eq0(self, number: int) -> Optional[Dict[str, Any]]:
        """Download the .eq0 file for the given asteroid number."""
        url = f"{self.base_url}{number}.eq0"
        try:
            resp = requests.get(url, timeout=self.config.request_timeout)
            resp.raise_for_status()
            return self._parse_eq0(resp.text, number)
        except requests.HTTPError as e:
            self.logger.warning(f"NEODyS .eq0 not found for {number}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"NEODyS fetch failed for {number}: {e}")
            return None

    def _parse_eq0(self, text: str, number: int) -> Optional[Dict[str, Any]]:
        """
        Parse OEF2.0 equinoctial element file.

        File format (after END_OF_HEADER):
          {number}
          ! comment
           EQU  a  e*sin(LP)  e*cos(LP)  tan(i/2)*sin(LN)  tan(i/2)*cos(LN)  mean_long
           MJD  epoch_mjd  TDT
           MAG  H  G
        """
        lines = [l.rstrip() for l in text.splitlines()]
        # Skip header
        try:
            header_end = next(i for i, l in enumerate(lines) if "END_OF_HEADER" in l)
        except StopIteration:
            return None

        data_lines = [l for l in lines[header_end + 1:] if l and not l.startswith("!")]

        equ_line = mjd_line = mag_line = None
        for line in data_lines:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "EQU" and len(tokens) >= 7:
                equ_line = tokens
            elif tokens[0] == "MJD" and len(tokens) >= 2:
                mjd_line = tokens
            elif tokens[0] == "MAG" and len(tokens) >= 3:
                mag_line = tokens

        if equ_line is None:
            return None

        a       = float(equ_line[1])
        esinLP  = float(equ_line[2])
        ecosLP  = float(equ_line[3])
        hLN     = float(equ_line[4])   # tan(i/2)*sin(LN)
        kLN     = float(equ_line[5])   # tan(i/2)*cos(LN)
        mean_L  = float(equ_line[6])   # mean longitude

        result = self._equinoctial_to_keplerian(a, esinLP, ecosLP, hLN, kLN, mean_L)

        if mjd_line:
            result["_epoch_mjd"] = float(mjd_line[1])

        if mag_line:
            result["_H"] = float(mag_line[1])
            result["_G"] = float(mag_line[2]) if len(mag_line) > 2 else None

        return result

    @staticmethod
    def _equinoctial_to_keplerian(
        a: float,
        esinLP: float, ecosLP: float,
        hLN: float, kLN: float,
        mean_L: float,
    ) -> Dict[str, float]:
        """
        Convert OEF2.0 equinoctial elements to classical Keplerian elements.

        Equinoctial elements:
          a         — semi-major axis (AU)
          e*sin(LP) — where LP = omega + Omega (longitude of perihelion)
          e*cos(LP)
          tan(i/2)*sin(LN)  — where LN = Omega (longitude of ascending node)
          tan(i/2)*cos(LN)
          L = mean longitude = M + LP

        Output (degrees):
          a, e, i, ra_of_ascending_node (Omega), arg_of_periapsis (omega), mean_anomaly (M)
        """
        e = math.sqrt(esinLP ** 2 + ecosLP ** 2)

        LP_rad = math.atan2(esinLP, ecosLP)          # longitude of perihelion
        LN_rad = math.atan2(hLN, kLN)               # longitude of ascending node
        i_rad  = 2.0 * math.atan(math.sqrt(hLN ** 2 + kLN ** 2))

        omega_deg = math.degrees(LP_rad - LN_rad)    # argument of perihelion
        Omega_deg = math.degrees(LN_rad)             # longitude of ascending node
        i_deg     = math.degrees(i_rad)

        # Mean anomaly M = L - LP  (all in degrees)
        LP_deg = math.degrees(LP_rad)
        M_deg  = (mean_L - LP_deg) % 360.0

        return {
            "semi_major_axis":      a,
            "eccentricity":         e,
            "inclination":          i_deg % 180.0,
            "ra_of_ascending_node": Omega_deg % 360.0,
            "arg_of_periapsis":     omega_deg % 360.0,
            "mean_anomaly":         M_deg,
        }

    def get_object_summary(self, designation: str) -> Optional[Dict[str, Any]]:
        """Return a combined summary dict."""
        try:
            orbital = self.fetch_orbital_elements(designation)
            return {
                "designation": designation,
                "orbital_elements": orbital,
                "data_source": "NEODyS",
                "retrieved_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"NEODyS summary error for {designation}: {e}")
            return None
