"""
JPL Horizons data source implementation.

Uses the JPL Horizons REST API (ssd.jpl.nasa.gov/api/horizons.api) to
fetch high-precision osculating orbital elements, physical parameters,
and ephemeris data for Near-Earth Objects.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import requests

from .base import HTTPDataSource
from ...config.settings import APIConfig
from ..cache import CacheManager

logger = logging.getLogger(__name__)

_HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


class HorizonsSource(HTTPDataSource):
    """
    NASA JPL Horizons data source.

    Provides high-precision heliocentric osculating orbital elements and
    physical properties via the Horizons REST API.
    """

    def __init__(self, config: Optional[APIConfig] = None, cache_manager: Optional[CacheManager] = None):
        if config is None:
            config = APIConfig()
        super().__init__(
            name="Horizons",
            base_url=_HORIZONS_URL,
            config=config,
            cache_manager=cache_manager,
        )

    # ------------------------------------------------------------------ #
    # Async implementation (satisfies HTTPDataSource abstract method)     #
    # ------------------------------------------------------------------ #

    async def fetch_orbital_elements(self, designation: str):
        """Async wrapper — delegates to the sync implementation."""
        from .base import FetchResult
        result = self._fetch_orbital_elements_sync(designation)
        if result is None:
            return FetchResult(
                success=False,
                error_message=f"Horizons returned no data for {designation}",
                source=self.name,
            )
        return FetchResult(success=True, data=result, source=self.name)

    # ------------------------------------------------------------------ #
    # Synchronous core implementation                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _horizons_command(designation: str) -> str:
        """
        Convert a designation string to a Horizons COMMAND value.

        Horizons accepts quoted asteroid numbers ('99942') or names ('Apophis')
        but rejects combined forms like '99942 Apophis'.  Provisional designations
        such as '2004 MN4' must be passed as-is (they contain non-letter chars).
        """
        stripped = designation.strip()
        # Named asteroid: digits + space + pure letters (e.g. "99942 Apophis")
        m = re.match(r'^(\d+)\s+([A-Za-z]+)$', stripped)
        if m:
            return f"'{m.group(1)}'"
        return f"'{stripped}'"

    def _fetch_orbital_elements_sync(self, designation: str) -> Optional[Dict[str, Any]]:
        """Fetch orbital elements from Horizons using OBJ_DATA mode."""
        try:
            resp = requests.get(
                _HORIZONS_URL,
                params={
                    "format":     "json",
                    "COMMAND":    self._horizons_command(designation),
                    "OBJ_DATA":   "YES",
                    "MAKE_EPHEM": "NO",
                },
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            result_text = data.get("result", "")
            if not result_text or "No matches found" in result_text:
                self.logger.warning(f"Horizons: no match for {designation!r}")
                return None
            if "Multiple major-bodies match" in result_text or "ambiguous" in result_text.lower():
                # Try with DES= prefix to force small-body lookup
                return self._fetch_as_smallbody(designation)

            elements = self._parse_obj_data(result_text)
            if not elements:
                return None

            # Metadata
            elements["_source"] = self.name
            elements["_designation"] = designation
            elements["_fetched_at"] = datetime.utcnow().isoformat()
            return elements

        except requests.HTTPError as e:
            self.logger.error(f"Horizons HTTP error for {designation}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Horizons fetch error for {designation}: {e}")
            return None

    def _fetch_as_smallbody(self, designation: str) -> Optional[Dict[str, Any]]:
        """Re-query using DES= prefix to force small-body (asteroid/comet) resolution."""
        try:
            resp = requests.get(
                _HORIZONS_URL,
                params={
                    "format":     "json",
                    "COMMAND":    f"DES={designation};",
                    "OBJ_DATA":   "YES",
                    "MAKE_EPHEM": "NO",
                },
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            result_text = resp.json().get("result", "")
            if not result_text or "No matches found" in result_text:
                return None
            elements = self._parse_obj_data(result_text)
            if elements:
                elements["_source"] = self.name
            return elements
        except Exception as e:
            self.logger.error(f"Horizons DES fetch error for {designation}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Text parser for OBJ_DATA output                                     #
    # ------------------------------------------------------------------ #

    def _parse_obj_data(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse orbital elements and physical parameters from Horizons OBJ_DATA text.

        The relevant lines look like:
          EC= .1915216893501022   QR= .7458270478466523   TP= ...
          OM= 204.0389272089208   W=  126.6520518368553   IN= 3.336751320066756
          A= .9225071817289903    MA= 127.3225632013606   ADIST= 1.099187315611328
          GM= n.a.                RAD= .170               ROTPER= 30.56
          H= 19.09                G= .240
        """
        elements: Dict[str, Any] = {}

        # Regex helper: find KEY= <value> where value may be scientific notation
        _num = r'([\d.E+\-]+)'

        def _find(key: str) -> Optional[float]:
            """Extract value following KEY= in text."""
            # Use word-boundary so 'A=' doesn't match 'MA=' or 'ADIST='
            pattern = rf'(?<![A-Z]){re.escape(key)}=\s*{_num}'
            m = re.search(pattern, text)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
            return None

        # Orbital elements
        ec   = _find("EC")
        a    = _find("A")
        inc  = _find("IN")
        om   = _find("OM")
        ma   = _find("MA")

        # W= and W = both appear depending on Horizons version
        w = None
        m_w = re.search(r'\bW\s*=\s*' + _num, text)
        if m_w:
            try:
                w = float(m_w.group(1))
            except ValueError:
                pass

        # Physical parameters
        rad    = _find("RAD")    # radius in km
        h_mag  = _find("H")
        rotper = _find("ROTPER")

        if ec is None and a is None:
            return None  # not enough data

        if ec is not None:
            elements["eccentricity"] = ec
        if a is not None:
            elements["semi_major_axis"] = a
        if inc is not None:
            elements["inclination"] = inc
        if om is not None:
            elements["ra_of_ascending_node"] = om
        if w is not None:
            elements["arg_of_periapsis"] = w
        if ma is not None:
            elements["mean_anomaly"] = ma

        # Physical: diameter = 2 * RAD (km)
        if rad is not None:
            elements["diameter"] = round(2.0 * rad, 4)
        if rotper is not None:
            elements["rot_per"] = rotper

        return elements if elements else None

    # ------------------------------------------------------------------ #
    # Ephemeris (bonus — not used by DataFetcher but useful standalone)  #
    # ------------------------------------------------------------------ #

    def fetch_ephemeris(
        self,
        designation: str,
        start_date: datetime,
        end_date: datetime,
        step_size: str = "1d",
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch observer ephemeris from Horizons."""
        try:
            resp = requests.get(
                _HORIZONS_URL,
                params={
                    "format":     "json",
                    "COMMAND":    f"'{designation}'",
                    "EPHEM_TYPE": "OBSERVER",
                    "CENTER":     "399",
                    "START_TIME": start_date.strftime("%Y-%m-%d"),
                    "STOP_TIME":  end_date.strftime("%Y-%m-%d"),
                    "STEP_SIZE":  step_size,
                    "QUANTITIES": "1,3,9,19,20,23,24",
                    "OBJ_DATA":   "NO",
                    "MAKE_EPHEM": "YES",
                },
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            return self._parse_ephemeris(resp.json().get("result", ""))
        except Exception as e:
            self.logger.error(f"Horizons ephemeris error for {designation}: {e}")
            return None

    def fetch_orbital_history(self, designation: str, years: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch a time-series of osculating orbital elements from Horizons ELEMENTS ephemeris.
        Returns one entry per year over `years` years centred on today.
        """
        today = datetime.utcnow()
        start = (today - timedelta(days=years * 365 // 2)).strftime("%Y-%m-%d")
        stop = (today + timedelta(days=years * 365 // 2)).strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                _HORIZONS_URL,
                params={
                    "format": "json",
                    "COMMAND": self._horizons_command(designation),
                    "OBJ_DATA": "NO",
                    "MAKE_EPHEM": "YES",
                    "EPHEM_TYPE": "ELEMENTS",
                    "CENTER": "500@10",
                    "START_TIME": start,
                    "STOP_TIME": stop,
                    "STEP_SIZE": "365d",
                    "OUT_UNITS": "AU-D",
                },
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            result_text = resp.json().get("result", "")
            return self._parse_elements_table(result_text)
        except Exception as e:
            self.logger.error(f"Horizons orbital history failed for {designation}: {e}")
            return []

    def _parse_elements_table(self, text: str) -> List[Dict[str, Any]]:
        """Parse the $$SOE ... $$EOE block from a Horizons ELEMENTS ephemeris."""
        import re as _re
        rows: List[Dict[str, Any]] = []
        in_block = False
        pending_epoch: Optional[str] = None
        for line in text.splitlines():
            if "$$SOE" in line:
                in_block = True
                continue
            if "$$EOE" in line:
                break
            if not in_block or not line.strip():
                continue
            # Try to parse element values from the line
            element_keys = [
                ("a",    r"A=\s*([-\d.E+]+)"),
                ("e",    r"EC=\s*([-\d.E+]+)"),
                ("i",    r"IN=\s*([-\d.E+]+)"),
                ("node", r"OM=\s*([-\d.E+]+)"),
                ("peri", r"W=\s*([-\d.E+]+)"),
                ("M",    r"MA=\s*([-\d.E+]+)"),
            ]
            parsed = {}
            for key, pattern in element_keys:
                m = _re.search(pattern, line)
                if m:
                    try:
                        parsed[key] = float(m.group(1))
                    except ValueError:
                        pass
            if parsed:
                if pending_epoch:
                    parsed["epoch"] = pending_epoch
                    pending_epoch = None
                rows.append(parsed)
            elif not any(c in line for c in ["=", "!"]):
                # Likely a date/epoch line
                pending_epoch = line.strip()
        return rows

    def _parse_ephemeris(self, text: str) -> List[Dict[str, Any]]:
        """Parse $$SOE…$$EOE ephemeris block."""
        ephemeris = []
        in_soe = False
        for line in text.splitlines():
            if "$$SOE" in line:
                in_soe = True
                continue
            if "$$EOE" in line:
                break
            if not in_soe or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    ephemeris.append({
                        "date":         parts[0] + " " + parts[1],
                        "ra":           float(parts[2]),
                        "dec":          float(parts[3]),
                        "distance_au":  float(parts[4]),
                    })
                except (ValueError, IndexError):
                    pass
        return ephemeris

    def get_object_summary(self, designation: str) -> Optional[Dict[str, Any]]:
        """Return a combined summary dict."""
        try:
            orbital = self._fetch_orbital_elements_sync(designation)
            start = datetime.now()
            ephemeris = self.fetch_ephemeris(designation, start, start + timedelta(days=30))
            return {
                "designation": designation,
                "orbital_elements": orbital,
                "ephemeris": ephemeris,
                "data_source": "Horizons",
                "retrieved_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Horizons summary error for {designation}: {e}")
            return None
