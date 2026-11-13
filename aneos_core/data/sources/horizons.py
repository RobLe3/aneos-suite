"""
JPL Horizons data source implementation.

Uses the JPL Horizons REST API (ssd.jpl.nasa.gov/api/horizons.api) to
fetch high-precision osculating orbital elements, physical parameters,
and ephemeris data for Near-Earth Objects.
"""

from aneos_core.data.provider_http import get as provider_get
import re
import hashlib
import math
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc

import requests

from .base import HTTPDataSource
from ...config.settings import APIConfig
from ..cache import CacheManager

logger = logging.getLogger(__name__)

_HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


class OrbitalHistoryQuery(list):
    """List-compatible computed history plus explicit bounded query outcome."""

    def __init__(self, records, metadata):
        super().__init__(records)
        import copy

        self.query_metadata = copy.deepcopy(metadata)


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
            resp = provider_get(
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

            return self._attach_orbit_metadata(elements,result_text,designation,data.get('signature'))

        except requests.HTTPError as e:
            self.logger.error(f"Horizons HTTP error for {designation}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Horizons fetch error for {designation}: {e}")
            return None

    def _fetch_as_smallbody(self, designation: str) -> Optional[Dict[str, Any]]:
        """Re-query using DES= prefix to force small-body (asteroid/comet) resolution."""
        try:
            resp = provider_get(
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
            data=resp.json()
            result_text = data.get("result", "")
            if not result_text or "No matches found" in result_text:
                return None
            elements = self._parse_obj_data(result_text)
            return self._attach_orbit_metadata(elements,result_text,designation,data.get('signature')) if elements else None
        except Exception as e:
            self.logger.error(f"Horizons DES fetch error for {designation}: {e}")
            return None

    def _attach_orbit_metadata(self,elements,text,designation,signature=None):
        retrieved=datetime.now(UTC).isoformat()
        elements.update(_source=self.name,_designation=designation,_fetched_at=retrieved,
            _evidence_metadata={'schema_version':'horizons-object-evidence-v1','solution':{
                'source':'JPL Horizons','native_epoch':elements.get('native_epoch'),
                'retrieved_at':retrieved,'result_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'signature':signature,'designation_requested':designation,
                'basis':'catalogue_object_data_not_observation'}})
        return elements

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

        # The object-data epoch is native TDB/TT/etc only when explicitly labeled.
        from ..epoch_evidence import NativeEpoch
        match = re.search(r'(?m)^\s*EPOCH\s*=\s*([+\-\d.Ee]+)([^\n]*)', text)
        if match:
            label=match.group(0).strip()
            scale_match=re.search(r'\b(TDB|TDT|TT|UTC|UT1|UT)\b',match.group(2))
            scale=scale_match.group(1) if scale_match else 'UNKNOWN'
            elements['native_epoch']=NativeEpoch(match.group(1),'JD',scale,'Horizons','osculating_elements',label).to_dict()

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
            resp = provider_get(
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

    def fetch_orbital_history(
        self, designation: str, years: int = 10
    ) -> List[Dict[str, Any]]:
        """Bounded computed samples, with request scope/outage metadata.

        List compatibility is retained for existing callers; an outage is NOT an
        empty observed history. HTTP consumers must inspect query_metadata.status.
        """
        if type(years) is not int or not 1 <= years <= 1000:
            raise ValueError("History years must be a positive bounded integer")
        today = datetime.now(UTC)
        start = (today - timedelta(days=years * 365 // 2)).strftime("%Y-%m-%d")
        stop = (today + timedelta(days=years * 365 // 2)).strftime("%Y-%m-%d")
        params = {
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
            "REF_SYSTEM": "ICRF",
            "REF_PLANE": "ECLIPTIC",
            "TIME_TYPE": "TDB",
            "ELM_LABELS": "YES",
        }
        metadata = {
            "schema_version": "orbital-history-query-v1",
            "designation_requested": designation,
            "requested_years": years,
            "request_parameters": params,
            "attempted_at": today.isoformat(),
            "retrieved_at": None,
            "status": "unavailable",
            "returned_points": 0,
            "result_text_sha256": None,
            "signature": None,
            "schema_status": "not_available",
            "evidence_basis": "computed_ephemeris_not_observations",
            "coverage": "bounded_computed_samples_not_observation_coverage",
        }
        response = None
        try:
            response = provider_get(
                _HORIZONS_URL, params=params, timeout=self.config.request_timeout
            )
            response.raise_for_status()
            metadata["retrieved_at"] = datetime.now(UTC).isoformat()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Invalid Horizons response")
            signature = data.get("signature")
            metadata["signature"] = signature
            if signature is not None:
                if not isinstance(signature, dict) or signature.get("version") not in (
                    "1.2",
                    "1.3",
                ):
                    metadata["schema_status"] = "unsupported"
                    raise ValueError("Unsupported Horizons API schema")
                metadata["schema_status"] = "supported_elements_1.2_1.3"
            else:
                metadata["schema_status"] = "unverified_missing_signature"
            metadata["signature"] = signature
            text = data.get("result")
            if (
                data.get("error")
                or not isinstance(text, str)
                or "$$SOE" not in text
                or "$$EOE" not in text
            ):
                raise ValueError("Horizons did not return a bounded elements block")
            metadata["result_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            rows = self._parse_elements_table(text)
            metadata.update(
                status="available" if rows else "empty", returned_points=len(rows)
            )
            return OrbitalHistoryQuery(rows, metadata)
        except (ValueError, TypeError) as exc:
            metadata.update(
                status="invalid_evidence", error_code="UNSUPPORTED_HISTORY_EVIDENCE"
            )
            self.logger.warning("Horizons history evidence invalid for %s", designation)
        except Exception:
            metadata.update(
                status="unavailable", error_code="HISTORY_SOURCE_UNAVAILABLE"
            )
            self.logger.warning(
                "Horizons history source unavailable for %s", designation
            )
        finally:
            if response is not None:
                response.close()
        return OrbitalHistoryQuery([], metadata)

    def _parse_elements_table(self, text: str) -> List[Dict[str, Any]]:
        """Keep complete computed element rows bound to their native epoch header.

        A real Horizons header contains JD = calendar TDB, unlike an element line.
        Its equals sign must not cause the entire history to be discarded.
        """
        from ..epoch_evidence import horizons_header_epoch
        rows=[];in_block=False;pending_epoch=None;pending_record={}
        element_keys=[('a','A'),('e','EC'),('i','IN'),('node','OM'),('peri','W'),('M','MA')]
        def flush():
            if pending_epoch is not None and pending_record:
                rows.append({'epoch':pending_epoch.label,'native_epoch':pending_epoch.to_dict(),**pending_record})
        for line in text.splitlines():
            if '$$SOE' in line:in_block=True;continue
            if '$$EOE' in line:break
            if not in_block or not line.strip() or line.lstrip().startswith('!'):continue
            epoch=horizons_header_epoch(line)
            if epoch is not None:
                flush();pending_epoch=epoch;pending_record={};continue
            parsed={}
            for key,token in element_keys:
                matches=re.findall(r'(?<![A-Za-z])'+token+r'\s*=\s*([-\d.Ee+]+)',line)
                if len(matches)>1:raise ValueError('Duplicate element in one Horizons line')
                if matches:
                    value=float(matches[0])
                    if not math.isfinite(value):raise ValueError('Nonfinite Horizons orbital element')
                    parsed[key]=value
            if parsed:
                if pending_epoch is None:
                    raise ValueError('Unbound Horizons orbital elements without a recognized epoch header')
                if set(parsed)&set(pending_record):raise ValueError('Duplicate element in one Horizons epoch')
                pending_record.update(parsed)
        flush()
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
