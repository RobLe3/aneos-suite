"""
NEODyS data source implementation.

Fetches osculating orbital elements from the University of Pisa NEODyS-2
catalogue.  Objects are stored as equinoctial-element files (.eq0) at:
  https://newton.spacedys.com/~neodys2/epoch/{number-or-provisional-identifier}.eq0

Equinoctial elements are converted to classical Keplerian elements
(a, e, i, Omega, omega, M) for use with the rest of the pipeline.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import requests

from .base import DataSourceBase
from ...config.settings import APIConfig
from ..cache import CacheManager

logger = logging.getLogger(__name__)


class NEODySSource(DataSourceBase):
    """
    NEODyS-2 (Near Earth Objects Dynamic Site) data source.

    Downloads the osculating .eq0 element file for each object and converts
    the equinoctial elements to classical orbital elements.
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        if config is None:
            config = APIConfig()
        super().__init__(name="NEODyS", config=config, cache_manager=cache_manager)
        self.base_url = config.neodys_url
        self.rest_url = config.neodys_rest_url

    def get_base_url(self) -> str:
        return self.base_url

    async def health_check(self) -> bool:
        """Check that the NEODyS epoch directory is accessible."""
        import aiohttp

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8)
            ) as session:
                async with session.get(self.base_url) as response:
                    return response.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Synchronous implementation (overrides the abstract async stub)      #
    # ------------------------------------------------------------------ #

    def _catalogue_target(self, designation):
        from ..neodys_evidence import catalogue_identifier
        from .mpc import MPCSource

        try:
            return catalogue_identifier(designation), None
        except ValueError:
            # Names/number+name require actual native binding, not a leading-digit guess.
            row = MPCSource(config=self.config)._query_mpc(designation)
            if row is None:
                return None, None
            number = row.get("number")
            if number is None:
                raise ValueError("Matched catalogue row supplies no permanent number")
            target = catalogue_identifier(number)
            if not target.isdigit():
                raise ValueError("Invalid permanent number")
            return target, {
                "native_catalogue_row": dict(row),
                "identity_assessment": getattr(row, "identity_assessment", None),
            }

    def fetch_acquisition(self, designation):
        """Fetch a native numbered/provisional OEF file, with explicit outcomes."""
        from .base import FetchResult
        from ...utils.errors import AmbiguousObjectError, SourceIdentityError

        try:
            target, resolver = self._catalogue_target(designation)
            if target is None:
                return FetchResult(
                    False,
                    source=self.name,
                    outcome="missing",
                    error_code="NEODYS_ALIAS_NO_MATCH",
                )
            result = self._download_eq0(target)
            if result.success and resolver is not None:
                result.data["_evidence_metadata"]["alias_resolution"] = resolver
            return result
        except AmbiguousObjectError as exc:
            return FetchResult(
                False,
                source=self.name,
                outcome="ambiguous",
                error_code="NEODYS_ALIAS_AMBIGUOUS",
                raw_response=getattr(exc, "native_response", None),
            )
        except (SourceIdentityError, ValueError, TypeError) as exc:
            return FetchResult(
                False,
                source=self.name,
                outcome="invalid",
                error_code="NEODYS_INVALID_IDENTITY",
                raw_response=getattr(exc, "native_response", None),
            )
        except ImportError:
            return FetchResult(
                False,
                source=self.name,
                outcome="unavailable",
                error_code="NEODYS_ALIAS_DEPENDENCY_UNAVAILABLE",
            )
        except Exception:
            return FetchResult(
                False,
                source=self.name,
                outcome="unavailable",
                error_code="NEODYS_ALIAS_UNAVAILABLE",
            )

    def fetch_orbital_elements(self, designation):
        """Compatibility dict/None API; acquisition retains detailed outcome/native data."""
        result = self.fetch_acquisition(designation)
        return result.data if result.success else None

    def _resolve_number(self, designation):
        """Legacy permanent-number resolver with native MPC identity binding.

        Ordinary provisional acquisition uses its documented direct filename instead.
        """
        from ..neodys_evidence import catalogue_identifier

        try:
            target = catalogue_identifier(designation)
            if target.isdigit():
                return int(target)
        except ValueError:
            pass
        from .mpc import MPCSource

        row = MPCSource(config=self.config)._query_mpc(designation)
        if row is None:
            return None
        number = catalogue_identifier(row.get("number"))
        return int(number) if number.isdigit() else None

    def _download_eq0(self, target):
        from .base import FetchResult
        from ..provider_http import APPLICATION_HEADERS
        from ..neodys_evidence import parse_oef, catalogue_identifier

        target = catalogue_identifier(target)
        raw = None
        response = None
        try:
            response = requests.get(
                self.base_url.rstrip("/") + "/" + target + ".eq0",
                timeout=self.config.request_timeout,
                headers=APPLICATION_HEADERS,
                allow_redirects=False,
            )
            raw = {"equinoctial_text": response.text}
            if response.status_code == 404:
                return FetchResult(
                    False,
                    source=self.name,
                    outcome="missing",
                    raw_response=raw,
                    error_code="NEODYS_FILE_NOT_FOUND",
                )
            if response.status_code != 200:
                return FetchResult(
                    False,
                    source=self.name,
                    outcome="unavailable",
                    raw_response=raw,
                    error_code="NEODYS_HTTP_UNAVAILABLE",
                )
            parsed = parse_oef(response.text, target)
            parsed["_native_response"] = raw
            return FetchResult(
                True,
                data=parsed,
                source=self.name,
                outcome="available",
                raw_response=raw,
            )
        except (ValueError, TypeError, OverflowError):
            return FetchResult(
                False,
                source=self.name,
                outcome="invalid",
                raw_response=raw,
                error_code="NEODYS_INVALID_OEF",
            )
        except Exception:
            return FetchResult(
                False,
                source=self.name,
                outcome="unavailable",
                raw_response=raw,
                error_code="NEODYS_REQUEST_UNAVAILABLE",
            )
        finally:
            if response is not None:
                response.close()

    def _fetch_eq0(self, number):
        """Compatibility path; never replaces a failed request with invented elements."""
        result = self._download_eq0(number)
        return result.data if result.success else None

    def _parse_eq0(self, text, number):
        from ..neodys_evidence import parse_oef

        return parse_oef(text, number)

    @staticmethod
    def _equinoctial_to_keplerian(a, esinLP, ecosLP, hLN, kLN, mean_L):
        from ..neodys_evidence import equinoctial_to_keplerian

        return equinoctial_to_keplerian(a, esinLP, ecosLP, hLN, kLN, mean_L)

    def _make_request(self, designation):
        """Controlled legacy projection through validated OEF acquisition, not undocumented REST/zero defaults."""
        result = self.fetch_acquisition(designation)
        if not result.success:
            return None
        data = result.data
        return {
            "a": data.get("semi_major_axis"),
            "e": data.get("eccentricity"),
            "i": data.get("inclination"),
            "omega": data.get("ra_of_ascending_node"),
            "w": data.get("arg_of_periapsis"),
            "M": data.get("mean_anomaly"),
            "H": data.get("_physical", {}).get("absolute_magnitude_h"),
            "source": "NEODyS",
        }

    def get_object_summary(self, designation: str) -> Optional[Dict[str, Any]]:
        """Return a combined summary dict."""
        try:
            orbital = self.fetch_orbital_elements(designation)
            return {
                "designation": designation,
                "orbital_elements": orbital,
                "data_source": "NEODyS",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"NEODyS summary error for {designation}: {e}")
            return None
