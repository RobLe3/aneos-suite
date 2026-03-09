"""
SBDB (Small-Body Database) data source implementation.

This module provides integration with NASA's Small-Body Database API
for fetching Near Earth Object orbital elements and physical parameters.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime, UTC

from .base import HTTPDataSource, FetchResult, DataSourceException
from ...config.settings import APIConfig
from ..cache import CacheManager

logger = logging.getLogger(__name__)

class SBDBSource(HTTPDataSource):
    """NASA Small-Body Database data source."""
    
    def __init__(self, config: APIConfig, cache_manager: Optional[CacheManager] = None):
        super().__init__(
            name="SBDB",
            base_url=config.sbdb_url,
            config=config,
            cache_manager=cache_manager
        )
    
    async def fetch_orbital_elements(self, designation: str) -> FetchResult:
        """Fetch orbital elements from SBDB API."""
        try:
            params = {
                "sstr": designation,
                "phys-par": "true",
            }

            response_data = await self._http_get("", params)

            # Optionally enrich with non-gravitational parameters (only ~3% of NEOs
            # have these; requesting nongrav=1 for the rest returns HTTP 400).
            try:
                ng_params = {"sstr": designation, "phys-par": "false", "nongrav": "1"}
                ng_response = await self._http_get("", ng_params)
                if "nongrav_params" in ng_response:
                    response_data["nongrav_params"] = ng_response["nongrav_params"]
            except Exception as _ng_exc:
                logger.debug("Non-gravitational fetch skipped for %s: %s", designation, _ng_exc)
            
            if "orbit" not in response_data:
                return FetchResult(
                    success=False,
                    error_message=f"No orbital data found for {designation}",
                    source=self.name
                )
            
            # Parse SBDB response format
            orbital_data = self._parse_sbdb_response(response_data, designation)
            
            return FetchResult(
                success=True,
                data=orbital_data,
                source=self.name
            )
            
        except Exception as e:
            error_msg = f"SBDB API error for {designation}: {str(e)}"
            logger.error(error_msg)
            
            return FetchResult(
                success=False,
                error_message=error_msg,
                source=self.name
            )
    
    def _parse_sbdb_response(self, data: Dict[str, Any], designation: str) -> Dict[str, Any]:
        """Parse SBDB API response into standardized format."""
        orbital_data = {}

        # Extract orbital elements (list of {name, value, ...} dicts)
        if "orbit" in data:
            elements = data["orbit"].get("elements", [])

            # Maps the SBDB 'name' field to OrbitalElements field names
            element_mapping = {
                "e":  "eccentricity",
                "i":  "inclination",
                "a":  "semi_major_axis",
                "om": "ra_of_ascending_node",
                "w":  "arg_of_periapsis",
                "ma": "mean_anomaly",
            }

            for element in elements:
                sbdb_name = element.get("name")
                value = element.get("value")
                if sbdb_name in element_mapping and value is not None:
                    try:
                        orbital_data[element_mapping[sbdb_name]] = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing {sbdb_name}='{value}' for {designation}: {e}")

        # Extract physical parameters (list of {name, value, ...} dicts → convert to dict)
        phys_list = data.get("phys_par", [])
        phys_par = {p["name"]: p.get("value") for p in phys_list if p.get("value") is not None}

        # Build physical sub-dict for PhysicalProperties construction
        physical_data: Dict[str, Any] = {}
        for float_field in ("diameter", "albedo", "rot_per"):
            if float_field in phys_par:
                try:
                    physical_data[float_field] = float(phys_par[float_field])
                except (ValueError, TypeError):
                    pass

        if "spec_T" in phys_par:
            physical_data["spectral_type"] = str(phys_par["spec_T"])

        if "H" in phys_par:
            try:
                physical_data["absolute_magnitude_h"] = float(phys_par["H"])
            except (ValueError, TypeError):
                pass

        # Store physical data separately for consumers that use NEOData.physical_properties
        orbital_data["_physical"] = physical_data

        # Non-gravitational parameters (present for ~3% of NEOs)
        ng_list = data.get("nongrav_params", [])
        nongrav_data: Dict[str, Any] = {}
        for item in ng_list:
            name = item.get("name", "")
            value = item.get("value")
            if value is not None:
                try:
                    if name == "A1":
                        nongrav_data["a1"] = float(value)
                    elif name == "A2":
                        nongrav_data["a2"] = float(value)
                    elif name == "A3":
                        nongrav_data["a3"] = float(value)
                    elif name in ("model", "Model"):
                        nongrav_data["model"] = str(value)
                except (ValueError, TypeError):
                    pass
        orbital_data["_nongrav"] = nongrav_data if nongrav_data else None

        # Observation window (first_obs / last_obs from orbit section)
        orbit_section = data.get("orbit", {})
        for sbdb_key, out_key in (("first_obs", "first_observation_date"),
                                   ("last_obs",  "last_observation_date")):
            raw_date = orbit_section.get(sbdb_key)
            if raw_date:
                orbital_data[out_key] = str(raw_date)

        # Source metadata (filtered out before OrbitalElements construction)
        orbital_data["_source"] = self.name
        orbital_data["_designation"] = designation
        orbital_data["_fetched_at"] = datetime.now(UTC).isoformat()

        return orbital_data
    
    def _parse_epoch(self, epoch_str: str) -> Optional[str]:
        """Parse SBDB epoch format."""
        try:
            # SBDB typically uses format like "2023-01-01"
            parsed_date = datetime.strptime(epoch_str, "%Y-%m-%d")
            return parsed_date.isoformat()
        except ValueError:
            try:
                # Try alternative format
                parsed_date = datetime.strptime(epoch_str, "%Y-%b-%d")
                return parsed_date.isoformat()
            except ValueError:
                logger.warning(f"Could not parse SBDB epoch format: {epoch_str}")
                return None
    
    async def health_check(self) -> bool:
        """Perform health check by testing a known object (Apophis / 99942)."""
        try:
            result = await self.fetch_orbital_elements("99942")
            return result.success

        except Exception as e:
            logger.error(f"SBDB health check failed: {e}")
            return False