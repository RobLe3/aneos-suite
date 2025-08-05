"""
SBDB (Small-Body Database) data source implementation.

This module provides integration with NASA's Small-Body Database API
for fetching Near Earth Object orbital elements and physical parameters.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

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
                "des": designation,
                "phys_par": 1  # Include physical parameters
            }
            
            response_data = await self._http_get("", params)
            
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
        
        # Extract orbital elements
        if "orbit" in data:
            orbit = data["orbit"]
            elements = orbit.get("elements", [])
            
            # Map SBDB element names to our standard names
            element_mapping = {
                "e": "eccentricity",
                "i": "inclination", 
                "a": "semi_major_axis",
                "node": "ra_of_ascending_node",
                "w": "arg_of_periapsis",
                "M": "mean_anomaly",
                "epoch": "epoch"
            }
            
            for element in elements:
                name = element.get("name")
                value = element.get("value")
                
                if name in element_mapping and value is not None:
                    standard_name = element_mapping[name]
                    
                    try:
                        if name == "epoch":
                            # Parse epoch date
                            orbital_data[standard_name] = self._parse_epoch(value)
                        else:
                            orbital_data[standard_name] = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing {name} value '{value}' for {designation}: {e}")
        
        # Extract physical parameters
        if "phys_par" in data:
            phys_par = data["phys_par"]
            
            if "diameter" in phys_par and phys_par["diameter"]:
                try:
                    orbital_data["diameter"] = float(phys_par["diameter"])
                except (ValueError, TypeError):
                    pass
            
            if "albedo" in phys_par and phys_par["albedo"]:
                try:
                    orbital_data["albedo"] = float(phys_par["albedo"])
                except (ValueError, TypeError):
                    pass
            
            if "rot_per" in phys_par and phys_par["rot_per"]:
                try:
                    orbital_data["rot_per"] = float(phys_par["rot_per"])
                except (ValueError, TypeError):
                    pass
        
        # Add source metadata
        orbital_data["_source"] = self.name
        orbital_data["_designation"] = designation
        orbital_data["_fetched_at"] = datetime.utcnow().isoformat()
        
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
        """Perform health check by testing a known object."""
        try:
            # Test with a well-known asteroid (Ceres)
            result = await self.fetch_orbital_elements("1 Ceres")
            return result.success
            
        except Exception as e:
            logger.error(f"SBDB health check failed: {e}")
            return False