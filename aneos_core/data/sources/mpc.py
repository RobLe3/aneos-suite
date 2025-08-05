"""
MPC (Minor Planet Center) data source implementation.

Minor Planet Center API implementation for fetching orbital elements
and observational data of Near-Earth Objects.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import DataSourceBase


class MPCSource(DataSourceBase):
    """
    Minor Planet Center (MPC) data source.
    
    Provides access to orbital elements and observational data
    from the IAU Minor Planet Center.
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize MPC data source.
        
        Args:
            timeout: Request timeout in seconds
        """
        super().__init__(
            name="MPC",
            base_url="https://www.minorplanetcenter.net/web_service",
            timeout=timeout
        )
    
    def _get_health_check_endpoint(self) -> str:
        """Get health check endpoint for MPC API."""
        return "search_orbits"
    
    def fetch_orbital_elements(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch orbital elements from MPC API.
        
        Args:
            designation: NEO designation or name
            
        Returns:
            Dictionary containing orbital elements or None if not found
        """
        try:
            params = {
                "designation": designation,
                "format": "json"
            }
            
            response = self._make_request("GET", "search_orbits", params=params)
            data = response.json()
            
            if not data or len(data) == 0:
                self.logger.warning(f"No MPC data found for {designation}")
                return None
            
            # MPC returns array of results, take first match
            mpc_data = data[0]
            
            orbital_elements = {
                "semi_major_axis": self._safe_float(mpc_data.get("a")),
                "eccentricity": self._safe_float(mpc_data.get("e")),
                "inclination": self._safe_float(mpc_data.get("i")),
                "longitude_of_ascending_node": self._safe_float(mpc_data.get("Node")),
                "argument_of_periapsis": self._safe_float(mpc_data.get("Peri")),
                "mean_anomaly": self._safe_float(mpc_data.get("M")),
                "epoch_jd": self._safe_float(mpc_data.get("Epoch")),
                "perihelion_distance": self._safe_float(mpc_data.get("q")),
                "orbital_period": self._safe_float(mpc_data.get("P")),
                "data_source": "MPC"
            }
            
            # Calculate aphelion distance if not provided
            if (orbital_elements["semi_major_axis"] and 
                orbital_elements["eccentricity"] and 
                not orbital_elements.get("aphelion_distance")):
                a = orbital_elements["semi_major_axis"]
                e = orbital_elements["eccentricity"]
                orbital_elements["aphelion_distance"] = a * (1 + e)
            
            return orbital_elements
            
        except Exception as e:
            self.logger.error(f"Error fetching MPC orbital elements for {designation}: {e}")
            return None
    
    def fetch_physical_properties(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch physical properties from MPC.
        
        Note: MPC has limited physical property data.
        
        Args:
            designation: NEO designation or name
            
        Returns:
            Dictionary containing physical properties or None if not found
        """
        try:
            params = {
                "designation": designation,
                "format": "json"
            }
            
            response = self._make_request("GET", "search_orbits", params=params)
            data = response.json()
            
            if not data or len(data) == 0:
                return None
            
            mpc_data = data[0]
            
            properties = {
                "absolute_magnitude": self._safe_float(mpc_data.get("H")),
                "data_source": "MPC"
            }
            
            # MPC occasionally has additional physical parameters
            if "G" in mpc_data:
                properties["slope_parameter"] = self._safe_float(mpc_data["G"])
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error fetching MPC physical properties for {designation}: {e}")
            return None
    
    def fetch_observations(self, designation: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch observational data from MPC.
        
        Args:
            designation: NEO designation or name
            limit: Maximum number of observations to fetch
            
        Returns:
            List of observation dictionaries
        """
        try:
            params = {
                "designation": designation,
                "limit": limit,
                "format": "json"
            }
            
            response = self._make_request("GET", "search_observations", params=params)
            data = response.json()
            
            if not data:
                self.logger.warning(f"No MPC observations found for {designation}")
                return None
            
            observations = []
            for obs in data:
                observation = {
                    "date": self._parse_date(obs.get("Date")),
                    "ra": self._safe_float(obs.get("RA")),  # Right ascension
                    "dec": self._safe_float(obs.get("Dec")),  # Declination
                    "magnitude": self._safe_float(obs.get("Mag")),
                    "observatory_code": obs.get("Obs"),
                    "note": obs.get("Note"),
                    "data_source": "MPC"
                }
                observations.append(observation)
            
            return observations
            
        except Exception as e:
            self.logger.error(f"Error fetching MPC observations for {designation}: {e}")
            return None
    
    def search_neos(self, **criteria) -> Optional[List[str]]:
        """
        Search for NEOs matching criteria.
        
        Args:
            **criteria: Search criteria
            
        Returns:
            List of designations matching criteria
        """
        try:
            params = {
                "format": "json",
                "neo": "true"  # Search for NEOs only
            }
            params.update(criteria)
            
            response = self._make_request("GET", "search_orbits", params=params)
            data = response.json()
            
            if not data:
                return []
            
            designations = []
            for obj in data:
                if "Principal_desig" in obj:
                    designations.append(obj["Principal_desig"])
                elif "Name" in obj:
                    designations.append(obj["Name"])
            
            return designations
            
        except Exception as e:
            self.logger.error(f"Error searching MPC NEOs: {e}")
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # MPC date format: "2023 01 01.12345"
            if " " in date_str:
                parts = date_str.strip().split()
                if len(parts) >= 3:
                    year = int(parts[0])
                    month = int(parts[1])
                    day_frac = float(parts[2])
                    day = int(day_frac)
                    hour_frac = (day_frac - day) * 24
                    hour = int(hour_frac)
                    minute = int((hour_frac - hour) * 60)
                    
                    return datetime(year, month, day, hour, minute)
            
            # Fallback to ISO format
            return datetime.fromisoformat(date_str)
            
        except (ValueError, IndexError):
            self.logger.warning(f"Could not parse MPC date: {date_str}")
            return None
    
    def get_object_summary(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive object summary from MPC.
        
        Args:
            designation: NEO designation or name
            
        Returns:
            Dictionary containing all available data
        """
        try:
            orbital_elements = self.fetch_orbital_elements(designation)
            physical_properties = self.fetch_physical_properties(designation)
            observations = self.fetch_observations(designation, limit=10)
            
            summary = {
                "designation": designation,
                "orbital_elements": orbital_elements,
                "physical_properties": physical_properties,
                "observations": observations,
                "data_source": "MPC",
                "retrieved_at": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting MPC summary for {designation}: {e}")
            return None