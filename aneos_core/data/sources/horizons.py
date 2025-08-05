"""
JPL Horizons data source implementation.

NASA JPL Horizons system implementation for fetching high-precision
orbital elements and ephemeris data of Near-Earth Objects.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .base import DataSourceBase


class HorizonsSource(DataSourceBase):
    """
    NASA JPL Horizons data source.
    
    Provides access to high-precision orbital elements and ephemeris data
    from NASA's JPL Horizons system.
    """
    
    def __init__(self, timeout: int = 15):
        """
        Initialize Horizons data source.
        
        Args:
            timeout: Request timeout in seconds (longer for Horizons)
        """
        super().__init__(
            name="Horizons",
            base_url="https://ssd.jpl.nasa.gov/api",
            timeout=timeout
        )
    
    def _get_health_check_endpoint(self) -> str:
        """Get health check endpoint for Horizons API."""
        return "horizons.api"
    
    def fetch_orbital_elements(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch orbital elements from Horizons API.
        
        Args:
            designation: NEO designation or name
            
        Returns:
            Dictionary containing orbital elements or None if not found
        """
        try:
            params = {
                "format": "json",
                "COMMAND": f"'{designation}'",
                "OBJ_DATA": "YES",
                "MAKE_EPHEM": "NO"
            }
            
            response = self._make_request("GET", "horizons.api", params=params)
            data = response.json()
            
            if "result" not in data or not data["result"]:
                self.logger.warning(f"No Horizons data found for {designation}")
                return None
            
            result_text = data["result"]
            
            # Parse orbital elements from result text
            # Horizons returns data in a specific text format
            orbital_elements = self._parse_horizons_elements(result_text)
            orbital_elements["data_source"] = "Horizons"
            
            return orbital_elements
            
        except Exception as e:
            self.logger.error(f"Error fetching Horizons orbital elements for {designation}: {e}")
            return None
    
    def fetch_physical_properties(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch physical properties from Horizons.
        
        Args:
            designation: NEO designation or name
            
        Returns:
            Dictionary containing physical properties or None if not found
        """
        try:
            params = {
                "format": "json",
                "COMMAND": f"'{designation}'",
                "OBJ_DATA": "YES",
                "MAKE_EPHEM": "NO"
            }
            
            response = self._make_request("GET", "horizons.api", params=params)
            data = response.json()
            
            if "result" not in data:
                return None
            
            result_text = data["result"]
            
            # Parse physical properties from result text
            properties = self._parse_horizons_physical(result_text)
            properties["data_source"] = "Horizons"
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error fetching Horizons physical properties for {designation}: {e}")
            return None
    
    def fetch_ephemeris(
        self,
        designation: str,
        start_date: datetime,
        end_date: datetime,
        step_size: str = "1d"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch ephemeris data from Horizons.
        
        Args:
            designation: NEO designation or name
            start_date: Start date for ephemeris
            end_date: End date for ephemeris
            step_size: Step size (e.g., '1d', '1h', '1m')
            
        Returns:
            List of ephemeris points
        """
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            params = {
                "format": "json",
                "COMMAND": f"'{designation}'",
                "OBJ_DATA": "NO",
                "MAKE_EPHEM": "YES",
                "EPHEM_TYPE": "OBSERVER",
                "CENTER": "399",  # Earth
                "START_TIME": start_str,
                "STOP_TIME": end_str,
                "STEP_SIZE": step_size,
                "QUANTITIES": "1,3,9,19,20,23,24"  # Selected quantities
            }
            
            response = self._make_request("GET", "horizons.api", params=params)
            data = response.json()
            
            if "result" not in data:
                return None
            
            result_text = data["result"]
            
            # Parse ephemeris data from result text
            ephemeris = self._parse_horizons_ephemeris(result_text)
            
            return ephemeris
            
        except Exception as e:
            self.logger.error(f"Error fetching Horizons ephemeris for {designation}: {e}")
            return None
    
    def _parse_horizons_elements(self, result_text: str) -> Dict[str, Any]:
        """Parse orbital elements from Horizons result text."""
        elements = {}
        
        lines = result_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse semi-major axis
            if "A=" in line:
                try:
                    a_part = line.split("A=")[1].split()[0]
                    elements["semi_major_axis"] = float(a_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse eccentricity
            if "EC=" in line:
                try:
                    e_part = line.split("EC=")[1].split()[0]
                    elements["eccentricity"] = float(e_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse inclination
            if "IN=" in line:
                try:
                    i_part = line.split("IN=")[1].split()[0]
                    elements["inclination"] = float(i_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse longitude of ascending node
            if "OM=" in line:
                try:
                    om_part = line.split("OM=")[1].split()[0]
                    elements["longitude_of_ascending_node"] = float(om_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse argument of periapsis
            if "W =" in line:
                try:
                    w_part = line.split("W =")[1].split()[0]
                    elements["argument_of_periapsis"] = float(w_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse mean anomaly
            if "MA=" in line:
                try:
                    ma_part = line.split("MA=")[1].split()[0]
                    elements["mean_anomaly"] = float(ma_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse perihelion distance
            if "QR=" in line:
                try:
                    q_part = line.split("QR=")[1].split()[0]
                    elements["perihelion_distance"] = float(q_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse aphelion distance
            if "AD=" in line:
                try:
                    ad_part = line.split("AD=")[1].split()[0]
                    elements["aphelion_distance"] = float(ad_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse orbital period
            if "PR=" in line:
                try:
                    pr_part = line.split("PR=")[1].split()[0]
                    elements["orbital_period"] = float(pr_part) / 365.25  # Convert to years
                except (IndexError, ValueError):
                    pass
        
        return elements
    
    def _parse_horizons_physical(self, result_text: str) -> Dict[str, Any]:
        """Parse physical properties from Horizons result text."""
        properties = {}
        
        lines = result_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse absolute magnitude
            if "H=" in line:
                try:
                    h_part = line.split("H=")[1].split()[0]
                    properties["absolute_magnitude"] = float(h_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse slope parameter
            if "G=" in line:
                try:
                    g_part = line.split("G=")[1].split()[0]
                    properties["slope_parameter"] = float(g_part)
                except (IndexError, ValueError):
                    pass
            
            # Parse diameter
            if "DIAMETER" in line.upper():
                try:
                    # Extract diameter value (format varies)
                    diameter_str = line.split("=")[-1].strip()
                    diameter_val = float(diameter_str.split()[0])
                    properties["diameter"] = diameter_val
                except (IndexError, ValueError):
                    pass
        
        return properties
    
    def _parse_horizons_ephemeris(self, result_text: str) -> List[Dict[str, Any]]:
        """Parse ephemeris data from Horizons result text."""
        ephemeris = []
        
        lines = result_text.split('\n')
        data_section = False
        
        for line in lines:
            line = line.strip()
            
            # Start of data section
            if "$$SOE" in line:
                data_section = True
                continue
            
            # End of data section
            if "$$EOE" in line:
                break
            
            if data_section and line and not line.startswith("Date"):
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        point = {
                            "date": self._parse_horizons_date(parts[0] + " " + parts[1]),
                            "ra": float(parts[2]),  # Right ascension
                            "dec": float(parts[3]),  # Declination
                            "distance_au": float(parts[4]),
                            "distance_rate": float(parts[5]) if len(parts) > 5 else None
                        }
                        ephemeris.append(point)
                except (ValueError, IndexError):
                    continue
        
        return ephemeris
    
    def _parse_horizons_date(self, date_str: str) -> Optional[datetime]:
        """Parse Horizons date format."""
        try:
            # Horizons format: "2023-Jan-01 12:00"
            return datetime.strptime(date_str, "%Y-%b-%d %H:%M")
        except ValueError:
            try:
                # Alternative format: "2023-01-01 12:00"
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            except ValueError:
                self.logger.warning(f"Could not parse Horizons date: {date_str}")
                return None
    
    def get_object_summary(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive object summary from Horizons.
        
        Args:
            designation: NEO designation or name
            
        Returns:
            Dictionary containing all available data
        """
        try:
            orbital_elements = self.fetch_orbital_elements(designation)
            physical_properties = self.fetch_physical_properties(designation)
            
            # Get short ephemeris for next 30 days
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30)
            ephemeris = self.fetch_ephemeris(designation, start_date, end_date, "1d")
            
            summary = {
                "designation": designation,
                "orbital_elements": orbital_elements,
                "physical_properties": physical_properties,
                "ephemeris": ephemeris,
                "data_source": "Horizons",
                "retrieved_at": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting Horizons summary for {designation}: {e}")
            return None