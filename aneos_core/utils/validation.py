"""
Input validation utilities for aNEOS Core.

Provides validation functions for NEO data, orbital elements,
and other inputs to ensure data quality and consistency.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from ..config.constants import *


def validate_neo_designation(designation: str) -> Tuple[bool, str]:
    """
    Validate NEO designation format.
    
    Args:
        designation: NEO designation string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not designation or not isinstance(designation, str):
        return False, "Designation must be a non-empty string"
    
    designation = designation.strip()
    
    # Common NEO designation patterns
    patterns = [
        r'^\d{4}\s[A-Z]{2}\d*$',  # 2023 AB1
        r'^\(\d+\)$',             # (12345)
        r'^\d+\s[A-Za-z\s]+$',    # 433 Eros
        r'^[A-Za-z\s]+$',         # Apophis
        r'^\d{4}\s[A-Z]{2}\d*[A-Z]*$',  # 2023 AB123C
    ]
    
    for pattern in patterns:
        if re.match(pattern, designation):
            return True, ""
    
    return False, f"Invalid designation format: {designation}"


def validate_orbital_elements(elements: Dict[str, Any]) -> List[str]:
    """
    Validate orbital elements for physical consistency.
    
    Args:
        elements: Dictionary of orbital elements
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Semi-major axis validation
    a = elements.get("semi_major_axis")
    if a is not None:
        if not isinstance(a, (int, float)) or a <= 0:
            errors.append("Semi-major axis must be positive")
        elif a > 1000:  # Unreasonably large for NEOs
            errors.append("Semi-major axis too large for NEO")
    
    # Eccentricity validation
    e = elements.get("eccentricity")
    if e is not None:
        if not isinstance(e, (int, float)) or e < 0 or e >= 1:
            errors.append("Eccentricity must be between 0 and 1")
    
    # Inclination validation
    i = elements.get("inclination")
    if i is not None:
        if not isinstance(i, (int, float)) or i < 0 or i > 180:
            errors.append("Inclination must be between 0 and 180 degrees")
    
    # Angular elements validation (0-360 degrees)
    angular_elements = [
        "longitude_of_ascending_node",
        "argument_of_periapsis",
        "mean_anomaly"
    ]
    
    for elem_name in angular_elements:
        value = elements.get(elem_name)
        if value is not None:
            if not isinstance(value, (int, float)):
                errors.append(f"{elem_name} must be numeric")
            elif value < 0 or value >= 360:
                errors.append(f"{elem_name} must be between 0 and 360 degrees")
    
    # Perihelion distance validation (NEO criterion)
    q = elements.get("perihelion_distance")
    if q is not None:
        if not isinstance(q, (int, float)) or q <= 0:
            errors.append("Perihelion distance must be positive")
        elif q > NEO_MIN_PERIHELION:
            errors.append(f"Perihelion distance > {NEO_MIN_PERIHELION} AU - may not be NEO")
    
    # Consistency checks
    if a is not None and e is not None and q is not None:
        expected_q = a * (1 - e)
        if abs(q - expected_q) > 0.01:  # Small tolerance for rounding
            errors.append("Inconsistent perihelion distance with a and e")
    
    # Orbital period validation
    period = elements.get("orbital_period")
    if period is not None and a is not None:
        # Kepler's third law: P² = a³ (in appropriate units)
        expected_period = a ** 1.5  # Simplified for AU and years
        if abs(period - expected_period) > 0.1 * expected_period:
            errors.append("Orbital period inconsistent with semi-major axis")
    
    return errors


def validate_physical_properties(properties: Dict[str, Any]) -> List[str]:
    """
    Validate physical properties for reasonableness.
    
    Args:
        properties: Dictionary of physical properties
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Diameter validation
    diameter = properties.get("diameter")
    if diameter is not None:
        if not isinstance(diameter, (int, float)) or diameter <= 0:
            errors.append("Diameter must be positive")
        elif diameter > 50:  # km - Very large for NEO
            errors.append("Diameter unusually large for NEO")
    
    # Albedo validation
    albedo = properties.get("albedo")
    if albedo is not None:
        if not isinstance(albedo, (int, float)) or albedo < 0 or albedo > 1:
            errors.append("Albedo must be between 0 and 1")
    
    # Absolute magnitude validation
    h_mag = properties.get("absolute_magnitude")
    if h_mag is not None:
        if not isinstance(h_mag, (int, float)):
            errors.append("Absolute magnitude must be numeric")
        elif h_mag < 10 or h_mag > 35:
            errors.append("Absolute magnitude outside typical NEO range")
    
    # Rotation period validation
    rot_period = properties.get("rotation_period")
    if rot_period is not None:
        if not isinstance(rot_period, (int, float)) or rot_period <= 0:
            errors.append("Rotation period must be positive")
        elif rot_period < 0.1:  # hours - Very fast rotation
            errors.append("Rotation period unusually fast")
        elif rot_period > 1000:  # hours - Very slow rotation
            errors.append("Rotation period unusually slow")
    
    # Consistency check: diameter vs H magnitude
    if diameter is not None and h_mag is not None and albedo is not None:
        # Standard formula: D = 1329 * 10^(-0.2*H) / sqrt(A)
        expected_diameter = 1329 * (10 ** (-0.2 * h_mag)) / (albedo ** 0.5)
        if abs(diameter - expected_diameter) > 0.5 * expected_diameter:
            errors.append("Diameter inconsistent with H magnitude and albedo")
    
    return errors


def validate_close_approach(approach: Dict[str, Any]) -> List[str]:
    """
    Validate close approach data.
    
    Args:
        approach: Close approach dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Date validation
    date = approach.get("date")
    if date is not None:
        if not isinstance(date, datetime):
            errors.append("Close approach date must be datetime object")
    
    # Distance validation
    distance_au = approach.get("distance_au")
    if distance_au is not None:
        if not isinstance(distance_au, (int, float)) or distance_au <= 0:
            errors.append("Distance must be positive")
        elif distance_au > 10:  # AU - Very distant
            errors.append("Close approach distance unusually large")
    
    # Velocity validation
    velocity = approach.get("velocity_km_s")
    if velocity is not None:
        if not isinstance(velocity, (int, float)) or velocity <= 0:
            errors.append("Velocity must be positive")
        elif velocity > SOLAR_ESCAPE_VELOCITY:
            errors.append("Velocity exceeds solar escape velocity")
    
    return errors


def validate_coordinate(value: float, coord_type: str) -> bool:
    """
    Validate coordinate values.
    
    Args:
        value: Coordinate value
        coord_type: Type of coordinate ('ra', 'dec', 'lat', 'lon')
        
    Returns:
        True if valid
    """
    if not isinstance(value, (int, float)):
        return False
    
    if coord_type in ['ra', 'lon']:
        return 0 <= value < 360
    elif coord_type in ['dec', 'lat']:
        return -90 <= value <= 90
    
    return False


def validate_date_range(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> Tuple[bool, str]:
    """
    Validate date range for data queries.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        if start_date >= end_date:
            return False, "Start date must be before end date"
        
        # Check for reasonable date range
        date_diff = (end_date - start_date).days
        if date_diff > 36500:  # 100 years
            return False, "Date range too large (max 100 years)"
        
        return True, ""
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid date format: {e}"


def validate_search_criteria(criteria: Dict[str, Any]) -> List[str]:
    """
    Validate search criteria for NEO queries.
    
    Args:
        criteria: Search criteria dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate numeric ranges
    numeric_fields = {
        'diameter_min': (0, 50),  # km
        'diameter_max': (0, 50),
        'albedo_min': (0, 1),
        'albedo_max': (0, 1),
        'h_min': (10, 35),  # magnitude
        'h_max': (10, 35),
        'period_min': (0, 1000),  # years
        'period_max': (0, 1000),
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        value = criteria.get(field)
        if value is not None:
            if not isinstance(value, (int, float)):
                errors.append(f"{field} must be numeric")
            elif value < min_val or value > max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}")
    
    # Validate min/max pairs
    pairs = [
        ('diameter_min', 'diameter_max'),
        ('albedo_min', 'albedo_max'),
        ('h_min', 'h_max'),
        ('period_min', 'period_max')
    ]
    
    for min_field, max_field in pairs:
        min_val = criteria.get(min_field)
        max_val = criteria.get(max_field)
        if min_val is not None and max_val is not None:
            if min_val >= max_val:
                errors.append(f"{min_field} must be less than {max_field}")
    
    return errors


def sanitize_designation(designation: str) -> str:
    """
    Sanitize NEO designation for safe use in queries.
    
    Args:
        designation: Raw designation string
        
    Returns:
        Sanitized designation
    """
    if not designation:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[^\w\s\(\)\-\.]', '', designation)
    
    # Normalize whitespace
    sanitized = ' '.join(sanitized.split())
    
    return sanitized.strip()


class DataValidator:
    """
    Comprehensive data validator for NEO data structures.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_cache = {}
    
    def validate_neo_data(self, neo_data: Dict[str, Any], strict: bool = False) -> Dict[str, List[str]]:
        """
        Validate complete NEO data structure.
        
        Args:
            neo_data: NEO data dictionary
            strict: Use strict validation rules
            
        Returns:
            Dictionary of validation results by category
        """
        results = {
            "designation": [],
            "orbital_elements": [],
            "physical_properties": [],
            "close_approaches": [],
            "general": []
        }
        
        # Validate designation
        designation = neo_data.get("designation")
        if designation:
            is_valid, error = validate_neo_designation(designation)
            if not is_valid:
                results["designation"].append(error)
        else:
            results["designation"].append("Missing designation")
        
        # Validate orbital elements
        orbital_elements = neo_data.get("orbital_elements")
        if orbital_elements:
            results["orbital_elements"] = validate_orbital_elements(orbital_elements)
        
        # Validate physical properties
        physical_properties = neo_data.get("physical_properties")
        if physical_properties:
            results["physical_properties"] = validate_physical_properties(physical_properties)
        
        # Validate close approaches
        close_approaches = neo_data.get("close_approaches", [])
        for i, approach in enumerate(close_approaches):
            approach_errors = validate_close_approach(approach)
            for error in approach_errors:
                results["close_approaches"].append(f"Approach {i+1}: {error}")
        
        # Overall data consistency
        if not orbital_elements and not physical_properties:
            results["general"].append("No orbital elements or physical properties available")
        
        return results
    
    def is_data_complete(self, neo_data: Dict[str, Any], min_completeness: float = 0.5) -> bool:
        """
        Check if NEO data meets minimum completeness requirements.
        
        Args:
            neo_data: NEO data dictionary
            min_completeness: Minimum completeness score (0-1)
            
        Returns:
            True if data is sufficiently complete
        """
        completeness_score = neo_data.get("completeness_score", 0.0)
        return completeness_score >= min_completeness
    
    def get_data_quality_score(self, neo_data: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            neo_data: NEO data dictionary
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        validation_results = self.validate_neo_data(neo_data)
        
        # Count total errors
        total_errors = sum(len(errors) for errors in validation_results.values())
        
        # Base score from completeness
        base_score = neo_data.get("completeness_score", 0.0)
        
        # Penalty for validation errors
        error_penalty = min(total_errors * 0.1, 0.5)  # Max 50% penalty
        
        quality_score = max(0.0, base_score - error_penalty)
        
        return quality_score