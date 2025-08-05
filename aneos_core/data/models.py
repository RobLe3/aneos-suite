"""
Data models for aNEOS - structured representations of NEO data.

This module provides type-safe data models to replace the dictionary-based
approach in the original monolithic script.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from pathlib import Path

@dataclass
class OrbitalElements:
    """Orbital elements for a Near Earth Object."""
    
    # Primary orbital elements
    eccentricity: Optional[float] = None
    inclination: Optional[float] = None  # degrees
    semi_major_axis: Optional[float] = None  # AU
    ra_of_ascending_node: Optional[float] = None  # degrees
    arg_of_periapsis: Optional[float] = None  # degrees
    mean_anomaly: Optional[float] = None  # degrees
    epoch: Optional[datetime] = None
    
    # Physical parameters
    diameter: Optional[float] = None  # km
    albedo: Optional[float] = None
    rot_per: Optional[float] = None  # rotation period in hours
    spectral_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate orbital elements after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate orbital element values against physical constraints."""
        errors = []
        
        if self.eccentricity is not None:
            if not (0 <= self.eccentricity < 1):
                errors.append(f"Eccentricity {self.eccentricity} outside valid range [0, 1)")
        
        if self.inclination is not None:
            if not (0 <= self.inclination <= 180):
                errors.append(f"Inclination {self.inclination} outside valid range [0, 180] degrees")
        
        if self.semi_major_axis is not None:
            if self.semi_major_axis <= 0:
                errors.append(f"Semi-major axis {self.semi_major_axis} must be positive")
        
        if self.albedo is not None:
            if not (0 <= self.albedo <= 1):
                errors.append(f"Albedo {self.albedo} outside valid range [0, 1]")
        
        if self.diameter is not None:
            if self.diameter <= 0:
                errors.append(f"Diameter {self.diameter} must be positive")
        
        if errors:
            raise ValueError("Orbital elements validation failed: " + "; ".join(errors))
    
    def is_complete(self) -> bool:
        """Check if all essential orbital elements are present."""
        essential = [
            self.eccentricity, self.inclination, self.semi_major_axis,
            self.ra_of_ascending_node, self.arg_of_periapsis, self.mean_anomaly
        ]
        return all(elem is not None for elem in essential)
    
    def completeness_score(self) -> float:
        """Calculate completeness score (0-1) based on available data."""
        all_fields = [
            self.eccentricity, self.inclination, self.semi_major_axis,
            self.ra_of_ascending_node, self.arg_of_periapsis, self.mean_anomaly,
            self.epoch, self.diameter, self.albedo
        ]
        present_count = sum(1 for field in all_fields if field is not None)
        return present_count / len(all_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "eccentricity": self.eccentricity,
            "inclination": self.inclination,
            "semi_major_axis": self.semi_major_axis,
            "ra_of_ascending_node": self.ra_of_ascending_node,
            "arg_of_periapsis": self.arg_of_periapsis,
            "mean_anomaly": self.mean_anomaly,
            "epoch": self.epoch.isoformat() if self.epoch else None,
            "diameter": self.diameter,
            "albedo": self.albedo,
            "rot_per": self.rot_per,
            "spectral_type": self.spectral_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrbitalElements':
        """Create from dictionary data."""
        # Handle epoch conversion
        epoch_data = data.get("epoch")
        epoch = None
        if epoch_data:
            if isinstance(epoch_data, str):
                try:
                    epoch = datetime.fromisoformat(epoch_data.replace('Z', '+00:00'))
                except ValueError:
                    # Try other common formats
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            epoch = datetime.strptime(epoch_data, fmt)
                            break
                        except ValueError:
                            continue
            elif isinstance(epoch_data, datetime):
                epoch = epoch_data
        
        return cls(
            eccentricity=data.get("eccentricity"),
            inclination=data.get("inclination"),
            semi_major_axis=data.get("semi_major_axis"),
            ra_of_ascending_node=data.get("ra_of_ascending_node"),
            arg_of_periapsis=data.get("arg_of_periapsis"),
            mean_anomaly=data.get("mean_anomaly"),
            epoch=epoch,
            diameter=data.get("diameter"),
            albedo=data.get("albedo"),
            rot_per=data.get("rot_per"),
            spectral_type=data.get("spectral_type")
        )

@dataclass
class CloseApproach:
    """Data for a close approach event."""
    
    designation: str
    orbit_id: Optional[str] = None
    close_approach_date: Optional[datetime] = None
    distance_au: Optional[float] = None  # AU
    distance_km: Optional[float] = None  # km
    relative_velocity_km_s: Optional[float] = None  # km/s
    infinity_velocity_km_s: Optional[float] = None  # km/s
    subpoint: Optional[tuple] = None  # (lat, lon) in degrees
    
    def __post_init__(self):
        """Calculate derived values."""
        # Convert distance units if only one is provided
        if self.distance_au is not None and self.distance_km is None:
            self.distance_km = self.distance_au * 149597870.7
        elif self.distance_km is not None and self.distance_au is None:
            self.distance_au = self.distance_km / 149597870.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "designation": self.designation,
            "orbit_id": self.orbit_id,
            "cd": self.close_approach_date.isoformat() if self.close_approach_date else None,
            "dist": self.distance_au,
            "distance_km": self.distance_km,
            "v_rel": self.relative_velocity_km_s,
            "v_inf": self.infinity_velocity_km_s,
            "subpoint": self.subpoint
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CloseApproach':
        """Create from dictionary data."""
        # Handle date conversion
        cd_data = data.get("cd")
        cd = None
        if cd_data:
            if isinstance(cd_data, str):
                try:
                    cd = datetime.fromisoformat(cd_data.replace('Z', '+00:00'))
                except ValueError:
                    # Try legacy format
                    try:
                        cd = datetime.strptime(cd_data, "%Y-%b-%d %H:%M")
                    except ValueError:
                        pass
            elif isinstance(cd_data, datetime):
                cd = cd_data
        
        return cls(
            designation=data.get("designation", ""),
            orbit_id=data.get("orbit_id"),
            close_approach_date=cd,
            distance_au=data.get("dist"),
            distance_km=data.get("distance_km"),
            relative_velocity_km_s=data.get("v_rel"),
            infinity_velocity_km_s=data.get("v_inf"),
            subpoint=data.get("subpoint")
        )

@dataclass
class NEOData:
    """Complete Near Earth Object data structure."""
    
    designation: str
    orbital_elements: Optional[OrbitalElements] = None
    close_approaches: List[CloseApproach] = field(default_factory=list)
    
    # Data source information
    sources_used: List[str] = field(default_factory=list)
    source_contributions: Dict[str, float] = field(default_factory=dict)
    completeness: float = 0.0
    
    # Analysis results
    raw_anomaly_score: Optional[float] = None
    dynamic_anomaly_score: Optional[float] = None
    anomaly_category: Optional[str] = None
    score_components: Dict[str, float] = field(default_factory=dict)
    
    # Observation period
    first_observation: Optional[datetime] = None
    last_observation: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.orbital_elements:
            self.completeness = self.orbital_elements.completeness_score()
        
        # Calculate observation period from close approaches
        if self.close_approaches and not self.first_observation:
            dates = [ca.close_approach_date for ca in self.close_approaches if ca.close_approach_date]
            if dates:
                self.first_observation = min(dates)
                self.last_observation = max(dates)
    
    def add_close_approach(self, approach: Union[CloseApproach, Dict[str, Any]]) -> None:
        """Add a close approach event."""
        if isinstance(approach, dict):
            approach = CloseApproach.from_dict(approach)
        
        self.close_approaches.append(approach)
        self.updated_at = datetime.utcnow()
        
        # Update observation period
        if approach.close_approach_date:
            if not self.first_observation or approach.close_approach_date < self.first_observation:
                self.first_observation = approach.close_approach_date
            if not self.last_observation or approach.close_approach_date > self.last_observation:
                self.last_observation = approach.close_approach_date
    
    def set_orbital_elements(self, elements: Union[OrbitalElements, Dict[str, Any]]) -> None:
        """Set orbital elements."""
        if isinstance(elements, dict):
            elements = OrbitalElements.from_dict(elements)
        
        self.orbital_elements = elements
        self.completeness = elements.completeness_score()
        self.updated_at = datetime.utcnow()
    
    def update_anomaly_analysis(self, raw_score: float, dynamic_score: float, 
                              category: str, components: Dict[str, float]) -> None:
        """Update anomaly analysis results."""
        self.raw_anomaly_score = raw_score
        self.dynamic_anomaly_score = dynamic_score
        self.anomaly_category = category
        self.score_components = components.copy()
        self.updated_at = datetime.utcnow()
    
    def is_highly_anomalous(self, threshold: float = 2.0) -> bool:
        """Check if NEO is highly anomalous based on dynamic score."""
        return (self.dynamic_anomaly_score is not None and 
                self.dynamic_anomaly_score >= threshold)
    
    def get_observation_span_days(self) -> Optional[int]:
        """Get observation span in days."""
        if self.first_observation and self.last_observation:
            return (self.last_observation - self.first_observation).days
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "designation": self.designation,
            "orbital_elements": self.orbital_elements.to_dict() if self.orbital_elements else None,
            "close_approaches": [ca.to_dict() for ca in self.close_approaches],
            "sources_used": self.sources_used,
            "source_contributions": self.source_contributions,
            "completeness": self.completeness,
            "raw_TAS": self.raw_anomaly_score,
            "dynamic_TAS": self.dynamic_anomaly_score,
            "dynamic_category": self.anomaly_category,
            "score_components": self.score_components,
            "first_observation": self.first_observation.isoformat() if self.first_observation else None,
            "last_observation": self.last_observation.isoformat() if self.last_observation else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NEOData':
        """Create from dictionary data."""
        # Handle orbital elements
        orbital_data = data.get("orbital_elements")
        orbital_elements = None
        if orbital_data:
            orbital_elements = OrbitalElements.from_dict(orbital_data)
        
        # Handle close approaches
        ca_data = data.get("close_approaches", [])
        close_approaches = []
        for ca in ca_data:
            if isinstance(ca, dict):
                close_approaches.append(CloseApproach.from_dict(ca))
        
        # Handle datetime fields
        def parse_datetime(dt_str):
            if dt_str:
                try:
                    return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    return None
            return None
        
        return cls(
            designation=data.get("designation", ""),
            orbital_elements=orbital_elements,
            close_approaches=close_approaches,
            sources_used=data.get("sources_used", []),
            source_contributions=data.get("source_contributions", {}),
            completeness=data.get("completeness", 0.0),
            raw_anomaly_score=data.get("raw_TAS"),
            dynamic_anomaly_score=data.get("dynamic_TAS"),
            anomaly_category=data.get("dynamic_category"),
            score_components=data.get("score_components", {}),
            first_observation=parse_datetime(data.get("first_observation")),
            last_observation=parse_datetime(data.get("last_observation")),
            created_at=parse_datetime(data.get("created_at")) or datetime.utcnow(),
            updated_at=parse_datetime(data.get("updated_at")) or datetime.utcnow()
        )
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save NEO data to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'NEOData':
        """Load NEO data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

@dataclass
class AnalysisResult:
    """Results from NEO analysis pipeline."""
    
    total_neos_analyzed: int
    high_anomaly_neos: List[NEOData]
    category_counts: Dict[str, int]
    processing_time_seconds: float
    source_statistics: Dict[str, Dict[str, Any]]
    
    # Statistical summary
    mean_raw_score: float = 0.0
    mean_dynamic_score: float = 0.0
    max_dynamic_score: float = 0.0
    min_dynamic_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "total_neos_analyzed": self.total_neos_analyzed,
            "high_anomaly_neos": [neo.to_dict() for neo in self.high_anomaly_neos],
            "category_counts": self.category_counts,
            "processing_time_seconds": self.processing_time_seconds,
            "source_statistics": self.source_statistics,
            "mean_raw_score": self.mean_raw_score,
            "mean_dynamic_score": self.mean_dynamic_score,
            "max_dynamic_score": self.max_dynamic_score,
            "min_dynamic_score": self.min_dynamic_score
        }