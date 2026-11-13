"""
Data models for aNEOS - structured representations of NEO data.

This module provides type-safe data models to replace the dictionary-based
approach in the original monolithic script.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc
import json
import logging
import copy
import math
from .epoch_evidence import NativeEpoch, decimal_value
from .orbit_evidence import OrbitModelParameter, OrbitalCovariance, AU_DAY2_TO_M_S2, finite_number
from pathlib import Path


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalise naive datetimes to UTC-aware values."""

    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)

# Physical upper bound: |a_i| > 1e-9 AU/day² is anomalously large for any known NEO.
# Typical Yarkovsky A2 range: 1e-15 to 1e-11 AU/day² (Farnocchia et al. 2013).
_NONGRAV_BOUND = 1e-9  # AU/day²


@dataclass
class NonGravitationalParameters:
    """Yarkovsky / Marsden non-gravitational acceleration parameters.

    Units: a1, a2, a3 in AU/day². Typical Yarkovsky A2: 1e-15 to 1e-11 AU/day².
    Reference: Farnocchia et al. 2013, Icarus 224, 1-13.
    """
    a1: Optional[float] = None   # Radial component (AU/day²)
    a2: Optional[float] = None   # Transverse Yarkovsky (dominant)
    a3: Optional[float] = None   # Normal component
    model: Optional[str] = None  # "marsden", "chebyshev", etc.
    epoch: Optional[datetime] = None  # Legacy UTC timestamp; never used for provider TDB epochs
    parameters: Dict[str, OrbitModelParameter] = field(default_factory=dict)
    covariance: Optional[OrbitalCovariance] = None
    solution: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = 'nongrav-evidence-v1'

    def __post_init__(self):
        _log = logging.getLogger(__name__)
        if self.schema_version != 'nongrav-evidence-v1':
            raise ValueError(f'Unsupported non-gravitational evidence schema: {self.schema_version}')
        for name, parameter in self.parameters.items():
            if name != parameter.name:
                raise ValueError('Parameter key/name mismatch')
        for label in ('A1', 'A2', 'A3'):
            record = self.parameters.get(label)
            if record is not None:
                current = getattr(self, label.lower())
                if current is not None and current != record.value:
                    raise ValueError(f'{label} scalar conflicts with parameter record')
                # Legacy scalar attributes retain native values; only publish a
                # scalar for documented units. Unknown units remain in records.
                if record.units in {'au/d^2', 'AU/day^2'}:
                    setattr(self, label.lower(), record.value)
                elif current is not None:
                    raise ValueError(f'{label} scalar has unsupported native units')
        for _name, _val in [("a1", self.a1), ("a2", self.a2), ("a3", self.a3)]:
            if _val is not None:
                _val = finite_number(_val, _name)
                setattr(self, _name, _val)
            if _val is not None and abs(_val) > _NONGRAV_BOUND:
                _log.warning(
                    f"NonGravitationalParameters.{_name}={_val:.3e} AU/day² exceeds "
                    f"physical bound ±{_NONGRAV_BOUND:.0e}; data quality suspect"
                )

    def to_dict(self):
        return {'schema_version': self.schema_version, 'a1': self.a1, 'a2': self.a2, 'a3': self.a3,
                'model': self.model, 'epoch': self.epoch.isoformat() if self.epoch else None,
                'parameters': {key: value.to_dict() for key, value in self.parameters.items()},
                'covariance': self.covariance.to_dict() if self.covariance else None,
                'solution': copy.deepcopy(self.solution)}

    @classmethod
    def from_dict(cls, data):
        epoch = data.get('epoch')
        if isinstance(epoch, str):
            epoch = _ensure_utc(datetime.fromisoformat(epoch.replace('Z', '+00:00')))
        return cls(a1=data.get('a1'), a2=data.get('a2'), a3=data.get('a3'),
                   model=data.get('model'), epoch=epoch,
                   parameters={key: OrbitModelParameter.from_dict(value)
                               for key, value in data.get('parameters', {}).items()},
                   covariance=OrbitalCovariance.from_dict(data['covariance']) if data.get('covariance') else None,
                   solution=copy.deepcopy(data.get('solution', {})),
                   schema_version=data.get('schema_version', 'nongrav-evidence-v1'))

    def evidence_report(self):
        """Describe all three axes without turning coefficients into thrust."""
        components = {}
        for name in ('A1', 'A2', 'A3'):
            parameter = self.parameters.get(name)
            legacy = getattr(self, name.lower())
            if parameter is not None:
                try:
                    si = parameter.in_si()
                except ValueError:
                    si = None
                components[name] = {
                    'status': 'available', 'native': parameter.to_dict(), 'si_coefficient': si,
                    'evidence_basis': 'fitted_parameter' if parameter.kind == 'EST' else
                                      'considered_parameter' if parameter.kind == 'CON' else 'fixed_model_parameter',
                }
            else:
                components[name] = {'status': 'not_available' if legacy is None else 'unverified_legacy_scalar',
                                    'value': legacy, 'sigma': None, 'units': 'AU/day^2',
                                    'kind': None, 'evidence_basis': 'unknown'}
        return {'schema_version': self.schema_version, 'components': components,
                'solution': copy.deepcopy(self.solution),
                'model_parameters': {key: parameter.to_dict() for key, parameter in self.parameters.items()},
                'covariance': self.covariance.to_dict() if self.covariance else None,
                'covariance_status': 'available' if self.covariance else 'not_available',
                'covariance_applies_to_display_epoch': (
                    decimal_value(self.covariance.epoch_jd_tdb) ==
                    decimal_value(self.solution['epoch_jd_tdb'])
                    if self.covariance and self.solution.get('epoch_jd_tdb') is not None else None),
                'origin_inference': 'not_assessed',
                'interpretation': 'Orbit-fit coefficients; not direct propulsion or origin evidence'}

    def acceleration_covariance_si(self):
        if self.covariance is None:
            return None
        labels = ['A1', 'A2', 'A3']
        for label in labels:
            if label not in self.parameters:
                return None  # No fabricated component or zero variance.
            self.parameters[label].in_si()  # Validate units before conversion.
        if any(label not in self.covariance.labels for label in labels):
            return None
        block = self.covariance.select(labels)
        block['data'] = [[value * AU_DAY2_TO_M_S2**2 for value in row] for row in block['data']]
        block['units'] = '(m/s^2)^2'
        block['kind'] = 'marginal_covariance_submatrix'
        block['interpretation'] = 'Coefficient covariance at the labelled solution epoch, not an instantaneous force covariance'
        return block


@dataclass
class OrbitalElements:
    """Orbital elements for a Near Earth Object."""
    
    # Identification
    designation: Optional[str] = None  # NEO designation (e.g., "2024 YR4")
    
    # Primary orbital elements
    eccentricity: Optional[float] = None
    inclination: Optional[float] = None  # degrees
    semi_major_axis: Optional[float] = None  # AU
    ra_of_ascending_node: Optional[float] = None  # degrees
    arg_of_periapsis: Optional[float] = None  # degrees
    # Common alternate names
    ascending_node: Optional[float] = None  # degrees
    argument_of_perihelion: Optional[float] = None  # degrees
    mean_anomaly: Optional[float] = None  # degrees
    epoch: Optional[datetime] = None
    native_epoch: Optional[NativeEpoch] = None
    
    def __post_init__(self):
        """Normalize aliases/native evidence and validate orbital values."""
        if isinstance(self.native_epoch, dict):
            self.native_epoch = NativeEpoch.from_dict(self.native_epoch)
        if self.native_epoch is not None and not isinstance(self.native_epoch, NativeEpoch):
            raise ValueError('Invalid native orbital epoch')
        self._synchronize_aliases()
        self._validate()

    def _synchronize_aliases(self) -> None:
        """Keep legacy and alternate orbital element names in sync."""
        if self.ascending_node is not None and self.ra_of_ascending_node is None:
            self.ra_of_ascending_node = self.ascending_node
        elif self.ascending_node is None and self.ra_of_ascending_node is not None:
            self.ascending_node = self.ra_of_ascending_node

        if self.argument_of_perihelion is not None and self.arg_of_periapsis is None:
            self.arg_of_periapsis = self.argument_of_perihelion
        elif self.argument_of_perihelion is None and self.arg_of_periapsis is not None:
            self.argument_of_perihelion = self.arg_of_periapsis
    
    def _validate(self) -> None:
        """Validate orbital element values against physical constraints."""
        errors = []
        
        if self.eccentricity is not None:
            if self.eccentricity < 0:
                errors.append(f"Eccentricity {self.eccentricity} must be non-negative")

        if self.inclination is not None:
            if not (0 <= self.inclination <= 180):
                errors.append(f"Inclination {self.inclination} outside valid range [0, 180] degrees")

        if self.semi_major_axis is not None:
            # Hyperbolic orbits (e >= 1) have negative semi-major axis — physically valid.
            # Reject negative a only for elliptical orbits (e < 1 or e unknown).
            is_hyperbolic = self.eccentricity is not None and self.eccentricity >= 1
            if self.semi_major_axis <= 0 and not is_hyperbolic:
                errors.append(f"Semi-major axis {self.semi_major_axis} must be positive for elliptical orbits")
        
        if errors:
            raise ValueError("Orbital elements validation failed: " + "; ".join(errors))
    
    def is_complete(self) -> bool:
        """Check if all essential orbital elements are present."""
        essential = [
            self.eccentricity, self.inclination, self.semi_major_axis,
            self.ascending_node, self.argument_of_perihelion, self.mean_anomaly
        ]
        return all(elem is not None for elem in essential)
    
    def completeness_score(self) -> float:
        """Calculate completeness score (0-1) based on available data."""
        all_fields = [
            self.eccentricity, self.inclination, self.semi_major_axis,
            self.ascending_node, self.argument_of_perihelion, self.mean_anomaly,
            self.native_epoch if self.native_epoch is not None else self.epoch
        ]
        present_count = sum(1 for field in all_fields if field is not None)
        return present_count / len(all_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        result = {
            "eccentricity": self.eccentricity,
            "inclination": self.inclination,
            "semi_major_axis": self.semi_major_axis,
            "ra_of_ascending_node": self.ra_of_ascending_node,
            "ascending_node": self.ascending_node,
            "arg_of_periapsis": self.arg_of_periapsis,
            "argument_of_perihelion": self.argument_of_perihelion,
            "mean_anomaly": self.mean_anomaly,
            "epoch": self.epoch.isoformat() if self.epoch else None,
        }
        if self.native_epoch is not None:
            result['native_epoch'] = self.native_epoch.to_dict()
        return result

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
            ra_of_ascending_node=data.get("ra_of_ascending_node") if data.get("ra_of_ascending_node") is not None else data.get("ascending_node"),
            arg_of_periapsis=data.get("arg_of_periapsis") if data.get("arg_of_periapsis") is not None else data.get("argument_of_perihelion"),
            ascending_node=data.get("ascending_node"),
            argument_of_perihelion=data.get("argument_of_perihelion"),
            mean_anomaly=data.get("mean_anomaly"),
            epoch=epoch,
            native_epoch=NativeEpoch.from_dict(data["native_epoch"]) if data.get("native_epoch") is not None else None,
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
    
    epoch_jd_tdb: Optional[float] = None
    time_scale: Optional[str] = None
    native_calendar: Optional[str] = None
    body: Optional[str] = None
    evidence_metadata: Dict[str, Any] = field(default_factory=dict)

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
            "epoch_jd_tdb": self.epoch_jd_tdb,
            "time_scale": self.time_scale,
            "native_calendar": self.native_calendar,
            "body": self.body,
            "evidence_metadata": copy.deepcopy(self.evidence_metadata),
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
            subpoint=data.get("subpoint"), epoch_jd_tdb=data.get("epoch_jd_tdb"),
            time_scale=data.get("time_scale"), native_calendar=data.get("native_calendar"), body=data.get("body"),
            evidence_metadata=copy.deepcopy(data.get("evidence_metadata", {}))
        )

@dataclass
class NEOData:
    """Complete Near Earth Object data structure."""

    designation: str
    orbital_elements: Optional[OrbitalElements] = None
    physical_properties: Optional["PhysicalProperties"] = None  # Canonical physical data location
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
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    fetched_at: Optional[datetime] = None  # Timestamp when data was fetched from source
    nongrav: Optional["NonGravitationalParameters"] = None
    evidence_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.orbital_elements:
            self.completeness = self.orbital_elements.completeness_score()
        
        from .physical_evidence import verify_neo_physical
        verify_neo_physical(self)
        from .epoch_evidence import verify_neo_epoch
        verify_neo_epoch(self)

        # Predicted encounters are not observations of the fitted orbit.
        # Only source first_obs/last_obs may establish the observation arc.

    def add_close_approach(self, approach: Union[CloseApproach, Dict[str, Any]]) -> None:
        """Add a close approach event."""
        if isinstance(approach, dict):
            approach = CloseApproach.from_dict(approach)
        
        self.close_approaches.append(approach)
        self.updated_at = _utcnow()
        
    def set_orbital_elements(self, elements: Union[OrbitalElements, Dict[str, Any]]) -> None:
        """Set orbital elements."""
        if isinstance(elements, dict):
            elements = OrbitalElements.from_dict(elements)
        
        self.orbital_elements = elements
        self.completeness = elements.completeness_score()
        self.updated_at = _utcnow()
    
    def update_anomaly_analysis(self, raw_score: float, dynamic_score: float, 
                              category: str, components: Dict[str, float]) -> None:
        """Update anomaly analysis results."""
        self.raw_anomaly_score = raw_score
        self.dynamic_anomaly_score = dynamic_score
        self.anomaly_category = category
        self.score_components = components.copy()
        self.updated_at = _utcnow()
    
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
            "evidence_metadata": self.evidence_metadata,
            "completeness": self.completeness,
            "raw_TAS": self.raw_anomaly_score,
            "dynamic_TAS": self.dynamic_anomaly_score,
            "dynamic_category": self.anomaly_category,
            "score_components": self.score_components,
            "first_observation": self.first_observation.isoformat() if self.first_observation else None,
            "last_observation": self.last_observation.isoformat() if self.last_observation else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "physical_properties": copy.deepcopy(self.physical_properties.to_dict()) if self.physical_properties else None,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "nongrav": self.nongrav.to_dict() if self.nongrav else None,
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
        
        # Reconstruct PhysicalProperties from cache
        pp_data = data.get("physical_properties")
        physical_properties = None
        if pp_data:
            physical_properties = PhysicalProperties.from_dict(copy.deepcopy(pp_data))

        ng_data = data.get("nongrav")
        nongrav = None
        if ng_data:
            nongrav = NonGravitationalParameters.from_dict(ng_data)

        return cls(
            designation=data.get("designation", ""),
            orbital_elements=orbital_elements,
            physical_properties=physical_properties,
            close_approaches=close_approaches,
            sources_used=data.get("sources_used", []),
            source_contributions=data.get("source_contributions", {}),
            evidence_metadata=copy.deepcopy(data.get("evidence_metadata", {})),
            completeness=data.get("completeness", 0.0),
            raw_anomaly_score=data.get("raw_TAS"),
            dynamic_anomaly_score=data.get("dynamic_TAS"),
            anomaly_category=data.get("dynamic_category"),
            score_components=data.get("score_components", {}),
            first_observation=_ensure_utc(parse_datetime(data.get("first_observation"))),
            last_observation=_ensure_utc(parse_datetime(data.get("last_observation"))),
            created_at=_ensure_utc(parse_datetime(data.get("created_at")) or _utcnow()),
            updated_at=_ensure_utc(parse_datetime(data.get("updated_at")) or _utcnow()),
            fetched_at=_ensure_utc(parse_datetime(data.get("fetched_at"))),
            nongrav=nongrav,
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

@dataclass
class PhysicalProperties:
    """Physical properties of a Near Earth Object."""
    
    # Size and mass properties
    diameter_km: Optional[float] = None
    diameter_uncertainty: Optional[float] = None
    mass_kg: Optional[float] = None
    density_g_cm3: Optional[float] = None
    
    # Optical properties
    absolute_magnitude_h: Optional[float] = None
    albedo: Optional[float] = None
    albedo_uncertainty: Optional[float] = None
    
    # Rotational properties
    rotation_period_hours: Optional[float] = None
    rotation_period_uncertainty: Optional[float] = None
    pole_ecliptic_lat: Optional[float] = None  # degrees
    pole_ecliptic_lon: Optional[float] = None  # degrees
    
    # Spectral properties
    spectral_type: Optional[str] = None
    spectral_class: Optional[str] = None
    color_indices: Dict[str, float] = field(default_factory=dict)
    
    # Thermal properties
    thermal_inertia: Optional[float] = None  # SI units
    emissivity: Optional[float] = None
    
    # Data quality indicators
    data_quality_flag: Optional[str] = None
    measurement_method: Optional[str] = None
    reference_source: Optional[str] = None
    
    def __post_init__(self):
        """Validate physical properties after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate physical property values."""
        errors = []
        
        if self.diameter_km is not None:
            if self.diameter_km <= 0:
                errors.append(f"Diameter {self.diameter_km} km must be positive")
        
        if self.mass_kg is not None:
            if self.mass_kg <= 0:
                errors.append(f"Mass {self.mass_kg} kg must be positive")
        
        if self.density_g_cm3 is not None:
            if self.density_g_cm3 <= 0:
                errors.append(f"Density {self.density_g_cm3} g/cm³ must be positive")
        
        if self.albedo is not None:
            if not math.isfinite(self.albedo) or self.albedo < 0:
                errors.append(f"Albedo {self.albedo} must be finite and nonnegative")
        
        if self.rotation_period_hours is not None:
            if self.rotation_period_hours <= 0:
                errors.append(f"Rotation period {self.rotation_period_hours} hours must be positive")
        
        if self.emissivity is not None:
            if not (0 <= self.emissivity <= 1):
                errors.append(f"Emissivity {self.emissivity} outside valid range [0, 1]")
        
        if errors:
            raise ValueError("Physical properties validation failed: " + "; ".join(errors))
    
    def completeness_score(self) -> float:
        """Calculate completeness score (0-1) based on available physical data."""
        all_fields = [
            self.diameter_km, self.mass_kg, self.absolute_magnitude_h, self.albedo,
            self.rotation_period_hours, self.spectral_type, self.thermal_inertia
        ]
        present_count = sum(1 for field in all_fields if field is not None)
        return present_count / len(all_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "diameter_km": self.diameter_km,
            "diameter_uncertainty": self.diameter_uncertainty,
            "mass_kg": self.mass_kg,
            "density_g_cm3": self.density_g_cm3,
            "absolute_magnitude_h": self.absolute_magnitude_h,
            "albedo": self.albedo,
            "albedo_uncertainty": self.albedo_uncertainty,
            "rotation_period_hours": self.rotation_period_hours,
            "rotation_period_uncertainty": self.rotation_period_uncertainty,
            "pole_ecliptic_lat": self.pole_ecliptic_lat,
            "pole_ecliptic_lon": self.pole_ecliptic_lon,
            "spectral_type": self.spectral_type,
            "spectral_class": self.spectral_class,
            "color_indices": self.color_indices,
            "thermal_inertia": self.thermal_inertia,
            "emissivity": self.emissivity,
            "data_quality_flag": self.data_quality_flag,
            "measurement_method": self.measurement_method,
            "reference_source": self.reference_source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicalProperties':
        """Create from dictionary data."""
        return cls(
            diameter_km=data.get("diameter_km"),
            diameter_uncertainty=data.get("diameter_uncertainty"),
            mass_kg=data.get("mass_kg"),
            density_g_cm3=data.get("density_g_cm3"),
            absolute_magnitude_h=data.get("absolute_magnitude_h"),
            albedo=data.get("albedo"),
            albedo_uncertainty=data.get("albedo_uncertainty"),
            rotation_period_hours=data.get("rotation_period_hours"),
            rotation_period_uncertainty=data.get("rotation_period_uncertainty"),
            pole_ecliptic_lat=data.get("pole_ecliptic_lat"),
            pole_ecliptic_lon=data.get("pole_ecliptic_lon"),
            spectral_type=data.get("spectral_type"),
            spectral_class=data.get("spectral_class"),
            color_indices=data.get("color_indices", {}),
            thermal_inertia=data.get("thermal_inertia"),
            emissivity=data.get("emissivity"),
            data_quality_flag=data.get("data_quality_flag"),
            measurement_method=data.get("measurement_method"),
            reference_source=data.get("reference_source")
        )

@dataclass
class ImpactAssessment:
    """
    Legacy impact record container, not evidence of a fitted scientific model.

    Empty fields remain unavailable/not-assessed. Scoped source evidence uses the
    replayable Sentry report contract, not this record's presence or defaults.
    """
    
    designation: str
    calculation_method: str
    last_updated: datetime = field(default_factory=_utcnow)
    
    # Core impact metrics
    collision_probability: Optional[float] = None  # Overall probability of Earth impact
    collision_probability_per_year: Optional[float] = None  # Annual impact rate
    time_to_impact_years: Optional[float] = None  # Most probable impact time
    
    # Uncertainty and confidence
    probability_uncertainty: Optional[Tuple[float, float]] = None  # (lower, upper bounds)
    calculation_confidence: Optional[float] = None  # 0-1, quality of calculation
    data_arc_years: Optional[float] = None  # Observation arc length
    
    # Physical impact assessment
    impact_energy_mt: Optional[float] = None  # Megatons TNT equivalent
    impact_velocity_km_s: Optional[float] = None  # Impact velocity
    crater_diameter_km: Optional[float] = None  # Expected crater size
    damage_radius_km: Optional[float] = None  # Damage radius estimate
    
    # Spatial distribution
    impact_latitude_distribution: Dict[str, float] = field(default_factory=dict)
    most_probable_impact_region: Optional[str] = None
    
    # Temporal evolution
    impact_probability_by_decade: Dict[str, float] = field(default_factory=dict)
    peak_risk_period: Optional[Tuple[int, int]] = None  # (start_year, end_year)
    
    # Special considerations
    keyhole_passages: List[Dict[str, Any]] = field(default_factory=list)
    artificial_object_considerations: Optional[Dict[str, Any]] = None
    
    # Scientific rationale
    primary_risk_factors: List[str] = field(default_factory=list)
    calculation_assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Risk classification
    risk_level: str = "not_assessed"  # negligible, very_low, low, moderate, high, extreme
    comparative_risk: Optional[str] = None
    
    # Moon impact assessment
    moon_collision_probability: Optional[float] = None  # Probability of Moon impact
    moon_impact_energy_mt: Optional[float] = None  # Moon impact energy
    earth_vs_moon_impact_ratio: Optional[float] = None  # Relative likelihood
    moon_impact_effects: Optional[Dict[str, Any]] = None  # Moon impact consequences
    
    def __post_init__(self):
        """Validate impact assessment data."""
        if self.collision_probability is not None and not (0 <= self.collision_probability <= 1):
            raise ValueError(f"Collision probability {self.collision_probability} must be in [0,1]")
        
        if self.calculation_confidence is not None and not (0 <= self.calculation_confidence <= 1):
            raise ValueError(f"Calculation confidence {self.calculation_confidence} must be in [0,1]")
        
        # Ensure last_updated is UTC-aware
        self.last_updated = _ensure_utc(self.last_updated)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "designation": self.designation,
            "calculation_method": self.calculation_method,
            "last_updated": self.last_updated.isoformat(),
            
            # Core metrics
            "collision_probability": self.collision_probability,
            "collision_probability_per_year": self.collision_probability_per_year,
            "time_to_impact_years": self.time_to_impact_years,
            
            # Uncertainty
            "probability_uncertainty": list(self.probability_uncertainty) if self.probability_uncertainty is not None else None,
            "calculation_confidence": self.calculation_confidence,
            "data_arc_years": self.data_arc_years,
            
            # Physical assessment
            "impact_energy_mt": self.impact_energy_mt,
            "impact_velocity_km_s": self.impact_velocity_km_s,
            "crater_diameter_km": self.crater_diameter_km,
            "damage_radius_km": self.damage_radius_km,
            
            # Spatial/temporal
            "impact_latitude_distribution": self.impact_latitude_distribution,
            "most_probable_impact_region": self.most_probable_impact_region,
            "impact_probability_by_decade": self.impact_probability_by_decade,
            "peak_risk_period": list(self.peak_risk_period) if self.peak_risk_period else None,
            
            # Special analysis
            "keyhole_passages": self.keyhole_passages,
            "artificial_object_considerations": self.artificial_object_considerations,
            
            # Scientific rationale
            "primary_risk_factors": self.primary_risk_factors,
            "calculation_assumptions": self.calculation_assumptions,
            "limitations": self.limitations,
            
            # Risk classification
            "risk_level": self.risk_level,
            "comparative_risk": self.comparative_risk,
            
            # Moon impact assessment
            "moon_collision_probability": self.moon_collision_probability,
            "moon_impact_energy_mt": self.moon_impact_energy_mt,
            "earth_vs_moon_impact_ratio": self.earth_vs_moon_impact_ratio,
            "moon_impact_effects": self.moon_impact_effects
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImpactAssessment':
        """Create from dictionary data."""
        
        # Handle datetime conversion
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
        elif last_updated is None:
            last_updated = _utcnow()
        
        # Handle tuple conversion
        uncertainty = data.get("probability_uncertainty")
        if isinstance(uncertainty, list):
            uncertainty = tuple(uncertainty)
        
        peak_risk = data.get("peak_risk_period")
        if isinstance(peak_risk, list) and len(peak_risk) == 2:
            peak_risk = tuple(peak_risk)
        
        return cls(
            designation=data.get("designation", "Unknown"),
            calculation_method=data.get("calculation_method", "unknown"),
            last_updated=last_updated,
            
            collision_probability=data.get("collision_probability"),
            collision_probability_per_year=data.get("collision_probability_per_year"),
            time_to_impact_years=data.get("time_to_impact_years"),
            
            probability_uncertainty=uncertainty,
            calculation_confidence=data.get("calculation_confidence"),
            data_arc_years=data.get("data_arc_years"),
            
            impact_energy_mt=data.get("impact_energy_mt"),
            impact_velocity_km_s=data.get("impact_velocity_km_s"),
            crater_diameter_km=data.get("crater_diameter_km"),
            damage_radius_km=data.get("damage_radius_km"),
            
            impact_latitude_distribution=data.get("impact_latitude_distribution", {}),
            most_probable_impact_region=data.get("most_probable_impact_region"),
            impact_probability_by_decade=data.get("impact_probability_by_decade", {}),
            peak_risk_period=peak_risk,
            
            keyhole_passages=data.get("keyhole_passages", []),
            artificial_object_considerations=data.get("artificial_object_considerations"),
            
            primary_risk_factors=data.get("primary_risk_factors", []),
            calculation_assumptions=data.get("calculation_assumptions", []),
            limitations=data.get("limitations", []),
            
            risk_level=data.get("risk_level", "not_assessed"),
            comparative_risk=data.get("comparative_risk"),
            
            moon_collision_probability=data.get("moon_collision_probability"),
            moon_impact_energy_mt=data.get("moon_impact_energy_mt"),
            earth_vs_moon_impact_ratio=data.get("earth_vs_moon_impact_ratio"),
            moon_impact_effects=data.get("moon_impact_effects")
        )

@dataclass
class ImpactScenario:
    """
    Specific impact scenario analysis for detailed risk assessment.
    
    This model represents a particular impact scenario with specific
    conditions and consequences.
    """
    
    impact_date: datetime
    probability: float  # Probability of this specific scenario
    impact_location: Tuple[float, float]  # (latitude, longitude) in degrees
    impact_velocity: float  # km/s
    impact_angle: float  # degrees from horizontal
    energy_mt: float  # Megatons TNT equivalent
    scenario_type: str  # direct, resonant_return, keyhole_passage
    
    # Consequence assessment
    crater_size_km: Optional[float] = None
    damage_assessment: Optional[Dict[str, Any]] = None
    affected_population: Optional[int] = None
    economic_impact_usd: Optional[float] = None
    
    # Environmental effects
    dust_injection_atmosphere: Optional[float] = None  # Tons of material
    climate_impact_duration_years: Optional[float] = None
    tsunami_risk: Optional[bool] = None  # If ocean impact
    
    def __post_init__(self):
        """Validate impact scenario data."""
        if not (0 <= self.probability <= 1):
            raise ValueError(f"Scenario probability {self.probability} must be in [0,1]")
        
        lat, lon = self.impact_location
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} must be in [-90, 90]")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude {lon} must be in [-180, 180]")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "impact_date": self.impact_date.isoformat(),
            "probability": self.probability,
            "impact_location": list(self.impact_location),
            "impact_velocity": self.impact_velocity,
            "impact_angle": self.impact_angle,
            "energy_mt": self.energy_mt,
            "scenario_type": self.scenario_type,
            "crater_size_km": self.crater_size_km,
            "damage_assessment": self.damage_assessment,
            "affected_population": self.affected_population,
            "economic_impact_usd": self.economic_impact_usd,
            "dust_injection_atmosphere": self.dust_injection_atmosphere,
            "climate_impact_duration_years": self.climate_impact_duration_years,
            "tsunami_risk": self.tsunami_risk
        }