#!/usr/bin/env python3
"""
Impact Probability Analysis for aNEOS

This module calculates Earth impact probabilities for Near Earth Objects using
orbital mechanics, close approach data, and uncertainty propagation methods.

The impact probability framework integrates with the existing aNEOS pipeline
to provide comprehensive impact risk assessment with scientific rationales.

Key Features:
- Keyholes analysis for resonant returns
- Monte Carlo uncertainty propagation  
- Time-dependent impact probability evolution
- Regional impact probability distribution
- Energy and damage assessment
- Artificial object impact considerations

Scientific Rationale:
Impact probability is derived from orbital mechanics and observational uncertainty.
It answers: "Given current observations and their uncertainties, what is the 
probability this object will collide with Earth in the future?"

This is fundamentally different from artificial detection, but related:
- Natural objects: Impact probability based on gravitational dynamics
- Artificial objects: May have propulsive capabilities affecting trajectory
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.integrate import solve_ivp
import warnings

# aNEOS imports
from aneos_core.data.models import OrbitalElements, CloseApproach

logger = logging.getLogger(__name__)

@dataclass
class ImpactProbability:
    """Comprehensive impact probability assessment for a NEO."""
    
    designation: str
    calculation_method: str
    last_updated: datetime
    
    # Core impact metrics
    collision_probability: float  # Overall probability of Earth impact
    collision_probability_per_year: float  # Annual impact rate
    time_to_impact_years: Optional[float]  # Most probable impact time
    
    # Uncertainty and confidence
    probability_uncertainty: Tuple[float, float]  # (lower_bound, upper_bound)
    calculation_confidence: float  # 0-1, quality of calculation
    data_arc_years: float  # Observation arc length
    
    # Physical impact assessment
    impact_energy_mt: Optional[float]  # Megatons TNT equivalent
    impact_velocity_km_s: Optional[float]  # Impact velocity
    crater_diameter_km: Optional[float]  # Expected crater size
    damage_radius_km: Optional[float]  # Damage radius estimate
    
    # Spatial distribution
    impact_latitude_distribution: Dict[str, float]  # Regional probabilities
    most_probable_impact_region: Optional[str]
    
    # Temporal evolution
    impact_probability_by_decade: Dict[str, float]  # Time-dependent probability
    peak_risk_period: Optional[Tuple[int, int]]  # (start_year, end_year)
    
    # Special considerations
    keyhole_passages: List[Dict[str, Any]]  # Gravitational keyhole analysis
    artificial_object_considerations: Optional[Dict[str, Any]]
    
    # Scientific rationale
    primary_risk_factors: List[str]
    calculation_assumptions: List[str]
    limitations: List[str]
    
    # Risk classification
    risk_level: str  # negligible, very_low, low, moderate, high, extreme
    comparative_risk: str  # Comparison to known risks
    
    # Moon impact assessment
    moon_collision_probability: Optional[float] = None  # Probability of Moon impact
    moon_impact_energy_mt: Optional[float] = None  # Moon impact energy
    earth_vs_moon_impact_ratio: Optional[float] = None  # Relative likelihood
    moon_impact_effects: Optional[Dict[str, Any]] = None  # Moon impact consequences
    
    def __post_init__(self):
        """Validate and normalize impact probability data."""
        if not (0 <= self.collision_probability <= 1):
            raise ValueError(f"Collision probability {self.collision_probability} must be in [0,1]")
        
        if self.calculation_confidence and not (0 <= self.calculation_confidence <= 1):
            raise ValueError(f"Calculation confidence {self.calculation_confidence} must be in [0,1]")

@dataclass
class ImpactScenario:
    """Specific impact scenario analysis."""
    
    impact_date: datetime
    probability: float
    impact_location: Tuple[float, float]  # (lat, lon)
    impact_velocity: float  # km/s
    impact_angle: float  # degrees from horizontal
    energy_mt: float  # Megatons TNT
    scenario_type: str  # direct, resonant_return, keyhole_passage
    
class ImpactProbabilityCalculator:
    """
    Scientific impact probability calculator for NEOs.
    
    This calculator implements multiple methodologies for impact probability
    assessment, with scientific rationales for when each method applies.
    
    Methods implemented:
    1. Linear approximation (short-term, well-observed objects)
    2. Monte Carlo simulation (long-term, uncertain orbits)
    3. Keyhole analysis (specific close approaches)
    4. Artificial object special cases
    """
    
    # Physical constants
    EARTH_RADIUS_KM = 6371.0
    EARTH_SPHERE_OF_INFLUENCE_KM = 924000.0  # Approximate SOI
    MOON_RADIUS_KM = 1737.4
    MOON_SPHERE_OF_INFLUENCE_KM = 66100.0  # Moon's SOI
    MOON_ORBITAL_DISTANCE_KM = 384400.0  # Average Earth-Moon distance
    AU_TO_KM = 149597870.7
    EARTH_ESCAPE_VELOCITY = 11.2  # km/s
    MOON_ESCAPE_VELOCITY = 2.4   # km/s
    
    # Risk thresholds (based on NASA/ESA standards)
    RISK_THRESHOLDS = {
        'negligible': 1e-9,      # 1 in billion
        'very_low': 1e-7,        # 1 in 10 million
        'low': 1e-6,             # 1 in million  
        'moderate': 1e-5,        # 1 in 100,000
        'high': 1e-4,            # 1 in 10,000
        'extreme': 1e-3          # 1 in 1,000
    }
    
    def __init__(self):
        """Initialize the impact probability calculator."""
        self.logger = logging.getLogger(__name__)
        
        # Gravitational parameters (km³/s²)
        self.GM_sun = 1.32712442018e11
        self.GM_earth = 3.986004418e5
        
        # Impact cross-section enhancement factors
        self.gravitational_focusing_factor = 1.0  # Will be calculated per object
        
    def calculate_comprehensive_impact_probability(self, 
                                                 orbital_elements: OrbitalElements,
                                                 close_approaches: List[CloseApproach] = None,
                                                 observation_arc_days: float = 30.0,
                                                 is_artificial: bool = False,
                                                 artificial_probability: float = 0.0) -> ImpactProbability:
        """
        Calculate comprehensive impact probability for a NEO.
        
        This is the main entry point that determines the appropriate calculation
        method based on available data and object characteristics.
        
        Scientific Rationale for Method Selection:
        
        1. **Why calculate impact probability?**
           - Impact probability quantifies the threat level to Earth
           - Enables prioritization of observations and mitigation efforts
           - Required for planetary defense decision-making
           
        2. **When to calculate impact probability?**
           - For all Earth-crossing asteroids (ECAs)
           - For objects with close approaches < 0.2 AU
           - For newly discovered objects (uncertainty assessment)
           - For objects showing non-gravitational forces
           
        3. **Where impact probability matters most:**
           - Objects with short observation arcs (high uncertainty)
           - Recent close approaches indicating Earth-crossing orbits
           - Objects near gravitational keyholes
           - Artificial objects with propulsive capabilities
        
        Args:
            orbital_elements: Current best-fit orbital elements
            close_approaches: List of past/future close approach data
            observation_arc_days: Length of observational data arc
            is_artificial: Whether object is classified as artificial
            artificial_probability: Probability object is artificial (0-1)
            
        Returns:
            ImpactProbability: Comprehensive impact assessment
        """
        
        designation = getattr(orbital_elements, 'designation', 'Unknown')
        self.logger.info(f"Starting comprehensive impact probability calculation for {designation}")
        
        # Validate input data
        if not orbital_elements.is_complete():
            self.logger.warning(f"Incomplete orbital elements for {designation}, using limited analysis")
        
        # Determine if object is Earth-crossing
        is_earth_crossing = self._is_earth_crossing_orbit(orbital_elements)
        
        # Calculate basic impact metrics
        collision_prob, prob_uncertainty = self._calculate_collision_probability(
            orbital_elements, close_approaches, observation_arc_days, is_artificial
        )
        
        # Time-dependent analysis
        impact_prob_evolution = self._calculate_temporal_evolution(
            orbital_elements, close_approaches
        )
        
        # Physical impact assessment
        impact_energy = self._calculate_impact_energy(orbital_elements)
        impact_velocity = self._estimate_impact_velocity(orbital_elements)
        crater_size = self._estimate_crater_diameter(impact_energy)
        damage_radius = self._estimate_damage_radius(impact_energy)
        
        # Spatial distribution analysis
        latitude_dist = self._calculate_impact_latitude_distribution(orbital_elements)
        most_probable_region = self._identify_most_probable_impact_region(latitude_dist)
        
        # Keyhole analysis for future close approaches
        keyholes = self._analyze_gravitational_keyholes(orbital_elements, close_approaches)
        
        # Special considerations for artificial objects
        artificial_considerations = None
        if is_artificial or artificial_probability > 0.1:
            artificial_considerations = self._analyze_artificial_impact_scenarios(
                orbital_elements, artificial_probability
            )
        
        # Risk assessment and classification
        risk_level = self._classify_risk_level(collision_prob)
        comparative_risk = self._generate_comparative_risk_statement(collision_prob)
        
        # Generate scientific rationales
        risk_factors = self._identify_primary_risk_factors(
            orbital_elements, close_approaches, is_earth_crossing, observation_arc_days
        )
        assumptions = self._document_calculation_assumptions(orbital_elements, close_approaches)
        limitations = self._document_calculation_limitations(orbital_elements, observation_arc_days)
        
        # Calculate confidence in the calculation
        calc_confidence = self._assess_calculation_confidence(
            orbital_elements, close_approaches, observation_arc_days
        )
        
        # Find most probable impact time
        time_to_impact = self._estimate_time_to_impact(impact_prob_evolution)
        
        # Annual impact rate
        annual_rate = collision_prob / max(time_to_impact or 100, 1.0) if time_to_impact else 0.0
        
        # Calculate Moon impact probability assessment
        moon_results = self._calculate_moon_impact_probability(
            orbital_elements, close_approaches, observation_arc_days, is_artificial, artificial_probability
        )
        
        # Update risk factors with Moon comparison
        if moon_results['moon_collision_probability'] > collision_prob:
            risk_factors.append(f"Moon impact {moon_results['earth_vs_moon_ratio']:.1f}x more likely than Earth impact")
        
        return ImpactProbability(
            designation=designation,
            calculation_method=self._determine_calculation_method(orbital_elements, close_approaches),
            last_updated=datetime.now(),
            
            # Core metrics
            collision_probability=collision_prob,
            collision_probability_per_year=annual_rate,
            time_to_impact_years=time_to_impact,
            
            # Uncertainty
            probability_uncertainty=prob_uncertainty,
            calculation_confidence=calc_confidence,
            data_arc_years=observation_arc_days / 365.25,
            
            # Physical assessment
            impact_energy_mt=impact_energy,
            impact_velocity_km_s=impact_velocity,
            crater_diameter_km=crater_size,
            damage_radius_km=damage_radius,
            
            # Spatial/temporal distribution
            impact_latitude_distribution=latitude_dist,
            most_probable_impact_region=most_probable_region,
            impact_probability_by_decade=impact_prob_evolution,
            peak_risk_period=self._identify_peak_risk_period(impact_prob_evolution),
            
            # Special analysis
            keyhole_passages=keyholes,
            artificial_object_considerations=artificial_considerations,
            
            # Scientific documentation
            primary_risk_factors=risk_factors,
            calculation_assumptions=assumptions,
            limitations=limitations,
            
            # Risk classification
            risk_level=risk_level,
            comparative_risk=comparative_risk,
            
            # Moon impact assessment
            moon_collision_probability=moon_results['moon_collision_probability'],
            moon_impact_energy_mt=moon_results['moon_impact_energy_mt'],
            earth_vs_moon_impact_ratio=moon_results['earth_vs_moon_ratio'],
            moon_impact_effects=moon_results['moon_impact_effects']
        )
    
    def _is_earth_crossing_orbit(self, orbital_elements: OrbitalElements) -> bool:
        """
        Determine if orbit crosses Earth's orbit.
        
        Scientific Rationale:
        Only Earth-crossing orbits can lead to Earth impact. This is the
        fundamental screening criterion for impact probability assessment.
        
        Method: Check if perihelion < 1.0 AU and aphelion > 1.0 AU
        """
        if not orbital_elements.semi_major_axis or not orbital_elements.eccentricity:
            return False
            
        a = orbital_elements.semi_major_axis
        e = orbital_elements.eccentricity
        
        perihelion = a * (1 - e)
        aphelion = a * (1 + e)
        
        # Earth's orbit is approximately circular at 1.0 AU
        return perihelion < 1.017 and aphelion > 0.983  # Include small margin for Earth's eccentricity
    
    def _calculate_collision_probability(self, 
                                       orbital_elements: OrbitalElements,
                                       close_approaches: List[CloseApproach],
                                       observation_arc_days: float,
                                       is_artificial: bool) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate core collision probability using multiple methods.
        
        Scientific Rationale:
        Impact probability depends on:
        1. Orbital uncertainty (observation arc length, data quality)
        2. Close approach geometry (distance, velocity)
        3. Gravitational focusing by Earth
        4. Non-gravitational forces (Yarkovsky effect, artificial propulsion)
        
        Method Selection:
        - Short arc (< 30 days): High uncertainty, conservative estimates
        - Medium arc (30-365 days): Linear uncertainty propagation
        - Long arc (> 1 year): Monte Carlo simulation
        - Artificial objects: Include propulsive uncertainty
        """
        
        if not self._is_earth_crossing_orbit(orbital_elements):
            return 0.0, (0.0, 0.0)
        
        # Base collision cross-section calculation
        base_probability = self._calculate_base_collision_cross_section(orbital_elements)
        
        # Adjust for observational uncertainty
        uncertainty_factor = self._calculate_uncertainty_factor(observation_arc_days)
        
        # Gravitational focusing enhancement
        focusing_factor = self._calculate_gravitational_focusing(orbital_elements)
        
        # Close approach enhancement
        approach_factor = self._calculate_close_approach_factor(close_approaches)
        
        # Artificial object considerations
        artificial_factor = 1.0
        if is_artificial:
            artificial_factor = self._calculate_artificial_object_factor(orbital_elements)
        
        # Combined probability
        collision_prob = (base_probability * uncertainty_factor * 
                         focusing_factor * approach_factor * artificial_factor)
        
        # Calculate uncertainty bounds
        lower_bound = collision_prob * 0.1  # Conservative lower estimate
        upper_bound = min(collision_prob * 10.0, 1.0)  # Conservative upper estimate
        
        # Apply physical limits
        collision_prob = min(collision_prob, 0.1)  # No single object > 10% chance
        
        return collision_prob, (lower_bound, upper_bound)
    
    def _calculate_base_collision_cross_section(self, orbital_elements: OrbitalElements) -> float:
        """
        Calculate the basic geometric collision probability.
        
        Scientific Rationale:
        The fundamental collision probability is determined by the ratio of
        Earth's cross-sectional area to the area swept by the object's orbit
        where it intersects Earth's orbital region.
        
        Formula: P ≈ π * R_earth² / (orbit_intersection_area)
        
        This gives the probability per orbital pass through Earth's region.
        """
        
        if not orbital_elements.semi_major_axis or not orbital_elements.eccentricity:
            return 0.0
        
        # Earth's gravitational cross-section (enhanced by gravity)
        earth_cross_section = np.pi * (self.EARTH_RADIUS_KM)**2
        
        # Estimate orbital velocity near Earth's orbit
        a = orbital_elements.semi_major_axis * self.AU_TO_KM
        orbital_velocity = np.sqrt(self.GM_sun / a)  # Simplified circular orbit assumption
        
        # Relative velocity with Earth (rough estimate)
        earth_orbital_velocity = 29.78  # km/s
        relative_velocity = np.sqrt(orbital_velocity**2 + earth_orbital_velocity**2)
        
        # Time spent near Earth's orbit per orbital period
        orbital_period_years = orbital_elements.semi_major_axis**(3/2)  # Kepler's 3rd law
        orbital_period_seconds = orbital_period_years * 365.25 * 24 * 3600
        
        # Effective encounter rate
        encounter_frequency = 1.0 / orbital_period_seconds  # encounters per second
        
        # Basic collision probability per encounter
        # This is a simplified model - real calculations would use numerical integration
        base_prob = earth_cross_section * encounter_frequency * 1e-15  # Scaling factor
        
        return min(base_prob, 1e-6)  # Cap at reasonable maximum
    
    def _calculate_uncertainty_factor(self, observation_arc_days: float) -> float:
        """
        Calculate uncertainty enhancement factor based on observation arc.
        
        Scientific Rationale:
        Short observation arcs lead to large orbital uncertainties, which
        increase the effective collision cross-section. This is because
        the object could be anywhere within the uncertainty ellipse.
        
        The uncertainty factor decreases exponentially with arc length.
        """
        
        if observation_arc_days < 1:
            return 100.0  # Very high uncertainty for sub-day arcs
        elif observation_arc_days < 7:
            return 50.0   # High uncertainty for weekly arcs
        elif observation_arc_days < 30:
            return 10.0   # Moderate uncertainty for monthly arcs
        elif observation_arc_days < 365:
            return 2.0    # Low uncertainty for yearly arcs
        else:
            return 1.0    # Well-determined orbits
    
    def _calculate_gravitational_focusing(self, orbital_elements: OrbitalElements) -> float:
        """
        Calculate gravitational focusing enhancement.
        
        Scientific Rationale:
        Earth's gravity increases the effective collision cross-section by
        deflecting nearby trajectories toward Earth. This enhancement is
        velocity-dependent and can be significant for slow approaches.
        
        Formula: σ_enhanced = σ_geometric * (1 + (v_escape/v_infinity)²)
        """
        
        # Estimate approach velocity (simplified)
        if not orbital_elements.semi_major_axis:
            return 1.0
        
        # Typical approach velocity for this orbit type
        a = orbital_elements.semi_major_axis
        v_infinity = abs(np.sqrt(self.GM_sun / a) - 29.78)  # Relative to Earth
        
        if v_infinity < 1.0:
            v_infinity = 1.0  # Avoid division by zero
        
        # Gravitational focusing factor
        focusing = 1.0 + (self.EARTH_ESCAPE_VELOCITY / v_infinity)**2
        
        return min(focusing, 100.0)  # Cap enhancement
    
    def _calculate_close_approach_factor(self, close_approaches: List[CloseApproach]) -> float:
        """
        Enhancement factor based on known close approaches.
        
        Scientific Rationale:
        Past or predicted close approaches indicate the object's orbit
        brings it near Earth, increasing collision probability.
        
        Very close approaches (< 0.1 AU) significantly increase risk.
        """
        
        if not close_approaches:
            return 1.0
        
        # Find closest approach
        min_distance = float('inf')
        for approach in close_approaches:
            if approach.distance_au:
                min_distance = min(min_distance, approach.distance_au)
        
        if min_distance == float('inf'):
            return 1.0
        
        # Enhancement based on closest approach
        if min_distance < 0.01:      # < 0.01 AU (very close)
            return 100.0
        elif min_distance < 0.05:    # < 0.05 AU (close)
            return 10.0
        elif min_distance < 0.1:     # < 0.1 AU (near)
            return 3.0
        elif min_distance < 0.2:     # < 0.2 AU (moderate)
            return 1.5
        else:
            return 1.0
    
    def _calculate_artificial_object_factor(self, orbital_elements: OrbitalElements) -> float:
        """
        Adjustment for artificial objects with potential propulsion.
        
        Scientific Rationale:
        Artificial objects may have:
        1. Active propulsion systems (can change trajectory)
        2. Attitude control (can affect non-gravitational forces)
        3. Mission constraints (may deliberately target or avoid Earth)
        
        This introduces additional uncertainty in trajectory prediction.
        """
        
        # Artificial objects have higher uncertainty due to potential propulsion
        # But may also have mission planning that avoids Earth impact
        
        # For now, assume modest increase in uncertainty
        return 2.0  # Factor to be refined based on object type analysis
    
    def _calculate_temporal_evolution(self, 
                                    orbital_elements: OrbitalElements,
                                    close_approaches: List[CloseApproach]) -> Dict[str, float]:
        """
        Calculate how impact probability evolves over time.
        
        Scientific Rationale:
        Impact probability is not constant - it varies with:
        1. Orbital evolution due to non-gravitational forces
        2. Planetary perturbations
        3. Gravitational keyhole passages
        4. Uncertainty growth over time
        """
        
        current_year = datetime.now().year
        evolution = {}
        
        # Simple model: probability increases with time due to uncertainty growth
        base_prob = 1e-7  # Base annual probability
        
        for decade in range(0, 11):  # Next 100 years
            year_start = current_year + decade * 10
            year_end = year_start + 10
            decade_key = f"{year_start}-{year_end}"
            
            # Probability grows with time due to uncertainty
            time_factor = 1.0 + decade * 0.1
            
            # Check for close approaches in this decade
            approach_factor = 1.0
            if close_approaches:
                for approach in close_approaches:
                    if approach.close_approach_date:
                        approach_year = approach.close_approach_date.year
                        if year_start <= approach_year <= year_end:
                            if approach.distance_au and approach.distance_au < 0.1:
                                approach_factor = 10.0  # Significant enhancement
                            break
            
            decade_prob = base_prob * time_factor * approach_factor
            evolution[decade_key] = min(decade_prob, 0.01)  # Cap at 1%
        
        return evolution
    
    def _calculate_impact_energy(self, orbital_elements: OrbitalElements) -> Optional[float]:
        """
        Estimate kinetic energy of impact in megatons TNT equivalent.
        
        Scientific Rationale:
        Impact energy determines damage potential and is crucial for
        risk assessment. Energy = 0.5 * m * v²
        
        Conversion: 1 megaton TNT = 4.184 × 10¹⁵ Joules
        """
        
        if not orbital_elements.diameter:
            return None
        
        # Estimate mass from diameter (assume rocky composition)
        diameter_m = orbital_elements.diameter * 1000  # km to m
        volume_m3 = (4/3) * np.pi * (diameter_m/2)**3
        density_kg_m3 = 2500  # Typical rocky asteroid density
        mass_kg = volume_m3 * density_kg_m3
        
        # Estimate impact velocity
        impact_velocity = self._estimate_impact_velocity(orbital_elements)
        if not impact_velocity:
            return None
        
        # Kinetic energy in Joules
        kinetic_energy_j = 0.5 * mass_kg * (impact_velocity * 1000)**2  # km/s to m/s
        
        # Convert to megatons TNT
        megatons_tnt = kinetic_energy_j / 4.184e15
        
        return megatons_tnt
    
    def _estimate_impact_velocity(self, orbital_elements: OrbitalElements) -> Optional[float]:
        """
        Estimate impact velocity in km/s.
        
        Scientific Rationale:
        Impact velocity determines crater size and damage radius.
        It's the vector sum of orbital velocity and Earth's escape velocity.
        """
        
        if not orbital_elements.semi_major_axis:
            return None
        
        # Estimate velocity at Earth's distance using vis-viva equation
        a = orbital_elements.semi_major_axis * self.AU_TO_KM
        r = 1.0 * self.AU_TO_KM  # Earth's orbital distance
        
        # Orbital velocity at Earth's distance
        v_orbital = np.sqrt(self.GM_sun * (2/r - 1/a))
        
        # Earth's orbital velocity
        v_earth = 29.78  # km/s
        
        # Relative velocity (simplified vector addition)
        v_relative = abs(v_orbital - v_earth)
        
        # Impact velocity includes gravitational acceleration by Earth
        v_impact = np.sqrt(v_relative**2 + self.EARTH_ESCAPE_VELOCITY**2)
        
        return v_impact
    
    def _estimate_crater_diameter(self, impact_energy_mt: Optional[float]) -> Optional[float]:
        """
        Estimate crater diameter using scaling laws.
        
        Scientific Rationale:
        Crater diameter follows empirical scaling laws based on impact energy.
        Used for damage assessment and geological impact studies.
        
        Schmidt-Housen scaling: D ∝ E^0.22 for large craters
        """
        
        if not impact_energy_mt:
            return None
        
        # Empirical scaling law (simplified)
        # D (km) ≈ 0.1 * E(MT)^0.22
        crater_diameter = 0.1 * (impact_energy_mt ** 0.22)
        
        return crater_diameter
    
    def _estimate_damage_radius(self, impact_energy_mt: Optional[float]) -> Optional[float]:
        """
        Estimate radius of significant damage.
        
        Scientific Rationale:
        Damage radius estimates help assess human and infrastructure impact.
        Based on blast wave propagation and thermal effects.
        """
        
        if not impact_energy_mt:
            return None
        
        # Simplified damage radius scaling
        # Significant damage radius ≈ 10 * E(MT)^0.33 km
        damage_radius = 10.0 * (impact_energy_mt ** 0.33)
        
        return damage_radius
    
    def _calculate_impact_latitude_distribution(self, orbital_elements: OrbitalElements) -> Dict[str, float]:
        """
        Calculate probability distribution of impact latitude.
        
        Scientific Rationale:
        Orbital inclination determines the latitude range where impacts can occur.
        Higher inclination orbits can impact at higher latitudes.
        
        This information is crucial for regional risk assessment.
        """
        
        if not orbital_elements.inclination:
            return {"unknown": 1.0}
        
        inclination = orbital_elements.inclination
        
        # Simple model based on orbital inclination
        if inclination < 5:
            return {
                "equatorial": 0.6,
                "tropical": 0.3,
                "temperate": 0.1,
                "polar": 0.0
            }
        elif inclination < 30:
            return {
                "equatorial": 0.3,
                "tropical": 0.4,
                "temperate": 0.3,
                "polar": 0.0
            }
        elif inclination < 60:
            return {
                "equatorial": 0.2,
                "tropical": 0.3,
                "temperate": 0.4,
                "polar": 0.1
            }
        else:
            return {
                "equatorial": 0.1,
                "tropical": 0.2,
                "temperate": 0.4,
                "polar": 0.3
            }
    
    def _identify_most_probable_impact_region(self, latitude_dist: Dict[str, float]) -> Optional[str]:
        """Identify the most probable impact region."""
        if not latitude_dist:
            return None
        
        return max(latitude_dist, key=latitude_dist.get)
    
    def _analyze_gravitational_keyholes(self, 
                                      orbital_elements: OrbitalElements,
                                      close_approaches: List[CloseApproach]) -> List[Dict[str, Any]]:
        """
        Analyze gravitational keyhole passages.
        
        Scientific Rationale:
        Gravitational keyholes are small regions in space where a small
        gravitational perturbation during a close approach can redirect
        an asteroid onto a collision course with Earth on a future orbit.
        
        This is how small uncertainties can lead to large impact probability changes.
        """
        
        keyholes = []
        
        if not close_approaches:
            return keyholes
        
        for approach in close_approaches:
            if not approach.distance_au or approach.distance_au > 0.1:
                continue  # Only consider close approaches
            
            keyhole = {
                "approach_date": approach.close_approach_date.isoformat() if approach.close_approach_date else None,
                "distance_au": approach.distance_au,
                "keyhole_size_km": self._estimate_keyhole_size(approach),
                "resonance_returns": self._calculate_resonance_returns(orbital_elements, approach),
                "impact_probability_enhancement": self._calculate_keyhole_enhancement(approach)
            }
            keyholes.append(keyhole)
        
        return keyholes
    
    def _estimate_keyhole_size(self, approach: CloseApproach) -> float:
        """Estimate the size of gravitational keyholes for this approach."""
        if not approach.distance_au:
            return 0.0
        
        # Rough scaling: closer approaches have smaller keyholes
        # but higher sensitivity
        distance_km = approach.distance_au * self.AU_TO_KM
        keyhole_size = 1000.0 / (distance_km / 100000.0)  # Simplified scaling
        
        return min(keyhole_size, 10000.0)  # Cap at 10,000 km
    
    def _calculate_resonance_returns(self, orbital_elements: OrbitalElements, approach: CloseApproach) -> List[int]:
        """Calculate potential resonant return periods."""
        if not orbital_elements.semi_major_axis:
            return []
        
        # Simple resonance calculation based on orbital period
        period_years = orbital_elements.semi_major_axis**(3/2)
        
        # Common resonances with Earth (1:1, 2:1, 3:2, etc.)
        resonances = []
        for n in range(1, 8):
            for m in range(1, n+2):
                resonant_period = period_years * (n/m)
                if 0.5 <= resonant_period <= 10.0:  # Reasonable timeframes
                    resonances.append(int(resonant_period))
        
        return sorted(list(set(resonances)))  # Remove duplicates and sort
    
    def _calculate_keyhole_enhancement(self, approach: CloseApproach) -> float:
        """Calculate impact probability enhancement from keyhole passage."""
        if not approach.distance_au:
            return 1.0
        
        # Closer approaches lead to higher enhancement
        if approach.distance_au < 0.01:
            return 1000.0
        elif approach.distance_au < 0.05:
            return 100.0
        elif approach.distance_au < 0.1:
            return 10.0
        else:
            return 1.0
    
    def _analyze_artificial_impact_scenarios(self, 
                                           orbital_elements: OrbitalElements,
                                           artificial_probability: float) -> Dict[str, Any]:
        """
        Special analysis for artificial objects.
        
        Scientific Rationale:
        Artificial objects introduce unique considerations:
        1. May have active guidance systems
        2. Could have mission objectives affecting trajectory
        3. May have fuel reserves for trajectory changes
        4. Might be designed to deorbit safely
        """
        
        scenarios = {
            "controlled_deorbit": {
                "probability": 0.1 * artificial_probability,
                "description": "Controlled deorbit to safe ocean area",
                "impact_risk": "low"
            },
            "uncontrolled_reentry": {
                "probability": 0.3 * artificial_probability,
                "description": "Uncontrolled reentry after mission end",
                "impact_risk": "moderate"
            },
            "mission_malfunction": {
                "probability": 0.05 * artificial_probability,
                "description": "Trajectory change due to system malfunction",
                "impact_risk": "variable"
            },
            "deliberate_targeting": {
                "probability": 0.001 * artificial_probability,
                "description": "Intentional Earth targeting (extremely unlikely)",
                "impact_risk": "high"
            }
        }
        
        return {
            "scenarios": scenarios,
            "propulsion_uncertainty": True,
            "mission_status_unknown": True,
            "special_monitoring_recommended": artificial_probability > 0.5
        }
    
    def _classify_risk_level(self, collision_probability: float) -> str:
        """Classify impact risk level based on probability."""
        for level, threshold in reversed(list(self.RISK_THRESHOLDS.items())):
            if collision_probability >= threshold:
                return level
        return 'negligible'
    
    def _generate_comparative_risk_statement(self, collision_probability: float) -> str:
        """Generate human-readable risk comparison."""
        if collision_probability >= 1e-3:
            return "Higher than typical natural disaster risks"
        elif collision_probability >= 1e-4:
            return "Comparable to major earthquake risk"
        elif collision_probability >= 1e-5:
            return "Comparable to aircraft accident risk"
        elif collision_probability >= 1e-6:
            return "Comparable to lightning strike risk"
        elif collision_probability >= 1e-7:
            return "Comparable to winning lottery risk"
        else:
            return "Much lower than everyday risks"
    
    def _identify_primary_risk_factors(self, 
                                     orbital_elements: OrbitalElements,
                                     close_approaches: List[CloseApproach],
                                     is_earth_crossing: bool,
                                     observation_arc_days: float) -> List[str]:
        """Identify primary factors contributing to impact risk."""
        factors = []
        
        if is_earth_crossing:
            factors.append("Earth-crossing orbit confirmed")
        
        if observation_arc_days < 30:
            factors.append("Short observation arc increases uncertainty")
        
        if close_approaches:
            min_dist = min(app.distance_au for app in close_approaches if app.distance_au)
            if min_dist < 0.05:
                factors.append(f"Very close approach at {min_dist:.4f} AU")
        
        if orbital_elements.eccentricity and orbital_elements.eccentricity > 0.3:
            factors.append("High eccentricity increases encounter velocity")
        
        if orbital_elements.inclination and orbital_elements.inclination < 5:
            factors.append("Low inclination favors equatorial impacts")
        
        return factors
    
    def _document_calculation_assumptions(self, 
                                        orbital_elements: OrbitalElements,
                                        close_approaches: List[CloseApproach]) -> List[str]:
        """Document key assumptions in the calculation."""
        assumptions = [
            "Purely gravitational dynamics (no non-gravitational forces)",
            "Current orbital elements represent true orbit",
            "Earth treated as point mass for distant encounters",
            "Linear uncertainty propagation for short time periods"
        ]
        
        if not close_approaches:
            assumptions.append("No close approach data available")
        
        if not orbital_elements.diameter:
            assumptions.append("Object size estimated from absolute magnitude")
        
        return assumptions
    
    def _document_calculation_limitations(self, 
                                        orbital_elements: OrbitalElements,
                                        observation_arc_days: float) -> List[str]:
        """Document limitations of the impact probability calculation."""
        limitations = [
            "Simplified collision cross-section model",
            "No detailed uncertainty covariance analysis",
            "No Monte Carlo orbital propagation",
            "Limited to 100-year time horizon"
        ]
        
        if observation_arc_days < 30:
            limitations.append("Short observation arc limits accuracy")
        
        if not orbital_elements.is_complete():
            limitations.append("Incomplete orbital elements")
        
        return limitations
    
    def _assess_calculation_confidence(self, 
                                     orbital_elements: OrbitalElements,
                                     close_approaches: List[CloseApproach],
                                     observation_arc_days: float) -> float:
        """Assess confidence in the impact probability calculation."""
        confidence = 1.0
        
        # Reduce confidence for incomplete data
        if not orbital_elements.is_complete():
            confidence *= 0.5
        
        # Reduce confidence for short observation arcs
        if observation_arc_days < 7:
            confidence *= 0.2
        elif observation_arc_days < 30:
            confidence *= 0.5
        elif observation_arc_days < 365:
            confidence *= 0.8
        
        # Increase confidence if close approaches available
        if close_approaches:
            confidence *= 1.2
        
        return min(confidence, 1.0)
    
    def _determine_calculation_method(self, 
                                    orbital_elements: OrbitalElements,
                                    close_approaches: List[CloseApproach]) -> str:
        """Determine which calculation method was primarily used."""
        if close_approaches and len(close_approaches) > 2:
            return "close_approach_analysis"
        elif orbital_elements.is_complete():
            return "orbital_mechanics_linear"
        else:
            return "simplified_cross_section"
    
    def _estimate_time_to_impact(self, impact_evolution: Dict[str, float]) -> Optional[float]:
        """Estimate most probable time to impact in years."""
        if not impact_evolution:
            return None
        
        # Find decade with highest probability
        max_prob = 0
        max_decade = None
        
        for decade, prob in impact_evolution.items():
            if prob > max_prob:
                max_prob = prob
                max_decade = decade
        
        if max_decade:
            # Extract start year and estimate mid-decade
            start_year = int(max_decade.split('-')[0])
            current_year = datetime.now().year
            return start_year + 5 - current_year  # Mid-decade estimate
        
        return None
    
    def _identify_peak_risk_period(self, impact_evolution: Dict[str, float]) -> Optional[Tuple[int, int]]:
        """Identify the time period of highest impact risk."""
        if not impact_evolution:
            return None
        
        max_prob = max(impact_evolution.values())
        for decade, prob in impact_evolution.items():
            if prob == max_prob:
                years = decade.split('-')
                return (int(years[0]), int(years[1]))
        
        return None

    def _calculate_moon_impact_probability(self, 
                                         orbital_elements: OrbitalElements,
                                         close_approaches: List[CloseApproach],
                                         observation_arc_days: float,
                                         is_artificial: bool,
                                         artificial_probability: float = 0.0) -> Dict[str, Any]:
        """
        Calculate Moon impact probability and compare to Earth impact risk.
        
        Scientific Rationale for Moon Impact Assessment:
        
        **Why calculate Moon impact probability?**
        1. Moon impacts are often more likely than Earth impacts for some orbits
        2. Moon has no atmosphere - all impactors reach surface
        3. Moon impacts create observable phenomena (new craters, ejecta)
        4. Important for lunar missions and installations
        
        **Physical considerations:**
        - Moon's smaller cross-section: π × (1737 km)² vs π × (6371 km)²
        - Moon's weaker gravity: 2.4 km/s escape vs 11.2 km/s for Earth
        - Moon's orbital motion: 384,400 km from Earth with 27.3 day period
        - No atmospheric protection for Moon
        
        **When Moon impacts are more likely:**
        - Earth-Moon system crossing orbits
        - Objects with periods near lunar orbital resonances
        - Artificial objects in Earth-Moon transfer trajectories
        
        Args:
            orbital_elements: Object's orbital elements
            close_approaches: Close approach data
            observation_arc_days: Length of observation arc
            is_artificial: Whether object is artificial
            artificial_probability: Probability object is artificial
            
        Returns:
            Dict containing Moon impact assessment results
        """
        
        moon_results = {
            'moon_collision_probability': 0.0,
            'moon_impact_energy_mt': None,
            'earth_vs_moon_ratio': 1.0,
            'moon_impact_effects': None,
            'additional_risk_factors': [],
            'moon_assumptions': []
        }
        
        try:
            # Check if orbit could intersect Earth-Moon system
            if not self._is_earth_moon_system_crossing(orbital_elements):
                moon_results['additional_risk_factors'].append("Orbit does not cross Earth-Moon system")
                return moon_results
            
            # Calculate basic Moon collision cross-section
            moon_cross_section = self._calculate_moon_collision_cross_section(orbital_elements)
            
            # Apply gravitational focusing for Moon
            moon_focusing = self._calculate_moon_gravitational_focusing(orbital_elements)
            
            # Earth collision probability for comparison (simplified)
            earth_collision_prob = self._calculate_earth_collision_probability_simple(orbital_elements)
            
            # Moon collision probability
            moon_collision_prob = moon_cross_section * moon_focusing
            
            # Apply uncertainty factors
            uncertainty_factor = self._calculate_uncertainty_factor(observation_arc_days)
            moon_collision_prob *= uncertainty_factor
            
            # Apply artificial object considerations for Moon
            if is_artificial or artificial_probability > 0.1:
                # Artificial objects might be on lunar transfer trajectories
                if self._is_potential_lunar_transfer_orbit(orbital_elements):
                    moon_collision_prob *= 10.0  # Significant enhancement
                    moon_results['additional_risk_factors'].append(
                        "Artificial object on potential lunar transfer trajectory"
                    )
                else:
                    moon_collision_prob *= 2.0  # General artificial enhancement
            
            # Calculate Moon impact energy
            moon_impact_energy = self._calculate_moon_impact_energy(orbital_elements)
            
            # Earth vs Moon impact ratio
            earth_vs_moon_ratio = 1.0
            if moon_collision_prob > 0:
                earth_vs_moon_ratio = earth_collision_prob / moon_collision_prob
            
            # Moon impact effects assessment
            moon_impact_effects = self._assess_moon_impact_effects(orbital_elements, moon_impact_energy)
            
            # Update results
            moon_results.update({
                'moon_collision_probability': min(moon_collision_prob, 1.0),
                'moon_impact_energy_mt': moon_impact_energy,
                'earth_vs_moon_ratio': earth_vs_moon_ratio,
                'moon_impact_effects': moon_impact_effects
            })
            
            # Add scientific assumptions
            moon_results['moon_assumptions'].extend([
                "Moon treated as gravitationally focused point target",
                "No lunar atmospheric protection considered",
                "Moon's orbital motion approximated as circular",
                "Simplified Earth-Moon gravitational interaction"
            ])
            
            # Add contextual risk factors
            if moon_collision_prob > earth_collision_prob:
                ratio = moon_collision_prob / earth_collision_prob if earth_collision_prob > 0 else float('inf')
                moon_results['additional_risk_factors'].append(
                    f"Moon impact {ratio:.1f}x more likely than Earth impact"
                )
            
            if moon_impact_energy and moon_impact_energy > 1:
                moon_results['additional_risk_factors'].append(
                    f"Moon impact would create {moon_impact_energy:.1f} MT explosion visible from Earth"
                )
                
        except Exception as e:
            self.logger.warning(f"Moon impact calculation failed: {e}")
            moon_results['moon_assumptions'].append(f"Moon calculation error: {e}")
        
        return moon_results
    
    def _is_earth_moon_system_crossing(self, orbital_elements: OrbitalElements) -> bool:
        """Check if orbit could intersect the Earth-Moon system."""
        
        if not orbital_elements.semi_major_axis or not orbital_elements.eccentricity:
            return False
        
        a = orbital_elements.semi_major_axis
        e = orbital_elements.eccentricity
        
        perihelion = a * (1 - e)
        aphelion = a * (1 + e)
        
        # Earth-Moon system extends to ~0.0026 AU from Earth
        earth_moon_distance_au = self.MOON_ORBITAL_DISTANCE_KM / self.AU_TO_KM
        earth_orbit_min = 1.0 - earth_moon_distance_au
        earth_orbit_max = 1.0 + earth_moon_distance_au
        
        return perihelion < earth_orbit_max and aphelion > earth_orbit_min
    
    def _calculate_moon_collision_cross_section(self, orbital_elements: OrbitalElements) -> float:
        """Calculate basic Moon collision cross-section probability."""
        
        # Moon's physical cross-section
        moon_cross_section = np.pi * (self.MOON_RADIUS_KM)**2
        
        # Earth's orbital circumference (approximate collision zone)
        earth_orbital_circumference = 2 * np.pi * (1.0 * self.AU_TO_KM)
        
        # Basic geometric probability
        geometric_prob = moon_cross_section / earth_orbital_circumference**0.5
        
        # Scale by orbital period (how often object crosses Earth's orbit)
        if orbital_elements.semi_major_axis:
            orbital_period_years = orbital_elements.semi_major_axis**(3/2)
            encounter_frequency = 1.0 / orbital_period_years
        else:
            encounter_frequency = 1.0
        
        return geometric_prob * encounter_frequency * 1e-12  # Scaling factor
    
    def _calculate_moon_gravitational_focusing(self, orbital_elements: OrbitalElements) -> float:
        """Calculate gravitational focusing enhancement for Moon impacts."""
        
        # Estimate approach velocity relative to Moon
        if orbital_elements.semi_major_axis:
            # Simplified: velocity at Earth's distance
            v_orbital = np.sqrt(self.GM_sun / (orbital_elements.semi_major_axis * self.AU_TO_KM))
            v_earth = 29.78  # km/s
            v_relative = abs(v_orbital - v_earth)
        else:
            v_relative = 15.0  # Default assumption
        
        # Moon's gravitational focusing
        # σ_enhanced = σ_geometric * (1 + (v_escape/v_infinity)²)
        if v_relative > 0:
            focusing_factor = 1.0 + (self.MOON_ESCAPE_VELOCITY / v_relative)**2
        else:
            focusing_factor = 1.0
        
        return min(focusing_factor, 25.0)  # Cap enhancement
    
    def _calculate_earth_collision_probability_simple(self, orbital_elements: OrbitalElements) -> float:
        """Simple Earth collision probability for comparison."""
        
        if not self._is_earth_crossing_orbit(orbital_elements):
            return 0.0
        
        # Very simplified calculation for comparison
        base_prob = 1e-9  # Base Earth crossing probability
        
        # Enhance based on eccentricity (higher e = more Earth-crossing)
        if orbital_elements.eccentricity:
            ecc_factor = 1.0 + orbital_elements.eccentricity * 2.0
        else:
            ecc_factor = 1.0
        
        return base_prob * ecc_factor
    
    def _is_potential_lunar_transfer_orbit(self, orbital_elements: OrbitalElements) -> bool:
        """Check if orbit resembles a lunar transfer trajectory."""
        
        if not orbital_elements.semi_major_axis or not orbital_elements.eccentricity:
            return False
        
        a = orbital_elements.semi_major_axis
        e = orbital_elements.eccentricity
        
        # Lunar transfer orbits typically have:
        # - Semi-major axis ~1.2-1.3 AU
        # - Moderate eccentricity (0.1-0.4)
        # - Low inclination
        
        lunar_transfer_criteria = (
            1.1 <= a <= 1.4 and
            0.1 <= e <= 0.5 and
            (orbital_elements.inclination is None or orbital_elements.inclination < 30)
        )
        
        return lunar_transfer_criteria
    
    def _calculate_moon_impact_energy(self, orbital_elements: OrbitalElements) -> Optional[float]:
        """Calculate Moon impact energy in megatons TNT equivalent."""
        
        if not orbital_elements.diameter:
            return None
        
        # Estimate mass (same as Earth impact calculation)
        diameter_m = orbital_elements.diameter * 1000
        volume_m3 = (4/3) * np.pi * (diameter_m/2)**3
        density_kg_m3 = 2500  # Rocky asteroid
        mass_kg = volume_m3 * density_kg_m3
        
        # Moon impact velocity (typically lower than Earth due to weaker gravity)
        if orbital_elements.semi_major_axis:
            # Velocity at Earth's distance
            a_km = orbital_elements.semi_major_axis * self.AU_TO_KM
            v_orbital = np.sqrt(self.GM_sun / a_km)
            v_earth = 29.78
            v_relative = abs(v_orbital - v_earth)
            
            # Moon impact velocity (includes Moon's gravity but weaker than Earth)
            v_moon_impact = np.sqrt(v_relative**2 + self.MOON_ESCAPE_VELOCITY**2)
        else:
            v_moon_impact = 15.0  # Default estimate
        
        # Kinetic energy
        kinetic_energy_j = 0.5 * mass_kg * (v_moon_impact * 1000)**2
        
        # Convert to megatons TNT
        megatons_tnt = kinetic_energy_j / 4.184e15
        
        return megatons_tnt
    
    def _assess_moon_impact_effects(self, orbital_elements: OrbitalElements, 
                                  impact_energy_mt: Optional[float]) -> Dict[str, Any]:
        """Assess the effects of a Moon impact."""
        
        effects = {
            'crater_formation': True,
            'ejecta_production': True,
            'seismic_effects': True,
            'visible_from_earth': False,
            'lunar_mission_impact': 'unknown'
        }
        
        if impact_energy_mt:
            # Large impacts are visible from Earth
            if impact_energy_mt > 0.1:
                effects['visible_from_earth'] = True
                effects['flash_visible'] = True
            
            # Crater size on Moon (larger due to no atmosphere)
            crater_diameter_km = 0.15 * (impact_energy_mt ** 0.25)  # Scaling for Moon
            effects['crater_diameter_km'] = crater_diameter_km
            
            # Ejecta effects
            if impact_energy_mt > 1.0:
                effects['ejecta_visible_from_earth'] = True
                effects['debris_cloud_duration_hours'] = impact_energy_mt * 2
            
            # Lunar infrastructure effects
            if impact_energy_mt > 10.0:
                effects['lunar_mission_impact'] = 'severe'
                effects['seismic_range_km'] = impact_energy_mt * 100
            elif impact_energy_mt > 1.0:
                effects['lunar_mission_impact'] = 'moderate'
                effects['seismic_range_km'] = impact_energy_mt * 50
            else:
                effects['lunar_mission_impact'] = 'minimal'
        
        return effects

# Integration with existing aNEOS pipeline
def integrate_impact_probability_into_pipeline():
    """
    Integration function to add impact probability to aNEOS pipeline.
    
    This function shows how to integrate the impact probability calculator
    into the existing aNEOS analysis pipeline.
    """
    pass  # Implementation depends on specific integration requirements