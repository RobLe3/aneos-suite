#!/usr/bin/env python3
"""
Physical Sanity Validation for aNEOS

This module implements physical sanity checks as specified in the Calibration Plan v1.2
to ensure all outputs are physically plausible and consistent.

Key Features:
- Energy/size/mass consistency validation
- Crater scaling law verification
- Risk label logic validation
- Unit consistency checks
- Physical bounds enforcement

Based on Calibration Plan v1.2 requirements.
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class PhysicalValidationResult:
    """Result of physical sanity validation."""
    
    status: ValidationResult
    issues: List[str]
    warnings: List[str]
    corrected_values: Dict[str, Any]
    validation_notes: List[str]


class PhysicalSanityValidator:
    """
    Physical sanity validation for NEO analysis outputs.
    
    Implements the physical sanity model from Calibration Plan v1.2:
    - Energy check: E = 0.5 * m * v²
    - Impact scaling for tens/hundreds of meters objects
    - Crater scaling: Earth rocky surface → 10-20× impactor diameter
    - Crater scaling: Moon → 15-25× impactor diameter (lower gravity)
    """
    
    # Physical constants
    TNT_JOULES_PER_MEGATON = 4.184e15  # Joules per megaton TNT
    TYPICAL_ASTEROID_DENSITY = 2500  # kg/m³ (rocky asteroid)
    EARTH_ESCAPE_VELOCITY = 11.2  # km/s
    MOON_ESCAPE_VELOCITY = 2.4  # km/s
    
    # Scaling factors from Calibration Plan (updated per interim assessment)
    EARTH_CRATER_SCALING_MIN_SMALL = 3   # Minimum for small objects (airburst effects)
    EARTH_CRATER_SCALING_MAX_SMALL = 7   # Maximum for small objects
    EARTH_CRATER_SCALING_MIN = 10  # Minimum crater/impactor diameter ratio for large objects
    EARTH_CRATER_SCALING_MAX = 20  # Maximum crater/impactor diameter ratio for large objects
    MOON_CRATER_SCALING_MIN = 15   # Minimum crater/impactor diameter ratio for Moon
    MOON_CRATER_SCALING_MAX = 25   # Maximum crater/impactor diameter ratio for Moon
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_impact_assessment(self, impact_data: Dict[str, Any]) -> PhysicalValidationResult:
        """
        Validate an impact assessment for physical consistency.
        
        Args:
            impact_data: Impact assessment data dictionary
            
        Returns:
            PhysicalValidationResult with validation status and any issues
        """
        issues = []
        warnings = []
        corrected_values = {}
        notes = []
        
        # Extract key values
        diameter_km = impact_data.get('object_diameter_km')
        energy_mt = impact_data.get('impact_energy_mt')
        velocity_km_s = impact_data.get('impact_velocity_km_s')
        crater_diameter_km = impact_data.get('crater_diameter_km')
        moon_crater_diameter_km = impact_data.get('moon_crater_diameter_km')
        collision_probability = impact_data.get('collision_probability', 0)
        risk_level = impact_data.get('risk_level', 'negligible')
        
        # 1. Energy consistency check
        if diameter_km and energy_mt and velocity_km_s:
            expected_energy = self._calculate_expected_energy(diameter_km, velocity_km_s)
            energy_ratio = energy_mt / expected_energy if expected_energy > 0 else float('inf')
            
            if energy_ratio > 10 or energy_ratio < 0.1:
                issues.append(
                    f"Energy inconsistency: {energy_mt:.1f} MT vs expected {expected_energy:.1f} MT "
                    f"(ratio: {energy_ratio:.1f}x)"
                )
                corrected_values['impact_energy_mt'] = expected_energy
                notes.append(f"Energy corrected using E = 0.5 * m * v² with density {self.TYPICAL_ASTEROID_DENSITY} kg/m³")
        
        # 2. Crater scaling validation
        if diameter_km and crater_diameter_km:
            crater_ratio = crater_diameter_km / (diameter_km / 1000)  # Convert km to m for diameter
            diameter_m = diameter_km * 1000
            
            # Use different scaling ranges for small vs large objects
            if diameter_m <= 100:  # Small objects (airburst effects)
                min_scaling = self.EARTH_CRATER_SCALING_MIN_SMALL
                max_scaling = self.EARTH_CRATER_SCALING_MAX_SMALL
                default_scaling = 5.0
            else:  # Large objects
                min_scaling = self.EARTH_CRATER_SCALING_MIN
                max_scaling = self.EARTH_CRATER_SCALING_MAX
                default_scaling = 15.0
                
            if crater_ratio < min_scaling or crater_ratio > max_scaling:
                issues.append(
                    f"Earth crater scaling violation: {crater_ratio:.1f}x impactor diameter "
                    f"(expected {min_scaling}-{max_scaling}x for {diameter_m:.0f}m object)"
                )
                # Correct to appropriate scaling
                corrected_crater = (diameter_km / 1000) * default_scaling
                corrected_values['crater_diameter_km'] = corrected_crater
                notes.append(f"Earth crater diameter corrected using {default_scaling}x scaling factor")
        
        # 3. Moon crater scaling validation
        if diameter_km and moon_crater_diameter_km:
            moon_crater_ratio = moon_crater_diameter_km / (diameter_km / 1000)
            if moon_crater_ratio < self.MOON_CRATER_SCALING_MIN or moon_crater_ratio > self.MOON_CRATER_SCALING_MAX:
                issues.append(
                    f"Moon crater scaling violation: {moon_crater_ratio:.1f}x impactor diameter "
                    f"(expected {self.MOON_CRATER_SCALING_MIN}-{self.MOON_CRATER_SCALING_MAX}x)"
                )
                # Correct to middle of range
                corrected_moon_crater = (diameter_km / 1000) * (self.MOON_CRATER_SCALING_MIN + self.MOON_CRATER_SCALING_MAX) / 2
                corrected_values['moon_crater_diameter_km'] = corrected_moon_crater
                notes.append(f"Moon crater diameter corrected using 20x scaling factor")
        
        # 4. Energy scaling validation by size category
        if diameter_km and energy_mt:
            diameter_m = diameter_km * 1000
            if 10 <= diameter_m < 100:  # Tens of meters → kiloton to low-megaton
                if energy_mt > 10:
                    warnings.append(f"High energy ({energy_mt:.1f} MT) for {diameter_m}m object (expected <10 MT)")
            elif 100 <= diameter_m < 1000:  # Hundreds of meters → hundreds to thousands of megatons
                if energy_mt > 10000:
                    warnings.append(f"Very high energy ({energy_mt:.1f} MT) for {diameter_m}m object (expected <10,000 MT)")
                elif energy_mt < 100:
                    warnings.append(f"Low energy ({energy_mt:.1f} MT) for {diameter_m}m object (expected >100 MT)")
        
        # 5. Risk label logic validation (from Calibration Plan Table 6)
        probability_risk_valid = self._validate_risk_label_logic(collision_probability, risk_level)
        if not probability_risk_valid:
            issues.append(
                f"Risk label contradiction: '{risk_level}' risk with {collision_probability:.2e} probability"
            )
            corrected_risk = self._get_correct_risk_level(collision_probability)
            corrected_values['risk_level'] = corrected_risk
            notes.append(f"Risk level corrected to '{corrected_risk}' based on probability threshold")
        
        # 6. Missing data validation
        if not diameter_km and energy_mt:
            warnings.append("Energy calculated without object diameter - calculation may be unreliable")
        
        # Determine overall status
        if issues:
            status = ValidationResult.FAIL
        elif warnings:
            status = ValidationResult.WARNING  
        else:
            status = ValidationResult.PASS
            
        return PhysicalValidationResult(
            status=status,
            issues=issues,
            warnings=warnings,
            corrected_values=corrected_values,
            validation_notes=notes
        )
    
    def _calculate_expected_energy(self, diameter_km: float, velocity_km_s: float) -> float:
        """Calculate expected impact energy using E = 0.5 * m * v²"""
        # Calculate mass from diameter
        diameter_m = diameter_km * 1000
        volume_m3 = (4/3) * math.pi * (diameter_m/2)**3
        mass_kg = volume_m3 * self.TYPICAL_ASTEROID_DENSITY
        
        # Calculate kinetic energy
        velocity_m_s = velocity_km_s * 1000
        energy_joules = 0.5 * mass_kg * velocity_m_s**2
        
        # Convert to megatons
        energy_mt = energy_joules / self.TNT_JOULES_PER_MEGATON
        return energy_mt
    
    def _validate_risk_label_logic(self, collision_probability: float, risk_level: str) -> bool:
        """
        Validate risk label against collision probability using Calibration Plan Table 6.
        
        Risk Label Logic from Calibration Plan:
        - 0: Cleared
        - 0 < P < 1e-6: Negligible  
        - 1e-6 ≤ P < 1e-4: Very Low
        - 1e-4 ≤ P < 1e-2: Low
        - ≥1e-2: Elevated
        """
        if collision_probability == 0:
            return risk_level.lower() in ['cleared', 'negligible']
        elif 0 < collision_probability < 1e-6:
            return risk_level.lower() == 'negligible'
        elif 1e-6 <= collision_probability < 1e-4:
            return risk_level.lower() in ['very_low', 'very low']
        elif 1e-4 <= collision_probability < 1e-2:
            return risk_level.lower() == 'low'
        elif collision_probability >= 1e-2:
            return risk_level.lower() == 'elevated'
        else:
            return False
    
    def _get_correct_risk_level(self, collision_probability: float) -> str:
        """Get correct risk level for given collision probability."""
        if collision_probability == 0:
            return 'cleared'
        elif 0 < collision_probability < 1e-6:
            return 'negligible'
        elif 1e-6 <= collision_probability < 1e-4:
            return 'very_low'
        elif 1e-4 <= collision_probability < 1e-2:
            return 'low'
        elif collision_probability >= 1e-2:
            return 'elevated'
        else:
            return 'unknown'
    
    def validate_artificial_assessment(self, artificial_data: Dict[str, Any]) -> PhysicalValidationResult:
        """
        Validate artificial object assessment for consistency.
        
        Args:
            artificial_data: Artificial assessment data dictionary
            
        Returns:
            PhysicalValidationResult with validation status
        """
        issues = []
        warnings = []
        corrected_values = {}
        notes = []
        
        # Extract key values
        statistical_significance = artificial_data.get('statistical_significance')
        sigma_level = artificial_data.get('sigma_level')
        artificial_probability = artificial_data.get('artificial_probability')
        prior_rate = artificial_data.get('prior_artificial_rate', 0.001)  # Default 0.1%
        
        # 1. Sigma to p-value consistency
        if sigma_level and statistical_significance:
            expected_significance = self._sigma_to_significance(sigma_level)
            significance_ratio = abs(statistical_significance - expected_significance) / expected_significance
            
            if significance_ratio > 0.1:  # 10% tolerance
                issues.append(
                    f"Sigma/significance mismatch: {sigma_level:.1f}σ → expected {expected_significance:.1%}, "
                    f"got {statistical_significance:.1%}"
                )
                corrected_values['statistical_significance'] = expected_significance
                notes.append("Statistical significance corrected to match sigma level")
        
        # 2. Artificial probability bounds check
        if artificial_probability:
            if artificial_probability > prior_rate * 1000:  # More than 1000x prior
                warnings.append(
                    f"Very high artificial probability ({artificial_probability:.3%}) vs prior ({prior_rate:.3%})"
                )
            
            if artificial_probability > 0.5:  # More than 50% is suspicious
                issues.append(f"Unrealistic artificial probability: {artificial_probability:.1%}")
                corrected_values['artificial_probability'] = min(artificial_probability, prior_rate * 100)
                notes.append("Artificial probability capped at reasonable threshold")
        
        # 3. Classification consistency
        classification = artificial_data.get('classification', '').lower()
        is_artificial_claim = artificial_data.get('is_artificial', False)
        
        if is_artificial_claim and artificial_probability and artificial_probability < 0.5:
            issues.append(
                f"Classification 'is_artificial=True' with low probability ({artificial_probability:.3%})"
            )
            corrected_values['is_artificial'] = False
            notes.append("Artificial classification corrected based on probability threshold")
        
        # Determine overall status
        if issues:
            status = ValidationResult.FAIL
        elif warnings:
            status = ValidationResult.WARNING
        else:
            status = ValidationResult.PASS
            
        return PhysicalValidationResult(
            status=status,
            issues=issues,
            warnings=warnings,
            corrected_values=corrected_values,
            validation_notes=notes
        )
    
    def _sigma_to_significance(self, sigma: float) -> float:
        """Convert sigma level to two-sided statistical significance."""
        from aneos_core.utils.statistical_utils import sigma_to_p_value
        return sigma_to_p_value(sigma)
    
    def create_validation_report(self, validation_result: PhysicalValidationResult) -> str:
        """Create a human-readable validation report."""
        report = []
        report.append(f"🔍 Physical Sanity Validation: {validation_result.status.value.upper()}")
        
        if validation_result.issues:
            report.append("\n❌ Issues Found:")
            for issue in validation_result.issues:
                report.append(f"  • {issue}")
        
        if validation_result.warnings:
            report.append("\n⚠️  Warnings:")
            for warning in validation_result.warnings:
                report.append(f"  • {warning}")
        
        if validation_result.corrected_values:
            report.append("\n🔧 Corrected Values:")
            for key, value in validation_result.corrected_values.items():
                if isinstance(value, float):
                    if value < 0.001:
                        report.append(f"  • {key}: {value:.2e}")
                    else:
                        report.append(f"  • {key}: {value:.3f}")
                else:
                    report.append(f"  • {key}: {value}")
        
        if validation_result.validation_notes:
            report.append("\n📋 Validation Notes:")
            for note in validation_result.validation_notes:
                report.append(f"  • {note}")
        
        return "\n".join(report)


def validate_neo_analysis_output(analysis_result: Dict[str, Any]) -> PhysicalValidationResult:
    """
    Convenience function to validate a complete NEO analysis output.
    
    Args:
        analysis_result: Complete analysis result dictionary
        
    Returns:
        PhysicalValidationResult with overall validation status
    """
    validator = PhysicalSanityValidator()
    
    # Combine impact and artificial assessments
    combined_issues = []
    combined_warnings = []
    combined_corrections = {}
    combined_notes = []
    
    # Validate impact assessment if present
    impact_data = analysis_result.get('impact_assessment', {})
    if impact_data:
        # Add object diameter to impact data for validation
        orbital_elements = analysis_result.get('orbital_elements', {})
        if 'diameter' in orbital_elements:
            impact_data['object_diameter_km'] = orbital_elements['diameter']
        
        impact_validation = validator.validate_impact_assessment(impact_data)
        combined_issues.extend(impact_validation.issues)
        combined_warnings.extend(impact_validation.warnings)
        combined_corrections.update(impact_validation.corrected_values)
        combined_notes.extend(impact_validation.validation_notes)
    
    # Validate artificial assessment if present
    artificial_data = analysis_result.get('calibrated_assessment', {})
    if artificial_data:
        artificial_validation = validator.validate_artificial_assessment(artificial_data)
        combined_issues.extend(artificial_validation.issues)
        combined_warnings.extend(artificial_validation.warnings)
        combined_corrections.update(artificial_validation.corrected_values)
        combined_notes.extend(artificial_validation.validation_notes)
    
    # Determine overall status
    if combined_issues:
        status = ValidationResult.FAIL
    elif combined_warnings:
        status = ValidationResult.WARNING
    else:
        status = ValidationResult.PASS
    
    return PhysicalValidationResult(
        status=status,
        issues=combined_issues,
        warnings=combined_warnings,
        corrected_values=combined_corrections,
        validation_notes=combined_notes
    )