#!/usr/bin/env python3
"""
Consistency Validator for aNEOS Analysis

Implements validation rules per interim assessment to block contradictory results:
- Risk=high with P=0 contradictions
- Artificial flag conflicting with classification
- Physics violations (sub-km crater for 500m object)
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConsistencyViolation(Enum):
    RISK_PROBABILITY_CONTRADICTION = "risk_probability_contradiction"
    ARTIFICIAL_CLASSIFICATION_MISMATCH = "artificial_classification_mismatch"
    PHYSICS_VIOLATION = "physics_violation"
    SEVERE_INCONSISTENCY = "severe_inconsistency"


@dataclass
class ConsistencyResult:
    """Result of consistency validation."""
    is_valid: bool
    violations: List[ConsistencyViolation]
    errors: List[str]
    warnings: List[str]
    corrected_values: Dict[str, Any]
    blocked_report: bool = False


class ConsistencyValidator:
    """
    Validates analysis results for logical consistency.
    
    Blocks reports that have contradictory or nonsensical results.
    """
    
    # Risk level hierarchy (lowest to highest)
    RISK_LEVELS = ['cleared', 'negligible', 'very_low', 'low', 'moderate', 'high', 'extreme']
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_analysis_result(self, analysis_result: Dict[str, Any]) -> ConsistencyResult:
        """
        Validate complete analysis result for consistency.
        
        Args:
            analysis_result: Complete analysis result dictionary
            
        Returns:
            ConsistencyResult indicating validation status
        """
        violations = []
        errors = []
        warnings = []
        corrected_values = {}
        
        # 1. Risk/Probability consistency
        impact_data = analysis_result.get('impact_assessment', {})
        self._validate_risk_probability_consistency(
            impact_data, violations, errors, corrected_values
        )
        
        # 2. Artificial classification consistency  
        calibrated_data = analysis_result.get('calibrated_assessment', {})
        self._validate_artificial_classification_consistency(
            analysis_result, calibrated_data, violations, errors, corrected_values
        )
        
        # 3. Physics sanity checks
        self._validate_physics_consistency(
            impact_data, violations, errors, corrected_values
        )
        
        # 4. Cross-validation between different assessments
        self._validate_cross_assessment_consistency(
            analysis_result, violations, warnings, corrected_values
        )
        
        # Determine if report should be blocked
        severe_violations = [
            ConsistencyViolation.RISK_PROBABILITY_CONTRADICTION,
            ConsistencyViolation.ARTIFICIAL_CLASSIFICATION_MISMATCH,
            ConsistencyViolation.PHYSICS_VIOLATION
        ]
        
        has_severe_violations = any(v in severe_violations for v in violations)
        blocked = has_severe_violations
        
        if blocked:
            self.logger.error(f"Analysis result blocked due to consistency violations: {violations}")
        
        return ConsistencyResult(
            is_valid=(len(violations) == 0),
            violations=violations,
            errors=errors,
            warnings=warnings,
            corrected_values=corrected_values,
            blocked_report=blocked
        )
    
    def _validate_risk_probability_consistency(self, impact_data: Dict[str, Any], 
                                             violations: List, errors: List, 
                                             corrected_values: Dict) -> None:
        """Validate risk level matches probability."""
        if not impact_data:
            return
            
        collision_prob = impact_data.get('collision_probability', 0.0)
        risk_level = impact_data.get('risk_level', 'negligible').lower()
        
        # Critical check: P=0 must have risk ≤ negligible
        if collision_prob == 0.0 and risk_level not in ['cleared', 'negligible']:
            violations.append(ConsistencyViolation.RISK_PROBABILITY_CONTRADICTION)
            errors.append(
                f"BLOCKED: Risk level '{risk_level}' with zero collision probability "
                f"(P=0 must have risk ≤ negligible)"
            )
            corrected_values['risk_level'] = 'cleared'
            
        # General probability/risk consistency
        elif collision_prob > 0:
            expected_risk = self._get_expected_risk_level(collision_prob)
            if expected_risk != risk_level:
                # Allow one level tolerance
                expected_idx = self.RISK_LEVELS.index(expected_risk) if expected_risk in self.RISK_LEVELS else 0
                actual_idx = self.RISK_LEVELS.index(risk_level) if risk_level in self.RISK_LEVELS else 0
                
                if abs(expected_idx - actual_idx) > 1:
                    violations.append(ConsistencyViolation.RISK_PROBABILITY_CONTRADICTION)
                    errors.append(
                        f"Risk level mismatch: '{risk_level}' for P={collision_prob:.2e} "
                        f"(expected '{expected_risk}')"
                    )
                    corrected_values['risk_level'] = expected_risk
    
    def _validate_artificial_classification_consistency(self, analysis_result: Dict, 
                                                      calibrated_data: Dict,
                                                      violations: List, errors: List,
                                                      corrected_values: Dict) -> None:
        """Validate artificial flag matches classification."""
        is_artificial = analysis_result.get('is_artificial', False)
        classification = analysis_result.get('classification', '').lower()
        
        # Get calibrated probability if available
        calibrated_prob = calibrated_data.get('calibrated_artificial_probability', 0.0)
        
        # Check for contradictions
        artificial_classifications = ['artificial', 'highly_suspicious', 'suspicious']
        natural_classifications = ['natural', 'cleared', 'normal']
        
        if is_artificial and classification in natural_classifications:
            violations.append(ConsistencyViolation.ARTIFICIAL_CLASSIFICATION_MISMATCH)
            errors.append(
                f"BLOCKED: is_artificial=True conflicts with classification='{classification}'"
            )
            corrected_values['is_artificial'] = False
            
        elif not is_artificial and classification in artificial_classifications:
            violations.append(ConsistencyViolation.ARTIFICIAL_CLASSIFICATION_MISMATCH)
            errors.append(
                f"BLOCKED: is_artificial=False conflicts with classification='{classification}'"
            )
            corrected_values['classification'] = 'natural'
        
        # Check calibrated probability consistency
        if is_artificial and calibrated_prob < 0.5:
            violations.append(ConsistencyViolation.ARTIFICIAL_CLASSIFICATION_MISMATCH)
            errors.append(
                f"BLOCKED: is_artificial=True with low calibrated probability ({calibrated_prob:.1%})"
            )
            corrected_values['is_artificial'] = False
    
    def _validate_physics_consistency(self, impact_data: Dict[str, Any],
                                    violations: List, errors: List,
                                    corrected_values: Dict) -> None:
        """Validate physics makes sense."""
        if not impact_data:
            return
            
        # Get object size and crater size
        orbital_elements = impact_data.get('orbital_elements', {})
        object_diameter_km = orbital_elements.get('diameter') or impact_data.get('object_diameter_km')
        crater_diameter_km = impact_data.get('crater_diameter_km')
        
        if object_diameter_km and crater_diameter_km:
            object_diameter_m = object_diameter_km * 1000
            
            # Block nonsense cases: sub-km crater for 500m+ object
            if object_diameter_m >= 500 and crater_diameter_km < 1.0:
                violations.append(ConsistencyViolation.PHYSICS_VIOLATION)
                errors.append(
                    f"BLOCKED: {crater_diameter_km:.1f}km crater for {object_diameter_m:.0f}m object "
                    f"(500m+ objects should create >1km craters)"
                )
                # Use minimum realistic scaling
                corrected_crater = object_diameter_km * 10  # 10x scaling minimum
                corrected_values['crater_diameter_km'] = corrected_crater
            
            # Check for unreasonably small craters
            crater_ratio = crater_diameter_km / object_diameter_km
            if crater_ratio < 1.0:  # Crater smaller than impactor
                violations.append(ConsistencyViolation.PHYSICS_VIOLATION)
                errors.append(
                    f"BLOCKED: Crater ({crater_diameter_km:.2f}km) smaller than impactor ({object_diameter_km:.2f}km)"
                )
                corrected_values['crater_diameter_km'] = object_diameter_km * 3  # Minimum 3x
    
    def _validate_cross_assessment_consistency(self, analysis_result: Dict,
                                             violations: List, warnings: List,
                                             corrected_values: Dict) -> None:
        """Cross-validate different assessment components."""
        impact_data = analysis_result.get('impact_assessment', {})
        calibrated_data = analysis_result.get('calibrated_assessment', {})
        
        # Check threat level consistency
        threat_level = analysis_result.get('threat_level', '').lower()
        risk_level = impact_data.get('risk_level', 'negligible').lower()
        
        if threat_level == 'high' and risk_level in ['negligible', 'very_low']:
            warnings.append(
                f"Threat level '{threat_level}' inconsistent with risk level '{risk_level}'"
            )
        
        # Check if artificial detection makes physical sense
        is_artificial = analysis_result.get('is_artificial', False)
        collision_prob = impact_data.get('collision_probability', 0.0)
        
        if is_artificial and collision_prob > 1e-3:  # Very high impact probability
            warnings.append(
                f"Artificial object with unusually high Earth impact probability ({collision_prob:.2e})"
            )
    
    def _get_expected_risk_level(self, collision_probability: float) -> str:
        """Get expected risk level for given collision probability."""
        # Based on Calibration Plan Table 6 risk thresholds
        if collision_probability == 0:
            return 'cleared'
        elif 0 < collision_probability < 1e-6:
            return 'negligible'
        elif 1e-6 <= collision_probability < 1e-5:
            return 'very_low'
        elif 1e-5 <= collision_probability < 1e-4:
            return 'low'  
        elif 1e-4 <= collision_probability < 1e-3:
            return 'moderate'
        elif 1e-3 <= collision_probability < 1e-2:
            return 'high'
        elif collision_probability >= 1e-2:
            return 'extreme'
        else:
            return 'negligible'
    
    def create_validation_report(self, result: ConsistencyResult) -> str:
        """Create human-readable validation report."""
        report = []
        
        if result.blocked_report:
            report.append("🚫 ANALYSIS RESULT BLOCKED")
            report.append("Report generation prevented due to critical consistency violations.")
        else:
            report.append(f"✅ Consistency Validation: {'PASS' if result.is_valid else 'ISSUES FOUND'}")
        
        if result.errors:
            report.append("\n❌ Critical Errors:")
            for error in result.errors:
                report.append(f"  • {error}")
        
        if result.warnings:
            report.append("\n⚠️  Warnings:")
            for warning in result.warnings:
                report.append(f"  • {warning}")
                
        if result.corrected_values:
            report.append("\n🔧 Auto-Corrections Applied:")
            for key, value in result.corrected_values.items():
                report.append(f"  • {key}: {value}")
        
        if result.violations:
            violation_types = [v.value for v in result.violations]
            report.append(f"\n🔍 Violation Types: {', '.join(violation_types)}")
        
        return "\n".join(report)


def validate_analysis_consistency(analysis_result: Dict[str, Any]) -> ConsistencyResult:
    """
    Convenience function to validate analysis result consistency.
    
    Args:
        analysis_result: Complete analysis result dictionary
        
    Returns:
        ConsistencyResult with validation status
    """
    validator = ConsistencyValidator()
    return validator.validate_analysis_result(analysis_result)