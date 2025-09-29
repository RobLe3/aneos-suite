#!/usr/bin/env python3
"""
Impact-Enhanced Analysis Pipeline for aNEOS

This module extends the enhanced analysis pipeline to include comprehensive
impact probability assessment. It integrates seamlessly with the existing
pipeline architecture while adding critical impact risk evaluation capabilities.

Scientific Rationale:
Impact probability assessment is essential for planetary defense because:

1. **WHY**: Quantifies the actual threat level to Earth and human civilization
2. **WHERE**: Most important for Earth-crossing asteroids and recent discoveries
3. **WHEN**: Critical during close approaches and for short observation arcs

The impact assessment provides actionable intelligence for:
- Observation prioritization
- Mission planning for deflection
- Emergency preparedness
- Public communication about asteroid threats

Integration Strategy:
- Preserves all existing aNEOS functionality
- Adds impact assessment as optional enhancement
- Fails gracefully if impact calculation unavailable
- Provides comprehensive scientific documentation
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio

# aNEOS imports
from .enhanced_pipeline import EnhancedAnalysisPipeline
from .impact_probability import ImpactProbabilityCalculator, ImpactProbability
from ..data.models import OrbitalElements, CloseApproach, ImpactAssessment
from ..validation import EnhancedAnalysisResult

logger = logging.getLogger(__name__)

class ImpactEnhancedAnalysisResult:
    """
    Extended analysis result that includes comprehensive impact assessment.
    
    This class wraps the existing EnhancedAnalysisResult and adds impact
    probability analysis while maintaining full backward compatibility.
    """
    
    def __init__(self, 
                 enhanced_result: EnhancedAnalysisResult,
                 impact_assessment: Optional[ImpactAssessment] = None):
        """
        Initialize impact-enhanced analysis result.
        
        Args:
            enhanced_result: Original enhanced analysis result
            impact_assessment: Comprehensive impact probability assessment
        """
        self.enhanced_result = enhanced_result
        self.impact_assessment = impact_assessment
        
        # Preserve all original attributes for backward compatibility
        for attr in dir(enhanced_result):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(enhanced_result, attr))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format including impact assessment."""
        result_dict = self.enhanced_result.to_dict() if hasattr(self.enhanced_result, 'to_dict') else {}
        
        if self.impact_assessment:
            result_dict['impact_assessment'] = self.impact_assessment.to_dict()
        
        return result_dict
    
    def get_impact_summary(self) -> Dict[str, Any]:
        """Get concise impact assessment summary for display."""
        if not self.impact_assessment:
            return {"status": "impact_assessment_unavailable"}
        
        impact = self.impact_assessment
        
        return {
            "collision_probability": impact.collision_probability,
            "risk_level": impact.risk_level,
            "comparative_risk": impact.comparative_risk,
            "time_to_impact_years": impact.time_to_impact_years,
            "impact_energy_mt": impact.impact_energy_mt,
            "calculation_confidence": impact.calculation_confidence,
            "primary_risk_factors": impact.primary_risk_factors[:3],  # Top 3 factors
            "most_probable_impact_region": impact.most_probable_impact_region
        }

class ImpactEnhancedPipeline:
    """
    Pipeline that integrates impact probability assessment with aNEOS analysis.
    
    This pipeline provides comprehensive impact risk evaluation by combining:
    1. Standard aNEOS artificial object detection
    2. Enhanced validation with statistical rigor
    3. Impact probability calculation and risk assessment
    
    Scientific Integration Rationale:
    
    **Why integrate impact probability with artificial detection?**
    - Both natural and artificial objects can pose impact threats
    - Artificial objects may have unpredictable trajectories due to propulsion
    - Detection confidence affects impact probability uncertainty
    - Combined analysis provides complete threat assessment
    
    **Where impact assessment is most critical:**
    - Earth-crossing asteroids (ECAs) with recent discoveries
    - Objects with close approaches < 0.2 AU
    - Short observation arcs with high uncertainty
    - Objects showing unusual orbital characteristics
    
    **When to prioritize impact assessment:**
    - Newly discovered objects (first 30 days)
    - Before and after close approaches
    - When artificial probability > 10% (propulsion uncertainty)
    - For objects flagged as "highly suspicious" 
    """
    
    def __init__(self, enhanced_pipeline: EnhancedAnalysisPipeline):
        """
        Initialize impact-enhanced pipeline.
        
        Args:
            enhanced_pipeline: Existing enhanced analysis pipeline
        """
        self.enhanced_pipeline = enhanced_pipeline
        self.impact_calculator = ImpactProbabilityCalculator()
        self.logger = logging.getLogger(__name__)
        
        # Impact assessment configuration
        self.impact_config = {
            'enable_for_earth_crossers': True,
            'enable_for_close_approaches': True,
            'enable_for_artificial_objects': True,
            'enable_for_high_uncertainty': True,
            'close_approach_threshold_au': 0.2,
            'artificial_probability_threshold': 0.1,
            'short_arc_threshold_days': 30
        }
        
        self.logger.info("Impact-enhanced analysis pipeline initialized")
    
    async def analyze_neo_comprehensive(self, 
                                      designation: str,
                                      neo_data: Optional[Any] = None,
                                      include_impact_assessment: bool = True) -> ImpactEnhancedAnalysisResult:
        """
        Perform comprehensive NEO analysis including impact probability assessment.
        
        This method provides the complete aNEOS analysis suite with impact risk
        evaluation. It determines automatically when impact assessment is needed
        based on scientific criteria.
        
        Scientific Decision Logic for Impact Assessment:
        
        1. **ALWAYS assess impact if:** Object is Earth-crossing
        2. **HIGH PRIORITY assess if:** Close approach < 0.2 AU
        3. **MEDIUM PRIORITY assess if:** Short observation arc < 30 days
        4. **SPECIAL CASE assess if:** Artificial probability > 10%
        5. **SKIP assessment if:** No Earth-crossing orbit detected
        
        Args:
            designation: NEO designation
            neo_data: Optional NEO data (orbital elements, close approaches)
            include_impact_assessment: Whether to perform impact assessment
            
        Returns:
            ImpactEnhancedAnalysisResult with complete analysis and impact assessment
        """
        
        self.logger.info(f"Starting comprehensive analysis for {designation}")
        
        # Step 1: Run enhanced aNEOS analysis
        enhanced_result = await self.enhanced_pipeline.analyze_neo_with_validation(
            designation, neo_data
        )
        
        # Step 2: Determine if impact assessment is needed
        should_assess_impact = self._should_assess_impact(enhanced_result, neo_data)
        
        impact_assessment = None
        if include_impact_assessment and should_assess_impact:
            try:
                # Step 3: Perform impact probability assessment
                self.logger.info(f"Performing impact assessment for {designation}")
                impact_assessment = await self._calculate_impact_assessment(
                    designation, enhanced_result, neo_data
                )
                
                # Step 4: Log impact assessment results
                self._log_impact_assessment_results(designation, impact_assessment)
                
            except Exception as e:
                self.logger.warning(f"Impact assessment failed for {designation}: {e}")
                # Continue without impact assessment - don't break the analysis
        
        elif not should_assess_impact:
            self.logger.info(f"Impact assessment skipped for {designation} - not Earth-crossing")
        
        # Step 5: Create comprehensive result
        return ImpactEnhancedAnalysisResult(enhanced_result, impact_assessment)
    
    def _should_assess_impact(self, enhanced_result: EnhancedAnalysisResult, neo_data: Optional[Any]) -> bool:
        """
        Determine if impact assessment should be performed based on scientific criteria.
        
        Scientific Rationale:
        Impact assessment is computationally expensive and should be focused on
        objects that actually pose potential impact threats. This screening
        prevents wasted computation on obviously non-threatening objects.
        
        Decision Criteria:
        1. Earth-crossing orbit (fundamental requirement)
        2. Close approaches indicating Earth proximity
        3. High uncertainty (short arcs) that could hide impact potential
        4. Artificial objects with propulsive uncertainty
        
        Args:
            enhanced_result: Result from enhanced aNEOS analysis
            neo_data: Original NEO data
            
        Returns:
            bool: Whether impact assessment should be performed
        """
        
        # Extract orbital elements and close approaches from available data
        orbital_elements = self._extract_orbital_elements(enhanced_result, neo_data)
        close_approaches = self._extract_close_approaches(enhanced_result, neo_data)
        
        if not orbital_elements:
            self.logger.debug("No orbital elements available - skipping impact assessment")
            return False
        
        # Criterion 1: Earth-crossing orbit (MANDATORY)
        is_earth_crossing = self._is_earth_crossing_orbit(orbital_elements)
        if not is_earth_crossing:
            return False
        
        # Criterion 2: Close approaches within threshold
        has_close_approaches = self._has_significant_close_approaches(close_approaches)
        
        # Criterion 3: High uncertainty from short observation arc
        has_high_uncertainty = self._has_high_orbital_uncertainty(enhanced_result)
        
        # Criterion 4: Artificial object with propulsive uncertainty
        is_artificial_candidate = self._is_artificial_candidate(enhanced_result)
        
        # Decision: Assess if Earth-crossing AND any additional criterion
        should_assess = is_earth_crossing and (
            has_close_approaches or 
            has_high_uncertainty or 
            is_artificial_candidate or
            self.impact_config['enable_for_earth_crossers']  # Force for all ECAs
        )
        
        if should_assess:
            criteria_met = []
            if is_earth_crossing:
                criteria_met.append("Earth-crossing")
            if has_close_approaches:
                criteria_met.append("close approaches")
            if has_high_uncertainty:
                criteria_met.append("high uncertainty")
            if is_artificial_candidate:
                criteria_met.append("artificial candidate")
            
            self.logger.info(f"Impact assessment criteria met: {', '.join(criteria_met)}")
        
        return should_assess
    
    async def _calculate_impact_assessment(self, 
                                         designation: str,
                                         enhanced_result: EnhancedAnalysisResult, 
                                         neo_data: Optional[Any]) -> ImpactAssessment:
        """
        Calculate comprehensive impact probability assessment.
        
        This method coordinates the impact probability calculation with
        scientific rationale and error handling.
        
        Args:
            designation: NEO designation
            enhanced_result: Enhanced aNEOS analysis result
            neo_data: Original NEO data
            
        Returns:
            ImpactAssessment: Comprehensive impact probability analysis
        """
        
        # Extract data for impact calculation
        orbital_elements = self._extract_orbital_elements(enhanced_result, neo_data)
        close_approaches = self._extract_close_approaches(enhanced_result, neo_data)
        observation_arc_days = self._estimate_observation_arc_length(enhanced_result, neo_data)
        
        # Determine if object is artificial and with what probability
        is_artificial = getattr(enhanced_result, 'is_artificial', False)
        artificial_probability = getattr(enhanced_result, 'artificial_probability', 0.0)
        
        # Perform impact probability calculation
        impact_prob_result = self.impact_calculator.calculate_comprehensive_impact_probability(
            orbital_elements=orbital_elements,
            close_approaches=close_approaches,
            observation_arc_days=observation_arc_days,
            is_artificial=is_artificial,
            artificial_probability=artificial_probability
        )
        
        # Convert to ImpactAssessment model
        impact_assessment = ImpactAssessment(
            designation=designation,
            calculation_method=impact_prob_result.calculation_method,
            last_updated=impact_prob_result.last_updated,
            
            # Core metrics
            collision_probability=impact_prob_result.collision_probability,
            collision_probability_per_year=impact_prob_result.collision_probability_per_year,
            time_to_impact_years=impact_prob_result.time_to_impact_years,
            
            # Uncertainty
            probability_uncertainty=impact_prob_result.probability_uncertainty,
            calculation_confidence=impact_prob_result.calculation_confidence,
            data_arc_years=impact_prob_result.data_arc_years,
            
            # Physical assessment
            impact_energy_mt=impact_prob_result.impact_energy_mt,
            impact_velocity_km_s=impact_prob_result.impact_velocity_km_s,
            crater_diameter_km=impact_prob_result.crater_diameter_km,
            damage_radius_km=impact_prob_result.damage_radius_km,
            
            # Spatial/temporal
            impact_latitude_distribution=impact_prob_result.impact_latitude_distribution,
            most_probable_impact_region=impact_prob_result.most_probable_impact_region,
            impact_probability_by_decade=impact_prob_result.impact_probability_by_decade,
            peak_risk_period=impact_prob_result.peak_risk_period,
            
            # Special considerations
            keyhole_passages=impact_prob_result.keyhole_passages,
            artificial_object_considerations=impact_prob_result.artificial_object_considerations,
            
            # Scientific rationale
            primary_risk_factors=impact_prob_result.primary_risk_factors,
            calculation_assumptions=impact_prob_result.calculation_assumptions,
            limitations=impact_prob_result.limitations,
            
            # Risk classification
            risk_level=impact_prob_result.risk_level,
            comparative_risk=impact_prob_result.comparative_risk
        )
        
        return impact_assessment
    
    def _extract_orbital_elements(self, enhanced_result: EnhancedAnalysisResult, neo_data: Optional[Any]) -> Optional[OrbitalElements]:
        """Extract orbital elements from available data sources."""
        
        # Try to get from enhanced result first
        if hasattr(enhanced_result, 'orbital_elements'):
            elements = enhanced_result.orbital_elements
            if isinstance(elements, OrbitalElements):
                return elements
            elif isinstance(elements, dict):
                return OrbitalElements.from_dict(elements)
        
        # Try to get from neo_data
        if neo_data and hasattr(neo_data, 'orbital_elements'):
            elements = neo_data.orbital_elements
            if isinstance(elements, OrbitalElements):
                return elements
            elif isinstance(elements, dict):
                return OrbitalElements.from_dict(elements)
        
        # Try to extract from nested data structures
        if hasattr(enhanced_result, 'original_result'):
            orig = enhanced_result.original_result
            if hasattr(orig, 'orbital_elements'):
                elements = orig.orbital_elements
                if isinstance(elements, dict):
                    return OrbitalElements.from_dict(elements)
        
        return None
    
    def _extract_close_approaches(self, enhanced_result: EnhancedAnalysisResult, neo_data: Optional[Any]) -> List[CloseApproach]:
        """Extract close approach data from available sources."""
        
        close_approaches = []
        
        # Try enhanced result
        if hasattr(enhanced_result, 'close_approaches'):
            approaches = enhanced_result.close_approaches
            if isinstance(approaches, list):
                for approach in approaches:
                    if isinstance(approach, CloseApproach):
                        close_approaches.append(approach)
                    elif isinstance(approach, dict):
                        close_approaches.append(CloseApproach(**approach))
        
        # Try neo_data
        if neo_data and hasattr(neo_data, 'close_approaches'):
            approaches = neo_data.close_approaches
            if isinstance(approaches, list):
                for approach in approaches:
                    if isinstance(approach, CloseApproach):
                        close_approaches.append(approach)
                    elif isinstance(approach, dict):
                        close_approaches.append(CloseApproach(**approach))
        
        return close_approaches
    
    def _estimate_observation_arc_length(self, enhanced_result: EnhancedAnalysisResult, neo_data: Optional[Any]) -> float:
        """Estimate observation arc length in days."""
        
        # Try to get from data quality metrics
        if hasattr(enhanced_result, 'data_quality_score'):
            # Rough estimate: higher quality = longer arc
            quality = enhanced_result.data_quality_score
            if quality > 0.8:
                return 365.0  # High quality suggests year+ arc
            elif quality > 0.6:
                return 90.0   # Medium quality suggests ~3 months
            elif quality > 0.3:
                return 30.0   # Low quality suggests ~1 month
            else:
                return 7.0    # Very low quality suggests week or less
        
        # Default conservative estimate for new objects
        return 30.0
    
    def _is_earth_crossing_orbit(self, orbital_elements: OrbitalElements) -> bool:
        """Check if orbit crosses Earth's orbit."""
        
        if not orbital_elements.semi_major_axis or not orbital_elements.eccentricity:
            return False
        
        a = orbital_elements.semi_major_axis
        e = orbital_elements.eccentricity
        
        perihelion = a * (1 - e)
        aphelion = a * (1 + e)
        
        # Earth's orbit: ~0.98 to ~1.02 AU
        return perihelion < 1.02 and aphelion > 0.98
    
    def _has_significant_close_approaches(self, close_approaches: List[CloseApproach]) -> bool:
        """Check if there are significant close approaches."""
        
        threshold_au = self.impact_config['close_approach_threshold_au']
        
        for approach in close_approaches:
            if approach.distance_au and approach.distance_au < threshold_au:
                return True
        
        return False
    
    def _has_high_orbital_uncertainty(self, enhanced_result: EnhancedAnalysisResult) -> bool:
        """Check if orbital uncertainty is high (short observation arc)."""
        
        # Check data quality as proxy for uncertainty
        if hasattr(enhanced_result, 'data_quality_score'):
            return enhanced_result.data_quality_score < 0.5
        
        # If no quality data, assume high uncertainty for safety
        return True
    
    def _is_artificial_candidate(self, enhanced_result: EnhancedAnalysisResult) -> bool:
        """Check if object is a candidate artificial object."""
        
        threshold = self.impact_config['artificial_probability_threshold']
        
        if hasattr(enhanced_result, 'artificial_probability'):
            return enhanced_result.artificial_probability > threshold
        
        if hasattr(enhanced_result, 'is_artificial'):
            return enhanced_result.is_artificial
        
        return False
    
    def _log_impact_assessment_results(self, designation: str, impact_assessment: ImpactAssessment):
        """Log key impact assessment results for monitoring."""
        
        impact = impact_assessment
        
        # Log key metrics
        self.logger.info(f"Impact assessment complete for {designation}:")
        self.logger.info(f"  Collision probability: {impact.collision_probability:.2e}")
        self.logger.info(f"  Risk level: {impact.risk_level}")
        self.logger.info(f"  Calculation confidence: {impact.calculation_confidence:.2f}")
        
        # Log significant risks
        if impact.collision_probability > 1e-6:
            self.logger.warning(f"Elevated impact risk detected for {designation}: "
                              f"P={impact.collision_probability:.2e} ({impact.comparative_risk})")
        
        if impact.impact_energy_mt and impact.impact_energy_mt > 100:
            self.logger.warning(f"High-energy impact potential for {designation}: "
                              f"{impact.impact_energy_mt:.1f} MT")
        
        # Log artificial object considerations
        if impact.artificial_object_considerations:
            self.logger.info(f"Artificial object impact considerations apply for {designation}")
    
    # Convenience methods that preserve original pipeline interface
    async def analyze_neo(self, designation: str, neo_data: Optional[Any] = None) -> ImpactEnhancedAnalysisResult:
        """Convenience method with same signature as original pipeline."""
        return await self.analyze_neo_comprehensive(designation, neo_data, include_impact_assessment=True)
    
    async def analyze_neo_with_validation(self, designation: str, neo_data: Optional[Any] = None) -> ImpactEnhancedAnalysisResult:
        """Convenience method that always includes impact assessment."""
        return await self.analyze_neo_comprehensive(designation, neo_data, include_impact_assessment=True)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of impact-enhanced pipeline."""
        
        base_status = self.enhanced_pipeline.get_validation_status()
        
        impact_status = {
            'impact_calculator_available': self.impact_calculator is not None,
            'impact_assessment_config': self.impact_config,
            'components': {
                'enhanced_pipeline': True,
                'impact_probability_calculator': True,
                'orbital_elements_extraction': True,
                'close_approach_analysis': True,
                'keyhole_analysis': True,
                'artificial_object_considerations': True
            }
        }
        
        return {**base_status, 'impact_enhancement': impact_status}

# Factory function for easy integration
def create_impact_enhanced_pipeline(original_pipeline: Any) -> ImpactEnhancedPipeline:
    """
    Factory function to create an impact-enhanced pipeline from an original pipeline.
    
    This function provides the recommended way to add impact assessment capabilities
    to any existing aNEOS pipeline.
    
    Args:
        original_pipeline: Original aNEOS pipeline instance
        
    Returns:
        ImpactEnhancedPipeline: Pipeline with comprehensive impact assessment
    """
    
    # Create enhanced pipeline if not already enhanced
    if isinstance(original_pipeline, EnhancedAnalysisPipeline):
        enhanced_pipeline = original_pipeline
    else:
        enhanced_pipeline = EnhancedAnalysisPipeline(original_pipeline)
    
    # Create impact-enhanced pipeline
    return ImpactEnhancedPipeline(enhanced_pipeline)