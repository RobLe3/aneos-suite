"""
Enhanced Analysis Pipeline for aNEOS Scientific Rigor Enhancement.

This module provides a wrapper around the existing analysis pipeline that
adds scientific rigor validation without modifying the original analysis logic.
Uses additive architecture principles to preserve all existing functionality.

Key Features:
- Wraps existing AnalysisPipeline without modification
- Adds 5-stage validation pipeline for false positive prevention
- Provides statistical testing and uncertainty quantification
- Maintains backward compatibility with all existing code
- Optional enhanced functionality that gracefully degrades
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import asyncio

from ..validation import (
    MultiStageValidator, 
    EnhancedAnalysisResult,
    StatisticalTesting,
    UncertaintyAnalysis
)
from .advanced_scoring import AdvancedScoreCalculator, AdvancedAnomalyScore
from ..config.settings import WeightConfig, ThresholdConfig
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedAnalysisPipeline:
    """
    Enhanced wrapper around existing AnalysisPipeline with scientific rigor validation.
    
    This class wraps the original analysis pipeline and adds comprehensive
    validation capabilities while preserving all original functionality.
    The enhanced features are additive and fail gracefully if unavailable.
    """
    
    def __init__(
        self, 
        original_pipeline: Any,
        validation_config: Optional[Dict[str, Any]] = None,
        enable_validation: bool = True
    ):
        """
        Initialize enhanced analysis pipeline wrapper.
        
        Args:
            original_pipeline: Original aNEOS AnalysisPipeline instance
            validation_config: Optional configuration for validation system
            enable_validation: Whether to enable validation features (default: True)
        """
        self.original_pipeline = original_pipeline
        self.enable_validation = enable_validation
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation components if enabled
        if self.enable_validation:
            try:
                self.validator = MultiStageValidator(config=validation_config)
                self.statistical_testing = StatisticalTesting()
                self.uncertainty_analysis = UncertaintyAnalysis()
                
                # Initialize XVIII SWARM advanced scoring system
                config_path = Path(__file__).parent.parent / "config" / "advanced_scoring_weights.json"
                self.advanced_scorer = AdvancedScoreCalculator(config_path)
                
                self.validation_available = True
            except Exception as e:
                logger.warning("Enhanced validation init failed: %s — features disabled", e)
                self.validation_available = False
        else:
            self.validation_available = False
    
    async def analyze_neo(
        self, 
        designation: str, 
        neo_data: Optional[Any] = None,
        enhanced: bool = True
    ) -> Union[Any, EnhancedAnalysisResult]:
        """
        Analyze NEO with optional enhanced validation.
        
        This method preserves the exact signature and behavior of the original
        analyze_neo method while optionally adding enhanced validation.
        
        Args:
            designation: NEO designation (same as original)
            neo_data: Optional NEO data (same as original) 
            enhanced: Whether to apply enhanced validation (default: True)
            
        Returns:
            Original analysis result OR EnhancedAnalysisResult if validation enabled
        """
        try:
            # Step 1: Run original analysis (COMPLETELY UNCHANGED)
            self.logger.info(f"Running original analysis for {designation}")
            original_result = await self.original_pipeline.analyze_neo(designation, neo_data)
            
            # Step 2: Apply enhanced validation if enabled and requested
            if enhanced and self.validation_available and original_result:
                try:
                    self.logger.info(f"Applying enhanced validation for {designation}")
                    enhanced_result = await self.validator.validate_analysis_result(
                        neo_data if neo_data is not None else getattr(original_result, 'neo_data', None), original_result
                    )
                    
                    # Log validation results
                    fp_prob = enhanced_result.validation_result.overall_false_positive_probability
                    recommendation = enhanced_result.validation_result.recommendation
                    self.logger.info(
                        f"Validation complete for {designation}: "
                        f"FP probability: {fp_prob if fp_prob is not None else 'unavailable'}, Recommendation: {recommendation}"
                    )
                    
                    return enhanced_result
                    
                except Exception as e:
                    self.logger.warning(f"Enhanced validation failed for {designation}: {e}")
                    self.logger.warning("Returning explicit unavailable validation with original result preserved")
                    # Preserve explicit unavailability; absence is never a successful gate.
                    return self._create_basic_enhanced_result(original_result)
            else:
                # Explicitly disabled/unrequested enhancement preserves original behavior.
                if enhanced and self.enable_validation:
                    return self._create_basic_enhanced_result(original_result)
                return original_result
                
        except Exception as e:
            self.logger.error(f"Analysis failed for {designation}: {e}")
            # Let original error propagate - don't modify error handling behavior
            raise
    
    async def analyze_neo_with_validation(
        self, 
        designation: str, 
        neo_data: Optional[Any] = None
    ) -> EnhancedAnalysisResult:
        """
        Convenience method that always applies enhanced validation.
        
        This method guarantees enhanced validation will be applied,
        falling back to basic enhancement if full validation fails.
        
        Args:
            designation: NEO designation
            neo_data: Optional NEO data
            
        Returns:
            EnhancedAnalysisResult with validation data
        """
        result = await self.analyze_neo(designation, neo_data, enhanced=True)
        
        if isinstance(result, EnhancedAnalysisResult):
            return result
        else:
            # Create basic enhanced result if validation wasn't applied
            return self._create_basic_enhanced_result(result)
    
    async def analyze_neo_original(
        self, 
        designation: str, 
        neo_data: Optional[Any] = None
    ) -> Any:
        """
        Run original analysis without any enhancement.
        
        This method provides access to the original analysis behavior
        exactly as it was before enhancement.
        
        Args:
            designation: NEO designation
            neo_data: Optional NEO data
            
        Returns:
            Original analysis result (unchanged)
        """
        return await self.original_pipeline.analyze_neo(designation, neo_data)
    
    def get_validation_status(self) -> Dict[str, Any]:
        """
        Get status of enhanced validation system.
        
        Returns:
            Dict with validation system status information
        """
        return {
            'validation_enabled': self.enable_validation,
            'validation_available': self.validation_available,
            'availability_semantics': 'component_initialized_not_scientific_validation',
            'scientific_validation_status': 'unavailable_uncalibrated',
            'components': {
                'multi_stage_validator': hasattr(self, 'validator') and self.validator is not None,
                'statistical_testing': hasattr(self, 'statistical_testing') and self.statistical_testing is not None,
                'uncertainty_analysis': hasattr(self, 'uncertainty_analysis') and self.uncertainty_analysis is not None
            },
            'original_pipeline_type': type(self.original_pipeline).__name__
        }
    
    def _create_basic_enhanced_result(self, original_result: Any) -> EnhancedAnalysisResult:
        """
        Create a basic enhanced result when full validation is unavailable.
        
        This provides minimal enhancement structure while preserving the original result.
        """
        from ..validation.multi_stage_validator import ValidationResult, ValidationStageResult
        from ..validation.statistical_testing import MultipleTestingResult
        
        assessment = None
        original_payload = (getattr(original_result, 'analysis', {}) or {}).get('assessment')
        if isinstance(original_payload, dict) and original_payload.get('schema_version') == 'screening-assessment-v1':
            try:
                from .screening import ScreeningAnalysisService
                replay = ScreeningAnalysisService().replay(original_payload['evidence_snapshot'], original_payload['evidence_id'])
                if replay.policy_id == original_payload['policy_id']:
                    assessment = replay.to_payload()
            except (KeyError, TypeError, ValueError):
                pass  # Preserve unavailable inference; never trust a malformed legacy claim.

        # Create minimal validation result
        minimal_validation = ValidationResult(
            overall_validation_passed=False,  # Unavailable validation cannot grant a pass
            overall_false_positive_probability=None,  # Not estimated
            overall_confidence=None,  # Not estimated
            stage_results=[],
            space_debris_matches=[],
            synthetic_population_percentile=None,
            statistical_significance_summary=None,
            recommendation='additional_evidence_required',
            contract_version='required-review-v1', validation_status='unavailable',
            expert_review_priority='medium',
            total_processing_time_ms=0.0,
            validation_timestamp=datetime.now()
        )
        
        return EnhancedAnalysisResult(
            original_result=original_result,
            validation_result=minimal_validation,
            statistical_tests={},
            uncertainty_analysis={'status': 'unavailable'},
            enhancement_timestamp=datetime.now(),
            enhancement_version="minimal", screening_assessment=assessment
        )
    
    # Proxy all other methods to original pipeline
    def __getattr__(self, name: str) -> Any:
        """
        Proxy unknown attributes/methods to the original pipeline.
        
        This ensures complete backward compatibility - any method or attribute
        that doesn't exist on the enhanced wrapper is automatically forwarded
        to the original pipeline.
        """
        return getattr(self.original_pipeline, name)
    
    async def analyze_neo_with_advanced_scoring(self, designation, neo_data=None):
        """Compatibility wrapper: unavailable stages cannot become measured clues."""
        enhanced = await self.analyze_neo_with_validation(designation, neo_data)
        return {'designation': designation, 'enhanced_analysis': enhanced,
            'screening_assessment': enhanced.screening_assessment,
            'advanced_score': None, 'advanced_scoring_status': 'unavailable_scientific_prerequisites',
            'scoring_explanation': 'Legacy stage-to-clue fusion withdrawn: observational and calibration prerequisites unavailable',
            'xviii_swarm_version': 'legacy_fusion_withdrawn'}

    # Additional convenience methods for enhanced functionality
    
    async def bulk_analyze_with_validation(
        self, 
        designations: List[str],
        neo_data_list: Optional[List[Any]] = None
    ) -> List[EnhancedAnalysisResult]:
        """
        Bulk analysis with enhanced validation for multiple NEOs.
        
        Args:
            designations: List of NEO designations
            neo_data_list: Optional list of NEO data objects
            
        Returns:
            List of EnhancedAnalysisResult objects
        """
        results = []
        
        if neo_data_list is None:
            neo_data_list = [None] * len(designations)
        
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent analyses
        
        async def analyze_single(designation, neo_data):
            async with semaphore:
                return await self.analyze_neo_with_validation(designation, neo_data)
        
        # Create tasks for parallel execution
        tasks = [
            analyze_single(designation, neo_data)
            for designation, neo_data in zip(designations, neo_data_list)
        ]
        
        # Execute all analyses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Bulk analysis failed for {designations[i]}: {result}")
                results[i] = self._create_basic_enhanced_result(None)
        
        return results
    
    def get_validation_summary(
        self, 
        enhanced_results: List[EnhancedAnalysisResult]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from multiple enhanced analysis results.
        
        Args:
            enhanced_results: List of enhanced analysis results
            
        Returns:
            Dict with summary statistics
        """
        if not enhanced_results:
            return {'error': 'No results provided'}
        
        try:
            # Extract key metrics
            fp_probabilities = []
            recommendations = {'accept': 0, 'reject': 0, 'expert_review': 0, 'unknown': 0}
            overall_scores = []
            
            for result in enhanced_results:
                if hasattr(result, 'validation_result'):
                    fp_prob = result.validation_result.overall_false_positive_probability
                    recommendation = result.validation_result.recommendation
                    
                    if fp_prob is not None:
                        fp_probabilities.append(fp_prob)
                    recommendations[recommendation] = recommendations.get(recommendation, 0) + 1
                    
                    if hasattr(result, 'overall_score'):
                        overall_scores.append(result.overall_score)
            
            # Calculate summary statistics
            summary = {
                'total_analyses': len(enhanced_results),
                'false_positive_statistics': {
                    'available_count': len(fp_probabilities),
                    'unavailable_count': len(enhanced_results) - len(fp_probabilities),
                    'mean_probability': sum(fp_probabilities) / len(fp_probabilities) if fp_probabilities else None,
                    'median_probability': sorted(fp_probabilities)[len(fp_probabilities)//2] if fp_probabilities else None,
                    'high_fp_risk_count': sum(1 for fp in fp_probabilities if fp > 0.5),
                    'low_fp_risk_count': sum(1 for fp in fp_probabilities if fp < 0.1)
                },
                'recommendations': recommendations,
                'overall_score_statistics': {
                    'mean': sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
                    'min': min(overall_scores) if overall_scores else 0.0,
                    'max': max(overall_scores) if overall_scores else 0.0
                },
                'validation_system_status': self.get_validation_status()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Validation summary generation failed: {e}")
            return {'error': str(e)}

# Convenience function for creating enhanced pipeline
def create_enhanced_pipeline(
    original_pipeline: Any,
    validation_config: Optional[Dict[str, Any]] = None,
    enable_validation: bool = True
) -> EnhancedAnalysisPipeline:
    """
    Create an enhanced analysis pipeline wrapper.
    
    Args:
        original_pipeline: Original aNEOS AnalysisPipeline instance
        validation_config: Optional validation configuration
        enable_validation: Whether to enable enhanced validation
        
    Returns:
        EnhancedAnalysisPipeline instance
    """
    return EnhancedAnalysisPipeline(
        original_pipeline=original_pipeline,
        validation_config=validation_config,
        enable_validation=enable_validation
    )