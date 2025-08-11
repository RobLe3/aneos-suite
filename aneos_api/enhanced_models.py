"""
Enhanced API models for scientific rigor validation.

These models extend the existing API models to include validation
and uncertainty information without breaking existing endpoints.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

class ValidationStageResultModel(BaseModel):
    """API model for validation stage results."""
    stage_number: int
    stage_name: str
    passed: bool
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    false_positive_reduction: float = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any] = {}
    processing_time_ms: float

class SpaceDebrisMatchModel(BaseModel):
    """API model for space debris matches."""
    catalog: str
    object_id: str
    object_name: str
    match_confidence: float = Field(..., ge=0.0, le=1.0)
    delta_v: float = Field(..., description="Delta-V in m/s")
    orbital_similarity: float = Field(..., ge=0.0, le=1.0)
    epoch_difference: float = Field(..., description="Epoch difference in days")
    match_criteria: Dict[str, Any] = {}

class StatisticalTestResultModel(BaseModel):
    """API model for statistical test results."""
    p_value: float = Field(..., ge=0.0, le=1.0)
    effect_size: float = Field(..., ge=0.0)
    confidence_interval: Tuple[float, float]
    is_significant: bool
    test_statistic: float
    test_type: str
    alpha_level: float = Field(default=0.05, ge=0.0, le=1.0)

class UncertaintyAnalysisModel(BaseModel):
    """API model for uncertainty analysis results."""
    overall_uncertainty: float = Field(..., ge=0.0, le=1.0)
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float] 
    indicator_uncertainties: Dict[str, float] = {}
    sensitivity_analysis: Dict[str, float] = {}
    uncertainty_breakdown: Dict[str, float] = {}
    analysis_method: str = Field(default="monte_carlo")

class ValidationResultModel(BaseModel):
    """API model for comprehensive validation results."""
    overall_validation_passed: bool
    overall_false_positive_probability: float = Field(..., ge=0.0, le=1.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    
    stage_results: List[ValidationStageResultModel]
    space_debris_matches: List[SpaceDebrisMatchModel] = []
    synthetic_population_percentile: float = Field(..., ge=0.0, le=100.0)
    
    recommendation: str = Field(..., regex="^(accept|reject|expert_review|unknown)$")
    expert_review_priority: str = Field(..., regex="^(low|medium|high|urgent)$")
    
    total_processing_time_ms: float
    validation_timestamp: datetime

class EnhancedAnalysisResponse(BaseModel):
    """
    Enhanced analysis response with validation and uncertainty data.
    
    This extends the basic analysis response while preserving
    all original fields for backward compatibility.
    """
    # Original analysis fields (preserved)
    designation: str
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall anomaly score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    classification: str
    processing_time_ms: float
    timestamp: datetime
    
    # Enhanced validation fields (new)
    validation_result: ValidationResultModel
    statistical_tests: Dict[str, StatisticalTestResultModel] = {}
    uncertainty_analysis: UncertaintyAnalysisModel
    
    # Enhancement metadata
    enhancement_timestamp: datetime
    enhancement_version: str = Field(default="1.0.0")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "designation": "2024 AB123",
                "overall_score": 0.75,
                "confidence": 0.85,
                "classification": "potentially_artificial",
                "processing_time_ms": 1250.5,
                "timestamp": "2025-08-06T15:30:00Z",
                "validation_result": {
                    "overall_validation_passed": True,
                    "overall_false_positive_probability": 0.12,
                    "overall_confidence": 0.88,
                    "stage_results": [],
                    "recommendation": "expert_review",
                    "expert_review_priority": "high",
                    "total_processing_time_ms": 2150.3,
                    "validation_timestamp": "2025-08-06T15:30:02Z"
                },
                "statistical_tests": {},
                "uncertainty_analysis": {
                    "overall_uncertainty": 0.08,
                    "confidence_interval_95": [0.65, 0.85],
                    "confidence_interval_99": [0.62, 0.88],
                    "analysis_method": "monte_carlo"
                },
                "enhancement_timestamp": "2025-08-06T15:30:02Z",
                "enhancement_version": "1.0.0"
            }
        }

class EnhancedBatchAnalysisResponse(BaseModel):
    """Response model for enhanced batch analysis."""
    batch_id: str
    results: List[EnhancedAnalysisResponse]
    summary_statistics: Dict[str, Any] = {}
    total_processing_time_ms: float
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Request models (reuse existing ones with optional enhancement flags)
class EnhancedAnalysisRequest(BaseModel):
    """Enhanced analysis request with validation options."""
    designation: str = Field(..., description="NEO designation (e.g., '2024 AB123')")
    force_refresh: bool = Field(default=False, description="Force refresh of cached data")
    include_raw_data: bool = Field(default=False, description="Include raw NEO data in response")
    include_indicators: bool = Field(default=True, description="Include individual indicator results")
    
    # Enhanced validation options
    enable_validation: bool = Field(default=True, description="Enable enhanced validation")
    validation_method: str = Field(default="full", regex="^(full|fast|statistical_only)$")
    uncertainty_method: str = Field(default="monte_carlo", regex="^(monte_carlo|bootstrap|analytical)$")
    monte_carlo_samples: Optional[int] = Field(default=None, ge=100, le=50000)
    
    class Config:
        schema_extra = {
            "example": {
                "designation": "2024 AB123",
                "force_refresh": False,
                "include_raw_data": False,
                "include_indicators": True,
                "enable_validation": True,
                "validation_method": "full",
                "uncertainty_method": "monte_carlo",
                "monte_carlo_samples": 10000
            }
        }

class ValidationSummaryResponse(BaseModel):
    """Summary statistics for multiple validation results."""
    total_analyses: int
    false_positive_statistics: Dict[str, Any]
    recommendations: Dict[str, int]
    overall_score_statistics: Dict[str, float]
    validation_system_status: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "total_analyses": 150,
                "false_positive_statistics": {
                    "mean_probability": 0.25,
                    "median_probability": 0.18,
                    "high_fp_risk_count": 12,
                    "low_fp_risk_count": 89
                },
                "recommendations": {
                    "accept": 89,
                    "expert_review": 45,
                    "reject": 16
                },
                "overall_score_statistics": {
                    "mean": 0.68,
                    "min": 0.12,
                    "max": 0.95
                },
                "validation_system_status": {
                    "validation_enabled": True,
                    "validation_available": True
                }
            }
        }