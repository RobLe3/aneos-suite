"""Enhanced REST transport: screening, review prerequisites and missing inference.

No installed calibrated scientific model is implied by an available endpoint.
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ScientificAPIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, allow_inf_nan=False)


class ValidationStageResultModel(ScientificAPIModel):
    stage_number: int
    stage_name: str
    passed: bool
    score: Optional[float] = Field(None, ge=0., le=1.)
    confidence: Optional[float] = Field(None, ge=0., le=1.)
    false_positive_reduction: Optional[float] = Field(None, ge=0., le=1.)
    details: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: Optional[float] = None
    timing_status: str = "unavailable_not_instrumented"
    validation_status: Optional[str] = None


class SpaceDebrisMatchModel(ScientificAPIModel):
    catalog: str
    object_id: str
    object_name: str
    match_confidence: float = Field(..., ge=0., le=1.)
    delta_v: float
    orbital_similarity: float = Field(..., ge=0., le=1.)
    epoch_difference: float
    match_criteria: Dict[str, Any] = Field(default_factory=dict)


class StatisticalTestResultModel(ScientificAPIModel):
    p_value: Optional[float] = Field(None, ge=0., le=1.)
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    is_significant: Optional[bool] = None
    test_statistic: Optional[float] = None
    test_type: Optional[str] = None
    alpha_level: Optional[float] = Field(None, ge=0., le=1.)


class UncertaintyAnalysisModel(ScientificAPIModel):
    status: str = 'unavailable'
    reason: Optional[str] = None
    overall_uncertainty: Optional[float] = Field(None, ge=0., le=1.)
    confidence_interval_95: Optional[Tuple[float, float]] = None
    confidence_interval_99: Optional[Tuple[float, float]] = None
    indicator_uncertainties: Dict[str, float] = Field(default_factory=dict)
    sensitivity_analysis: Dict[str, float] = Field(default_factory=dict)
    uncertainty_breakdown: Dict[str, float] = Field(default_factory=dict)
    analysis_method: Optional[str] = None


class ValidationResultModel(ScientificAPIModel):
    overall_validation_passed: bool
    overall_false_positive_probability: Optional[float] = Field(None, ge=0., le=1.)
    overall_confidence: Optional[float] = Field(None, ge=0., le=1.)
    stage_results: List[ValidationStageResultModel] = Field(default_factory=list)
    space_debris_matches: List[SpaceDebrisMatchModel] = Field(default_factory=list)
    synthetic_population_percentile: Optional[float] = Field(None, ge=0., le=100.)
    statistical_significance_summary: Optional[Dict[str, Any]] = None
    recommendation: str
    expert_review_priority: str = Field(..., pattern='^(low|medium|high|urgent)$')
    total_processing_time_ms: Optional[float] = None
    timing_status: str = "unavailable_not_instrumented"
    validation_timestamp: datetime
    contract_version: Optional[str] = None
    validation_status: Optional[str] = None
    evidence_id: Optional[str] = None
    policy_id: Optional[str] = None
    validation_evidence_id: Optional[str] = None
    screening_score: Optional[float] = None
    non_gravitational_evidence: Optional[Dict[str, Any]] = None
    transport_limitations: List[str] = Field(default_factory=list)


class EnhancedAnalysisResponse(ScientificAPIModel):
    designation: str
    overall_score: Optional[float] = Field(None, ge=0., description='Uncalibrated ranking units, not sigma or probability')
    score_semantics: str = 'uncalibrated_screening_rank'
    confidence: Optional[float] = Field(None, ge=0., le=1.)
    classification: str
    status: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float
    timestamp: datetime
    screening_assessment: Optional[Dict[str, Any]] = None
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_neo_data: Optional[Dict[str, Any]] = None
    indicator_scores: Dict[str, Any] = Field(default_factory=dict)
    validation_result: Optional[ValidationResultModel] = None
    statistical_tests: Dict[str, StatisticalTestResultModel] = Field(default_factory=dict)
    uncertainty_analysis: UncertaintyAnalysisModel = Field(default_factory=UncertaintyAnalysisModel)
    execution_options: Dict[str, Any] = Field(default_factory=dict)
    enhancement_timestamp: Optional[datetime] = None
    enhancement_version: str = 'enhanced-rest-v1'


class EnhancedBatchAnalysisResponse(ScientificAPIModel):
    batch_id: str
    results: List[EnhancedAnalysisResponse]
    summary_statistics: Dict[str, Any] = Field(default_factory=dict)
    total_processing_time_ms: float
    timestamp: datetime


class EnhancedAnalysisRequest(ScientificAPIModel):
    designation: str = Field(..., min_length=1, max_length=50)
    force_refresh: bool = False
    include_raw_data: bool = False
    include_indicators: bool = True
    enable_validation: bool = True
    validation_method: str = Field('full', pattern='^(full|fast|statistical_only)$')
    uncertainty_method: str = Field('monte_carlo', pattern='^(monte_carlo|bootstrap|analytical)$')
    monte_carlo_samples: Optional[int] = Field(None, ge=100, le=50000)

    @field_validator('designation')
    @classmethod
    def validate_designation(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('Designation must not be blank')
        return value


class ValidationSummaryResponse(ScientificAPIModel):
    total_analyses: int
    included_screening_assessments: int = 0
    excluded_rows: Dict[str, int] = Field(default_factory=dict)
    screening_counts: Dict[str, int] = Field(default_factory=dict)
    policy_groups: Dict[str, Any] = Field(default_factory=dict)
    false_positive_statistics: Dict[str, Any]
    recommendations: Dict[str, int]
    overall_score_statistics: Dict[str, Any]
    validation_system_status: Dict[str, Any]
