"""
API data models and schemas for aNEOS REST services.

This module defines Pydantic models for request/response serialization
and validation for all API endpoints.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

try:
    from pydantic import BaseModel, Field, validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback base class
    class BaseModel:
        def dict(self):
            return self.__dict__

# Base API Response Models
class APIResponse(BaseModel):
    """Base API response model."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[float] = None

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List[Any]
    total: int
    page: int = 1
    page_size: int = 50
    total_pages: int
    
    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values):
        total = values.get('total', 0)
        page_size = values.get('page_size', 50)
        return max(1, (total + page_size - 1) // page_size)

# Analysis Models
class AnalysisRequest(BaseModel):
    """Request model for NEO analysis."""
    designation: str = Field(..., description="NEO designation (e.g., '2024 AB123')")
    force_refresh: bool = Field(False, description="Force refresh of cached data")
    include_raw_data: bool = Field(False, description="Include raw NEO data in response")
    include_indicators: bool = Field(True, description="Include individual indicator results")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch NEO analysis."""
    designations: List[str] = Field(..., min_items=1, max_items=100, description="List of NEO designations")
    force_refresh: bool = Field(False, description="Force refresh of cached data")
    include_raw_data: bool = Field(False, description="Include raw NEO data in responses")
    progress_webhook: Optional[str] = Field(None, description="Webhook URL for progress updates")

class OrbitalElementsResponse(BaseModel):
    """Orbital elements response model."""
    eccentricity: Optional[float] = None
    inclination: Optional[float] = None
    semi_major_axis: Optional[float] = None
    ascending_node: Optional[float] = None
    argument_of_perihelion: Optional[float] = None
    mean_anomaly: Optional[float] = None
    epoch: Optional[datetime] = None
    orbital_period: Optional[float] = None

class CloseApproachResponse(BaseModel):
    """Close approach response model."""
    close_approach_date: Optional[datetime] = None
    distance_au: Optional[float] = None
    distance_km: Optional[float] = None
    relative_velocity_km_s: Optional[float] = None
    infinity_velocity_km_s: Optional[float] = None
    subpoint: Optional[List[float]] = Field(None, description="[latitude, longitude]")

class IndicatorResultResponse(BaseModel):
    """Indicator result response model."""
    indicator_name: str
    raw_score: float
    weighted_score: float
    confidence: float
    contributing_factors: List[str] = []
    metadata: Dict[str, Any] = {}

class AnomalyScoreResponse(BaseModel):
    """Anomaly score response model."""
    overall_score: float
    confidence: float
    classification: str = Field(..., description="natural, suspicious, highly_suspicious, or artificial")
    risk_factors: List[str] = []
    indicator_scores: Dict[str, IndicatorResultResponse] = {}
    statistical_summary: Dict[str, Any] = {}

class AnalysisResponse(APIResponse):
    """Complete analysis response model."""
    designation: str
    anomaly_score: AnomalyScoreResponse
    processing_time: float
    data_quality: Dict[str, Any]
    orbital_elements: Optional[OrbitalElementsResponse] = None
    close_approaches: List[CloseApproachResponse] = []
    raw_neo_data: Optional[Dict[str, Any]] = None

# ML Prediction Models
class PredictionRequest(BaseModel):
    """Request model for ML prediction."""
    designation: str = Field(..., description="NEO designation")
    use_cache: bool = Field(True, description="Use cached predictions if available")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")

class BatchPredictionRequest(BaseModel):
    """Request model for batch ML prediction."""
    designations: List[str] = Field(..., min_items=1, max_items=50)
    use_cache: bool = Field(True, description="Use cached predictions if available")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")

class FeatureContributionResponse(BaseModel):
    """Feature contribution response model."""
    feature_name: str
    contribution: float
    feature_value: float

class PredictionResponse(APIResponse):
    """ML prediction response model."""
    designation: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_probability: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_id: str
    feature_contributions: List[FeatureContributionResponse] = []
    model_predictions: Dict[str, float] = Field({}, description="Individual model predictions in ensemble")

# Training Models
class TrainingRequest(BaseModel):
    """Request model for model training."""
    designations: List[str] = Field(..., min_items=50, description="NEO designations for training")
    model_types: List[str] = Field(["isolation_forest"], description="Model types to train")
    use_ensemble: bool = Field(True, description="Create ensemble model")
    hyperparameter_optimization: bool = Field(True, description="Optimize hyperparameters")
    validation_split: float = Field(0.2, ge=0.1, le=0.4, description="Validation split ratio")

class TrainingResponse(APIResponse):
    """Training response model."""
    session_id: str
    status: str = Field(..., description="training, completed, failed")
    models_trained: List[str] = []
    training_score: Optional[float] = None
    validation_score: Optional[float] = None
    training_time: Optional[float] = None
    model_paths: List[str] = []

# Monitoring Models
class SystemMetricsResponse(BaseModel):
    """System metrics response model."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int

class AnalysisMetricsResponse(BaseModel):
    """Analysis metrics response model."""
    timestamp: datetime
    total_analyses: int
    successful_analyses: int
    failed_analyses: int
    average_processing_time: float
    cache_hit_rate: float
    anomaly_detection_rate: float

class MLMetricsResponse(BaseModel):
    """ML metrics response model."""
    timestamp: datetime
    model_predictions: int
    prediction_latency: float
    feature_quality: float
    ensemble_agreement: float
    alert_count: int

class AlertResponse(BaseModel):
    """Alert response model."""
    alert_id: str
    alert_type: str
    alert_level: str = Field(..., description="low, medium, high, critical")
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    data: Dict[str, Any] = {}

class MetricsResponse(APIResponse):
    """Comprehensive metrics response model."""
    system_metrics: Optional[SystemMetricsResponse] = None
    analysis_metrics: Optional[AnalysisMetricsResponse] = None
    ml_metrics: Optional[MLMetricsResponse] = None
    recent_alerts: List[AlertResponse] = []
    performance_summary: Dict[str, Any] = {}

# Admin Models
class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

class UserResponse(BaseModel):
    """User response model."""
    user_id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True

class CreateUserRequest(BaseModel):
    """Create user request model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.VIEWER

class ConfigResponse(BaseModel):
    """Configuration response model."""
    api_config: Dict[str, Any]
    analysis_config: Dict[str, Any]
    ml_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]

class SystemStatusResponse(APIResponse):
    """System status response model."""
    status: str = Field(..., description="healthy, degraded, critical")
    uptime_seconds: float
    services: Dict[str, bool]
    version: str
    deployment_info: Dict[str, Any] = {}

# Streaming Models
class StreamingEventType(str, Enum):
    """Streaming event types."""
    ANALYSIS_COMPLETE = "analysis_complete"
    PREDICTION_COMPLETE = "prediction_complete"
    ALERT_GENERATED = "alert_generated"
    SYSTEM_STATUS = "system_status"
    METRICS_UPDATE = "metrics_update"

class StreamingEvent(BaseModel):
    """Streaming event model."""
    event_type: StreamingEventType
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any]
    session_id: Optional[str] = None

# Search and Filter Models
class SortOrder(str, Enum):
    """Sort order enumeration."""
    ASC = "asc"
    DESC = "desc"

class AnalysisFilter(BaseModel):
    """Analysis results filter model."""
    classification: Optional[List[str]] = Field(None, description="Filter by classification")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    designations: Optional[List[str]] = None

class SearchRequest(BaseModel):
    """Search request model."""
    query: Optional[str] = Field(None, description="Search query")
    filters: Optional[AnalysisFilter] = None
    sort_by: str = Field("timestamp", description="Field to sort by")
    sort_order: SortOrder = SortOrder.DESC
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=100)

# Export Models
class ExportFormat(str, Enum):
    """Export format enumeration."""
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PDF = "pdf"

class ExportRequest(BaseModel):
    """Export request model."""
    format: ExportFormat = ExportFormat.JSON
    filters: Optional[AnalysisFilter] = None
    include_raw_data: bool = Field(False, description="Include raw NEO data")
    date_range: Optional[int] = Field(None, description="Days of data to export")

class ExportResponse(APIResponse):
    """Export response model."""
    export_id: str
    format: ExportFormat
    status: str = Field(..., description="processing, completed, failed")
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    expires_at: Optional[datetime] = None

# Validation helpers
if HAS_PYDANTIC:
    # Add custom validators
    
    @validator('designation', pre=True)
    def validate_designation(cls, v):
        """Validate NEO designation format."""
        if not isinstance(v, str):
            raise ValueError('Designation must be a string')
        
        # Basic validation - could be enhanced with regex
        if len(v.strip()) < 3:
            raise ValueError('Designation too short')
        
        return v.strip().upper()

# Response model registry for OpenAPI documentation
RESPONSE_MODELS = {
    'AnalysisResponse': AnalysisResponse,
    'PredictionResponse': PredictionResponse,
    'TrainingResponse': TrainingResponse,
    'MetricsResponse': MetricsResponse,
    'SystemStatusResponse': SystemStatusResponse,
    'ErrorResponse': ErrorResponse,
    'PaginatedResponse': PaginatedResponse
}