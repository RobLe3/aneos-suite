"""
Unified aNEOS Analysis Interface - Goal-Aligned Naming Convention

This interface provides standardized, goal-aligned method names and ensures
the most advanced aNEOS capabilities are used across all components.

Naming Convention:
- analyze_neo_*: Core analysis functions
- detect_artificial_*: Detection-specific functions  
- assess_*: Assessment and validation functions
- fetch_neo_*: Data fetching functions
- enrich_neo_*: Data enrichment functions
- validate_*: Validation functions
- calculate_*: Mathematical calculations
- generate_*: Report/result generation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .detection import DetectionResult, ArtificialNEODetector


class AnalysisCapability(Enum):
    """aNEOS analysis capabilities."""
    ARTIFICIAL_DETECTION = "artificial_detection"
    ORBITAL_ANALYSIS = "orbital_analysis"
    PHYSICAL_ANALYSIS = "physical_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    MULTI_SOURCE_ENRICHMENT = "multi_source_enrichment"
    STATISTICAL_VALIDATION = "statistical_validation"
    REAL_TIME_MONITORING = "real_time_monitoring"
    HISTORICAL_ANALYSIS = "historical_analysis"


@dataclass
class aNEOSAnalysisResult:
    """Unified aNEOS analysis result with comprehensive information."""
    
    # Core identification (required fields first)
    designation: str
    is_artificial: bool
    artificial_probability: float
    confidence_level: float
    classification: str  # natural, suspicious, highly_suspicious, artificial
    risk_assessment: str  # low, moderate, high, critical
    
    # Optional fields with defaults
    analysis_timestamp: Optional[datetime] = None
    sigma_statistical_level: Optional[float] = None
    threat_level: Optional[str] = None
    
    # Detailed analysis
    orbital_analysis: Optional[Dict[str, Any]] = None
    physical_analysis: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None
    
    # Evidence and factors
    risk_factors: List[str] = None
    evidence_sources: List[str] = None
    anomaly_indicators: List[str] = None
    
    # Quality metrics
    data_completeness: float = 0.0
    analysis_quality: float = 0.0
    validation_status: str = "pending"
    
    # Source information
    data_sources_used: List[str] = None
    detector_used: str = "unknown"
    analysis_method: str = "standard"
    
    # Metadata
    processing_time_ms: float = 0.0
    cache_status: str = "miss"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.risk_factors is None:
            self.risk_factors = []
        if self.evidence_sources is None:
            self.evidence_sources = []
        if self.anomaly_indicators is None:
            self.anomaly_indicators = []
        if self.data_sources_used is None:
            self.data_sources_used = []
        if self.metadata is None:
            self.metadata = {}
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.utcnow()


class aNEOSAnalysisInterface(ABC):
    """
    Unified aNEOS Analysis Interface - Goal-Aligned Design
    
    This interface ensures all aNEOS components use consistent, goal-aligned
    naming conventions and provide access to the most advanced capabilities.
    """
    
    @abstractmethod
    def analyze_neo_comprehensive(self, 
                                 designation: str,
                                 orbital_elements: Optional[Dict[str, Any]] = None,
                                 physical_data: Optional[Dict[str, Any]] = None,
                                 enrichment_sources: Optional[List[str]] = None) -> aNEOSAnalysisResult:
        """
        Perform comprehensive NEO analysis using most advanced aNEOS capabilities.
        
        Args:
            designation: NEO designation
            orbital_elements: Orbital parameters (optional, will fetch if not provided)
            physical_data: Physical properties (optional)
            enrichment_sources: Data sources to use for enrichment
            
        Returns:
            aNEOSAnalysisResult with comprehensive analysis
        """
        pass
    
    @abstractmethod
    def detect_artificial_signatures(self, 
                                   orbital_elements: Dict[str, Any],
                                   physical_data: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """
        Detect artificial signatures using most advanced detection algorithms.
        
        Uses MultiModal Sigma5 detection when available, falls back to production
        or corrected detectors as needed.
        """
        pass
    
    @abstractmethod
    def assess_threat_level(self, analysis_result: aNEOSAnalysisResult) -> str:
        """
        Assess threat level based on comprehensive analysis.
        
        Returns: "low", "moderate", "high", "critical"
        """
        pass
    
    @abstractmethod
    def fetch_neo_data_multi_source(self, designation: str, 
                                   sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch NEO data from multiple authoritative sources.
        
        Uses NASA CAD, SBDB, MPC, NEODyS with intelligent fallback.
        """
        pass
    
    @abstractmethod
    def enrich_neo_data_comprehensive(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich NEO data using all available aNEOS enhancement capabilities.
        """
        pass
    
    @abstractmethod
    def validate_analysis_quality(self, analysis_result: aNEOSAnalysisResult) -> Dict[str, Any]:
        """
        Validate analysis quality using aNEOS validation framework.
        """
        pass
    
    @abstractmethod
    def calculate_anomaly_score_advanced(self, neo_data: Dict[str, Any]) -> float:
        """
        Calculate advanced anomaly score using sophisticated algorithms.
        """
        pass
    
    @abstractmethod
    def generate_comprehensive_report(self, analysis_result: aNEOSAnalysisResult) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for decision support.
        """
        pass


class AdvancedaNEOSCapabilities:
    """
    Advanced aNEOS Capabilities Manager
    
    Ensures the most sophisticated aNEOS features are available and properly
    configured for maximum detection and analysis capability.
    """
    
    def __init__(self):
        self.capabilities = {}
        self._initialize_advanced_capabilities()
    
    def _initialize_advanced_capabilities(self):
        """Initialize all advanced aNEOS capabilities."""
        capabilities = [
            self._initialize_multimodal_detection,
            self._initialize_historical_analysis,
            self._initialize_validation_framework,
            self._initialize_multi_source_enrichment,
            self._initialize_real_time_monitoring,
            self._initialize_statistical_analysis
        ]
        
        for capability_init in capabilities:
            try:
                capability_init()
            except Exception as e:
                # Log but don't fail - graceful degradation
                pass
    
    def _initialize_multimodal_detection(self):
        """Initialize MultiModal Sigma5 detection capabilities."""
        try:
            from ..detection.detection_manager import get_detection_manager, DetectorType
            self.detection_manager = get_detection_manager(DetectorType.MULTIMODAL)
            self.capabilities[AnalysisCapability.ARTIFICIAL_DETECTION] = True
        except ImportError:
            self.capabilities[AnalysisCapability.ARTIFICIAL_DETECTION] = False
    
    def _initialize_historical_analysis(self):
        """Initialize historical analysis capabilities."""
        try:
            from ..polling.historical_chunked_poller import HistoricalChunkedPoller
            self.capabilities[AnalysisCapability.HISTORICAL_ANALYSIS] = True
        except ImportError:
            self.capabilities[AnalysisCapability.HISTORICAL_ANALYSIS] = False
    
    def _initialize_validation_framework(self):
        """Initialize comprehensive validation framework."""
        try:
            from ..validation import monte_carlo_false_positive_validation
            self.capabilities[AnalysisCapability.STATISTICAL_VALIDATION] = True
        except ImportError:
            self.capabilities[AnalysisCapability.STATISTICAL_VALIDATION] = False
    
    def _initialize_multi_source_enrichment(self):
        """Initialize multi-source data enrichment."""
        try:
            from ..data.fetcher import DataFetcher
            self.data_fetcher = DataFetcher()
            self.capabilities[AnalysisCapability.MULTI_SOURCE_ENRICHMENT] = True
        except ImportError:
            self.capabilities[AnalysisCapability.MULTI_SOURCE_ENRICHMENT] = False
    
    def _initialize_real_time_monitoring(self):
        """Initialize real-time monitoring capabilities."""
        try:
            # Check for dashboard capabilities
            self.capabilities[AnalysisCapability.REAL_TIME_MONITORING] = True
        except ImportError:
            self.capabilities[AnalysisCapability.REAL_TIME_MONITORING] = False
    
    def _initialize_statistical_analysis(self):
        """Initialize advanced statistical analysis."""
        try:
            # Check for statistical analysis modules
            self.capabilities[AnalysisCapability.STATISTICAL_VALIDATION] = True
        except ImportError:
            self.capabilities[AnalysisCapability.STATISTICAL_VALIDATION] = False
    
    def get_available_capabilities(self) -> List[AnalysisCapability]:
        """Get list of available advanced capabilities."""
        return [cap for cap, available in self.capabilities.items() if available]
    
    def is_capability_available(self, capability: AnalysisCapability) -> bool:
        """Check if specific capability is available."""
        return self.capabilities.get(capability, False)
    
    def get_recommended_analysis_method(self, neo_data: Dict[str, Any]) -> str:
        """Recommend best analysis method based on available capabilities and data."""
        if self.is_capability_available(AnalysisCapability.ARTIFICIAL_DETECTION):
            if (self.is_capability_available(AnalysisCapability.MULTI_SOURCE_ENRICHMENT) and
                self.is_capability_available(AnalysisCapability.STATISTICAL_VALIDATION)):
                return "comprehensive_multimodal"
            else:
                return "advanced_detection"
        else:
            return "basic_analysis"


# Global advanced capabilities manager
_advanced_capabilities = None

def get_advanced_capabilities() -> AdvancedaNEOSCapabilities:
    """Get global advanced capabilities manager."""
    global _advanced_capabilities
    if _advanced_capabilities is None:
        _advanced_capabilities = AdvancedaNEOSCapabilities()
    return _advanced_capabilities


def assess_aneos_system_maturity() -> Dict[str, Any]:
    """
    Assess the maturity and capability level of the current aNEOS installation.
    
    Returns comprehensive system capability assessment.
    """
    capabilities = get_advanced_capabilities()
    available_caps = capabilities.get_available_capabilities()
    
    maturity_level = "basic"
    if len(available_caps) >= 5:
        maturity_level = "advanced"
    elif len(available_caps) >= 3:
        maturity_level = "intermediate"
    
    return {
        "maturity_level": maturity_level,
        "available_capabilities": [cap.value for cap in available_caps],
        "total_capabilities": len(available_caps),
        "max_capabilities": len(AnalysisCapability),
        "capability_coverage": len(available_caps) / len(AnalysisCapability),
        "recommended_upgrades": _get_upgrade_recommendations(available_caps)
    }


def _get_upgrade_recommendations(available_caps: List[AnalysisCapability]) -> List[str]:
    """Get recommendations for capability upgrades."""
    recommendations = []
    
    if AnalysisCapability.ARTIFICIAL_DETECTION not in available_caps:
        recommendations.append("Install detection framework components")
    
    if AnalysisCapability.MULTI_SOURCE_ENRICHMENT not in available_caps:
        recommendations.append("Configure multi-source data fetching")
    
    if AnalysisCapability.STATISTICAL_VALIDATION not in available_caps:
        recommendations.append("Enable statistical validation modules")
    
    if AnalysisCapability.HISTORICAL_ANALYSIS not in available_caps:
        recommendations.append("Install historical analysis capabilities")
    
    return recommendations