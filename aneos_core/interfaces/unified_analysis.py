"""Compatibility analysis contracts with nullable scientific inference.

Historical method names describe interfaces, not installed physical/null models.
Implementation inventory is separate from provider health and scientific validity.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc

from .detection import DetectionResult, ArtificialNEODetector


class AnalysisCapability(Enum):
    """aNEOS analysis capabilities."""
    CANDIDATE_SCREENING = "candidate_screening"
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
    is_artificial: Optional[bool]
    artificial_probability: Optional[float]
    confidence_level: Optional[float]
    classification: str  # follow-up screening status, not origin
    risk_assessment: str  # not_assessed without a scoped impact model
    
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
    data_completeness: Optional[float] = None
    analysis_quality: Optional[float] = None
    validation_status: str = "pending"
    
    # Source information
    data_sources_used: List[str] = None
    detector_used: str = "unknown"
    analysis_method: str = "standard"
    
    # Metadata
    processing_time_ms: float = 0.0
    cache_status: str = "not_reported"
    metadata: Dict[str, Any] = None
    
    screening_assessment: Optional[Dict[str, Any]] = None
    neo_data: Any = None
    status: str = 'inconclusive'
    errors: Optional[List[str]] = None
    impact_assessment: Optional[Dict[str, Any]] = None
    data_quality: Optional[Dict[str, Any]] = None

    @property
    def screening_score(self):
        return (self.screening_assessment or {}).get('screening_score')

    @property
    def analysis_metadata(self):
        return {**(self.screening_assessment or {}).get('analysis_metadata', {}),
                **(self.metadata or {})}

    def __post_init__(self):
        """Post-initialization setup."""
        if self.errors is None:
            self.errors = []
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
            self.analysis_timestamp = datetime.now(UTC)


class aNEOSAnalysisInterface(ABC):
    """
    Unified aNEOS Analysis Interface - Goal-Aligned Design
    
    This interface standardizes transport; historical method names do not
    imply that physical/null models or independent validation are installed.
    """
    
    @abstractmethod
    def analyze_neo_comprehensive(self, 
                                 designation: str,
                                 orbital_elements: Optional[Dict[str, Any]] = None,
                                 physical_data: Optional[Dict[str, Any]] = None,
                                 enrichment_sources: Optional[List[str]] = None) -> aNEOSAnalysisResult:
        """
        Perform structured candidate screening with explicit unavailable inference.
        
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
        Screen supplied evidence without interpreting ranking as origin probability.

        An unavailable source/model cannot select a prototype fallback detector.
        """
        pass
    
    @abstractmethod
    def assess_threat_level(self, analysis_result: aNEOSAnalysisResult) -> str:
        """
        Assess threat level based on comprehensive analysis.
        
        Returns "not_assessed" without an appropriate scoped impact model.
        """
        pass
    
    @abstractmethod
    def fetch_neo_data_multi_source(self, designation: str, 
                                   sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch NEO data from multiple authoritative sources.
        
        Use typed acquisition; unsupported per-call selection must not be silently ignored.
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
    def calculate_anomaly_score_advanced(self, neo_data: Dict[str, Any]) -> Optional[float]:
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
    """Implementation inventory, explicitly not provider health/model validation."""
    def __init__(self):
        self.capabilities = {cap:False for cap in AnalysisCapability}
        self.component_inventory = {}
        self._initialize_advanced_capabilities()

    def _initialize_advanced_capabilities(self):
        try:
            from ..analysis.screening import ScreeningAnalysisService
            ScreeningAnalysisService()
            self.capabilities[AnalysisCapability.CANDIDATE_SCREENING] = True
            self.component_inventory['candidate_screening'] = 'initialized_uncalibrated'
        except ImportError:
            self.component_inventory['candidate_screening'] = 'unavailable'
        self._initialize_multimodal_detection()
        self._initialize_historical_analysis()
        self._initialize_validation_framework()
        self._initialize_multi_source_enrichment()
        self._initialize_real_time_monitoring()
        self._initialize_statistical_analysis()

    def _initialize_multimodal_detection(self):
        self.capabilities[AnalysisCapability.ARTIFICIAL_DETECTION] = False
        self.component_inventory['origin_model'] = 'not_installed'

    def _initialize_historical_analysis(self):
        self.component_inventory['historical_scientific_validation'] = 'not_established'

    def _initialize_validation_framework(self):
        self.capabilities[AnalysisCapability.STATISTICAL_VALIDATION] = False
        self.component_inventory['required_review'] = 'inventory_not_scientific_validation'

    def _initialize_multi_source_enrichment(self):
        try:
            from ..data.fetcher import DataFetcher
            self.capabilities[AnalysisCapability.MULTI_SOURCE_ENRICHMENT] = True
            self.component_inventory['typed_acquisition'] = 'importable_provider_health_not_tested'
        except ImportError:
            self.component_inventory['typed_acquisition'] = 'unavailable'

    def _initialize_real_time_monitoring(self):
        self.component_inventory['monitoring'] = 'not_verified_by_this_inventory'

    def _initialize_statistical_analysis(self):
        self.capabilities[AnalysisCapability.STATISTICAL_VALIDATION] = False
        self.component_inventory['calibration_artifact'] = 'not_installed'

    def get_available_capabilities(self):
        return [cap for cap,available in self.capabilities.items() if available]

    def is_capability_available(self, capability):
        return self.capabilities.get(capability, False)

    def get_recommended_analysis_method(self, neo_data):
        return 'canonical_candidate_screening' if self.is_capability_available(
            AnalysisCapability.CANDIDATE_SCREENING) else 'unavailable'


# Global advanced capabilities manager
_advanced_capabilities = None

def get_advanced_capabilities() -> AdvancedaNEOSCapabilities:
    """Get global advanced capabilities manager."""
    global _advanced_capabilities
    if _advanced_capabilities is None:
        _advanced_capabilities = AdvancedaNEOSCapabilities()
    return _advanced_capabilities


def assess_aneos_system_maturity():
    capabilities = get_advanced_capabilities()
    return {'maturity_level':'research_unvalidated',
        'available_capabilities':[cap.value for cap in capabilities.get_available_capabilities()],
        'total_capabilities':len(capabilities.get_available_capabilities()),
        'max_capabilities':len(AnalysisCapability), 'capability_coverage':None,
        'availability_semantics':'implemented_components_not_provider_health_or_scientific_validation',
        'component_inventory':dict(capabilities.component_inventory),
        'scientific_validation_status':'unavailable', 'five_sigma_claim_permitted':False,
        'recommended_upgrades':['Independent observational fits, calibration and scientific review required']}


def _get_upgrade_recommendations(available_caps: List[AnalysisCapability]) -> List[str]:
    """Get recommendations for capability upgrades."""
    return ['Independent observational fits, calibration and scientific review are required; installation alone is not validation']
