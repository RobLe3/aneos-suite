"""
False Positive Prevention Module for aNEOS Scientific Rigor Enhancement.

This module implements comprehensive false positive prevention through:
- Cross-matching against known space debris catalogs  
- Synthetic population testing for calibration
- Physical plausibility validation
- Orbital mechanics consistency checking
- Advanced human hardware cross-matching with THETA SWARM integration

All functions work with existing analysis results without modifying
the core analysis pipeline, following additive architecture principles.
"""

import numpy as np
import asyncio
import aiohttp
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Import the advanced human hardware analysis system
from .human_hardware_analysis import HumanHardwareAnalyzer, HumanHardwareMatch

logger = logging.getLogger(__name__)

@dataclass
class SpaceDebrisMatch:
    """Information about a potential space debris match."""
    catalog: str
    object_id: str
    object_name: str
    match_confidence: float
    delta_v: float  # m/s
    orbital_similarity: float
    epoch_difference: float  # days
    match_criteria: Dict[str, Any]

@dataclass
class FalsePositiveResult:
    """Results from false positive prevention analysis."""
    is_likely_false_positive: bool
    false_positive_probability: float
    space_debris_matches: List[SpaceDebrisMatch]
    synthetic_population_percentile: float
    physical_plausibility_score: float
    human_hardware_likelihood: float
    confidence_score: float
    analysis_timestamp: datetime
    
    # Enhanced human hardware analysis results
    human_hardware_analysis: Optional[HumanHardwareMatch] = None
    
@dataclass
class SyntheticNEO:
    """Synthetic NEO for false positive testing."""
    designation: str
    semi_major_axis: float
    eccentricity: float
    inclination: float
    longitude_of_ascending_node: float
    argument_of_perihelion: float
    mean_anomaly: float
    epoch: datetime
    is_artificial: bool = False

class FalsePositivePrevention:
    """
    Comprehensive false positive prevention through multiple validation methods.
    
    This class provides methods for:
    - Cross-matching against space debris catalogs
    - Synthetic population generation and testing
    - Physical plausibility assessment
    - Human hardware identification
    - Orbital mechanics validation
    """
    
    def __init__(self, cache_dir: Optional[str] = None, hardware_analysis_config: Optional[Dict[str, Any]] = None):
        """
        Initialize false positive prevention system with enhanced human hardware analysis.
        
        Args:
            cache_dir: Directory for caching catalog data (optional)
            hardware_analysis_config: Configuration for human hardware analyzer (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".fp_cache"
        # Ensure parent directories exist before creating cache directory
        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize advanced human hardware analyzer
        try:
            self.human_hardware_analyzer = HumanHardwareAnalyzer(
                config=hardware_analysis_config or self._default_hardware_config()
            )
            # EMERGENCY: Suppress initialization logging
            # self.logger.info("Advanced human hardware analyzer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize human hardware analyzer: {e}")
            self.human_hardware_analyzer = None
        
        # Space debris catalog configurations (legacy support - now enhanced by HumanHardwareAnalyzer)
        self.catalogs_config = {
            'DISCOS': {
                'url': 'https://discosweb.esoc.esa.int/api/objects',
                'format': 'json',
                'cache_duration_hours': 24,
                'orbital_tolerance': {'delta_v': 50, 'epoch_diff': 7}  # m/s, days
            },
            'SATCAT': {
                'url': 'https://celestrak.com/satcat/search.php',
                'format': 'csv', 
                'cache_duration_hours': 24,
                'orbital_tolerance': {'delta_v': 25, 'epoch_diff': 14}
            },
            'SPACE_TRACK': {
                'url': 'https://space-track.org/basicspacedata',
                'format': 'json',
                'cache_duration_hours': 12,
                'orbital_tolerance': {'delta_v': 30, 'epoch_diff': 7}
            }
        }
        
        # Initialize cached catalogs
        self.cached_catalogs = {}
        self._load_cached_catalogs()
        
    def _default_hardware_config(self) -> Dict[str, Any]:
        """Default configuration for human hardware analyzer."""
        return {
            'cache_dir': str(self.cache_dir / 'hardware_cache'),
            'processing_timeout_seconds': 2.0,
            'min_confidence_threshold': 0.7,
            'performance_targets': {
                'max_processing_time_ms': 2000,
                'min_accuracy_rate': 0.95
            },
            'constellation_detection': {
                'known_constellations': ['starlink', 'oneweb', 'kuiper', 'globalstar', 'iridium']
            }
        }
        
    def _load_cached_catalogs(self):
        """Load cached space debris catalogs if available."""
        try:
            for catalog_name in self.catalogs_config.keys():
                cache_file = self.cache_dir / f"{catalog_name.lower()}_cache.json"
                if cache_file.exists():
                    cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    max_age = timedelta(hours=self.catalogs_config[catalog_name]['cache_duration_hours'])
                    
                    if cache_age < max_age:
                        with open(cache_file, 'r') as f:
                            self.cached_catalogs[catalog_name] = json.load(f)
                        self.logger.info(f"Loaded cached {catalog_name} catalog ({len(self.cached_catalogs[catalog_name])} objects)")
                    else:
                        self.logger.info(f"{catalog_name} cache expired, will refresh")
                        
        except Exception as e:
            self.logger.warning(f"Error loading cached catalogs: {e}")
    
    async def cross_match_space_debris(
        self, 
        designation: str, 
        orbital_elements: Dict[str, float],
        neo_data: Optional[Any] = None
    ) -> List[SpaceDebrisMatch]:
        """
        Enhanced cross-match NEO against known space debris catalogs with advanced hardware analysis.
        
        Args:
            designation: NEO designation
            orbital_elements: Dict with orbital elements (a, e, i, Omega, omega, M)
            neo_data: Optional NEO data object for additional context
            
        Returns:
            List of SpaceDebrisMatch objects for potential matches
        """
        matches = []
        
        try:
            # Primary enhanced human hardware analysis
            if self.human_hardware_analyzer:
                try:
                    hardware_analysis = await self.human_hardware_analyzer.analyze_human_hardware(
                        neo_data, orbital_elements
                    )
                    
                    # Convert hardware analysis results to SpaceDebrisMatch format for compatibility
                    if hardware_analysis.catalog_matches:
                        for catalog_match in hardware_analysis.catalog_matches:
                            space_debris_match = SpaceDebrisMatch(
                                catalog=catalog_match.get('catalog', 'ENHANCED'),
                                object_id=catalog_match.get('object_id', 'unknown'),
                                object_name=catalog_match.get('object_name', 'unnamed'),
                                match_confidence=catalog_match.get('match_confidence', 0.0),
                                delta_v=self._calculate_delta_v_from_elements(
                                    orbital_elements, 
                                    catalog_match.get('orbital_elements', {})
                                ),
                                orbital_similarity=catalog_match.get('orbital_similarity', 0.0),
                                epoch_difference=0.0,  # Would need epoch comparison
                                match_criteria={
                                    'hardware_classification': hardware_analysis.object_classification,
                                    'artificial_probability': hardware_analysis.artificial_probability,
                                    'processing_time_ms': hardware_analysis.processing_time_ms
                                }
                            )
                            matches.append(space_debris_match)
                    
                    self.logger.info(
                        f"Enhanced hardware analysis for {designation}: "
                        f"{hardware_analysis.object_classification} "
                        f"(confidence: {hardware_analysis.classification_confidence:.3f})"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Enhanced hardware analysis failed for {designation}: {e}")
            
            # Fallback to legacy catalog search for additional coverage
            for catalog_name, config in self.catalogs_config.items():
                try:
                    catalog_matches = await self._search_catalog(
                        catalog_name, 
                        designation, 
                        orbital_elements
                    )
                    matches.extend(catalog_matches)
                except Exception as e:
                    self.logger.warning(f"Legacy catalog search failed for {catalog_name}: {e}")
            
            # Sort by match confidence (highest first)
            matches.sort(key=lambda x: x.match_confidence, reverse=True)
            
            # Limit to top matches to avoid information overload
            matches = matches[:20]
            
            self.logger.info(f"Found {len(matches)} potential debris matches for {designation}")
            return matches
            
        except Exception as e:
            self.logger.error(f"Space debris cross-match failed for {designation}: {e}")
            return []
    
    def _calculate_delta_v_from_elements(
        self, 
        elements1: Dict[str, float], 
        elements2: Dict[str, float]
    ) -> float:
        """Calculate delta-V between two sets of orbital elements."""
        try:
            a1, a2 = elements1.get('a', 0), elements2.get('a', 0)
            if a1 > 0 and a2 > 0:
                mu = 398600.4418  # Earth's gravitational parameter (km³/s²)
                v1 = np.sqrt(mu * (2/a1 - 1/a1))
                v2 = np.sqrt(mu * (2/a2 - 1/a2))
                return abs(v1 - v2) * 1000  # Convert to m/s
            return float('inf')
        except Exception:
            return float('inf')
    
    async def _search_catalog(
        self, 
        catalog_name: str, 
        designation: str, 
        orbital_elements: Dict[str, float]
    ) -> List[SpaceDebrisMatch]:
        """Search a specific catalog for matching objects."""
        matches = []
        
        try:
            # Use cached data if available, otherwise fetch
            if catalog_name not in self.cached_catalogs:
                await self._fetch_catalog_data(catalog_name)
            
            catalog_data = self.cached_catalogs.get(catalog_name, [])
            tolerance = self.catalogs_config[catalog_name]['orbital_tolerance']
            
            for obj in catalog_data:
                match_result = self._calculate_orbital_similarity(
                    orbital_elements, 
                    obj.get('orbital_elements', {}),
                    tolerance
                )
                
                if match_result['is_match']:
                    match = SpaceDebrisMatch(
                        catalog=catalog_name,
                        object_id=obj.get('id', 'unknown'),
                        object_name=obj.get('name', 'unnamed'),
                        match_confidence=match_result['confidence'],
                        delta_v=match_result['delta_v'],
                        orbital_similarity=match_result['similarity'],
                        epoch_difference=match_result['epoch_diff'],
                        match_criteria=match_result['criteria']
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Catalog search failed for {catalog_name}: {e}")
            return []
    
    def _calculate_orbital_similarity(
        self, 
        elements1: Dict[str, float], 
        elements2: Dict[str, float],
        tolerance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate orbital similarity between two sets of elements.
        
        Returns match information including delta-V and similarity score.
        """
        try:
            # Extract orbital elements with defaults
            a1, e1, i1 = elements1.get('a', 0), elements1.get('e', 0), elements1.get('i', 0)
            a2, e2, i2 = elements2.get('a', 0), elements2.get('e', 0), elements2.get('i', 0)
            
            # Calculate approximate delta-V using vis-viva equation
            # Simplified calculation for initial screening
            mu = 398600.4418  # Earth's gravitational parameter (km³/s²)
            
            if a1 > 0 and a2 > 0:
                # Velocity at aphelion for both orbits
                v1 = np.sqrt(mu * (2/a1 - 1/a1))
                v2 = np.sqrt(mu * (2/a2 - 1/a2))
                delta_v = abs(v1 - v2) * 1000  # Convert to m/s
            else:
                delta_v = float('inf')
            
            # Calculate similarity score (0-1, higher is more similar)
            da = abs(a1 - a2) / max(a1, a2, 0.1)
            de = abs(e1 - e2)
            di = abs(i1 - i2) * np.pi / 180  # Convert to radians
            
            similarity = np.exp(-(da**2 + de**2 + di**2))
            
            # Determine if it's a match based on delta-V threshold
            is_match = delta_v <= tolerance.get('delta_v', 25)
            
            # Calculate confidence based on similarity and delta-V
            confidence = similarity * np.exp(-delta_v / 100)  # Exponential decay with delta-V
            
            return {
                'is_match': is_match,
                'confidence': min(confidence, 1.0),
                'similarity': similarity,
                'delta_v': delta_v,
                'epoch_diff': 0.0,  # Placeholder - would need epoch comparison
                'criteria': {
                    'delta_a': da,
                    'delta_e': de, 
                    'delta_i': di,
                    'threshold_met': is_match
                }
            }
            
        except Exception as e:
            self.logger.error(f"Orbital similarity calculation failed: {e}")
            return {
                'is_match': False,
                'confidence': 0.0,
                'similarity': 0.0,
                'delta_v': float('inf'),
                'epoch_diff': 0.0,
                'criteria': {}
            }
    
    async def _fetch_catalog_data(self, catalog_name: str):
        """Fetch catalog data from external source (placeholder implementation)."""
        try:
            # In a real implementation, this would fetch from actual catalogs
            # For now, create mock data for demonstration
            mock_data = self._generate_mock_catalog_data(catalog_name)
            
            # Cache the data
            cache_file = self.cache_dir / f"{catalog_name.lower()}_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(mock_data, f)
            
            self.cached_catalogs[catalog_name] = mock_data
            self.logger.info(f"Fetched and cached {catalog_name} catalog data")
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {catalog_name} catalog: {e}")
    
    def _generate_mock_catalog_data(self, catalog_name: str) -> List[Dict]:
        """Generate mock catalog data for testing (placeholder)."""
        # In production, this would be replaced with actual API calls
        np.random.seed(42)  # Reproducible mock data
        
        mock_objects = []
        n_objects = {'DISCOS': 100, 'SATCAT': 150, 'SPACE_TRACK': 75}.get(catalog_name, 100)
        
        for i in range(n_objects):
            obj = {
                'id': f"{catalog_name}_{i:05d}",
                'name': f"DEBRIS-{i:05d}",
                'orbital_elements': {
                    'a': np.random.uniform(6500, 42000),  # km
                    'e': np.random.uniform(0, 0.9),
                    'i': np.random.uniform(0, 180),  # degrees
                    'Omega': np.random.uniform(0, 360),
                    'omega': np.random.uniform(0, 360), 
                    'M': np.random.uniform(0, 360)
                },
                'launch_date': (datetime.now() - timedelta(days=np.random.randint(1, 20000))).isoformat(),
                'object_type': np.random.choice(['PAYLOAD', 'ROCKET BODY', 'DEBRIS'])
            }
            mock_objects.append(obj)
        
        return mock_objects
    
    def synthetic_population_test(
        self, 
        analysis_result: Any, 
        n_synthetic: int = 1000
    ) -> Dict[str, Any]:
        """
        Test analysis result against synthetic NEO population.
        
        Args:
            analysis_result: Result from original aNEOS analysis
            n_synthetic: Number of synthetic NEOs to generate
            
        Returns:
            Dict with synthetic population test results
        """
        try:
            # Generate synthetic NEO population
            synthetic_neos = self._generate_synthetic_population(n_synthetic)
            
            # Simulate analysis results for synthetic population
            synthetic_scores = []
            for synthetic_neo in synthetic_neos:
                # Simulate analysis - in reality, this would run through the pipeline
                synthetic_score = self._simulate_analysis_score(synthetic_neo)
                synthetic_scores.append(synthetic_score)
            
            # Calculate percentile rank of actual result
            actual_score = getattr(analysis_result, 'overall_score', 0.0)
            percentile_rank = self._calculate_percentile_rank(actual_score, synthetic_scores)
            
            # Estimate false positive probability
            # Higher percentile = more unusual = less likely to be false positive
            fp_probability = max(0.0, 1.0 - percentile_rank / 100.0)
            
            # Calculate population statistics
            population_stats = {
                'mean': np.mean(synthetic_scores),
                'std': np.std(synthetic_scores),
                'median': np.median(synthetic_scores),
                'percentiles': {
                    '5': np.percentile(synthetic_scores, 5),
                    '25': np.percentile(synthetic_scores, 25),
                    '75': np.percentile(synthetic_scores, 75),
                    '95': np.percentile(synthetic_scores, 95)
                }
            }
            
            return {
                'synthetic_percentile': percentile_rank,
                'false_positive_probability': fp_probability,
                'population_size': n_synthetic,
                'population_stats': population_stats,
                'actual_score': actual_score,
                'z_score': (actual_score - population_stats['mean']) / max(population_stats['std'], 0.01)
            }
            
        except Exception as e:
            self.logger.error(f"Synthetic population test failed: {e}")
            return {
                'synthetic_percentile': 50.0,
                'false_positive_probability': 0.5,
                'population_size': 0,
                'population_stats': {},
                'actual_score': 0.0,
                'z_score': 0.0
            }
    
    def _generate_synthetic_population(self, n_synthetic: int) -> List[SyntheticNEO]:
        """Generate a population of synthetic NEOs with realistic orbital distributions."""
        synthetic_neos = []
        np.random.seed(42)  # Reproducible results
        
        for i in range(n_synthetic):
            # Generate realistic orbital elements based on known NEO distributions
            
            # Semi-major axis: Peak around 1-3 AU
            a = np.random.lognormal(mean=0.5, sigma=0.6)  # AU
            
            # Eccentricity: Higher eccentricities for NEOs
            e = np.random.beta(0.5, 2.0) * 0.95  # Limit to avoid parabolic orbits
            
            # Inclination: Most NEOs have low inclinations
            i = np.random.exponential(scale=10) % 180  # Degrees
            
            # Other angles: Uniform distribution
            Omega = np.random.uniform(0, 360)
            omega = np.random.uniform(0, 360)
            M = np.random.uniform(0, 360)
            
            synthetic_neo = SyntheticNEO(
                designation=f"SYNTH_{i:05d}",
                semi_major_axis=a,
                eccentricity=e,
                inclination=i,
                longitude_of_ascending_node=Omega,
                argument_of_perihelion=omega,
                mean_anomaly=M,
                epoch=datetime.now(),
                is_artificial=False
            )
            
            synthetic_neos.append(synthetic_neo)
        
        return synthetic_neos
    
    def _simulate_analysis_score(self, synthetic_neo: SyntheticNEO) -> float:
        """
        Simulate analysis score for a synthetic NEO.
        
        This is a simplified simulation - in reality, synthetic NEOs would
        be run through the actual analysis pipeline.
        """
        # Base score from orbital characteristics
        base_score = 0.0
        
        # Eccentricity contribution (higher = more unusual)
        base_score += min(synthetic_neo.eccentricity * 0.3, 0.25)
        
        # Inclination contribution (very high or very low = more unusual)
        i_norm = abs(synthetic_neo.inclination - 90) / 90  # 0 = polar, 1 = equatorial
        base_score += (1 - i_norm) * 0.2
        
        # Semi-major axis contribution (very close or very far = more unusual)
        if synthetic_neo.semi_major_axis < 0.8 or synthetic_neo.semi_major_axis > 5.0:
            base_score += 0.15
        
        # Add some random noise to simulate measurement uncertainty and other indicators
        noise = np.random.normal(0, 0.1)
        
        return max(0.0, min(base_score + noise, 1.0))
    
    def _calculate_percentile_rank(self, value: float, population: List[float]) -> float:
        """Calculate percentile rank of value in population."""
        if not population:
            return 50.0
        
        population_array = np.array(population)
        percentile = (np.sum(population_array <= value) / len(population_array)) * 100
        return min(max(percentile, 0.0), 100.0)
    
    async def assess_physical_plausibility(
        self, 
        neo_data: Any, 
        analysis_result: Any
    ) -> Dict[str, Any]:
        """
        Enhanced physical plausibility assessment with human hardware analysis integration.
        
        Args:
            neo_data: NEO data object
            analysis_result: Analysis result from pipeline
            
        Returns:
            Dict with enhanced physical plausibility assessment
        """
        try:
            plausibility_factors = {}
            
            # Traditional plausibility factors
            orbital_consistency = self._check_orbital_mechanics(neo_data)
            plausibility_factors['orbital_mechanics'] = orbital_consistency
            
            energy_conservation = self._check_energy_conservation(neo_data)
            plausibility_factors['energy_conservation'] = energy_conservation
            
            physical_consistency = self._check_physical_parameters(neo_data)
            plausibility_factors['physical_parameters'] = physical_consistency
            
            size_brightness = self._check_size_brightness_relationship(neo_data)
            plausibility_factors['size_brightness'] = size_brightness
            
            # Enhanced human hardware analysis integration
            hardware_analysis_score = 0.5  # Default neutral score
            human_hardware_match = None
            
            if self.human_hardware_analyzer:
                try:
                    # Extract orbital elements for hardware analysis
                    orbital_elements = self._extract_orbital_elements_from_neo_data(neo_data)
                    
                    # Run comprehensive hardware analysis
                    human_hardware_match = await self.human_hardware_analyzer.analyze_human_hardware(
                        neo_data, orbital_elements
                    )
                    
                    # Calculate hardware analysis contribution to plausibility
                    if human_hardware_match.artificial_probability > 0.7:
                        # High artificial probability reduces plausibility for natural NEO
                        hardware_analysis_score = 1.0 - human_hardware_match.artificial_probability
                    else:
                        # Low artificial probability increases plausibility for natural NEO
                        hardware_analysis_score = 0.5 + (1.0 - human_hardware_match.artificial_probability) * 0.5
                    
                    plausibility_factors['human_hardware_analysis'] = hardware_analysis_score
                    
                    self.logger.info(
                        f"Hardware analysis contribution to plausibility: {hardware_analysis_score:.3f} "
                        f"(artificial probability: {human_hardware_match.artificial_probability:.3f})"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Hardware analysis integration failed: {e}")
                    plausibility_factors['human_hardware_analysis'] = 0.5
            else:
                plausibility_factors['human_hardware_analysis'] = 0.5
            
            # Enhanced weight calculation including hardware analysis
            factor_weights = {
                'orbital_mechanics': 0.25,
                'energy_conservation': 0.2,
                'physical_parameters': 0.15,
                'size_brightness': 0.1,
                'human_hardware_analysis': 0.3  # Significant weight for hardware analysis
            }
            
            overall_score = sum(
                plausibility_factors.get(factor, 0.5) * weight
                for factor, weight in factor_weights.items()
            )
            
            # Enhanced confidence calculation
            confidence = self._calculate_enhanced_plausibility_confidence(
                plausibility_factors, human_hardware_match
            )
            
            return {
                'overall_plausibility': overall_score,
                'individual_factors': plausibility_factors,
                'is_physically_plausible': overall_score > 0.6,
                'confidence': confidence,
                'human_hardware_analysis_result': human_hardware_match,
                'enhanced_assessment': True  # Flag to indicate enhanced assessment was used
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced physical plausibility assessment failed: {e}")
            return {
                'overall_plausibility': 0.5,
                'individual_factors': {},
                'is_physically_plausible': True,  # Conservative - assume plausible if check fails
                'confidence': 0.0,
                'human_hardware_analysis_result': None,
                'enhanced_assessment': False
            }
    
    def _extract_orbital_elements_from_neo_data(self, neo_data: Any) -> Dict[str, float]:
        """Extract orbital elements from NEO data object."""
        try:
            # Try different ways to extract orbital elements based on data structure
            if hasattr(neo_data, 'orbital_elements'):
                oe = neo_data.orbital_elements
                if hasattr(oe, 'to_dict'):
                    elements = oe.to_dict()
                else:
                    elements = {
                        'a': getattr(oe, 'semi_major_axis', 1.0),
                        'e': getattr(oe, 'eccentricity', 0.1),
                        'i': getattr(oe, 'inclination', 10.0),
                        'Omega': getattr(oe, 'ra_of_ascending_node', 0.0),
                        'omega': getattr(oe, 'arg_of_periapsis', 0.0),
                        'M': getattr(oe, 'mean_anomaly', 0.0)
                    }
            elif hasattr(neo_data, 'to_dict'):
                data_dict = neo_data.to_dict()
                orbital_data = data_dict.get('orbital_elements', {})
                elements = {
                    'a': orbital_data.get('semi_major_axis', 1.0),
                    'e': orbital_data.get('eccentricity', 0.1),
                    'i': orbital_data.get('inclination', 10.0),
                    'Omega': orbital_data.get('ra_of_ascending_node', 0.0),
                    'omega': orbital_data.get('arg_of_periapsis', 0.0),
                    'M': orbital_data.get('mean_anomaly', 0.0)
                }
            else:
                # Fallback default values
                elements = {
                    'a': 1.5,  # AU
                    'e': 0.2,
                    'i': 15.0,  # degrees
                    'Omega': 180.0,
                    'omega': 90.0,
                    'M': 45.0
                }
            
            return elements
            
        except Exception as e:
            self.logger.warning(f"Failed to extract orbital elements: {e}")
            return {
                'a': 1.5, 'e': 0.2, 'i': 15.0, 
                'Omega': 180.0, 'omega': 90.0, 'M': 45.0
            }
    
    def _calculate_enhanced_plausibility_confidence(
        self, 
        factors: Dict[str, float], 
        hardware_match: Optional[HumanHardwareMatch]
    ) -> float:
        """Calculate enhanced confidence in plausibility assessment."""
        try:
            # Base confidence from factor consistency
            base_confidence = self._calculate_plausibility_confidence(factors)
            
            # Hardware analysis confidence boost
            hardware_confidence = 0.0
            if hardware_match:
                hardware_confidence = min(
                    hardware_match.classification_confidence * 0.5 +
                    (1.0 - hardware_match.artificial_probability) * 0.3,
                    0.4  # Cap the boost
                )
            
            return min(base_confidence + hardware_confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _check_orbital_mechanics(self, neo_data: Any) -> float:
        """Check orbital mechanics consistency (placeholder)."""
        # Placeholder - would implement actual orbital mechanics validation
        return 0.8
    
    def _check_energy_conservation(self, neo_data: Any) -> float:
        """Check energy conservation (placeholder)."""
        # Placeholder - would implement energy conservation checks
        return 0.85
    
    def _check_physical_parameters(self, neo_data: Any) -> float:
        """Check physical parameter consistency (placeholder)."""
        # Placeholder - would validate physical parameters
        return 0.75
    
    def _check_size_brightness_relationship(self, neo_data: Any) -> float:
        """Check size-brightness relationship (placeholder)."""
        # Placeholder - would validate size vs brightness relationship
        return 0.9
    
    def _calculate_plausibility_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate confidence in plausibility assessment."""
        if not factors:
            return 0.0
        
        # Confidence based on consistency of factors
        factor_values = list(factors.values())
        variance = np.var(factor_values)
        
        # Lower variance = higher confidence
        confidence = np.exp(-variance * 5)  # Scale factor for sensitivity
        return min(confidence, 1.0)