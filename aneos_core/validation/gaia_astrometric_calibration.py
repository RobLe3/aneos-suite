"""
MU SWARM - Gaia Astrometric Precision Calibration System for aNEOS

Ultra-high precision Gaia astrometric calibration system for enhanced positional 
validation and artificial object detection in aNEOS pipeline.

SCIENTIFIC BACKGROUND:
- Gaia EDR3/DR3 catalog integration with 1.8 billion sources
- Sub-milliarcsecond proper motion precision for G<15 mag sources
- Parallax accuracy ~0.01-0.04 mas for nearby objects
- Systematic accuracy ~0.1 mas absolute in ICRS reference frame

KEY CAPABILITIES:
- Astrometric cross-matching and validation
- Proper motion analysis with uncertainty propagation
- Artificial object detection via anomalous kinematics
- Positional precision enhancement for orbit refinement
- Statistical significance testing for astrometric anomalies

Author: MU SWARM - Gaia Astrometric Precision Calibration Team
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
from pathlib import Path
import json
import warnings

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, ICRS, FK5, get_body_barycentric
    from astropy.time import Time
    from astropy.table import Table
    from astroquery.gaia import Gaia
    from astroquery.simbad import Simbad
    from astroquery.exceptions import RemoteServiceError
    import astropy.constants as const
    ASTROPY_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Astropy/astroquery not available: {e}")
    ASTROPY_AVAILABLE = False

from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class GaiaSource:
    """Gaia catalog source with astrometric data."""
    source_id: int
    ra: float                    # Right ascension (degrees)
    dec: float                   # Declination (degrees) 
    ra_error: float              # RA uncertainty (mas)
    dec_error: float             # Dec uncertainty (mas)
    pmra: Optional[float] = None         # Proper motion RA (mas/yr)
    pmdec: Optional[float] = None        # Proper motion Dec (mas/yr)
    pmra_error: Optional[float] = None   # PM RA uncertainty (mas/yr)
    pmdec_error: Optional[float] = None  # PM Dec uncertainty (mas/yr)
    parallax: Optional[float] = None     # Parallax (mas)
    parallax_error: Optional[float] = None  # Parallax uncertainty (mas)
    g_mag: Optional[float] = None        # G-band magnitude
    bp_rp: Optional[float] = None        # BP-RP color
    ruwe: Optional[float] = None         # Renormalized unit weight error
    astrometric_excess_noise: Optional[float] = None  # Astrometric excess noise
    epoch: float = 2016.0               # Reference epoch (years)

@dataclass 
class ProperMotionAnalysis:
    """Proper motion analysis results."""
    pmra_mas_yr: float               # Proper motion RA (mas/yr)
    pmdec_mas_yr: float              # Proper motion Dec (mas/yr)
    pm_total: float                  # Total proper motion (mas/yr)
    pm_pa: float                     # Position angle of PM vector (deg)
    pmra_significance: float         # RA PM statistical significance 
    pmdec_significance: float        # Dec PM statistical significance
    pm_total_significance: float     # Total PM significance
    is_significant_motion: bool      # >3σ detection flag
    stellar_likelihood: float       # Likelihood of stellar motion (0-1)
    artificial_likelihood: float    # Likelihood of artificial motion (0-1)

@dataclass
class ParallaxAnalysis:
    """Parallax measurement analysis results."""
    parallax_mas: float              # Parallax measurement (mas)
    parallax_error_mas: float        # Parallax uncertainty (mas)
    parallax_significance: float     # Statistical significance (σ)
    distance_pc: Optional[float] = None      # Distance in parsecs (if significant)
    distance_error_pc: Optional[float] = None   # Distance uncertainty
    is_significant_parallax: bool = False    # >3σ detection flag
    consistency_check: Optional[float] = None   # Consistency with expected distance

@dataclass
class AstrometricPrecision:
    """Astrometric precision analysis results."""
    position_error_ellipse_major: float    # Major axis error ellipse (mas)
    position_error_ellipse_minor: float    # Minor axis error ellipse (mas)
    position_error_ellipse_pa: float       # Position angle of ellipse (deg)
    systematic_error_estimate: float       # Systematic error estimate (mas)
    total_position_uncertainty: float      # Combined uncertainty (mas)
    reference_frame_quality: str           # ICRS alignment quality
    epoch_propagation_error: float         # Error from epoch propagation (mas)

@dataclass
class ArtificialObjectSignature:
    """Artificial object detection signature."""
    anomalous_proper_motion: bool          # Non-stellar PM signature
    astrometric_excess_noise_flag: bool    # Excess noise detection
    color_magnitude_outlier: bool          # CMD position outlier
    parallax_inconsistency: bool           # Parallax-distance inconsistency
    artificial_probability: float          # Overall artificial likelihood (0-1)
    artificial_indicators: List[str]       # List of artificial indicators
    confidence: float                      # Detection confidence (0-1)

@dataclass
class GaiaAstrometricResult:
    """Comprehensive Gaia astrometric analysis result."""
    # Query metadata
    target_coords: Tuple[float, float]      # Target RA, Dec (degrees)
    search_radius_arcsec: float             # Search radius used
    n_sources_found: int                    # Number of Gaia sources found
    primary_source: Optional[GaiaSource] = None    # Best matching source
    
    # Analysis results
    proper_motion_analysis: Optional[ProperMotionAnalysis] = None
    parallax_analysis: Optional[ParallaxAnalysis] = None
    astrometric_precision: Optional[AstrometricPrecision] = None
    artificial_object_signature: Optional[ArtificialObjectSignature] = None
    
    # Validation metrics
    position_residual_mas: Optional[float] = None      # Position offset from target
    astrometric_quality_score: float = 0.0             # Overall quality (0-1)
    validation_passed: bool = False                     # Validation result
    validation_confidence: float = 0.0                 # Validation confidence
    
    # Processing metadata  
    processing_time_ms: float = 0.0
    gaia_data_release: str = "EDR3"
    analysis_timestamp: datetime = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now()
        if self.warnings is None:
            self.warnings = []

class GaiaAstrometricCalibrator:
    """
    Ultra-high precision Gaia astrometric calibration system.
    
    Provides comprehensive astrometric validation and artificial object detection
    capabilities using Gaia EDR3/DR3 catalog data with sub-milliarcsecond precision.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Gaia astrometric calibrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gaia query service
        if ASTROPY_AVAILABLE:
            Gaia.MAIN_GAIA_TABLE = self.config.get('gaia_table', 'gaiadr3.gaia_source')
            Gaia.ROW_LIMIT = self.config.get('max_sources', 1000)
        
        # Initialize cache
        self.cache_dir = Path(self.config.get('cache_dir', 'cache/gaia'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_enabled = self.config.get('enable_cache', True)
        
        # Performance tracking
        self._query_times = []
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info("Gaia Astrometric Calibrator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Gaia astrometric calibration."""
        return {
            # Gaia catalog settings
            'gaia_table': 'gaiadr3.gaia_source',
            'data_release': 'EDR3',
            'max_sources': 100,
            'default_radius_arcsec': 30.0,
            
            # Precision requirements  
            'position_precision_mas': 0.03,     # Target position precision
            'proper_motion_precision_mas_yr': 0.02,  # Target PM precision
            'parallax_precision_mas': 0.04,     # Target parallax precision
            'systematic_accuracy_mas': 0.1,     # Systematic accuracy requirement
            
            # Significance thresholds
            'pm_significance_threshold': 3.0,   # Proper motion significance (σ)
            'parallax_significance_threshold': 3.0,  # Parallax significance (σ)
            'artificial_probability_threshold': 0.7,  # Artificial object threshold
            'ruwe_threshold': 1.4,              # RUWE quality threshold
            
            # Artificial object detection
            'stellar_pm_percentiles': (5, 95),  # Expected stellar PM range
            'excess_noise_threshold': 0.5,      # Excess noise threshold
            'color_outlier_sigma': 3.0,         # Color-magnitude outlier threshold  
            'parallax_consistency_tolerance': 2.0,  # Parallax consistency (σ)
            
            # Performance settings
            'query_timeout_sec': 10.0,
            'max_processing_time_ms': 100,
            'enable_cache': True,
            'cache_expiry_hours': 24,
            'enable_parallel_queries': True,
            
            # Coordinate system settings
            'reference_epoch': 2016.0,          # Gaia reference epoch
            'target_epoch': None,               # Target epoch for propagation
            'reference_frame': 'ICRS',          # Reference frame
            
            # Database integration
            'enable_simbad_crossmatch': True,
            'simbad_radius_arcsec': 5.0,
            'enable_local_catalog': False,
            'local_catalog_path': None
        }
    
    async def analyze_position(
        self, 
        ra_deg: float, 
        dec_deg: float,
        target_epoch: Optional[float] = None,
        search_radius_arcsec: Optional[float] = None,
        designation: Optional[str] = None
    ) -> GaiaAstrometricResult:
        """
        Perform comprehensive Gaia astrometric analysis for target position.
        
        Args:
            ra_deg: Target right ascension (degrees)
            dec_deg: Target declination (degrees) 
            target_epoch: Target observation epoch (decimal years)
            search_radius_arcsec: Search radius (default from config)
            designation: Optional object designation for logging
            
        Returns:
            GaiaAstrometricResult with comprehensive analysis
        """
        start_time = time.time()
        
        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy/astroquery required for Gaia analysis")
        
        self.logger.info(f"Starting Gaia astrometric analysis for {designation or 'target'} "
                        f"at ({ra_deg:.6f}, {dec_deg:.6f})")
        
        try:
            # Set search radius
            radius = search_radius_arcsec or self.config['default_radius_arcsec']
            
            # Query Gaia catalog
            gaia_sources = await self._query_gaia_catalog(ra_deg, dec_deg, radius)
            
            if not gaia_sources:
                self.logger.warning("No Gaia sources found in search area")
                return GaiaAstrometricResult(
                    target_coords=(ra_deg, dec_deg),
                    search_radius_arcsec=radius,
                    n_sources_found=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    warnings=["No Gaia sources found"]
                )
            
            # Find primary (closest) source
            primary_source = self._find_primary_source(gaia_sources, ra_deg, dec_deg)
            
            # Perform astrometric analyses
            analyses = await asyncio.gather(
                self._analyze_proper_motion(primary_source),
                self._analyze_parallax(primary_source),
                self._analyze_precision(primary_source),
                self._detect_artificial_signatures(primary_source, gaia_sources),
                return_exceptions=True
            )
            
            proper_motion_analysis = analyses[0] if not isinstance(analyses[0], Exception) else None
            parallax_analysis = analyses[1] if not isinstance(analyses[1], Exception) else None
            precision_analysis = analyses[2] if not isinstance(analyses[2], Exception) else None
            artificial_signature = analyses[3] if not isinstance(analyses[3], Exception) else None
            
            # Calculate position residual
            position_residual = self._calculate_position_residual(
                primary_source, ra_deg, dec_deg
            )
            
            # Calculate overall quality metrics
            quality_score, validation_passed, confidence = self._calculate_validation_metrics(
                primary_source, proper_motion_analysis, parallax_analysis, 
                precision_analysis, artificial_signature
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            result = GaiaAstrometricResult(
                target_coords=(ra_deg, dec_deg),
                search_radius_arcsec=radius,
                n_sources_found=len(gaia_sources),
                primary_source=primary_source,
                proper_motion_analysis=proper_motion_analysis,
                parallax_analysis=parallax_analysis,
                astrometric_precision=precision_analysis,
                artificial_object_signature=artificial_signature,
                position_residual_mas=position_residual,
                astrometric_quality_score=quality_score,
                validation_passed=validation_passed,
                validation_confidence=confidence,
                processing_time_ms=processing_time,
                gaia_data_release=self.config['data_release']
            )
            
            self.logger.info(f"Gaia analysis complete: quality={quality_score:.3f}, "
                           f"validation={validation_passed}, artificial_prob="
                           f"{artificial_signature.artificial_probability:.3f if artificial_signature else 'N/A'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gaia astrometric analysis failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return GaiaAstrometricResult(
                target_coords=(ra_deg, dec_deg),
                search_radius_arcsec=search_radius_arcsec or self.config['default_radius_arcsec'],
                n_sources_found=0,
                processing_time_ms=processing_time,
                warnings=[f"Analysis failed: {str(e)}"]
            )
    
    async def _query_gaia_catalog(
        self, 
        ra_deg: float, 
        dec_deg: float, 
        radius_arcsec: float
    ) -> List[GaiaSource]:
        """Query Gaia catalog with caching and error handling."""
        
        # Check cache first
        cache_key = f"gaia_{ra_deg:.6f}_{dec_deg:.6f}_{radius_arcsec:.1f}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result
        
        self._cache_misses += 1
        
        try:
            query_start = time.time()
            
            # Construct ADQL query for Gaia
            query = f"""
            SELECT source_id, ra, dec, ra_error, dec_error,
                   pmra, pmdec, pmra_error, pmdec_error,
                   parallax, parallax_error, phot_g_mean_mag,
                   bp_rp, ruwe, astrometric_excess_noise
            FROM {self.config['gaia_table']}
            WHERE CONTAINS(
                POINT('ICRS', {ra_deg}, {dec_deg}),
                CIRCLE('ICRS', ra, dec, {radius_arcsec/3600.0})
            ) = 1
            AND phot_g_mean_mag < 20.0
            ORDER BY phot_g_mean_mag
            """
            
            # Execute query
            job = Gaia.launch_job_async(query, dump_to_file=False)
            table = job.get_results()
            
            query_time = (time.time() - query_start) * 1000
            self._query_times.append(query_time)
            
            # Convert to GaiaSource objects
            gaia_sources = []
            for row in table:
                source = GaiaSource(
                    source_id=int(row['source_id']),
                    ra=float(row['ra']),
                    dec=float(row['dec']),
                    ra_error=float(row['ra_error']) if row['ra_error'] is not None else np.nan,
                    dec_error=float(row['dec_error']) if row['dec_error'] is not None else np.nan,
                    pmra=float(row['pmra']) if row['pmra'] is not None else None,
                    pmdec=float(row['pmdec']) if row['pmdec'] is not None else None,
                    pmra_error=float(row['pmra_error']) if row['pmra_error'] is not None else None,
                    pmdec_error=float(row['pmdec_error']) if row['pmdec_error'] is not None else None,
                    parallax=float(row['parallax']) if row['parallax'] is not None else None,
                    parallax_error=float(row['parallax_error']) if row['parallax_error'] is not None else None,
                    g_mag=float(row['phot_g_mean_mag']) if row['phot_g_mean_mag'] is not None else None,
                    bp_rp=float(row['bp_rp']) if row['bp_rp'] is not None else None,
                    ruwe=float(row['ruwe']) if row['ruwe'] is not None else None,
                    astrometric_excess_noise=float(row['astrometric_excess_noise']) if row['astrometric_excess_noise'] is not None else None,
                    epoch=self.config['reference_epoch']
                )
                gaia_sources.append(source)
            
            # Cache results
            self._cache_result(cache_key, gaia_sources)
            
            self.logger.info(f"Gaia query returned {len(gaia_sources)} sources in {query_time:.1f}ms")
            return gaia_sources
            
        except Exception as e:
            self.logger.error(f"Gaia catalog query failed: {e}")
            raise
    
    def _find_primary_source(
        self, 
        gaia_sources: List[GaiaSource], 
        ra_deg: float, 
        dec_deg: float
    ) -> Optional[GaiaSource]:
        """Find the primary (closest) Gaia source to target position."""
        
        if not gaia_sources:
            return None
        
        # Calculate angular distances
        target_coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        
        min_distance = float('inf')
        primary_source = None
        
        for source in gaia_sources:
            source_coord = SkyCoord(ra=source.ra*u.deg, dec=source.dec*u.deg, frame='icrs')
            distance = target_coord.separation(source_coord).arcsec
            
            if distance < min_distance:
                min_distance = distance
                primary_source = source
        
        return primary_source
    
    async def _analyze_proper_motion(self, source: GaiaSource) -> Optional[ProperMotionAnalysis]:
        """Analyze proper motion data for stellar vs. artificial signatures."""
        
        if not source or source.pmra is None or source.pmdec is None:
            return None
        
        # Calculate proper motion statistics
        pmra = source.pmra
        pmdec = source.pmdec
        pmra_err = source.pmra_error or 0.1  # Default error if not available
        pmdec_err = source.pmdec_error or 0.1
        
        # Total proper motion
        pm_total = np.sqrt(pmra**2 + pmdec**2)
        pm_pa = np.degrees(np.arctan2(pmra, pmdec)) % 360
        
        # Statistical significance
        pmra_significance = abs(pmra) / pmra_err if pmra_err > 0 else 0
        pmdec_significance = abs(pmdec) / pmdec_err if pmdec_err > 0 else 0
        pm_total_significance = pm_total / np.sqrt(pmra_err**2 + pmdec_err**2)
        
        # Significance test
        is_significant = pm_total_significance >= self.config['pm_significance_threshold']
        
        # Stellar vs. artificial likelihood assessment
        stellar_likelihood, artificial_likelihood = self._assess_pm_naturality(
            pmra, pmdec, pm_total, source.g_mag
        )
        
        return ProperMotionAnalysis(
            pmra_mas_yr=pmra,
            pmdec_mas_yr=pmdec,
            pm_total=pm_total,
            pm_pa=pm_pa,
            pmra_significance=pmra_significance,
            pmdec_significance=pmdec_significance,
            pm_total_significance=pm_total_significance,
            is_significant_motion=is_significant,
            stellar_likelihood=stellar_likelihood,
            artificial_likelihood=artificial_likelihood
        )
    
    def _assess_pm_naturality(
        self, 
        pmra: float, 
        pmdec: float, 
        pm_total: float, 
        g_mag: Optional[float]
    ) -> Tuple[float, float]:
        """Assess whether proper motion is consistent with stellar or artificial motion."""
        
        # Stellar proper motion distribution (simplified model)
        # Real implementation would use Galactic kinematics models
        
        stellar_score = 1.0
        artificial_score = 0.0
        
        # Check for extremely high proper motion (>500 mas/yr)
        if pm_total > 500:
            stellar_score *= 0.1  # Very unlikely for normal stars
            artificial_score += 0.5
        elif pm_total > 100:
            stellar_score *= 0.5  # Uncommon but possible for nearby stars
            artificial_score += 0.3
        
        # Check magnitude consistency
        if g_mag is not None:
            # Bright stars with high PM more likely to be genuine
            if g_mag < 12 and pm_total > 100:
                stellar_score *= 1.2
            # Faint stars with very high PM suspicious
            elif g_mag > 18 and pm_total > 200:
                stellar_score *= 0.3
                artificial_score += 0.4
        
        # Check for peculiar proper motion patterns
        pm_ratio = abs(pmra / pmdec) if pmdec != 0 else float('inf')
        if pm_ratio > 10 or pm_ratio < 0.1:
            # Very elongated PM vectors can indicate artificial objects
            artificial_score += 0.2
            stellar_score *= 0.8
        
        # Normalize scores
        total = stellar_score + artificial_score
        if total > 0:
            stellar_score /= total
            artificial_score /= total
        else:
            stellar_score = 0.5
            artificial_score = 0.5
        
        return stellar_score, artificial_score
    
    async def _analyze_parallax(self, source: GaiaSource) -> Optional[ParallaxAnalysis]:
        """Analyze parallax measurements for distance validation."""
        
        if not source or source.parallax is None:
            return None
        
        parallax = source.parallax  # mas
        parallax_error = source.parallax_error or 0.1
        
        # Statistical significance
        parallax_significance = abs(parallax) / parallax_error if parallax_error > 0 else 0
        is_significant = parallax_significance >= self.config['parallax_significance_threshold']
        
        # Distance calculation (if significant positive parallax)
        distance_pc = None
        distance_error_pc = None
        if is_significant and parallax > 0:
            distance_pc = 1000.0 / parallax  # pc
            # Error propagation
            distance_error_pc = distance_pc * (parallax_error / parallax)
        
        return ParallaxAnalysis(
            parallax_mas=parallax,
            parallax_error_mas=parallax_error,
            parallax_significance=parallax_significance,
            distance_pc=distance_pc,
            distance_error_pc=distance_error_pc,
            is_significant_parallax=is_significant
        )
    
    async def _analyze_precision(self, source: GaiaSource) -> Optional[AstrometricPrecision]:
        """Analyze astrometric precision and error budget."""
        
        if not source:
            return None
        
        # Position uncertainty ellipse
        ra_error_mas = source.ra_error
        dec_error_mas = source.dec_error
        
        # Simple ellipse calculation (would be more sophisticated in practice)
        major_axis = max(ra_error_mas, dec_error_mas)
        minor_axis = min(ra_error_mas, dec_error_mas)
        pa = 0.0  # Position angle (would calculate from covariance)
        
        # Total position uncertainty 
        total_uncertainty = np.sqrt(ra_error_mas**2 + dec_error_mas**2)
        
        # Systematic error estimate
        systematic_error = self.config['systematic_accuracy_mas']
        
        # Epoch propagation error (if proper motion available)
        epoch_error = 0.0
        if source.pmra is not None and source.pmdec is not None:
            # Error propagation from reference epoch to current
            epoch_diff = 2025.0 - source.epoch  # Years
            pm_error = np.sqrt((source.pmra_error or 0.1)**2 + (source.pmdec_error or 0.1)**2)
            epoch_error = pm_error * epoch_diff
        
        # Reference frame quality assessment
        if source.ruwe is not None and source.ruwe < 1.2:
            frame_quality = "excellent"
        elif source.ruwe is not None and source.ruwe < 1.4:
            frame_quality = "good"
        else:
            frame_quality = "fair"
        
        return AstrometricPrecision(
            position_error_ellipse_major=major_axis,
            position_error_ellipse_minor=minor_axis,
            position_error_ellipse_pa=pa,
            systematic_error_estimate=systematic_error,
            total_position_uncertainty=total_uncertainty + systematic_error,
            reference_frame_quality=frame_quality,
            epoch_propagation_error=epoch_error
        )
    
    async def _detect_artificial_signatures(
        self, 
        primary_source: GaiaSource, 
        all_sources: List[GaiaSource]
    ) -> Optional[ArtificialObjectSignature]:
        """Detect signatures indicating artificial objects."""
        
        if not primary_source:
            return None
        
        indicators = []
        anomalous_pm = False
        excess_noise = False
        color_outlier = False
        parallax_inconsistent = False
        
        # 1. Anomalous proper motion check
        if (primary_source.pmra is not None and primary_source.pmdec is not None):
            pm_total = np.sqrt(primary_source.pmra**2 + primary_source.pmdec**2)
            
            # Check for very high proper motion
            if pm_total > 200:  # mas/yr
                anomalous_pm = True
                indicators.append("high_proper_motion")
            
            # Check for unusual proper motion patterns
            if primary_source.pmra != 0 and primary_source.pmdec != 0:
                pm_ratio = abs(primary_source.pmra / primary_source.pmdec)
                if pm_ratio > 5 or pm_ratio < 0.2:
                    anomalous_pm = True
                    indicators.append("anomalous_pm_pattern")
        
        # 2. Astrometric excess noise check
        if (primary_source.ruwe is not None and 
            primary_source.ruwe > self.config['ruwe_threshold']):
            excess_noise = True
            indicators.append("excess_astrometric_noise")
        
        if (primary_source.astrometric_excess_noise is not None and 
            primary_source.astrometric_excess_noise > self.config['excess_noise_threshold']):
            excess_noise = True
            indicators.append("high_excess_noise")
        
        # 3. Color-magnitude outlier check
        if primary_source.g_mag is not None and primary_source.bp_rp is not None:
            # Compare to stellar main sequence (simplified)
            expected_color = self._expected_stellar_color(primary_source.g_mag)
            color_deviation = abs(primary_source.bp_rp - expected_color)
            
            if color_deviation > 1.0:  # Significant color deviation
                color_outlier = True
                indicators.append("color_magnitude_outlier")
        
        # 4. Parallax consistency check
        if (primary_source.parallax is not None and 
            primary_source.parallax_error is not None and
            primary_source.g_mag is not None):
            
            # Check if parallax is consistent with magnitude
            expected_distance = self._expected_distance_from_magnitude(primary_source.g_mag)
            if expected_distance is not None:
                expected_parallax = 1000.0 / expected_distance  # mas
                parallax_diff = abs(primary_source.parallax - expected_parallax)
                
                if parallax_diff > 2 * primary_source.parallax_error:
                    parallax_inconsistent = True
                    indicators.append("parallax_inconsistency")
        
        # Calculate overall artificial probability
        n_indicators = len(indicators)
        base_probability = n_indicators / 4.0  # Base on number of indicators
        
        # Weight by severity
        severity_weights = {
            "high_proper_motion": 0.3,
            "anomalous_pm_pattern": 0.2,
            "excess_astrometric_noise": 0.15,
            "high_excess_noise": 0.15,
            "color_magnitude_outlier": 0.25,
            "parallax_inconsistency": 0.2
        }
        
        weighted_probability = sum(severity_weights.get(ind, 0.1) for ind in indicators)
        artificial_probability = min(weighted_probability, 0.95)
        
        # Confidence based on data quality
        confidence = 0.8 if primary_source.ruwe and primary_source.ruwe < 1.2 else 0.6
        
        return ArtificialObjectSignature(
            anomalous_proper_motion=anomalous_pm,
            astrometric_excess_noise_flag=excess_noise,
            color_magnitude_outlier=color_outlier,
            parallax_inconsistency=parallax_inconsistent,
            artificial_probability=artificial_probability,
            artificial_indicators=indicators,
            confidence=confidence
        )
    
    def _expected_stellar_color(self, g_mag: float) -> float:
        """Estimate expected BP-RP color for given G magnitude (main sequence)."""
        # Simplified main sequence color-magnitude relation
        # Real implementation would use detailed stellar models
        if g_mag < 10:
            return 0.8 + 0.1 * (g_mag - 5)  # G-type stars
        elif g_mag < 15:
            return 1.2 + 0.15 * (g_mag - 10)  # K-type stars  
        else:
            return 1.8 + 0.1 * (g_mag - 15)   # M-type stars
    
    def _expected_distance_from_magnitude(self, g_mag: float) -> Optional[float]:
        """Estimate expected distance for given G magnitude (very rough)."""
        # Extremely simplified - real implementation would use stellar luminosity functions
        if g_mag < 5:
            return None  # Bright stars, distance highly variable
        elif g_mag < 12:
            return 10**(0.2 * (g_mag - 5) + 1)  # pc
        else:
            return 10**(0.2 * (g_mag - 5) + 1.5)  # pc
    
    def _calculate_position_residual(
        self, 
        source: GaiaSource, 
        target_ra: float, 
        target_dec: float
    ) -> Optional[float]:
        """Calculate position residual between source and target."""
        
        if not source:
            return None
        
        # Calculate angular separation
        target_coord = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg, frame='icrs')
        source_coord = SkyCoord(ra=source.ra*u.deg, dec=source.dec*u.deg, frame='icrs')
        
        separation = target_coord.separation(source_coord)
        return separation.to(u.mas).value
    
    def _calculate_validation_metrics(
        self,
        source: GaiaSource,
        pm_analysis: Optional[ProperMotionAnalysis],
        parallax_analysis: Optional[ParallaxAnalysis], 
        precision_analysis: Optional[AstrometricPrecision],
        artificial_signature: Optional[ArtificialObjectSignature]
    ) -> Tuple[float, bool, float]:
        """Calculate overall validation metrics."""
        
        quality_components = []
        
        # Position quality
        if source and source.ra_error < self.config['position_precision_mas']:
            quality_components.append(0.9)
        elif source:
            quality_components.append(0.7)
        else:
            quality_components.append(0.3)
        
        # Proper motion quality
        if pm_analysis and pm_analysis.is_significant_motion:
            if pm_analysis.stellar_likelihood > 0.7:
                quality_components.append(0.9)
            else:
                quality_components.append(0.5)
        else:
            quality_components.append(0.6)
        
        # Parallax quality
        if parallax_analysis and parallax_analysis.is_significant_parallax:
            quality_components.append(0.8)
        else:
            quality_components.append(0.5)
        
        # Precision quality
        if precision_analysis and precision_analysis.reference_frame_quality == "excellent":
            quality_components.append(0.9)
        elif precision_analysis and precision_analysis.reference_frame_quality == "good":
            quality_components.append(0.7)
        else:
            quality_components.append(0.5)
        
        # Artificial object penalty
        if artificial_signature and artificial_signature.artificial_probability > 0.7:
            quality_components.append(0.2)
        elif artificial_signature and artificial_signature.artificial_probability > 0.3:
            quality_components.append(0.6)
        else:
            quality_components.append(0.9)
        
        # Overall quality score
        quality_score = np.mean(quality_components)
        
        # Validation decision
        validation_passed = (
            quality_score > 0.6 and
            (not artificial_signature or artificial_signature.artificial_probability < 0.7)
        )
        
        # Confidence based on data completeness and quality
        confidence_factors = []
        if source and source.ruwe is not None and source.ruwe < 1.2:
            confidence_factors.append(0.9)
        if pm_analysis and pm_analysis.pm_total_significance > 3:
            confidence_factors.append(0.8)
        if parallax_analysis and parallax_analysis.is_significant_parallax:
            confidence_factors.append(0.8)
        if not confidence_factors:
            confidence_factors.append(0.5)
        
        confidence = np.mean(confidence_factors)
        
        return quality_score, validation_passed, confidence
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[GaiaSource]]:
        """Get cached Gaia query result."""
        if not self._cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check expiry
                cache_time = datetime.fromisoformat(data['timestamp'])
                expiry = timedelta(hours=self.config['cache_expiry_hours'])
                
                if datetime.now() - cache_time < expiry:
                    # Convert back to GaiaSource objects
                    sources = []
                    for source_data in data['sources']:
                        source = GaiaSource(**source_data)
                        sources.append(source)
                    return sources
                else:
                    # Cache expired, remove file
                    cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Cache read failed for {cache_key}: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, sources: List[GaiaSource]):
        """Cache Gaia query result."""
        if not self._cache_enabled:
            return
        
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'sources': [asdict(source) for source in sources]
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, default=str)
                
        except Exception as e:
            self.logger.warning(f"Cache write failed for {cache_key}: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'average_query_time_ms': np.mean(self._query_times) if self._query_times else 0,
            'cache_hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            'total_queries': len(self._query_times),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        }

# Enhancement functions for integration with validation pipeline

async def enhance_stage2_with_gaia_precision(
    neo_data: Any,
    analysis_result: Any, 
    gaia_calibrator: GaiaAstrometricCalibrator
) -> Dict[str, Any]:
    """
    Enhance Stage 2 known object cross-match with Gaia precision astrometry.
    
    This function integrates Gaia astrometric calibration into the existing
    Stage 2 validation to provide ultra-high precision positional validation
    and artificial object detection capabilities.
    
    Args:
        neo_data: NEO observation data
        analysis_result: Original analysis result
        gaia_calibrator: Gaia astrometric calibrator instance
        
    Returns:
        Enhancement dictionary with Gaia analysis results
    """
    
    try:
        # Extract position from NEO data
        ra_deg = getattr(neo_data, 'ra', None) or getattr(neo_data, 'right_ascension', None)
        dec_deg = getattr(neo_data, 'dec', None) or getattr(neo_data, 'declination', None)
        designation = getattr(neo_data, 'designation', 'unknown')
        
        if ra_deg is None or dec_deg is None:
            return {
                'gaia_analysis_available': False,
                'gaia_analysis_error': 'No position data available'
            }
        
        # Perform Gaia astrometric analysis
        gaia_result = await gaia_calibrator.analyze_position(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            designation=designation
        )
        
        # Prepare enhancement data
        enhancement = {
            'gaia_analysis_available': True,
            'gaia_astrometric_result': gaia_result,
            'gaia_validation_passed': gaia_result.validation_passed,
            'gaia_quality_score': gaia_result.astrometric_quality_score,
            'gaia_artificial_probability': gaia_result.artificial_object_signature.artificial_probability if gaia_result.artificial_object_signature else 0.0,
            'gaia_position_precision_mas': gaia_result.position_residual_mas,
            'gaia_processing_time_ms': gaia_result.processing_time_ms
        }
        
        # Add specific analysis results
        if gaia_result.proper_motion_analysis:
            enhancement['gaia_proper_motion_analysis'] = {
                'pm_total_mas_yr': gaia_result.proper_motion_analysis.pm_total,
                'pm_significance': gaia_result.proper_motion_analysis.pm_total_significance,
                'stellar_likelihood': gaia_result.proper_motion_analysis.stellar_likelihood,
                'artificial_likelihood': gaia_result.proper_motion_analysis.artificial_likelihood
            }
        
        if gaia_result.parallax_analysis:
            enhancement['gaia_parallax_analysis'] = {
                'parallax_mas': gaia_result.parallax_analysis.parallax_mas,
                'distance_pc': gaia_result.parallax_analysis.distance_pc,
                'parallax_significance': gaia_result.parallax_analysis.parallax_significance
            }
        
        if gaia_result.astrometric_precision:
            enhancement['gaia_precision_analysis'] = {
                'total_uncertainty_mas': gaia_result.astrometric_precision.total_position_uncertainty,
                'systematic_error_mas': gaia_result.astrometric_precision.systematic_error_estimate,
                'reference_frame_quality': gaia_result.astrometric_precision.reference_frame_quality
            }
        
        # Calculate enhanced plausibility score
        base_plausibility = getattr(analysis_result, 'plausibility_score', 0.7)
        gaia_bonus = 0.2 if gaia_result.validation_passed else -0.3
        artificial_penalty = -0.5 * gaia_result.artificial_object_signature.artificial_probability if gaia_result.artificial_object_signature else 0
        
        enhanced_plausibility = max(0.0, min(1.0, base_plausibility + gaia_bonus + artificial_penalty))
        enhancement['enhanced_plausibility_score'] = enhanced_plausibility
        
        return enhancement
        
    except Exception as e:
        logger.error(f"Gaia enhancement failed: {e}")
        return {
            'gaia_analysis_available': False,
            'gaia_analysis_error': str(e),
            'enhanced_plausibility_score': getattr(analysis_result, 'plausibility_score', 0.7)
        }

def create_gaia_performance_tester() -> 'GaiaPerformanceTester':
    """Create Gaia performance testing suite."""
    
    class GaiaPerformanceTester:
        """Performance testing for Gaia astrometric calibration system."""
        
        def __init__(self):
            self.calibrator = GaiaAstrometricCalibrator()
            self.test_targets = [
                (45.0, 30.0, "Test Target 1"),
                (180.0, -30.0, "Test Target 2"), 
                (270.0, 60.0, "Test Target 3")
            ]
        
        async def run_performance_test(self) -> Dict[str, Any]:
            """Run comprehensive performance test."""
            
            results = []
            start_time = time.time()
            
            for ra, dec, name in self.test_targets:
                target_start = time.time()
                
                try:
                    result = await self.calibrator.analyze_position(ra, dec, designation=name)
                    
                    results.append({
                        'target': name,
                        'success': True,
                        'processing_time_ms': result.processing_time_ms,
                        'n_sources_found': result.n_sources_found,
                        'validation_passed': result.validation_passed,
                        'quality_score': result.astrometric_quality_score
                    })
                    
                except Exception as e:
                    results.append({
                        'target': name,
                        'success': False,
                        'error': str(e),
                        'processing_time_ms': (time.time() - target_start) * 1000
                    })
            
            total_time = (time.time() - start_time) * 1000
            
            # Calculate performance metrics
            successful_tests = [r for r in results if r['success']]
            processing_times = [r['processing_time_ms'] for r in successful_tests]
            
            performance_stats = self.calibrator.get_performance_stats()
            
            return {
                'total_tests': len(self.test_targets),
                'successful_tests': len(successful_tests),
                'success_rate': len(successful_tests) / len(self.test_targets),
                'total_test_time_ms': total_time,
                'average_processing_time_ms': np.mean(processing_times) if processing_times else 0,
                'max_processing_time_ms': max(processing_times) if processing_times else 0,
                'target_performance_met': np.mean(processing_times) < 100 if processing_times else False,
                'individual_results': results,
                'performance_stats': performance_stats,
                'timestamp': datetime.now().isoformat()
            }
    
    return GaiaPerformanceTester()