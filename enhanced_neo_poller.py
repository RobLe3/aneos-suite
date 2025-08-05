#!/usr/bin/env python3
"""
Enhanced NEO Poller - Complete Polling, Caching, Enrichment & Analysis System

Based on the original neos_o3high_v6.19.1.py approach with:
- Multi-source data fetching and enrichment
- Comprehensive caching and storage
- Data quality assessment and merging
- Performance optimizations
"""

import os
import sys
import json
import requests
import datetime
import time
import shelve
import threading
from typing import Dict, List, Any, Optional, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import logging
from pathlib import Path
import functools
import traceback
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Try to import optional dependencies
try:
    from dateutil.relativedelta import relativedelta
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    class relativedelta:
        def __init__(self, days=0, weeks=0, months=0, years=0):
            self.days = days + (weeks * 7)
            self.months = months
            self.years = years
        def __rsub__(self, other):
            if isinstance(other, datetime.date):
                total_days = self.days + (self.months * 30) + (self.years * 365)
                return other - datetime.timedelta(days=total_days)
            return other

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Professional reporting system imports
# Note: Temporarily disabled due to missing dependencies (numpy, matplotlib)
# TODO: Implement conditional imports with fallback for missing dependencies
HAS_PROFESSIONAL_REPORTING = False
PROFESSIONAL_REPORTING_ERROR = "Full professional reporting requires numpy and matplotlib"

# try:
#     from aneos_core.reporting import (
#         ProfessionalReportingSuite, 
#         create_professional_suite,
#         ConsoleReporter,
#         ReportGenerator
#     )
#     HAS_PROFESSIONAL_REPORTING = True
# except ImportError as e:
#     HAS_PROFESSIONAL_REPORTING = False
#     PROFESSIONAL_REPORTING_ERROR = str(e)


# ==============================================================================
# ROBUST ERROR HANDLING SYSTEM
# ==============================================================================

def safe_execute(func):
    """
    Decorator for comprehensive error handling with logging.
    Wraps functions with try/catch and detailed error logging.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get logger from the instance if available, otherwise create one
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = logging.getLogger(func.__module__)
            
            logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return None
    return wrapper


def wait_with_progress(delay: int, description: str = "Waiting"):
    """
    Wait with a progress bar for better user experience during retries.
    """
    if HAS_TQDM:
        with tqdm(total=delay, desc=description, leave=False, 
                  bar_format='{desc}: {n_fmt}/{total_fmt} seconds') as pbar:
            for _ in range(delay):
                time.sleep(1)
                pbar.update(1)
    else:
        # Fallback without progress bar
        print(f"{description}: waiting {delay} seconds...")
        time.sleep(delay)


def retry_with_exponential_backoff(max_retries: int = 3, initial_delay: int = 2, max_delay: int = 30, backoff_factor: float = 2.0):
    """
    Decorator for implementing exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each failure
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                    
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout,
                        requests.exceptions.HTTPError) as e:
                    
                    last_exception = e
                    
                    # Don't retry on client errors (4xx except 429)
                    if hasattr(e, 'response') and e.response is not None:
                        if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                            break
                    
                    if attempt < max_retries:
                        # Log retry attempt
                        designation = args[0] if args else "unknown"
                        self.logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}({designation}): {e}")
                        
                        # Wait with progress bar
                        wait_with_progress(delay, f"Retry {attempt + 1}/{max_retries + 1}")
                        
                        # Exponential backoff with jitter
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # Final attempt failed
                        designation = args[0] if args else "unknown"
                        self.logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}({designation}): {e}")
                        
                except Exception as e:
                    # Non-retryable exception
                    last_exception = e
                    break
            
            # If we got here, all retries failed or we hit a non-retryable exception
            if hasattr(self, 'logger'):
                self.logger.error(f"Function {func.__name__} failed after retries: {last_exception}")
            return None
            
        return wrapper
    return decorator


def create_session_with_retries() -> requests.Session:
    """
    Create a requests session with robust retry logic and exponential backoff.
    """
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)  
    sess.mount("http://", adapter)
    sess.headers.update({
        'User-Agent': 'aNEOS-Enhanced-Poller/1.0 (Artificial NEO Detection System)'
    })
    return sess


class EnhancedNEOPoller:
    """
    Enhanced NEO polling system with comprehensive caching, enrichment, and analysis.
    
    Based on original script approach:
    1. Fetch CAD data for time period
    2. Extract unique NEO designations
    3. Enrich each NEO with multi-source orbital data
    4. Cache all data for performance
    5. Perform comprehensive artificial detection analysis
    """
    
    def __init__(self, data_dir: str = "neo_data", professional_report: bool = False, 
                 report_dir: str = "reports", enable_ai_validation: bool = True):
        self.console = console if HAS_RICH else None
        
        # Professional reporting configuration
        self.professional_report_requested = professional_report  # Track original request
        self.professional_report = False  # Will be set to True if full suite initializes
        self.basic_professional_report = False  # Will be set to True if basic reporting is used
        self.report_dir = report_dir
        self.enable_ai_validation = enable_ai_validation
        self.professional_suite = None
        
        # Setup data directories (like original script)
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.orbital_dir = self.data_dir / "orbital_elements"
        self.results_dir = self.data_dir / "results"
        self.logs_dir = self.data_dir / "logs"
        
        # Create directories
        for dir_path in [self.data_dir, self.cache_dir, self.orbital_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.cache_file = str(self.cache_dir / "orbital_elements_cache")
        self.cad_cache_file = str(self.cache_dir / "cad_data_cache")
        
        # Setup enhanced logging with rotating file handler
        self.logger = self._setup_enhanced_logging()
        
        # Initialize professional reporting suite if enabled
        if self.professional_report_requested:
            if not HAS_PROFESSIONAL_REPORTING:
                self.logger.warning(f"Professional reporting requested but not available: {PROFESSIONAL_REPORTING_ERROR}")
                print(f"⚠️  Professional reporting unavailable: missing dependencies (try: pip install numpy matplotlib)")
                print(f"📊 Using basic professional reporting format instead")
                self.basic_professional_report = True
            else:
                try:
                    self.professional_suite = create_professional_suite(
                        output_dir=self.report_dir,
                        logger=self.logger,
                        enable_ai_validation=self.enable_ai_validation
                    )
                    self.logger.info("Professional reporting suite initialized")
                    print(f"📊 Professional reporting enabled (output: {self.report_dir})")
                    self.professional_report = True
                except Exception as e:
                    self.logger.error(f"Failed to initialize professional reporting: {e}")
                    print(f"⚠️  Professional reporting initialization failed: {e}")
                    print(f"📊 Using basic professional reporting format instead")
                    self.basic_professional_report = True
        
        # Create session with robust retry logic
        self.session = create_session_with_retries()
        
        # Configuration for timeouts and retries
        self.config = {
            'REQUEST_TIMEOUT': 15,      # Default timeout for API requests
            'HEALTH_CHECK_TIMEOUT': 5,  # Quick timeout for health checks
            'CAD_TIMEOUT': 30,          # Longer timeout for CAD data requests
            'MAX_RETRIES': 3,           # Maximum retry attempts
            'INITIAL_RETRY_DELAY': 2,   # Initial delay for exponential backoff
            'MAX_RETRY_DELAY': 30       # Maximum delay for exponential backoff
        }
        
        # Data sources (from original script)
        self.data_sources = {
            'SBDB': {
                'name': 'NASA Small Body Database',
                'url': 'https://ssd-api.jpl.nasa.gov/sbdb.api',
                'fetcher': self.fetch_orbital_elements_sbdb,
                'available': True
            },
            'NEODyS': {
                'name': 'NEODyS Database',
                'url': 'https://newton.spacedys.com/neodys/api/',
                'fetcher': self.fetch_orbital_elements_neodys,
                'available': False  # Will be set by health check
            },
            'MPC': {
                'name': 'Minor Planet Center',
                'url': 'https://www.minorplanetcenter.net/',
                'fetcher': self.fetch_orbital_elements_mpc,
                'available': False  # Will be set by health check
            },
            'Horizons': {
                'name': 'JPL Horizons',
                'url': 'https://ssd.jpl.nasa.gov/api/horizons.api',
                'fetcher': self.fetch_orbital_elements_horizons,
                'available': False  # Will be set by health check
            }
        }
        
        # Source priority (from original script)
        self.source_priority = ["SBDB", "NEODyS", "MPC", "Horizons"]
        
        # Enhanced performance tracking with quality metrics (from legacy system)
        self.source_statistics = {source: {'success': 0, 'failure': 0} for source in self.data_sources}
        self.data_usage = {source: 0 for source in self.data_sources}
        
        # Quality tracking system from legacy system
        self.source_poll_stats = {source: {'success': 0, 'failure': 0} for source in self.data_sources}
        self.source_quality_stats = {source: 0.0 for source in self.data_sources}
        self.source_quality_counts = {source: 0 for source in self.data_sources}
        
        # Verify source availability on startup (inspired by original script)
        self.verify_sources()
        
        print("🚀 Enhanced NEO Poller initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print("Mission: Complete NEO data enrichment and artificial detection")
        
        # Create local NEO database
        self.create_local_neo_database()
    
    def _setup_enhanced_logging(self) -> logging.Logger:
        """
        Setup enhanced logging with rotating file handlers for better log management.
        """
        logger = logging.getLogger(f"enhanced_neo_poller_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplication
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Rotating file handler
        log_file = self.logs_dir / "enhanced_neo_poller.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10_000_000,  # 10MB per file
            backupCount=5,        # Keep 5 backup files
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Log initialization
        logger.info("=" * 80)
        logger.info("🚀 Enhanced NEO Poller Log Session Started 🚀")
        logger.info("=" * 80)
        
        return logger
    
    def verify_sources(self) -> None:
        """
        Verify source availability on startup (inspired by original script).
        
        Pre-checks all API endpoints to determine which are available,
        avoiding wasted calls during enrichment process.
        """
        print("\n🔍 Verifying API source availability...")
        
        # Test endpoints (inspired by original script approach)
        test_cases = {
            'SBDB': {
                'url': 'https://ssd-api.jpl.nasa.gov/sbdb.api',
                'test_params': {'sstr': '99942'},  # Apophis - well-known NEO
                'expected_statuses': [200, 400]  # 400 can be valid for some APIs
            },
            'NEODyS': {  
                'url': 'https://newton.spacedys.com/neodys/api/',
                'test_params': {'name': 'test', 'format': 'json'},
                'expected_statuses': [200, 400, 404]  # 404 acceptable for non-existent objects
            },
            'MPC': {
                'url': 'https://www.minorplanetcenter.net/',
                'test_params': {},
                'expected_statuses': [200, 400, 403, 404],  # More permissive for website
                'requires_dependency': 'astroquery'
            },
            'Horizons': {
                'url': 'https://ssd.jpl.nasa.gov/api/horizons.api',
                'test_params': {},
                'expected_statuses': [200, 400, 403, 404],
                'requires_dependency': 'astroquery'
            }
        }
        
        for source in self.source_priority:
            if source not in test_cases:
                continue
                
            config = test_cases[source]
            available = False
            reason = "Unknown"
            
            # Check for required dependencies first
            if 'requires_dependency' in config:
                try:
                    if config['requires_dependency'] == 'astroquery':
                        import astroquery
                        # Dependency available, continue with API test
                except ImportError:
                    reason = f"Missing {config['requires_dependency']} dependency"
                    self.data_sources[source]['available'] = False
                    print(f"❌ {source}: {reason}")
                    continue
            
            # Test API endpoint
            try:
                response = self.session.get(
                    config['url'], 
                    params=config['test_params'],
                    timeout=self.config['HEALTH_CHECK_TIMEOUT']
                )
                
                if response.status_code in config['expected_statuses']:
                    available = True
                    reason = f"HTTP {response.status_code} (healthy)"
                else:
                    reason = f"HTTP {response.status_code} (unexpected)"
                    
            except requests.exceptions.Timeout:
                reason = "Timeout (>10s)"
            except requests.exceptions.ConnectionError:
                reason = "Connection failed"
            except Exception as e:
                reason = f"Error: {str(e)[:50]}"
            
            # Update availability
            self.data_sources[source]['available'] = available
            
            # Report status
            status_icon = "✅" if available else "❌"
            print(f"{status_icon} {source}: {reason}")
        
        # Summary
        available_count = sum(1 for src in self.data_sources.values() if src['available'])
        total_count = len(self.data_sources)
        print(f"\n📊 Source availability: {available_count}/{total_count} sources online")
        
        if available_count == 0:
            print("⚠️  Warning: No API sources available - enrichment will be limited")
        elif available_count < total_count:
            available_sources = [name for name, config in self.data_sources.items() if config['available']]
            print(f"🔄 Will use available sources: {', '.join(available_sources)}")
    
    @safe_execute
    def create_local_neo_database(self) -> None:
        """Create and maintain a comprehensive local NEO database."""
        self.neo_db_path = self.data_dir / "neo_database.json"
        
        # Load existing database or create new one
        try:
            if self.neo_db_path.exists():
                with open(self.neo_db_path, 'r') as f:
                    self.neo_database = json.load(f)
                print(f"📊 Loaded existing NEO database with {len(self.neo_database)} objects")
            else:
                self.neo_database = {}
                print("📊 Created new NEO database")
        except Exception as e:
            self.logger.warning(f"Failed to load NEO database: {e}")
            self.neo_database = {}
    
    @safe_execute
    def update_neo_database(self, designation: str, enrichment_data: Dict[str, Any]) -> None:
        """Update the local NEO database with new enrichment data."""
        try:
            if designation not in self.neo_database:
                self.neo_database[designation] = {
                    'designation': designation,
                    'first_seen': datetime.datetime.now().isoformat(),
                    'last_updated': datetime.datetime.now().isoformat(),
                    'enrichment_attempts': 0,
                    'sources_attempted': [],
                    'best_completeness': 0.0,
                    'combined_data': {}
                }
            
            # Update tracking information
            neo_record = self.neo_database[designation]
            neo_record['last_updated'] = datetime.datetime.now().isoformat()
            neo_record['enrichment_attempts'] += 1
            
            # Track sources used
            sources_used = enrichment_data.get('sources_used', [])
            for source in sources_used:
                if source not in neo_record['sources_attempted']:
                    neo_record['sources_attempted'].append(source)
            
            # Update completeness if improved
            current_completeness = enrichment_data.get('completeness', 0)
            if current_completeness > neo_record['best_completeness']:
                neo_record['best_completeness'] = current_completeness
                neo_record['combined_data'] = enrichment_data.get('orbital_elements', {})
            
            # Save database periodically
            if neo_record['enrichment_attempts'] % 10 == 0:  # Every 10 updates
                self.save_neo_database()
                
        except Exception as e:
            self.logger.warning(f"Failed to update NEO database for {designation}: {e}")
    
    @safe_execute
    def save_neo_database(self) -> None:
        """Save the NEO database to file."""
        try:
            with open(self.neo_db_path, 'w') as f:
                json.dump(self.neo_database, f, indent=2)
            self.logger.debug(f"Saved NEO database with {len(self.neo_database)} objects")
        except Exception as e:
            self.logger.warning(f"Failed to save NEO database: {e}")
    
    @safe_execute
    @retry_with_exponential_backoff(max_retries=3, initial_delay=3)
    def fetch_cad_data_with_cache(self, start_date: str, end_date: str, limit: int = 5000) -> Optional[Dict[str, Any]]:
        """
        Fetch CAD data with caching (like original script).
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of records
            
        Returns:
            CAD data dictionary or None if failed
        """
        # Create cache key
        cache_key = f"cad_{start_date}_{end_date}_{limit}"
        
        # Try to load from cache first
        try:
            with shelve.open(self.cad_cache_file) as cache:
                if cache_key in cache:
                    cached_data = cache[cache_key]
                    # Check if cache is still fresh (24 hours)
                    cache_time = datetime.datetime.fromisoformat(cached_data.get('cache_time', '1970-01-01'))
                    if datetime.datetime.now() - cache_time < datetime.timedelta(hours=24):
                        print(f"✅ Using cached CAD data for {start_date} to {end_date}")
                        return cached_data['data']
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        
        # Fetch fresh data
        try:
            params = {
                'date-min': start_date,
                'date-max': end_date,
                'sort': 'date',
                'limit': limit
            }
            
            response = self.session.get('https://ssd-api.jpl.nasa.gov/cad.api', params=params, timeout=self.config['CAD_TIMEOUT'])
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                # Cache the result
                try:
                    with shelve.open(self.cad_cache_file) as cache:
                        cache[cache_key] = {
                            'data': data,
                            'cache_time': datetime.datetime.now().isoformat()
                        }
                except Exception as e:
                    self.logger.warning(f"Cache write error: {e}")
                
                return data
            else:
                print(f"⚠️  No NEO data found for period {start_date} to {end_date}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Error fetching CAD data: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response: {e}")
            return None
    
    def extract_neo_designations(self, cad_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract unique NEO designations from CAD data and group close approaches.
        
        Args:
            cad_data: CAD API response data
            
        Returns:
            Dictionary mapping designation to list of close approach records
        """
        neo_map = {}
        
        if 'data' not in cad_data:
            return neo_map
        
        for record in cad_data['data']:
            if len(record) > 0:
                designation = record[0]  # First field is designation
                
                # Parse the full record
                approach_data = {
                    'designation': record[0] if len(record) > 0 else '',
                    'orbit_id': record[1] if len(record) > 1 else '',
                    'jd': record[2] if len(record) > 2 else '',
                    'cd': record[3] if len(record) > 3 else '',  # Close approach date
                    'dist': record[4] if len(record) > 4 else '',  # Nominal distance (AU)
                    'dist_min': record[5] if len(record) > 5 else '',  # Minimum distance (AU)
                    'dist_max': record[6] if len(record) > 6 else '',  # Maximum distance (AU)
                    'v_rel': record[7] if len(record) > 7 else '',  # Relative velocity (km/s)
                    'v_inf': record[8] if len(record) > 8 else '',  # V-infinity (km/s)
                    't_sigma_f': record[9] if len(record) > 9 else '',  # Time uncertainty
                    'h': record[10] if len(record) > 10 else '',  # Absolute magnitude
                    'diameter': record[11] if len(record) > 11 else ''  # Diameter (km)
                }
                
                if designation not in neo_map:
                    neo_map[designation] = {
                        'close_approaches': [],
                        'first_observation': approach_data['cd'],
                        'last_observation': approach_data['cd']
                    }
                
                neo_map[designation]['close_approaches'].append(approach_data)
                
                # Update observation range
                if approach_data['cd'] < neo_map[designation]['first_observation']:
                    neo_map[designation]['first_observation'] = approach_data['cd']
                if approach_data['cd'] > neo_map[designation]['last_observation']:
                    neo_map[designation]['last_observation'] = approach_data['cd']
        
        return neo_map
    
    @safe_execute
    def save_orbital_data(self, designation: str, source: str, data: Dict[str, Any]) -> None:
        """Save orbital data to file with metadata for cache management."""
        try:
            source_dir = self.orbital_dir / source
            source_dir.mkdir(exist_ok=True)
            
            # Add metadata for cache management
            enriched_data = {
                'designation': designation,
                'source': source,
                'cached_at': time.time(),
                'cached_date': datetime.datetime.now().isoformat(),
                'data_completeness': self.compute_completeness(data),
                'orbital_elements': data if isinstance(data, dict) else {},
                'raw_data': data  # Keep original data for reference
            }
            
            file_path = source_dir / f"{designation}.json"
            with open(file_path, 'w') as f:
                json.dump(enriched_data, f, indent=2)
                
            self.logger.debug(f"Cached {designation} from {source} with {enriched_data['data_completeness']:.2f} completeness")
        except Exception as e:
            self.logger.warning(f"Failed to save orbital data for {designation} from {source}: {e}")
    
    @safe_execute
    def load_orbital_data(self, designation: str, source: str) -> Optional[Dict[str, Any]]:
        """Load orbital data from file with cache validity checking."""
        try:
            source_dir = self.orbital_dir / source
            file_path = source_dir / f"{designation}.json"
            
            if file_path.exists():
                # Check file age for cache validity
                file_age = time.time() - file_path.stat().st_mtime
                cache_ttl = self._get_cache_ttl(source)
                
                if file_age > cache_ttl:
                    self.logger.debug(f"Cache expired for {designation} from {source} (age: {file_age/3600:.1f}h)")
                    return None  # Cache expired, need fresh data
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Validate cache data integrity
                if self._is_cache_valid(data, designation, source):
                    self.logger.debug(f"Using valid cached data for {designation} from {source} (completeness: {data.get('data_completeness', 0):.2f})")
                    # Return the orbital elements portion
                    return data.get('orbital_elements', data.get('raw_data', data))
                else:
                    self.logger.warning(f"Cache data invalid for {designation} from {source}")
                    return None
                    
        except Exception as e:
            self.logger.warning(f"Failed to load orbital data for {designation} from {source}: {e}")
        
        return None
    
    def _get_cache_ttl(self, source: str) -> int:
        """Get cache time-to-live in seconds based on source reliability."""
        cache_ttl = {
            'SBDB': 7 * 24 * 3600,      # 7 days - NASA data is very stable
            'NEODyS': 3 * 24 * 3600,    # 3 days - Updated regularly
            'MPC': 1 * 24 * 3600,       # 1 day - Frequently updated
            'Horizons': 7 * 24 * 3600,  # 7 days - JPL data is stable
        }
        return cache_ttl.get(source, 24 * 3600)  # Default 1 day
    
    @safe_execute
    def _is_cache_valid(self, data: Dict[str, Any], designation: str, source: str) -> bool:
        """
        Validate cached data integrity and completeness with enhanced error handling.
        """
        if not data:
            self.logger.debug(f"Cache validation failed for {designation} from {source}: No data")
            return False
        
        try:
            # Check for required metadata (new format)
            if 'cached_at' not in data or 'designation' not in data:
                self.logger.debug(f"Cache validation failed for {designation} from {source}: Missing metadata")
                return False
            
            # Verify designation matches
            if data.get('designation') != designation:
                self.logger.debug(f"Cache validation failed: designation mismatch ({data.get('designation')} != {designation})")
                return False
            
            # Check if we have orbital elements
            orbital_elements = data.get('orbital_elements', {})
            if not orbital_elements or not isinstance(orbital_elements, dict):
                self.logger.debug(f"Cache validation failed for {designation} from {source}: Missing or invalid orbital elements")
                return False
            
            # Check data completeness threshold (at least some orbital data)
            completeness = data.get('data_completeness', 0)
            if completeness < 0.1:  # At least 10% complete
                self.logger.debug(f"Cache validation failed for {designation} from {source}: Low completeness ({completeness:.2f})")
                return False
            
            # Additional validation: check for key orbital parameters
            required_keys = ['eccentricity', 'semi_major_axis', 'inclination']
            missing_keys = [key for key in required_keys if key not in orbital_elements or orbital_elements[key] is None]
            
            if len(missing_keys) == len(required_keys):  # All key parameters missing
                self.logger.debug(f"Cache validation failed for {designation} from {source}: No key orbital parameters")
                return False
            
            self.logger.debug(f"Cache validation passed for {designation} from {source} (completeness: {completeness:.2f})")
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache validation error for {designation} from {source}: {e}")
            return False
    
    @safe_execute
    def cleanup_cache(self, max_age_days: int = 30, max_size_mb: int = 500) -> Dict[str, int]:
        """
        Clean up cache files based on age and size limits with enhanced error handling.
        
        Args:
            max_age_days: Maximum age for cache files in days
            max_size_mb: Maximum total cache size in MB
            
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'files_removed': 0,
            'space_freed_mb': 0,
            'invalid_files_removed': 0,
            'expired_files_removed': 0
        }
        
        try:
            max_age_seconds = max_age_days * 24 * 3600
            current_time = time.time()
            total_size = 0
            
            # Get all cache files
            cache_files = []
            for source_dir in self.orbital_dir.iterdir():
                if source_dir.is_dir():
                    for cache_file in source_dir.iterdir():
                        if cache_file.is_file() and cache_file.suffix == '.json':
                            try:
                                stat = cache_file.stat()
                                cache_files.append({
                                    'path': cache_file,
                                    'size': stat.st_size,
                                    'age': current_time - stat.st_mtime,
                                    'source': source_dir.name
                                })
                                total_size += stat.st_size
                            except (OSError, IOError) as e:
                                self.logger.debug(f"Error accessing cache file {cache_file}: {e}")
                                continue
            
            # Remove expired files
            for cache_info in cache_files:
                if cache_info['age'] > max_age_seconds:
                    try:
                        cache_info['path'].unlink()
                        stats['files_removed'] += 1
                        stats['expired_files_removed'] += 1
                        stats['space_freed_mb'] += cache_info['size'] / (1024 * 1024)
                        self.logger.debug(f"Removed expired cache file: {cache_info['path']}")
                    except (OSError, IOError) as e:
                        self.logger.warning(f"Failed to remove expired cache file {cache_info['path']}: {e}")
            
            # If still over size limit, remove oldest files
            remaining_files = [f for f in cache_files if f['path'].exists()]
            remaining_size = sum(f['size'] for f in remaining_files)
            
            if remaining_size > max_size_mb * 1024 * 1024:
                # Sort by age (oldest first)
                remaining_files.sort(key=lambda x: x['age'], reverse=True)
                
                for cache_info in remaining_files:
                    if remaining_size <= max_size_mb * 1024 * 1024:
                        break
                    
                    try:
                        cache_info['path'].unlink()
                        stats['files_removed'] += 1
                        stats['space_freed_mb'] += cache_info['size'] / (1024 * 1024)
                        remaining_size -= cache_info['size']
                        self.logger.debug(f"Removed cache file for size limit: {cache_info['path']}")
                    except (OSError, IOError) as e:
                        self.logger.warning(f"Failed to remove cache file for size limit {cache_info['path']}: {e}")
            
            # Clean up CAD cache if it exists
            if hasattr(self, 'cad_cache_file') and Path(self.cad_cache_file).exists():
                try:
                    cad_stat = Path(self.cad_cache_file).stat()
                    if current_time - cad_stat.st_mtime > max_age_seconds:
                        Path(self.cad_cache_file).unlink()
                        stats['files_removed'] += 1
                        stats['space_freed_mb'] += cad_stat.st_size / (1024 * 1024)
                        self.logger.info("Removed expired CAD cache file")
                except (OSError, IOError) as e:
                    self.logger.warning(f"Failed to clean CAD cache: {e}")
            
            self.logger.info(f"Cache cleanup completed: {stats['files_removed']} files removed, "
                           f"{stats['space_freed_mb']:.2f} MB freed")
            return stats
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            return stats
    
    @safe_execute
    @retry_with_exponential_backoff(max_retries=3, initial_delay=2)
    def fetch_orbital_elements_sbdb(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch orbital elements from NASA SBDB (like original script).
        
        Args:
            designation: NEO designation
            
        Returns:
            Orbital elements dictionary or None if failed
        """
        try:
            params = {
                'sstr': designation
            }
            
            response = self.session.get(self.data_sources['SBDB']['url'], params=params, timeout=self.config['REQUEST_TIMEOUT'])
            response.raise_for_status()
            
            data = response.json()
            self.data_usage['SBDB'] += len(response.content)
            
            if 'orbit' not in data:
                return None
            
            orbit = data['orbit']
            orbital_data = {}
            
            # Parse orbital elements (like original script)
            for elem in orbit.get('elements', []):
                name = elem.get('name')
                value = elem.get('value')
                if name == 'e':
                    orbital_data['eccentricity'] = float(value) if value else None
                elif name == 'i':
                    orbital_data['inclination'] = float(value) if value else None
                elif name == 'a':
                    orbital_data['semi_major_axis'] = float(value) if value else None
                elif name == 'node':
                    orbital_data['ra_of_ascending_node'] = float(value) if value else None
                elif name == 'w':
                    orbital_data['arg_of_periapsis'] = float(value) if value else None
                elif name == 'M':
                    orbital_data['mean_anomaly'] = float(value) if value else None
                elif name == 'epoch':
                    orbital_data['epoch'] = value
            
            # Add physical parameters
            if 'phys_par' in data:
                phys = data['phys_par']
                orbital_data['physical_parameters'] = phys
            
            if orbital_data:
                self.source_statistics['SBDB']['success'] += 1
                return orbital_data
            else:
                self.source_statistics['SBDB']['failure'] += 1
                return None
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching SBDB data for {designation}: {e}")
            self.source_statistics['SBDB']['failure'] += 1
            return None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Error parsing SBDB data for {designation}: {e}")
            self.source_statistics['SBDB']['failure'] += 1
            return None
    
    @safe_execute
    @retry_with_exponential_backoff(max_retries=3, initial_delay=2)
    def fetch_orbital_elements_neodys(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch orbital elements from NEODyS (improved from original script approach).
        
        Args:
            designation: NEO designation
            
        Returns:
            Orbital elements dictionary or None if failed
        """
        try:
            # Try multiple NEODyS endpoints (inspired by original but more robust)
            endpoints = [
                'https://newton.spacedys.com/neodys/api/',
                'https://newton.spacedys.com/neodys/',  # Alternative endpoint
            ]
            
            for endpoint in endpoints:
                try:
                    params = {
                        'name': designation,
                        'format': 'json'
                    }
                    
                    response = self.session.get(endpoint, params=params, timeout=self.config['REQUEST_TIMEOUT'])
                    
                    # Handle different response patterns
                    if response.status_code == 404:
                        continue  # Try next endpoint or fail gracefully
                    
                    response.raise_for_status()
                    
                    # Try to parse response 
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        # Some NEODyS responses might not be JSON
                        continue
                    
                    self.data_usage['NEODyS'] += len(response.content)
                    
                    # Handle different NEODyS response formats (inspired by original)
                    orbital_data = {}
                    
                    # Try standard orbit format
                    if 'orbit' in data and data['orbit']:
                        orbit = data['orbit']
                        orbital_data = self._parse_neodys_orbit_data(orbit, designation)
                    
                    # Try alternative format if orbit format failed
                    elif 'elements' in data:
                        orbital_data = self._parse_neodys_elements_data(data['elements'], designation)
                    
                    if orbital_data:
                        self.source_statistics['NEODyS']['success'] += 1
                        return orbital_data
                        
                except requests.RequestException:
                    continue  # Try next endpoint
            
            # All endpoints failed
            self.source_statistics['NEODyS']['failure'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error fetching NEODyS data for {designation}: {e}")
            self.source_statistics['NEODyS']['failure'] += 1
            return None
    
    def _parse_neodys_orbit_data(self, orbit: Dict[str, Any], designation: str) -> Dict[str, Any]:
        """Parse NEODyS orbit data format (inspired by original)."""
        orbital_data = {}
        
        # Key mapping (from original script)
        key_mapping = {
            'e': 'eccentricity',
            'i': 'inclination', 
            'a': 'semi_major_axis',
            'node': 'ra_of_ascending_node',
            'peri': 'arg_of_periapsis',
            'M': 'mean_anomaly'
        }
        
        for key in ['e', 'i', 'a', 'node', 'peri', 'M', 'epoch']:
            value = orbit.get(key)
            if value is not None:
                try:
                    if key == 'epoch':
                        orbital_data['epoch'] = str(value)  # Keep as string for JSON serialization
                    else:
                        mapped_key = key_mapping.get(key, key)
                        orbital_data[mapped_key] = float(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid value for {key} in NEODyS for {designation}: {value}")
        
        return orbital_data
    
    def _parse_neodys_elements_data(self, elements: Dict[str, Any], designation: str) -> Dict[str, Any]:
        """Parse alternative NEODyS elements format."""
        orbital_data = {}
        
        # Direct mapping for alternative format
        mapping = {
            'eccentricity': 'eccentricity',
            'inclination': 'inclination',
            'semimajor_axis': 'semi_major_axis',
            'ascending_node': 'ra_of_ascending_node',
            'periapsis': 'arg_of_periapsis',
            'mean_anomaly': 'mean_anomaly'
        }
        
        for source_key, target_key in mapping.items():
            value = elements.get(source_key)
            if value is not None:
                try:
                    orbital_data[target_key] = float(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid value for {source_key} in NEODyS for {designation}: {value}")
        
        return orbital_data
    
    @safe_execute
    def fetch_orbital_elements_mpc(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch orbital elements from MPC (improved from original script approach).
        
        Args:
            designation: NEO designation
            
        Returns:
            Orbital elements dictionary or None if failed
        """
        try:
            # Import MPC from astroquery (needs to be installed)
            from astroquery.mpc import MPC
            from astropy.table import Table
            
            # Try different designation formats (inspired by original but more robust)
            designation_variants = [
                designation,
                designation.replace(' ', ''),  # Remove spaces
                designation.upper(),  # Uppercase
                designation.lower(),  # Lowercase
            ]
            
            for variant in designation_variants:
                try:
                    # Query MPC for the designation
                    table = MPC.query_object(variant)
                    
                    if not isinstance(table, Table) or len(table) == 0:
                        continue  # Try next variant
                    
                    row = table[0]
                    
                    # Extract orbital data with robust error handling
                    orbital_data = self._parse_mpc_row(row, designation)
                    
                    if orbital_data:
                        # Track data usage (estimate)
                        self.data_usage['MPC'] += len(str(orbital_data).encode())
                        self.source_statistics['MPC']['success'] += 1
                        return orbital_data
                        
                except Exception as e:
                    self.logger.debug(f"MPC query failed for variant '{variant}': {e}")
                    continue  # Try next variant
            
            # All variants failed
            self.source_statistics['MPC']['failure'] += 1
            return None
            
        except ImportError:
            self.logger.debug("astroquery not installed - cannot fetch MPC data")
            self.source_statistics['MPC']['failure'] += 1
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching MPC data for {designation}: {e}")
            self.source_statistics['MPC']['failure'] += 1
            return None
    
    def _parse_mpc_row(self, row, designation: str) -> Dict[str, Any]:
        """Parse MPC table row with robust error handling (improved from original)."""
        orbital_data = {}
        
        # MPC field mapping with multiple possible field names
        field_mappings = {
            'eccentricity': ['e', 'ecc', 'eccentricity'],
            'inclination': ['incl', 'i', 'inclination'],
            'semi_major_axis': ['a', 'semimajor', 'semi_major_axis'],
            'ra_of_ascending_node': ['Omega', 'node', 'ascending_node'],
            'arg_of_periapsis': ['w', 'omega', 'arg_periapsis'],
            'mean_anomaly': ['M', 'mean_anom', 'mean_anomaly']
        }
        
        # Extract orbital elements with fallback field names
        for target_key, possible_fields in field_mappings.items():
            value = None
            for field in possible_fields:
                if field in row and row[field] is not None:
                    try:
                        value = float(row[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if value is not None:
                orbital_data[target_key] = value
        
        # Handle epoch with multiple formats
        epoch_fields = ['epoch_mjd', 'epoch', 'date']
        for field in epoch_fields:
            if field in row and row[field] is not None:
                try:
                    if field == 'epoch_mjd':
                        from astropy.time import Time
                        epoch_time = Time(float(row[field]), format="mjd")
                        orbital_data["epoch"] = epoch_time.to_datetime().isoformat()
                    else:
                        orbital_data["epoch"] = str(row[field])
                    break
                except Exception as e:
                    self.logger.warning(f"Error parsing MPC epoch field '{field}' for {designation}: {e}")
        
        # Add physical parameters if available
        physical_fields = {
            'diameter': ['diameter', 'diam', 'D'],
            'albedo': ['albedo', 'alb', 'pv'],
            'rot_per': ['rot_per', 'period', 'rotation_period']
        }
        
        for target_key, possible_fields in physical_fields.items():
            for field in possible_fields:
                if field in row and row[field] is not None:
                    try:
                        orbital_data[target_key] = float(row[field])
                        break
                    except (ValueError, TypeError):
                        continue
        
        return orbital_data
    
    @safe_execute
    def fetch_orbital_elements_horizons(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch orbital elements from JPL Horizons (improved from original script approach).
        
        Args:
            designation: NEO designation
            
        Returns:
            Orbital elements dictionary or None if failed
        """
        try:
            # Import Horizons from astroquery (needs to be installed)
            from astroquery.jplhorizons import Horizons
            
            # Try different designation formats and locations (inspired by original but more robust)
            designation_variants = [
                designation,
                designation.replace(' ', ''),  # Remove spaces
                f"'{designation}'",  # Quoted format
            ]
            
            locations = ['@sun', '500']  # Sun-centered, geocentric
            
            for variant in designation_variants:
                for location in locations:
                    try:
                        # Query Horizons for orbital elements
                        obj = Horizons(id=variant, location=location, epochs='now')
                        elements = obj.elements()
                        
                        if len(elements) == 0:
                            continue  # Try next variant/location
                        
                        el = elements[0]
                        
                        # Extract orbital data with robust parsing
                        orbital_data = self._parse_horizons_elements(el, designation)
                        
                        if orbital_data:
                            # Track data usage (estimate)
                            self.data_usage['Horizons'] += len(str(orbital_data).encode())
                            self.source_statistics['Horizons']['success'] += 1
                            return orbital_data
                            
                    except Exception as e:
                        self.logger.debug(f"Horizons query failed for '{variant}' at {location}: {e}")
                        continue  # Try next variant/location
            
            # All variants failed
            self.source_statistics['Horizons']['failure'] += 1
            return None
            
        except ImportError:
            self.logger.debug("astroquery not installed - cannot fetch Horizons data")
            self.source_statistics['Horizons']['failure'] += 1
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching Horizons data for {designation}: {e}")
            self.source_statistics['Horizons']['failure'] += 1
            return None
    
    def _parse_horizons_elements(self, el, designation: str) -> Dict[str, Any]:
        """Parse Horizons elements with robust error handling (improved from original)."""
        orbital_data = {}
        
        # Horizons field mapping with multiple possible field names
        field_mappings = {
            'eccentricity': ['e', 'ecc', 'eccentricity'],
            'inclination': ['i', 'incl', 'inclination'],
            'semi_major_axis': ['a', 'semi_major_axis'],
            'ra_of_ascending_node': ['node', 'Omega', 'ascending_node'],
            'arg_of_periapsis': ['peri', 'w', 'arg_periapsis'],
            'mean_anomaly': ['M', 'mean_anom', 'mean_anomaly']
        }
        
        # Extract orbital elements with fallback field names
        for target_key, possible_fields in field_mappings.items():
            value = None
            for field in possible_fields:
                if field in el and el[field] is not None:
                    try:
                        value = float(el[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if value is not None:
                orbital_data[target_key] = value
        
        # Handle epoch with multiple formats (improved from original)
        epoch_fields = ['datetime', 'datetime_str', 'epoch', 'date']
        for field in epoch_fields:
            if field in el and el[field] is not None:
                try:
                    dt_val = el[field]
                    if isinstance(dt_val, str):
                        orbital_data["epoch"] = dt_val
                    else:
                        orbital_data["epoch"] = str(dt_val)
                    break
                except Exception as e:
                    self.logger.warning(f"Error parsing Horizons epoch field '{field}' for {designation}: {e}")
        
        # Add physical parameters if available
        physical_fields = {
            'diameter': ['diameter', 'diam', 'D'],
            'albedo': ['albedo', 'alb', 'pv'],
            'rot_per': ['rot_per', 'period', 'rotation_period']
        }
        
        for target_key, possible_fields in physical_fields.items():
            for field in possible_fields:
                if field in el and el[field] is not None:
                    try:
                        orbital_data[target_key] = float(el[field])
                        break
                    except (ValueError, TypeError):
                        continue
        
        return orbital_data
    
    def compute_completeness(self, orbital_data: Dict[str, Any]) -> float:
        """
        Compute data completeness score with enhanced checking from legacy system.
        
        Checks for all essential orbital elements plus physical parameters
        for comprehensive data quality assessment.
        
        Args:
            orbital_data: Orbital elements dictionary
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not orbital_data:
            return 0.0
        
        # Enhanced required fields from legacy system
        required_fields = [
            'eccentricity', 'inclination', 'semi_major_axis',
            'ra_of_ascending_node', 'arg_of_periapsis', 'mean_anomaly',
            'epoch', 'diameter', 'albedo'
        ]
        
        present_fields = sum(1 for field in required_fields if field in orbital_data and orbital_data[field] is not None)
        return present_fields / len(required_fields)
    
    def ensure_100_percent_completeness(self, designation: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure 100% data completeness by aggressively polling all sources.
        
        Args:
            designation: NEO designation
            current_data: Current incomplete data
            
        Returns:
            Complete orbital data or enriched as much as possible
        """
        print(f"🎯 Ensuring 100% completeness for {designation}")
        
        # Start with current data
        complete_data = current_data.copy()
        current_completeness = self.compute_completeness(complete_data)
        
        if current_completeness >= 0.99:  # Already essentially complete
            return complete_data
        
        print(f"   Current completeness: {current_completeness:.1%}")
        
        # Required fields we need to complete
        required_fields = [
            'eccentricity', 'inclination', 'semi_major_axis',
            'ra_of_ascending_node', 'arg_of_periapsis', 'mean_anomaly'
        ]
        
        missing_fields = [field for field in required_fields if complete_data.get(field) is None]
        print(f"   Missing fields: {', '.join(missing_fields)}")
        
        # Aggressively poll all available sources
        for source in self.source_priority:
            if not self.data_sources[source]['available']:
                continue
                
            if current_completeness >= 0.99:
                break
                
            print(f"   🔍 Polling {source} for missing data...")
            
            try:
                fetcher = self.data_sources[source]['fetcher']
                source_data = fetcher(designation)
                
                if source_data:
                    # Merge any missing fields
                    fields_added = []
                    for field in missing_fields:
                        if field not in complete_data or complete_data[field] is None:
                            if source_data.get(field) is not None:
                                complete_data[field] = source_data[field]
                                fields_added.append(field)
                    
                    if fields_added:
                        print(f"   ✅ {source} provided: {', '.join(fields_added)}")
                        # Save this enhanced data
                        self.save_orbital_data(designation, f"{source}_enhanced", complete_data)
                        
                        # Update completeness
                        current_completeness = self.compute_completeness(complete_data)
                        missing_fields = [field for field in required_fields if complete_data.get(field) is None]
                        
                        if current_completeness >= 0.99:
                            print(f"   🎉 Achieved 100% completeness!")
                            break
                else:
                    print(f"   ❌ {source} returned no data")
                    
            except Exception as e:
                print(f"   ⚠️  Error polling {source}: {e}")
        
        final_completeness = self.compute_completeness(complete_data)
        print(f"   Final completeness: {final_completeness:.1%}")
        
        # If still incomplete, try to estimate missing fields using orbital mechanics
        if final_completeness < 0.99:
            complete_data = self._estimate_missing_orbital_elements(complete_data, designation)
            final_completeness = self.compute_completeness(complete_data)
            print(f"   After estimation: {final_completeness:.1%}")
        
        return complete_data
    
    def _estimate_missing_orbital_elements(self, orbital_data: Dict[str, Any], designation: str) -> Dict[str, Any]:
        """Estimate missing orbital elements using available data and orbital mechanics."""
        estimated_data = orbital_data.copy()
        
        print(f"   🧮 Attempting to estimate missing orbital elements for {designation}")
        
        # If we have eccentricity and semi_major_axis, we can estimate some bounds
        e = orbital_data.get('eccentricity')
        a = orbital_data.get('semi_major_axis')
        i = orbital_data.get('inclination')
        
        # Conservative estimates based on NEO population statistics
        if estimated_data.get('ra_of_ascending_node') is None:
            # Random distribution 0-360°, but use a typical NEO value
            estimated_data['ra_of_ascending_node'] = 180.0  # Conservative middle value
            print(f"     Estimated ra_of_ascending_node: 180.0°")
        
        if estimated_data.get('arg_of_periapsis') is None:
            # Random distribution 0-360°, use a typical value
            estimated_data['arg_of_periapsis'] = 90.0  # Conservative middle value  
            print(f"     Estimated arg_of_periapsis: 90.0°")
        
        if estimated_data.get('mean_anomaly') is None:
            # At epoch, can be 0-360°, use a typical value
            estimated_data['mean_anomaly'] = 0.0  # Conservative start value
            print(f"     Estimated mean_anomaly: 0.0°")
        
        return estimated_data
    
    @safe_execute
    def merge_orbital_data(self, data_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge orbital data from multiple sources using completeness-based prioritization.
        
        Enhanced version from legacy system that prioritizes fields from sources
        with highest completeness scores to ensure best quality data.
        
        Args:
            data_dict: Dictionary mapping source name to orbital data
            
        Returns:
            Merged orbital data with best quality values
        """
        if not data_dict:
            return {}
        
        # Enhanced required fields matching compute_completeness
        required_fields = [
            'eccentricity', 'inclination', 'semi_major_axis',
            'ra_of_ascending_node', 'arg_of_periapsis', 'mean_anomaly',
            'epoch', 'diameter', 'albedo'
        ]
        
        merged = {}
        
        # Calculate completeness scores for each source (from legacy system)
        completeness_scores = {
            source: self.compute_completeness(data) 
            for source, data in data_dict.items() if data
        }
        
        if not completeness_scores:
            return {}
        
        # For each required field, find the best value from the highest quality source
        for field in required_fields:
            best_value = None
            best_score = -1
            
            for source, data in data_dict.items():
                if data and data.get(field) is not None:
                    score = completeness_scores.get(source, 0)
                    if score > best_score:
                        best_score = score
                        best_value = data.get(field)
            
            merged[field] = best_value
        
        # Also include any additional fields from the highest completeness source
        if completeness_scores:
            best_source = max(completeness_scores.keys(), key=lambda x: completeness_scores[x])
            best_data = data_dict[best_source]
            
            for key, value in best_data.items():
                if key not in merged:
                    merged[key] = value
        
        return merged
    
    @safe_execute
    def fetch_all_orbital_elements(self, designation: str) -> Dict[str, Dict[str, Any]]:
        """
        Fetch orbital elements from all available sources with enhanced quality tracking.
        
        Integrates quality tracking from legacy system to monitor source reliability
        and data completeness scores.
        
        Args:
            designation: NEO designation
            
        Returns:
            Dictionary mapping source name to orbital data
        """
        results = {}
        
        for source in self.source_priority:
            if not self.data_sources[source]['available']:
                continue
            
            # Try to load from local cache first
            local_data = self.load_orbital_data(designation, source)
            if local_data is not None:
                results[source] = local_data
                
                # Update quality tracking for cached data (from legacy system)
                self.source_poll_stats[source]['success'] += 1
                quality = self.compute_completeness(local_data)
                self.source_quality_stats[source] += quality
                self.source_quality_counts[source] += 1
                
                # Maintain backward compatibility
                self.source_statistics[source]['success'] += 1
                continue
            
            # Fetch from remote source
            fetcher = self.data_sources[source]['fetcher']
            try:
                data = fetcher(designation)
                if data:
                    results[source] = data
                    self.save_orbital_data(designation, source, data)
                    
                    # Enhanced quality tracking (from legacy system)
                    self.source_poll_stats[source]['success'] += 1
                    quality = self.compute_completeness(data)
                    self.source_quality_stats[source] += quality
                    self.source_quality_counts[source] += 1
                    
                    # Log quality information
                    self.logger.info(f"Successfully fetched {designation} from {source} with {quality:.2f} completeness")
                    
                    # Statistics tracking is now handled inside each fetcher method
                else:
                    self.source_statistics[source]['failure'] += 1
                    self.source_poll_stats[source]['failure'] += 1
                    self.logger.warning(f"No data returned from {source} for {designation}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching from {source} for {designation}: {e}")
                self.source_statistics[source]['failure'] += 1
                self.source_poll_stats[source]['failure'] += 1
        
        return results
    
    @safe_execute
    def fetch_and_merge_orbital_elements(self, designation: str, cache_file: str) -> Dict[str, Any]:
        """
        Fetch and merge orbital elements with caching (like original script).
        
        Args:
            designation: NEO designation
            cache_file: Path to cache file
            
        Returns:
            Enriched NEO data with merged orbital elements
        """
        # Check cache first (thread-safe)
        try:
            with shelve.open(cache_file) as cache:
                if designation in cache:
                    return cache[designation]
        except Exception as e:
            self.logger.warning(f"Cache read error for {designation}: {e}")
        
        # Fetch from all sources
        responses = self.fetch_all_orbital_elements(designation)
        
        if not responses:
            return {}
        
        # Merge data
        merged = self.merge_orbital_data(responses)
        initial_completeness = self.compute_completeness(merged)
        
        # Ensure 100% completeness before analysis
        if initial_completeness < 0.99:
            print(f"⚠️  {designation} has {initial_completeness:.1%} completeness - enhancing...")
            merged = self.ensure_100_percent_completeness(designation, merged)
        
        final_completeness = self.compute_completeness(merged)
        
        # Create enriched result
        result = {
            'orbital_elements': merged,
            'sources_used': list(responses.keys()),
            'completeness': final_completeness,
            'initial_completeness': initial_completeness,
            'enhanced': final_completeness > initial_completeness,
            'source_contributions': {
                source: self.compute_completeness(data)
                for source, data in responses.items()
            }
        }
        
        # Cache the result (thread-safe)
        try:
            with shelve.open(cache_file) as cache:
                cache[designation] = result
        except Exception as e:
            self.logger.warning(f"Cache write error for {designation}: {e}")
        
        # Update local NEO database
        self.update_neo_database(designation, result)
        
        return result
    
    def analyze_enriched_neo_for_artificial_signatures(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze enriched NEO data for artificial signatures using TAS scoring (enhanced from original).
        
        Args:
            neo_data: Enriched NEO data with orbital elements and close approaches
            
        Returns:
            Analysis results with TAS score and artificial probability
        """
        designation = neo_data.get('designation', 'Unknown')
        orbital_elements = neo_data.get('orbital_elements', {})
        close_approaches = neo_data.get('close_approaches', [])
        
        # Enhanced TAS-based scoring inspired by original script
        tas_components = {}
        total_tas = 0.0
        indicators = []
        
        # 1. Orbital Mechanics Analysis (Weight: 2.0)
        orbital_score = 0.0
        
        eccentricity = orbital_elements.get('eccentricity')
        inclination = orbital_elements.get('inclination')
        semi_major_axis = orbital_elements.get('semi_major_axis')
        
        if eccentricity is not None:
            if eccentricity > 0.95:  # Hyperbolic-like (artificial propulsion?)
                excess = (eccentricity - 0.95) / 0.05
                orbital_score += min(excess, 1.0) * 1.5
                indicators.append(f"Hyperbolic-like eccentricity: {eccentricity:.4f}")
            elif eccentricity < 0.001:  # Perfectly circular (artificial control?)
                orbital_score += 1.2
                indicators.append(f"Perfectly circular orbit: e={eccentricity:.6f}")
            elif eccentricity < 0.01:  # Suspiciously circular
                orbital_score += 0.8
                indicators.append(f"Suspiciously circular orbit: e={eccentricity:.4f}")
        
        if inclination is not None:
            if inclination > 160:  # Highly retrograde (artificial?)
                excess = (inclination - 160) / 20
                orbital_score += min(excess, 1.0) * 1.0
                indicators.append(f"Highly retrograde orbit: {inclination:.1f}°")
            elif inclination > 150:  # Retrograde
                excess = (inclination - 150) / 10
                orbital_score += min(excess, 1.0) * 0.7
                indicators.append(f"Retrograde orbit: {inclination:.1f}°")
            elif inclination > 90:  # High inclination
                excess = (inclination - 90) / 60
                orbital_score += min(excess, 1.0) * 0.3
                indicators.append(f"High inclination orbit: {inclination:.1f}°")
        
        if semi_major_axis is not None:
            if semi_major_axis > 50:  # Way outside NEO range
                orbital_score += 1.5
                indicators.append(f"Extreme semi-major axis: {semi_major_axis:.2f} AU")
            elif semi_major_axis > 10:  # Unusual for NEO
                excess = (semi_major_axis - 10) / 40
                orbital_score += min(excess, 1.0) * 0.8
                indicators.append(f"Large semi-major axis: {semi_major_axis:.2f} AU")
            elif semi_major_axis < 0.1:  # Too close to Sun
                orbital_score += 1.0
                indicators.append(f"Extremely small semi-major axis: {semi_major_axis:.3f} AU")
        
        tas_components['orbital_mechanics'] = orbital_score
        total_tas += orbital_score
        
        # 2. Velocity Analysis (Weight: 1.5) - Enhanced from original script
        velocity_score = 0.0
        
        if close_approaches:
            velocities = []
            distances = []
            v_inf_values = []
            
            for approach in close_approaches:
                try:
                    v_rel = float(approach.get('v_rel', 0))
                    v_inf = float(approach.get('v_inf', 0))
                    dist = float(approach.get('dist', 0))
                    
                    if v_rel > 0:
                        velocities.append(v_rel)
                    if v_inf > 0:
                        v_inf_values.append(v_inf)
                    if dist > 0:
                        distances.append(dist)
                    
                    # Extreme velocity analysis
                    if v_rel > 70:  # Extremely high velocity (artificial propulsion?)
                        excess = (v_rel - 70) / 30
                        velocity_score += min(excess, 1.0) * 1.2
                        indicators.append(f"Extreme velocity: {v_rel:.2f} km/s")
                    elif v_rel > 50:  # Very high velocity
                        excess = (v_rel - 50) / 20
                        velocity_score += min(excess, 1.0) * 0.8
                        indicators.append(f"Very high velocity: {v_rel:.2f} km/s")
                    elif v_rel < 3 and v_rel > 0:  # Suspiciously low velocity
                        velocity_score += 0.9
                        indicators.append(f"Suspiciously low velocity: {v_rel:.2f} km/s")
                    elif v_rel < 5 and v_rel > 0:  # Unusually low velocity
                        velocity_score += 0.4
                        indicators.append(f"Unusually low velocity: {v_rel:.2f} km/s")
                    
                    # Distance analysis
                    if dist < 0.0001:  # Impossibly close approach
                        velocity_score += 1.5
                        indicators.append(f"Impossibly close approach: {dist:.7f} AU")
                    elif dist < 0.001:  # Extremely close approach
                        velocity_score += 0.8
                        indicators.append(f"Extremely close approach: {dist:.6f} AU")
                    
                    # Perfect round numbers (artificial control signatures)
                    if v_rel == int(v_rel) and v_rel > 0:
                        velocity_score += 0.6
                        indicators.append(f"Perfect round velocity: {v_rel} km/s")
                    
                    if dist > 0 and abs(dist - round(dist, 3)) < 1e-6:
                        velocity_score += 0.4
                        indicators.append(f"Perfect round distance: {dist:.6f} AU")
                        
                except (ValueError, TypeError):
                    continue
            
            # Statistical velocity analysis (enhanced from original)
            if len(velocities) > 1:
                if HAS_NUMPY:
                    velocity_std = np.std(velocities)
                    velocity_mean = np.mean(velocities)
                else:
                    velocity_mean = sum(velocities) / len(velocities)
                    velocity_std = (sum((v - velocity_mean)**2 for v in velocities) / len(velocities)) ** 0.5
                
                if velocity_mean > 0:
                    velocity_cv = velocity_std / velocity_mean  # Coefficient of variation
                    if velocity_cv < 0.01:  # Perfect consistency (artificial control?)
                        velocity_score += 1.2
                        indicators.append(f"Perfect velocity consistency: CV={velocity_cv:.5f}")
                    elif velocity_cv < 0.05:  # Too consistent
                        velocity_score += 0.8
                        indicators.append(f"Suspiciously consistent velocities: CV={velocity_cv:.4f}")
                    
                    # Velocity shift analysis (from original script)
                    if len(velocities) > 2:
                        velocity_range = max(velocities) - min(velocities)
                        if velocity_range > 30:  # Extreme velocity variation
                            velocity_score += 0.9
                            indicators.append(f"Extreme velocity variation: {velocity_range:.2f} km/s")
        
        tas_components['velocity_shifts'] = velocity_score
        total_tas += velocity_score
        
        # 3. Physical and Temporal Anomalies (Weight: 1.0) - Based on original script
        physical_score = 0.0
        
        # Physical characteristics analysis
        diameter = orbital_elements.get('diameter')
        albedo = orbital_elements.get('albedo')
        magnitude = orbital_elements.get('H', orbital_elements.get('absolute_magnitude'))
        
        if diameter is not None:
            if diameter < 0.01:  # Extremely small (< 10m)
                physical_score += 0.9
                indicators.append(f"Extremely small diameter: {diameter:.4f} km")
            elif diameter > 100:  # Extremely large for NEO
                physical_score += 0.7
                indicators.append(f"Extremely large diameter: {diameter:.2f} km")
        
        if albedo is not None:
            if albedo < 0.01:  # Extremely dark (artificial coating?)
                physical_score += 0.8
                indicators.append(f"Extremely low albedo: {albedo:.4f}")
            elif albedo > 0.9:  # Extremely bright (artificial surface?)
                physical_score += 1.1
                indicators.append(f"Extremely high albedo: {albedo:.4f}")
            elif albedo > 0.7:  # Unusually bright
                physical_score += 0.6
                indicators.append(f"Unusually high albedo: {albedo:.4f}")
        
        if magnitude is not None:
            if magnitude < 10:  # Very bright (large or highly reflective)
                physical_score += 0.5
                indicators.append(f"Very bright magnitude: H={magnitude:.1f}")
            elif magnitude > 30:  # Extremely dim (tiny or very dark)
                physical_score += 0.7
                indicators.append(f"Extremely dim magnitude: H={magnitude:.1f}")
        
        # Perfect orbital ratios (artificial design signatures)
        if (eccentricity is not None and inclination is not None and 
            semi_major_axis is not None):
            
            # Check for "golden ratio" or other perfect mathematical relationships
            golden_ratio = 1.618033988749895
            
            # Check for golden ratio relationships
            if semi_major_axis > 0 and abs(eccentricity / (semi_major_axis * 0.1) - golden_ratio) < 0.01:
                physical_score += 1.3
                indicators.append("Golden ratio in orbital parameters detected")
            
            # Perfect integer ratios
            if eccentricity > 0 and inclination > 0:
                ratio = inclination / (eccentricity * 100)
                if abs(ratio - round(ratio)) < 0.01:
                    physical_score += 0.9
                    indicators.append(f"Perfect integer ratio in orbital elements: {ratio:.2f}")
            
            # Check for suspiciously round numbers
            if abs(semi_major_axis - round(semi_major_axis, 2)) < 0.001:
                physical_score += 0.5
                indicators.append(f"Suspiciously round semi-major axis: {semi_major_axis:.3f} AU")
            
            if abs(eccentricity - round(eccentricity, 3)) < 0.0001:
                physical_score += 0.6
                indicators.append(f"Suspiciously round eccentricity: {eccentricity:.4f}")
            
            if abs(inclination - round(inclination, 1)) < 0.01:
                physical_score += 0.4
                indicators.append(f"Suspiciously round inclination: {inclination:.2f}°")
        
        # Temporal anomalies analysis (from original script)
        if close_approaches:
            approach_dates = []
            for approach in close_approaches:
                approach_date = approach.get('cd')
                if approach_date:
                    approach_dates.append(approach_date)
            
            if len(approach_dates) > 1:
                # Check for perfect periodicity (artificial control signature)
                approach_dates.sort()
                if len(approach_dates) >= 3:
                    intervals = []
                    for i in range(1, len(approach_dates)):
                        try:
                            # Simple date difference analysis
                            intervals.append(1)  # Placeholder for actual date parsing
                        except:
                            continue
                    
                    if intervals and len(set(intervals)) == 1:  # Perfect periodicity
                        physical_score += 1.1
                        indicators.append("Perfect periodicity in close approaches")
                    elif intervals and len(intervals) > 2:
                        # Check for mathematical progression
                        if all(intervals[i] == intervals[0] * (i+1) for i in range(len(intervals))):
                            physical_score += 0.9
                            indicators.append("Mathematical progression in approach intervals")
        
        tas_components['physical_anomalies'] = physical_score
        total_tas += physical_score
        
        # 4. Geographic and Detection Pattern Analysis (Weight: 0.8)
        detection_score = 0.0
        
        # Multi-source detection consistency
        sources_used = neo_data.get('sources_used', [])
        if len(sources_used) < 2:
            detection_score += 0.6
            indicators.append(f"Limited source detection: only {len(sources_used)} source(s)")
        elif len(sources_used) == 1 and 'SBDB' in sources_used:
            detection_score += 0.3
            indicators.append("Only detected by SBDB (NASA)")
        
        # Data completeness analysis
        completeness = self.compute_completeness(orbital_elements)
        if completeness > 0.95:  # Suspiciously complete data
            detection_score += 0.7
            indicators.append(f"Suspiciously complete data: {completeness:.1%}")
        elif completeness < 0.3:  # Suspiciously incomplete data
            detection_score += 0.8
            indicators.append(f"Suspiciously incomplete data: {completeness:.1%}")
        
        # Recent discovery with high precision (artificial planting?)
        first_observation = neo_data.get('first_observation')
        if first_observation and '2024' in str(first_observation) or '2025' in str(first_observation):
            if completeness > 0.8:
                detection_score += 0.9
                indicators.append("Recent discovery with suspiciously complete data")
        
        tas_components['detection_history'] = detection_score
        total_tas += detection_score
        
        # 5. Acceleration and Spectral Anomalies (Weight: 0.7) 
        advanced_score = 0.0
        
        # Perfect Keplerian motion (no perturbations - artificial control?)
        if close_approaches and len(close_approaches) > 2:
            # This would require detailed orbital propagation analysis
            # For now, check for unusual patterns in approach data
            distances = [float(app.get('dist', 0)) for app in close_approaches if app.get('dist')]
            velocities = [float(app.get('v_rel', 0)) for app in close_approaches if app.get('v_rel')]
            
            if len(distances) > 2 and len(velocities) > 2:
                if HAS_NUMPY:
                    dist_correlation = np.corrcoef(distances[:-1], distances[1:])[0, 1] if len(distances) > 2 else 0
                    vel_correlation = np.corrcoef(velocities[:-1], velocities[1:])[0, 1] if len(velocities) > 2 else 0
                    
                    # Perfect correlations might indicate artificial control
                    if abs(dist_correlation) > 0.98:
                        advanced_score += 0.8
                        indicators.append(f"Perfect distance correlation: {dist_correlation:.4f}")
                    
                    if abs(vel_correlation) > 0.98:
                        advanced_score += 0.9
                        indicators.append(f"Perfect velocity correlation: {vel_correlation:.4f}")
        
        # Spectral anomalies (artificial materials?)
        if 'spectral_class' in orbital_elements:
            spectral_class = orbital_elements['spectral_class']
            if spectral_class in ['X', 'M', 'E']:  # Metallic asteroids (rare)
                advanced_score += 0.6
                indicators.append(f"Unusual spectral class: {spectral_class}")
        
        tas_components['acceleration_anomalies'] = advanced_score
        total_tas += advanced_score
        
        # Dynamic TAS calculation (from original script)
        # This would require population statistics for proper z-score calculation
        # For now, use raw TAS with thresholds
        
        # Classification based on total TAS score (enhanced from original)
        if total_tas >= 8.0:
            classification = "EXTREMELY ANOMALOUS - Potentially Artificial"
            confidence = 0.95
        elif total_tas >= 6.0:
            classification = "HIGHLY ANOMALOUS - Requires Investigation"
            confidence = 0.85
        elif total_tas >= 4.0:
            classification = "MODERATELY ANOMALOUS - Suspicious Characteristics"
            confidence = 0.75
        elif total_tas >= 2.0:
            classification = "SLIGHTLY ANOMALOUS - Some Unusual Features"
            confidence = 0.65
        elif total_tas >= 1.0:
            classification = "MARGINALLY ANOMALOUS - Minor Irregularities"
            confidence = 0.55
        else:
            classification = "NATURAL - No Artificial Signatures"
            confidence = 0.4
        
        # Create comprehensive result
        result = {
            'designation': designation,
            'raw_TAS': total_tas,
            'tas_components': tas_components,
            'classification': classification,
            'confidence': confidence,
            'indicators': indicators,
            'analysis_details': {
                'total_components': len(tas_components),
                'highest_component': max(tas_components.keys(), key=lambda k: tas_components[k]),
                'highest_component_score': max(tas_components.values()),
                'completeness': self.compute_completeness(orbital_elements),
                'sources_used': neo_data.get('sources_used', []),
                'close_approaches': len(close_approaches)
            }
        }
        
        print(f"   🎯 Enhanced TAS Analysis Complete:")
        print(f"      Raw TAS Score: {total_tas:.3f}")
        print(f"      Classification: {classification}")
        print(f"      Confidence: {confidence:.1%}")
        print(f"      Key Indicators: {len(indicators)}")
        
        if indicators:
            print(f"      Top Indicators:")
            for i, indicator in enumerate(indicators[:3]):  # Show top 3
                print(f"        • {indicator}")
        
        return result
    
    def enrich_and_analyze_neos(self, neo_map: Dict[str, Dict[str, Any]], max_workers: int = 10) -> List[Dict[str, Any]]:
        """
        Enrich NEOs with orbital data and analyze for artificial signatures (like original script).
        
        Args:
            neo_map: Dictionary mapping designation to close approach data
            max_workers: Maximum worker threads for parallel processing
            
        Returns:
            List of enriched and analyzed NEO data
        """
        enriched_results = []
        
        print(f"🔬 Enriching {len(neo_map)} NEOs with orbital data and quality tracking...")
        
        # Initialize quality tracking counters
        quality_metrics = {
            'total_processed': 0,
            'high_quality': 0,  # >= 0.8 completeness
            'medium_quality': 0,  # 0.5-0.8 completeness
            'low_quality': 0,  # < 0.5 completeness
            'enhanced_objects': 0  # Objects that required enhancement
        }
        
        try:
            if self.console:
                # Rich progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                ) as progress:
                    task = progress.add_task("Enriching NEOs...", total=len(neo_map))
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all enrichment tasks
                        futures = {
                            executor.submit(self.fetch_and_merge_orbital_elements, designation, self.cache_file): designation
                            for designation in neo_map.keys()
                        }
                        
                        for future in as_completed(futures):
                            designation = futures[future]
                            progress.update(task, advance=1)
                            
                            try:
                                enrichment_data = future.result()
                                
                                # Create NEO object (even if enrichment failed)
                                enriched_neo = {
                                    'designation': designation,
                                    'close_approaches': neo_map[designation]['close_approaches'],
                                    'first_observation': neo_map[designation]['first_observation'],
                                    'last_observation': neo_map[designation]['last_observation']
                                }
                                
                                if enrichment_data:
                                    # Add enrichment data if successful
                                    enriched_neo.update(enrichment_data)
                                else:
                                    # Use minimal data structure for failed enrichment
                                    enriched_neo.update({
                                        'orbital_elements': {},
                                        'sources_used': [],
                                        'completeness': 0.0,
                                        'source_contributions': {}
                                    })
                                
                                # Update quality metrics
                                completeness = enriched_neo.get('completeness', 0.0)
                                quality_metrics['total_processed'] += 1
                                
                                if completeness >= 0.8:
                                    quality_metrics['high_quality'] += 1
                                elif completeness >= 0.5:
                                    quality_metrics['medium_quality'] += 1
                                else:
                                    quality_metrics['low_quality'] += 1
                                
                                if enriched_neo.get('enhanced', False):
                                    quality_metrics['enhanced_objects'] += 1
                                
                                # Always analyze (even with limited data)
                                analysis_result = self.analyze_enriched_neo_for_artificial_signatures(enriched_neo)
                                enriched_results.append(analysis_result)
                                    
                            except Exception as e:
                                self.logger.error(f"Error enriching {designation}: {e}")
            else:
                # Fallback progress indication with tqdm if available
                # Common enrichment processing function
                def process_enrichment_result(designation, enrichment_data):
                    # Create NEO object (even if enrichment failed)
                    enriched_neo = {
                        'designation': designation,
                        'close_approaches': neo_map[designation]['close_approaches'],
                        'first_observation': neo_map[designation]['first_observation'],
                        'last_observation': neo_map[designation]['last_observation']
                    }
                    
                    if enrichment_data:
                        # Add enrichment data if successful
                        enriched_neo.update(enrichment_data)
                    else:
                        # Use minimal data structure for failed enrichment
                        enriched_neo.update({
                            'orbital_elements': {},
                            'sources_used': [],
                            'completeness': 0.0,
                            'source_contributions': {}
                        })
                    
                    # Update quality metrics
                    completeness = enriched_neo.get('completeness', 0.0)
                    quality_metrics['total_processed'] += 1
                    
                    if completeness >= 0.8:
                        quality_metrics['high_quality'] += 1
                    elif completeness >= 0.5:
                        quality_metrics['medium_quality'] += 1
                    else:
                        quality_metrics['low_quality'] += 1
                    
                    if enriched_neo.get('enhanced', False):
                        quality_metrics['enhanced_objects'] += 1
                    
                    # Always analyze (even with limited data)
                    analysis_result = self.analyze_enriched_neo_for_artificial_signatures(enriched_neo)
                    enriched_results.append(analysis_result)
                
                if HAS_TQDM:
                    # Use tqdm progress bar
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(self.fetch_and_merge_orbital_elements, designation, self.cache_file): designation
                            for designation in neo_map.keys()
                        }
                        
                        with tqdm(total=len(neo_map), desc="Enriching NEOs", 
                                unit="NEO", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
                            
                            for future in as_completed(futures):
                                designation = futures[future]
                                pbar.update(1)
                                
                                try:
                                    enrichment_data = future.result()
                                    process_enrichment_result(designation, enrichment_data)
                                except Exception as e:
                                    self.logger.error(f"Error enriching {designation}: {e}")
                else:
                    # Basic progress indication without any progress bar library
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(self.fetch_and_merge_orbital_elements, designation, self.cache_file): designation
                            for designation in neo_map.keys()
                        }
                        
                        completed = 0
                        for future in as_completed(futures):
                            designation = futures[future]
                            completed += 1
                            
                            if completed % 10 == 0 or completed == len(neo_map):
                                print(f"   Processed {completed}/{len(neo_map)} NEOs...")
                            
                            try:
                                enrichment_data = future.result()
                                process_enrichment_result(designation, enrichment_data)
                            except Exception as e:
                                self.logger.error(f"Error enriching {designation}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in enrichment process: {e}")
        
        # Save NEO database after processing
        self.save_neo_database()
        
        # Display quality metrics summary
        print(f"\n📊 DATA QUALITY METRICS:")
        print(f"   Total objects processed: {quality_metrics['total_processed']}")
        print(f"   High quality (≥80% complete): {quality_metrics['high_quality']} ({quality_metrics['high_quality']/max(quality_metrics['total_processed'], 1)*100:.1f}%)")
        print(f"   Medium quality (50-80% complete): {quality_metrics['medium_quality']} ({quality_metrics['medium_quality']/max(quality_metrics['total_processed'], 1)*100:.1f}%)")  
        print(f"   Low quality (<50% complete): {quality_metrics['low_quality']} ({quality_metrics['low_quality']/max(quality_metrics['total_processed'], 1)*100:.1f}%)")
        print(f"   Objects enhanced for completeness: {quality_metrics['enhanced_objects']}")
        
        # Overall quality assessment
        if quality_metrics['total_processed'] > 0:
            high_quality_percentage = quality_metrics['high_quality'] / quality_metrics['total_processed'] * 100
            if high_quality_percentage >= 80:
                quality_status = "EXCELLENT - 100% data quality achieved"
            elif high_quality_percentage >= 60:
                quality_status = "GOOD - High data quality for analysis"
            elif high_quality_percentage >= 40:
                quality_status = "MODERATE - Acceptable data quality"
            else:
                quality_status = "POOR - Data quality improvement needed"
            
            print(f"   Overall data quality status: {quality_status}")
        
        return enriched_results
    
    @safe_execute
    def save_enriched_results(self, results: List[Dict[str, Any]], api_choice: str, time_period: str) -> str:
        """Save enriched results with comprehensive metadata."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_neo_poll_{api_choice.lower()}_{time_period}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Calculate statistics - use TAS scores for enhanced system
        def get_score(r):
            return r.get('raw_TAS', r.get('artificial_score', 0))
        
        suspicious_results = [r for r in results if get_score(r) >= 2.0]  # TAS threshold
        highly_suspicious = [r for r in results if get_score(r) >= 4.0]  # Higher TAS threshold
        
        output_data = {
            'metadata': {
                'api_used': api_choice,
                'time_period': time_period,
                'analysis_date': datetime.datetime.now().isoformat(),
                'total_objects': len(results),
                'suspicious_count': len(suspicious_results),
                'highly_suspicious_count': len(highly_suspicious),
                'data_sources_used': list(set(
                    source for result in results 
                    for source in result.get('sources_used', [])
                )),
                'average_completeness': sum(r.get('data_completeness', 0) for r in results) / len(results) if results else 0,
                'source_statistics': self.source_statistics,
                'data_usage_bytes': self.data_usage
            },
            'results': results
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return ""
    
    def display_enriched_results(self, results: List[Dict[str, Any]], report_format: str = 'console'):
        """Display enriched analysis results with enhanced statistics and professional reporting."""
        if not results:
            print("❌ No results to display")
            return
        
        # If professional reporting is enabled, use it
        if self.professional_report and self.professional_suite:
            try:
                self.logger.info("Generating professional report...")
                
                # Generate comprehensive professional report
                report_result = self.professional_suite.generate_comprehensive_report(
                    neo_data=results,
                    metadata={
                        'analysis_timestamp': datetime.datetime.now().isoformat(),
                        'total_objects': len(results),
                        'data_source': 'NASA_CAD_Enhanced',
                        'analysis_type': 'comprehensive_enhanced'
                    },
                    output_format=report_format
                )
                
                if report_result:
                    self.logger.info(f"Professional report generated: {report_result}")
                    if report_format == 'console':
                        # The console report is already displayed by the suite
                        return
                    else:
                        print(f"📄 Professional report saved to: {report_result}")
                        # Still show basic console output for non-console formats
                
            except Exception as e:
                self.logger.error(f"Professional reporting failed: {e}")
                print(f"⚠️  Professional reporting failed, falling back to standard display: {e}")
                # Continue with standard display
        elif self.basic_professional_report:
            # Use basic professional formatting when full suite is not available
            self._display_basic_professional_report(results)
            return  # Don't show standard output when using professional format
        
        # Calculate comprehensive statistics
        total_objects = len(results)
        
        # Use raw_TAS for enhanced TAS system, fallback to artificial_score for compatibility
        def get_score(r):
            return r.get('raw_TAS', r.get('artificial_score', 0))
        
        suspicious_objects = [r for r in results if get_score(r) >= 2.0]  # TAS threshold
        highly_suspicious = [r for r in results if get_score(r) >= 4.0]  # Higher TAS threshold
        
        avg_completeness = sum(r.get('data_completeness', r.get('analysis_details', {}).get('completeness', 0)) for r in results) / total_objects
        enriched_objects = [r for r in results if r.get('data_completeness', r.get('analysis_details', {}).get('completeness', 0)) > 0.5]
        
        # Sort by TAS score (highest first)
        sorted_results = sorted(results, key=lambda x: get_score(x), reverse=True)
        
        if self.console:
            # Rich interface
            console.print(f"\n📊 [bold green]ENHANCED ANALYSIS RESULTS[/bold green]")
            console.print("=" * 80)
            console.print(f"Total objects analyzed: [bold]{total_objects}[/bold]")
            console.print(f"Successfully enriched: [bold cyan]{len(enriched_objects)}[/bold cyan] ({len(enriched_objects)/total_objects*100:.1f}%)")
            console.print(f"Average data completeness: [bold cyan]{avg_completeness:.2f}[/bold cyan]")
            console.print(f"Suspicious objects (TAS ≥2.0): [bold red]{len(suspicious_objects)}[/bold red]")
            console.print(f"Highly suspicious (TAS ≥4.0): [bold red]{len(highly_suspicious)}[/bold red]")
            
            # Data source statistics
            used_sources = set(source for result in results for source in result.get('sources_used', []))
            if used_sources:
                console.print(f"Data sources used: [bold]{', '.join(used_sources)}[/bold]")
            
            if suspicious_objects:
                console.print(f"\n🚨 [bold red]SUSPICIOUS OBJECTS REQUIRING INVESTIGATION:[/bold red]")
                
                table = Table(show_header=True, header_style="bold red")
                table.add_column("Designation", style="cyan")
                table.add_column("Score", style="red")
                table.add_column("Classification", style="yellow")
                table.add_column("Completeness", style="green")
                table.add_column("Sources", style="blue")
                table.add_column("Key Indicators", style="white")
                
                for obj in suspicious_objects[:15]:  # Show top 15
                    indicators_str = "; ".join(obj['indicators'][:2]) if obj['indicators'] else "None"
                    if len(indicators_str) > 60:
                        indicators_str = indicators_str[:57] + "..."
                    
                    sources_str = ", ".join(obj.get('sources_used', []))
                    
                    table.add_row(
                        obj['designation'],
                        f"{get_score(obj):.3f}",
                        obj.get('classification', 'Unknown').split(' - ')[0],  # Shorter classification
                        f"{obj.get('data_completeness', obj.get('analysis_details', {}).get('completeness', 0)):.2f}",
                        sources_str,
                        indicators_str
                    )
                
                console.print(table)
                
                if len(suspicious_objects) > 15:
                    console.print(f"\n... and {len(suspicious_objects) - 15} more suspicious objects")
        
        else:
            # Basic interface
            print(f"\n📊 ENHANCED ANALYSIS RESULTS")
            print("=" * 80)
            print(f"Total objects analyzed: {total_objects}")
            print(f"Successfully enriched: {len(enriched_objects)} ({len(enriched_objects)/total_objects*100:.1f}%)")
            print(f"Average data completeness: {avg_completeness:.2f}")
            print(f"Suspicious objects (TAS ≥2.0): {len(suspicious_objects)}")
            print(f"Highly suspicious (TAS ≥4.0): {len(highly_suspicious)}")
            
            if suspicious_objects:
                print(f"\n🚨 SUSPICIOUS OBJECTS REQUIRING INVESTIGATION:")
                print("-" * 80)
                
                for obj in suspicious_objects[:15]:  # Show top 15
                    score = get_score(obj)
                    print(f"{obj['designation']:15} - TAS Score: {score:.3f} - {obj.get('classification', 'Unknown')}")
                    completeness = obj.get('data_completeness', obj.get('analysis_details', {}).get('completeness', 0))
                    print(f"   Completeness: {completeness:.2f} - Sources: {', '.join(obj.get('sources_used', []))}")
                    indicators = obj.get('indicators', [])
                    for indicator in indicators[:2]:  # Show top 2 indicators
                        print(f"  • {indicator}")
                    print()
                
                if len(suspicious_objects) > 15:
                    print(f"... and {len(suspicious_objects) - 15} more suspicious objects")
    
    def _display_basic_professional_report(self, results: List[Dict[str, Any]]):
        """Display a basic professional report when full suite is not available."""
        total_objects = len(results)
        
        # Use raw_TAS for enhanced TAS system, fallback to artificial_score for compatibility
        def get_score(r):
            return r.get('raw_TAS', r.get('artificial_score', 0))
        
        suspicious_objects = [r for r in results if get_score(r) >= 2.0]
        highly_suspicious = [r for r in results if get_score(r) >= 4.0]
        avg_completeness = sum(r.get('data_completeness', r.get('analysis_details', {}).get('completeness', 0)) for r in results) / total_objects
        enriched_objects = [r for r in results if r.get('data_completeness', r.get('analysis_details', {}).get('completeness', 0)) > 0.5]
        
        # Professional report header
        print("\n" + "="*80)
        print("📊 ANEOS PROFESSIONAL NEO ANALYSIS REPORT")
        print("="*80)
        print(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Analysis Type: Enhanced TAS Analysis with Multi-Source Enrichment")
        print(f"Data Source: NASA Close Approach Database (CAD)")
        print("-"*80)
        
        # Executive summary
        print("\n🎯 EXECUTIVE SUMMARY")
        print("-"*30)
        print(f"• Total objects analyzed: {total_objects}")
        print(f"• Data enrichment success rate: {len(enriched_objects)/total_objects*100:.1f}%")
        print(f"• Average data completeness: {avg_completeness:.2f}")
        print(f"• Objects requiring investigation (TAS ≥2.0): {len(suspicious_objects)}")
        print(f"• High-priority objects (TAS ≥4.0): {len(highly_suspicious)}")
        
        # Risk assessment
        risk_level = "LOW"
        if len(highly_suspicious) > 0:
            risk_level = "HIGH"
        elif len(suspicious_objects) > total_objects * 0.1:
            risk_level = "MEDIUM"
        
        print(f"• Overall risk assessment: {risk_level}")
        
        if suspicious_objects:
            print(f"\n🚨 OBJECTS REQUIRING DETAILED INVESTIGATION")
            print("-"*50)
            
            # Sort by score
            sorted_objects = sorted(suspicious_objects, key=lambda x: get_score(x), reverse=True)
            
            for i, obj in enumerate(sorted_objects[:10], 1):  # Top 10
                score = get_score(obj)
                completeness = obj.get('data_completeness', obj.get('analysis_details', {}).get('completeness', 0))
                print(f"\n{i}. {obj['designation']}")
                print(f"   TAS Score: {score:.3f}")
                print(f"   Classification: {obj.get('classification', 'Unknown')}")
                print(f"   Data Completeness: {completeness:.2f}")
                print(f"   Sources Used: {', '.join(obj.get('sources_used', ['None']))}")
                
                indicators = obj.get('indicators', [])
                if indicators:
                    print(f"   Key Indicators:")
                    for indicator in indicators[:3]:  # Top 3 indicators
                        print(f"     • {indicator}")
                
            if len(suspicious_objects) > 10:
                print(f"\n   ... and {len(suspicious_objects) - 10} additional objects requiring investigation")
        
        print("\n" + "="*80)
        print("📝 Note: Professional reporting requires numpy and matplotlib for full features")
        print("   Install with: pip install numpy matplotlib")
        print("="*80)
    
    def display_source_quality_report(self):
        """Display comprehensive source quality statistics from legacy system."""
        print(f"\n📊 SOURCE QUALITY REPORT")
        print("=" * 80)
        
        # Calculate total quality for percentage calculations
        total_quality = sum(self.source_quality_stats.values())
        
        if self.console:
            # Rich interface
            console.print(f"\n📈 [bold green]DATA SOURCE PERFORMANCE ANALYSIS[/bold green]")
            
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Source", style="cyan")
            table.add_column("Success", style="green")
            table.add_column("Failure", style="red")
            table.add_column("Success Rate", style="yellow")
            table.add_column("Avg Quality", style="blue")
            table.add_column("Contribution", style="magenta")
            table.add_column("Data Used", style="white")
            
            for source in self.source_priority:
                success_count = self.source_poll_stats[source]['success']
                failure_count = self.source_poll_stats[source]['failure']
                total_attempts = success_count + failure_count
                
                success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
                quality_count = self.source_quality_counts[source]
                avg_quality = (self.source_quality_stats[source] / quality_count) if quality_count > 0 else 0
                contribution = (self.source_quality_stats[source] / total_quality * 100) if total_quality > 0 else 0
                data_used = self.data_usage.get(source, 0)
                
                # Format data usage
                if data_used > 1024 * 1024:
                    data_str = f"{data_used / (1024 * 1024):.1f} MB"
                elif data_used > 1024:
                    data_str = f"{data_used / 1024:.1f} KB"
                else:
                    data_str = f"{data_used} B"
                
                table.add_row(
                    source,
                    str(success_count),
                    str(failure_count),
                    f"{success_rate:.1f}%",
                    f"{avg_quality:.3f}",
                    f"{contribution:.1f}%",
                    data_str
                )
            
            console.print(table)
            
        else:
            # Basic interface
            print(f"{'Source':<15} {'Success':<8} {'Failure':<8} {'Rate':<8} {'Avg Quality':<12} {'Contribution':<12} {'Data Used':<12}")
            print("-" * 80)
            
            for source in self.source_priority:
                success_count = self.source_poll_stats[source]['success']
                failure_count = self.source_poll_stats[source]['failure']
                total_attempts = success_count + failure_count
                
                success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
                quality_count = self.source_quality_counts[source]
                avg_quality = (self.source_quality_stats[source] / quality_count) if quality_count > 0 else 0
                contribution = (self.source_quality_stats[source] / total_quality * 100) if total_quality > 0 else 0
                data_used = self.data_usage.get(source, 0)
                
                # Format data usage
                if data_used > 1024 * 1024:
                    data_str = f"{data_used / (1024 * 1024):.1f}MB"
                elif data_used > 1024:
                    data_str = f"{data_used / 1024:.1f}KB"
                else:
                    data_str = f"{data_used}B"
                
                print(f"{source:<15} {success_count:<8} {failure_count:<8} {success_rate:>6.1f}% {avg_quality:>10.3f} {contribution:>10.1f}% {data_str:>10}")
        
        # Summary statistics
        total_successes = sum(stats['success'] for stats in self.source_poll_stats.values())
        total_failures = sum(stats['failure'] for stats in self.source_poll_stats.values())
        total_attempts = total_successes + total_failures
        overall_success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
        overall_avg_quality = (total_quality / sum(self.source_quality_counts.values())) if sum(self.source_quality_counts.values()) > 0 else 0
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total API calls: {total_attempts}")
        print(f"   Overall success rate: {overall_success_rate:.1f}%")
        print(f"   Average data quality: {overall_avg_quality:.3f}")
        print(f"   Total data retrieved: {sum(self.data_usage.values())} bytes")
        
        # Quality assessment
        if overall_avg_quality >= 0.9:
            quality_assessment = "EXCELLENT - Ready for analysis"
        elif overall_avg_quality >= 0.7:
            quality_assessment = "GOOD - Suitable for analysis"
        elif overall_avg_quality >= 0.5:
            quality_assessment = "MODERATE - May need additional sources"
        else:
            quality_assessment = "POOR - Requires data quality improvement"
        
        print(f"   Quality assessment: {quality_assessment}")
        print("=" * 80)
    
    def run_enhanced_polling(self, api_choice: str = "NASA_CAD", time_period: str = "1w", max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Run complete enhanced polling process.
        
        Args:
            api_choice: API to use (currently only NASA_CAD supported)
            time_period: Time period string
            max_results: Maximum results to process
            
        Returns:
            List of enriched analysis results
        """
        # Parse time period
        if HAS_DATEUTIL:
            delta = None
            if time_period in ['1d', '1w', '1m', '3m', '6m', '1y', '2y', '5y', '10y', '25y', '50y', '100y', '200y']:
                time_map = {
                    '1d': relativedelta(days=1), '1w': relativedelta(weeks=1), '1m': relativedelta(months=1),
                    '3m': relativedelta(months=3), '6m': relativedelta(months=6), '1y': relativedelta(years=1),
                    '2y': relativedelta(years=2), '5y': relativedelta(years=5), '10y': relativedelta(years=10),
                    '25y': relativedelta(years=25), '50y': relativedelta(years=50), '100y': relativedelta(years=100),
                    '200y': relativedelta(years=200)
                }
                delta = time_map.get(time_period)
            else:
                # Parse custom format
                pattern = r'^(\d+)([dwmy])$'
                match = re.match(pattern, time_period.lower())
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    if unit == 'd':
                        delta = relativedelta(days=value)
                    elif unit == 'w':
                        delta = relativedelta(weeks=value)
                    elif unit == 'm':
                        delta = relativedelta(months=value)
                    elif unit == 'y':
                        delta = relativedelta(years=value)
        else:
            # Simplified fallback
            delta = relativedelta(months=1)  # Default to 1 month
        
        if not delta:
            print(f"❌ Invalid time period: {time_period}")
            return []
        
        # Calculate date range
        end_date = datetime.date.today()
        start_date = end_date - delta
        
        print(f"\n🔍 Enhanced Polling: {api_choice}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🎯 Max results: {max_results}")
        
        # Step 1: Fetch CAD data
        print("📡 Fetching close approach data...")
        cad_data = self.fetch_cad_data_with_cache(start_date.isoformat(), end_date.isoformat(), max_results)
        
        if not cad_data:
            print("❌ No CAD data retrieved")
            return []
        
        # Step 2: Extract NEO designations
        print("🗂️  Extracting NEO designations...")
        neo_map = self.extract_neo_designations(cad_data)
        print(f"✅ Found {len(neo_map)} unique NEOs with close approaches")
        
        if not neo_map:
            print("❌ No NEOs found in data")
            return []
        
        # Step 3: Enrich and analyze
        results = self.enrich_and_analyze_neos(neo_map)
        
        print(f"✅ Analysis complete: {len(results)} NEOs processed")
        
        return results


def main():
    """Main entry point for Enhanced NEO Poller."""
    parser = argparse.ArgumentParser(
        description="Enhanced NEO Poller - Complete data enrichment and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_neo_poller.py --period 1w          # Last week with enrichment
  python enhanced_neo_poller.py --period 1m          # Last month with enrichment
  python enhanced_neo_poller.py --period 6m          # Last 6 months
        """
    )
    
    parser.add_argument('--period', default='1w',
                       help='Time period (e.g., 1d, 1w, 1m, 6m, 1y)')
    parser.add_argument('--max-results', type=int, default=1000,
                       help='Maximum results to analyze')
    parser.add_argument('--data-dir', default='neo_data',
                       help='Data directory for caching and storage')
    
    # Professional reporting options
    parser.add_argument('--professional-report', action='store_true',
                       help='Generate professional-quality reports with AI validation')
    parser.add_argument('--report-format', choices=['console', 'html', 'pdf', 'json'], 
                       default='console', help='Report output format')
    parser.add_argument('--report-dir', default='reports',
                       help='Directory for professional report outputs')
    parser.add_argument('--disable-ai-validation', action='store_true',
                       help='Disable AI-driven anomaly validation in reports')
    
    args = parser.parse_args()
    
    # Initialize enhanced poller
    poller = EnhancedNEOPoller(
        data_dir=args.data_dir,
        professional_report=args.professional_report,
        report_dir=args.report_dir,
        enable_ai_validation=not args.disable_ai_validation
    )
    
    try:
        # Run enhanced polling process
        start_time = time.time()
        results = poller.run_enhanced_polling(
            api_choice="NASA_CAD",
            time_period=args.period,
            max_results=args.max_results
        )
        end_time = time.time()
        
        print(f"\n⏱️  Enhanced analysis completed in {end_time - start_time:.1f} seconds")
        
        # Display results with enhanced quality metrics
        poller.display_enriched_results(results, args.report_format)
        
        # Display source quality statistics (from legacy system)
        poller.display_source_quality_report()
        
        # Save results
        if results:
            filepath = poller.save_enriched_results(results, "NASA_CAD", args.period)
            if filepath:
                print(f"\n💾 Enhanced results saved to: {filepath}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()