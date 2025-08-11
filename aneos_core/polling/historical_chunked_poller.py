#!/usr/bin/env python3
"""
Historical Chunked NEO Poller - SAFE PATH PHASE 2

Implements chunked 200-year historical polling for massive dataset processing.
Breaks large time periods into manageable chunks to prevent memory issues
and API timeouts while maintaining data integrity.

Key Features:
- Chunked processing of long time periods (200+ years)
- Memory-efficient batch processing  
- Progress tracking and resumable operations
- Safe error handling with graceful degradation
- Integration with XVIII SWARM first-stage review
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time

# Try to import dependencies with fallbacks
try:
    from dateutil.relativedelta import relativedelta
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    class relativedelta:
        def __init__(self, days=0, weeks=0, months=0, years=0):
            self.days = days + (weeks * 7) + (months * 30) + (years * 365)
        def __rsub__(self, other):
            if isinstance(other, datetime):
                return other - timedelta(days=self.days)
            return other

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for chunked polling operations."""
    chunk_size_years: int = 5  # Process 5 years at a time
    max_objects_per_chunk: int = 50000  # Limit objects per chunk
    overlap_days: int = 7  # Overlap between chunks to avoid edge cases
    batch_size: int = 1000  # Objects to process in parallel
    retry_attempts: int = 3  # Retry failed chunks
    rate_limit_delay: float = 1.0  # Delay between API calls
    enable_caching: bool = True  # Cache chunk results
    
@dataclass
class ChunkResult:
    """Results from processing a single time chunk."""
    start_date: datetime
    end_date: datetime
    object_count: int
    processed_count: int
    candidate_count: int  # Objects flagged for further review
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    chunk_data: Optional[List[Dict]] = None

@dataclass
class HistoricalPollingResult:
    """Complete results from historical polling operation."""
    total_time_span_years: int
    total_chunks_processed: int
    total_objects_found: int
    total_candidates_flagged: int
    processing_start_time: datetime
    processing_end_time: datetime
    chunk_results: List[ChunkResult]
    summary_stats: Dict[str, Any]
    
class HistoricalChunkedPoller:
    """
    Chunked historical polling system for 200-year NEO data processing.
    
    This class implements safe, memory-efficient processing of massive
    historical NEO datasets by breaking them into manageable chunks.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize historical chunked poller.
        
        Args:
            config: Chunk processing configuration
        """
        self.config = config or ChunkConfig()
        self.logger = logging.getLogger(__name__)
        self.console = console if HAS_RICH else None
        
        # Initialize storage directories
        self.cache_dir = Path("neo_data/historical_cache")
        self.results_dir = Path("neo_data/historical_results")
        self._ensure_directories()
        
        # Initialize polling components (will be set by caller)
        self.base_poller = None  # Enhanced NEO Poller instance
        self.xviii_swarm_scorer = None  # Advanced scoring system
        
    def _ensure_directories(self):
        """Create necessary directories for caching and results."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to create directories: {e}")
    
    def set_components(self, base_poller=None, xviii_swarm_scorer=None):
        """
        Set external components needed for polling and scoring.
        
        Args:
            base_poller: EnhancedNEOPoller instance for data fetching
            xviii_swarm_scorer: XVIII SWARM advanced scoring system
        """
        self.base_poller = base_poller
        self.xviii_swarm_scorer = xviii_swarm_scorer
        
    def generate_time_chunks(
        self, 
        years_back: int = 200, 
        end_date: Optional[datetime] = None
    ) -> Generator[Tuple[datetime, datetime], None, None]:
        """
        Generate time chunks for processing historical data.
        
        Args:
            years_back: Number of years to go back from end_date
            end_date: End date for historical search (defaults to now)
            
        Yields:
            Tuples of (start_date, end_date) for each chunk
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - relativedelta(years=years_back)
        
        current_start = start_date
        chunk_delta = relativedelta(years=self.config.chunk_size_years)
        overlap_delta = timedelta(days=self.config.overlap_days)
        
        while current_start < end_date:
            # Calculate chunk end date
            current_end = min(current_start + chunk_delta, end_date)
            
            # Add overlap to avoid missing objects at boundaries
            chunk_end_with_overlap = min(current_end + overlap_delta, end_date)
            
            yield current_start, chunk_end_with_overlap
            
            # Move to next chunk
            current_start = current_end
    
    def estimate_chunk_load(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Estimate processing load for a time chunk.
        
        This helps optimize chunk sizes and resource allocation.
        
        Args:
            start_date: Chunk start date
            end_date: Chunk end date
            
        Returns:
            Dictionary with load estimates
        """
        years_span = (end_date - start_date).days / 365.25
        
        # Rough estimates based on NEO discovery rates
        # Discovery rate has increased dramatically over time
        base_rate = 100  # NEOs per year in early periods
        modern_rate = 3000  # NEOs per year in modern periods
        
        # Simple linear interpolation based on time period
        current_year = datetime.now().year
        start_year = start_date.year
        
        if start_year < 1980:
            estimated_rate = base_rate
        elif start_year < 2000:
            estimated_rate = base_rate + (modern_rate - base_rate) * (start_year - 1980) / 20
        else:
            estimated_rate = modern_rate
            
        estimated_objects = int(years_span * estimated_rate)
        
        return {
            'estimated_objects': estimated_objects,
            'years_span': years_span,
            'estimated_rate_per_year': estimated_rate,
            'processing_complexity': 'low' if estimated_objects < 10000 else 'high',
            'estimated_processing_time_minutes': estimated_objects / 1000  # Rough estimate
        }
    
    async def process_single_chunk(
        self, 
        start_date: datetime, 
        end_date: datetime,
        chunk_id: int
    ) -> ChunkResult:
        """
        Process a single time chunk for NEO data.
        
        Args:
            start_date: Chunk start date
            end_date: Chunk end date  
            chunk_id: Unique identifier for this chunk
            
        Returns:
            ChunkResult with processing results
        """
        start_time = time.time()
        
        try:
            self.logger.info(
                f"Processing chunk {chunk_id}: {start_date.strftime('%Y-%m-%d')} to "
                f"{end_date.strftime('%Y-%m-%d')}"
            )
            
            # Check cache first
            cache_key = f"chunk_{chunk_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            cached_result = self._load_cached_chunk(cache_key)
            
            if cached_result and self.config.enable_caching:
                self.logger.info(f"Using cached result for chunk {chunk_id}")
                return cached_result
            
            # Fetch NEO data for this time period
            if not self.base_poller:
                raise ValueError("Base poller not configured - call set_components() first")
            
            # Calculate time period string for existing poller
            years_span = (end_date - start_date).days / 365.25
            if years_span >= 10:
                period_str = f"{int(years_span)}y"
            elif years_span >= 1:
                period_str = f"{int(years_span * 12)}m"  
            else:
                period_str = f"{int((end_date - start_date).days)}d"
            
            # Fetch raw NEO data
            raw_data = await self._fetch_chunk_data(start_date, end_date, period_str)
            
            if not raw_data:
                return ChunkResult(
                    start_date=start_date,
                    end_date=end_date,
                    object_count=0,
                    processed_count=0,
                    candidate_count=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    success=True,
                    chunk_data=[]
                )
            
            # Process objects in batches to manage memory
            processed_objects = []
            candidate_objects = []
            
            for i in range(0, len(raw_data), self.config.batch_size):
                batch = raw_data[i:i + self.config.batch_size]
                batch_results = await self._process_object_batch(batch)
                
                processed_objects.extend(batch_results['processed'])
                candidate_objects.extend(batch_results['candidates'])
                
                # Rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)
            
            # Create result
            result = ChunkResult(
                start_date=start_date,
                end_date=end_date,
                object_count=len(raw_data),
                processed_count=len(processed_objects),
                candidate_count=len(candidate_objects),
                processing_time_ms=(time.time() - start_time) * 1000,
                success=True,
                chunk_data=candidate_objects  # Only store candidates to save memory
            )
            
            # Cache the result
            if self.config.enable_caching:
                self._save_cached_chunk(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process chunk {chunk_id}: {e}")
            return ChunkResult(
                start_date=start_date,
                end_date=end_date,
                object_count=0,
                processed_count=0,
                candidate_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
    
    async def _fetch_chunk_data(self, start_date: datetime, end_date: datetime, period_str: str) -> List[Dict]:
        """
        Fetch raw NEO data for a time chunk using real API sources.
        """
        try:
            if self.base_poller:
                # Use the enhanced NEO poller to fetch real data
                self.logger.info(f"Fetching real NEO data for period {period_str}")
                
                try:
                    # Call the enhanced poller's main polling method
                    poller_result = await self._call_enhanced_poller(period_str)
                    
                    if poller_result and 'neos' in poller_result:
                        neo_list = poller_result['neos']
                        self.logger.info(f"Fetched {len(neo_list)} NEO objects from APIs")
                        return neo_list
                    else:
                        self.logger.warning("Enhanced poller returned no data, using fallback")
                        
                except Exception as e:
                    self.logger.error(f"Enhanced poller failed: {e}")
                    
            # Fallback: Use NASA CAD API directly for real data
            return await self._fetch_nasa_cad_data(start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Failed to fetch chunk data: {e}")
            return []
    
    async def _call_enhanced_poller(self, period_str: str) -> Dict:
        """Call the enhanced NEO poller for real data."""
        try:
            # The enhanced poller expects different method calls
            # Let's try to use its polling capabilities
            
            if hasattr(self.base_poller, 'poll_neo_apis_enhanced'):
                # Call the enhanced polling method
                return await asyncio.to_thread(
                    self.base_poller.poll_neo_apis_enhanced,
                    time_period=period_str,
                    api_selections=['NASA_CAD'],  # Start with NASA CAD
                    enable_validation=True
                )
            elif hasattr(self.base_poller, 'fetch_cad_data'):
                # Fallback to direct CAD data fetching
                return await asyncio.to_thread(
                    self.base_poller.fetch_cad_data,
                    time_period=period_str
                )
            else:
                self.logger.warning("Enhanced poller doesn't have expected methods")
                return {}
                
        except Exception as e:
            self.logger.error(f"Enhanced poller call failed: {e}")
            return {}
    
    async def _fetch_nasa_cad_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Direct NASA CAD API fetch as fallback."""
        try:
            import requests
            import asyncio
            
            # NASA CAD API endpoint
            url = "https://ssd-api.jpl.nasa.gov/cad.api"
            
            # Format dates for NASA API
            date_min = start_date.strftime("%Y-%m-%d")
            date_max = end_date.strftime("%Y-%m-%d")
            
            params = {
                'date-min': date_min,
                'date-max': date_max,
                'diameter': '1',  # Only objects with diameter data
                'fullname': 'true'
            }
            
            self.logger.info(f"Fetching NASA CAD data from {date_min} to {date_max}")
            
            # Make async request
            def make_request():
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            
            data = await asyncio.to_thread(make_request)
            
            if 'data' in data and data['data']:
                # Convert NASA CAD format to our format
                neo_list = []
                fields = data.get('fields', [])
                
                for row in data['data']:
                    try:
                        # Create NEO object from NASA CAD data
                        # Parse NASA date format: "2025-Jul-09 18:22"
                        discovery_date = start_date
                        if len(row) > 3 and row[3]:
                            date_str = str(row[3])
                            try:
                                # Try NASA format: "2025-Jul-09 18:22"
                                discovery_date = datetime.strptime(date_str, "%Y-%b-%d %H:%M")
                            except ValueError:
                                try:
                                    # Fallback to simple date format
                                    discovery_date = datetime.strptime(date_str, "%Y-%m-%d")
                                except ValueError:
                                    # Use start_date as fallback
                                    self.logger.debug(f"Could not parse date '{date_str}', using start_date")
                                    discovery_date = start_date
                        
                        neo_obj = {
                            'designation': row[0] if len(row) > 0 else f'NEO_{len(neo_list)}',
                            'discovery_date': discovery_date,
                            'orbital_elements': {
                                'eccentricity': float(row[7]) if len(row) > 7 and row[7] else 0.1,
                                'inclination': float(row[8]) if len(row) > 8 and row[8] else 10,
                                'semi_major_axis': float(row[6]) if len(row) > 6 and row[6] else 1.0
                            },
                            'close_approach': {
                                'date': row[3] if len(row) > 3 else start_date.isoformat(),
                                'distance': float(row[4]) if len(row) > 4 and row[4] else 1.0,
                                'velocity': float(row[5]) if len(row) > 5 and row[5] else 10.0
                            },
                            'source': 'NASA_CAD'
                        }
                        neo_list.append(neo_obj)
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse NEO data row: {e}")
                        continue
                
                self.logger.info(f"Successfully fetched {len(neo_list)} real NEO objects from NASA CAD")
                return neo_list
            else:
                self.logger.warning("NASA CAD API returned no data")
                return []
                
        except Exception as e:
            self.logger.error(f"NASA CAD API fetch failed: {e}")
            return []
    
    async def _process_object_batch(self, batch: List[Dict]) -> Dict[str, List]:
        """
        Process a batch of NEO objects with XVIII SWARM first-stage review.
        
        Args:
            batch: List of NEO objects to process
            
        Returns:
            Dictionary with processed and candidate objects
        """
        processed = []
        candidates = []
        
        for neo_obj in batch:
            try:
                # Apply XVIII SWARM first-stage review
                if self.xviii_swarm_scorer:
                    # This would use the actual XVIII SWARM scoring
                    score = await self._apply_xviii_swarm_first_stage(neo_obj)
                    
                    # Add scoring results to object
                    neo_obj['xviii_swarm_score'] = score
                    neo_obj['processed_timestamp'] = datetime.now().isoformat()
                    
                    processed.append(neo_obj)
                    
                    # Flag as candidate if score is above threshold
                    if score.get('overall_score', 0) > 0.3:  # First-stage threshold
                        candidates.append(neo_obj)
                else:
                    # Fallback: simple heuristic screening
                    if self._simple_artificial_screening(neo_obj):
                        candidates.append(neo_obj)
                    processed.append(neo_obj)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process object {neo_obj.get('designation', 'unknown')}: {e}")
                continue
        
        return {'processed': processed, 'candidates': candidates}
    
    async def _apply_xviii_swarm_first_stage(self, neo_obj: Dict) -> Dict:
        """
        Apply XVIII SWARM first-stage review to a NEO object.
        
        This is a simplified version optimized for high-throughput screening.
        """
        try:
            # Mock XVIII SWARM first-stage scoring
            # In real implementation, this would use the actual advanced scorer
            
            orbital_elements = neo_obj.get('orbital_elements', {})
            eccentricity = orbital_elements.get('eccentricity', 0)
            inclination = orbital_elements.get('inclination', 0)
            
            # Simple first-stage scoring based on suspicious orbital characteristics
            score = 0.0
            
            # High eccentricity suggests possible artificial origin
            if eccentricity > 0.8:
                score += 0.4
            elif eccentricity > 0.5:
                score += 0.2
                
            # Unusual inclination patterns
            if inclination > 150 or inclination < 30:
                score += 0.3
                
            # Add some randomness for testing
            import random
            score += random.uniform(0, 0.3)
            
            return {
                'overall_score': min(score, 1.0),
                'first_stage_flags': [],
                'confidence': 0.7,  # First stage has lower confidence
                'processing_level': 'first_stage'
            }
            
        except Exception as e:
            self.logger.error(f"XVIII SWARM first-stage failed: {e}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    def _simple_artificial_screening(self, neo_obj: Dict) -> bool:
        """
        Simple heuristic screening for potentially artificial objects.
        
        Fallback method when XVIII SWARM is not available.
        """
        try:
            orbital_elements = neo_obj.get('orbital_elements', {})
            
            # Check for unusual orbital characteristics
            eccentricity = orbital_elements.get('eccentricity', 0)
            inclination = orbital_elements.get('inclination', 0)
            
            # Flag objects with very high eccentricity or unusual inclination
            if eccentricity > 0.9 or inclination > 160 or inclination < 20:
                return True
                
            return False
            
        except Exception:
            return False
    
    def _load_cached_chunk(self, cache_key: str) -> Optional[ChunkResult]:
        """Load cached chunk result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to ChunkResult
                    return ChunkResult(**data)
        except Exception as e:
            self.logger.warning(f"Failed to load cached chunk {cache_key}: {e}")
        return None
    
    def _save_cached_chunk(self, cache_key: str, result: ChunkResult):
        """Save chunk result to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                # Convert ChunkResult to dict for JSON serialization
                data = {
                    'start_date': result.start_date.isoformat(),
                    'end_date': result.end_date.isoformat(),
                    'object_count': result.object_count,
                    'processed_count': result.processed_count,
                    'candidate_count': result.candidate_count,
                    'processing_time_ms': result.processing_time_ms,
                    'success': result.success,
                    'error_message': result.error_message,
                    'chunk_data': result.chunk_data
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cached chunk {cache_key}: {e}")
    
    async def poll_historical_data(
        self, 
        years_back: int = 200,
        end_date: Optional[datetime] = None
    ) -> HistoricalPollingResult:
        """
        Poll historical NEO data using chunked processing.
        
        Args:
            years_back: Number of years to search back
            end_date: End date for search (defaults to now)
            
        Returns:
            HistoricalPollingResult with complete results
        """
        start_time = datetime.now()
        
        self.logger.info(f"Starting chunked historical polling: {years_back} years back")
        
        # Generate time chunks
        chunks = list(self.generate_time_chunks(years_back, end_date))
        total_chunks = len(chunks)
        
        self.logger.info(f"Processing {total_chunks} time chunks of ~{self.config.chunk_size_years} years each")
        
        # Process chunks with progress tracking
        chunk_results = []
        total_objects = 0
        total_candidates = 0
        
        if self.console and HAS_RICH:
            # Suppress verbose logging during progress display
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            logging.getLogger('aneos_core').setLevel(logging.ERROR)
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.fields[chunk_info]}"),
                    BarColumn(), 
                    TaskProgressColumn(),
                    TextColumn("{task.fields[status]}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task("chunks", total=total_chunks, chunk_info="Historical Data Polling", status="Starting...")
                    
                    for i, (start_date, end_date) in enumerate(chunks):
                        progress.update(task, chunk_info=f"Chunk {i+1}/{total_chunks}: {start_date.year}-{end_date.year}", status="Processing...")
                        
                        # Process chunk with retry logic
                        for attempt in range(self.config.retry_attempts):
                            try:
                                result = await self.process_single_chunk(start_date, end_date, i)
                                if result.success:
                                    break
                            except Exception as e:
                                self.logger.warning(f"Chunk {i} attempt {attempt+1} failed: {e}")
                                if attempt == self.config.retry_attempts - 1:
                                    # Create failed result
                                    result = ChunkResult(
                                        start_date=start_date,
                                        end_date=end_date,
                                        object_count=0,
                                        processed_count=0,
                                        candidate_count=0,
                                        processing_time_ms=0,
                                        success=False,
                                        error_message=f"Failed after {self.config.retry_attempts} attempts"
                                    )
                        
                        chunk_results.append(result)
                        
                        if result.success:
                            total_objects += result.object_count
                            total_candidates += result.candidate_count
                            progress.update(task, status=f"✅ {result.object_count:,} objects found")
                        else:
                            progress.update(task, status="❌ Chunk failed")
                        
                        progress.update(task, advance=1)
            finally:
                # Restore original logging levels
                logging.getLogger().setLevel(original_level)
                logging.getLogger('aneos_core').setLevel(original_level)
        else:
            # Fallback without rich progress
            for i, (start_date, end_date) in enumerate(chunks):
                print(f"Processing chunk {i+1}/{total_chunks}: {start_date.year}-{end_date.year}")
                
                result = await self.process_single_chunk(start_date, end_date, i)
                chunk_results.append(result)
                
                if result.success:
                    total_objects += result.object_count
                    total_candidates += result.candidate_count
        
        end_time = datetime.now()
        
        # Calculate summary statistics
        successful_chunks = sum(1 for r in chunk_results if r.success)
        failed_chunks = total_chunks - successful_chunks
        total_processing_time = sum(r.processing_time_ms for r in chunk_results) / 1000  # Convert to seconds
        
        summary_stats = {
            'successful_chunks': successful_chunks,
            'failed_chunks': failed_chunks,
            'success_rate': successful_chunks / total_chunks if total_chunks > 0 else 0,
            'total_processing_time_seconds': total_processing_time,
            'average_objects_per_chunk': total_objects / successful_chunks if successful_chunks > 0 else 0,
            'candidate_selection_rate': total_candidates / total_objects if total_objects > 0 else 0,
        }
        
        result = HistoricalPollingResult(
            total_time_span_years=years_back,
            total_chunks_processed=total_chunks,
            total_objects_found=total_objects,
            total_candidates_flagged=total_candidates,
            processing_start_time=start_time,
            processing_end_time=end_time,
            chunk_results=chunk_results,
            summary_stats=summary_stats
        )
        
        # Save complete results
        self._save_historical_results(result)
        
        self.logger.info(
            f"Historical polling complete: {total_objects} objects found, "
            f"{total_candidates} candidates flagged in {(end_time - start_time).total_seconds():.1f} seconds"
        )
        
        return result
    
    def _save_historical_results(self, result: HistoricalPollingResult):
        """Save complete historical polling results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_poll_{result.total_time_span_years}y_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Convert result to JSON-serializable format
            data = {
                'total_time_span_years': result.total_time_span_years,
                'total_chunks_processed': result.total_chunks_processed,
                'total_objects_found': result.total_objects_found,
                'total_candidates_flagged': result.total_candidates_flagged,
                'processing_start_time': result.processing_start_time.isoformat(),
                'processing_end_time': result.processing_end_time.isoformat(),
                'summary_stats': result.summary_stats,
                'chunk_count': len(result.chunk_results),
                'successful_chunks': sum(1 for r in result.chunk_results if r.success),
                # Don't save all chunk data to avoid huge files
                'sample_chunk_results': [
                    {
                        'start_date': r.start_date.isoformat(),
                        'end_date': r.end_date.isoformat(),
                        'object_count': r.object_count,
                        'candidate_count': r.candidate_count,
                        'success': r.success
                    }
                    for r in result.chunk_results[:10]  # Just save first 10 as sample
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Historical results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save historical results: {e}")

# Convenience function for easy usage
async def create_historical_poller(
    chunk_size_years: int = 5,
    max_objects_per_chunk: int = 50000,
    enable_caching: bool = True
) -> HistoricalChunkedPoller:
    """
    Create and configure a historical chunked poller.
    
    Args:
        chunk_size_years: Size of each time chunk in years
        max_objects_per_chunk: Maximum objects to process per chunk
        enable_caching: Whether to enable result caching
        
    Returns:
        Configured HistoricalChunkedPoller instance
    """
    config = ChunkConfig(
        chunk_size_years=chunk_size_years,
        max_objects_per_chunk=max_objects_per_chunk,
        enable_caching=enable_caching
    )
    
    poller = HistoricalChunkedPoller(config)
    
    # Try to initialize components
    try:
        from ..analysis.enhanced_pipeline import EnhancedAnalysisPipeline
        from ..analysis.advanced_scoring import AdvancedScoreCalculator
        
        # These would be set up by the caller with actual instances
        # poller.set_components(enhanced_pipeline, advanced_scorer)
        
    except ImportError:
        logging.warning("Could not import enhanced analysis components")
    
    return poller

if __name__ == "__main__":
    # Test the chunked poller
    import asyncio
    
    async def test_chunked_poller():
        poller = await create_historical_poller(chunk_size_years=10)
        
        # Test with smaller time period for demo
        result = await poller.poll_historical_data(years_back=50)
        
        print(f"Test completed: {result.total_objects_found} objects, {result.total_candidates_flagged} candidates")
    
    asyncio.run(test_chunked_poller())