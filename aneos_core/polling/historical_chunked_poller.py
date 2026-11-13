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

from aneos_core.data.provider_http import get as provider_get
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator, Tuple, Union
from dataclasses import dataclass, field
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
    cache_ttl_seconds: float = 86400
    
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
    schema_version: str = "cad-intake-v1"
    intake_metrics: Dict[str, Any] = field(default_factory=dict)

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
    
class _EncounterRecords(list):
    def __init__(self, records, metadata):
        super().__init__(records)
        self.metadata = metadata


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
        self.xviii_swarm_scorer = None  # Legacy injection retained; never selects a random model
        self.data_fetcher = None
        self._orbit_cache = {}
        
    def _ensure_directories(self):
        """Create necessary directories for caching and results."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to create directories: {e}")
    
    def set_components(self, base_poller=None, xviii_swarm_scorer=None, data_fetcher=None):
        """
        Set external components needed for polling and scoring.
        
        Args:
            base_poller: EnhancedNEOPoller instance for data fetching
            xviii_swarm_scorer: XVIII SWARM advanced scoring system
        """
        self.base_poller = base_poller
        self.xviii_swarm_scorer = xviii_swarm_scorer
        self.data_fetcher = data_fetcher
        
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
        if type(years_back) is not int or years_back <= 0:
            raise ValueError('Historical years_back must be a positive integer')
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
    
    async def process_single_chunk(self, start_date, end_date, chunk_id):
        """Acquire all encounters; ranking abstention never discards intake evidence."""
        start = time.time()
        key = f"cad_v1_chunk_{chunk_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_limit{self.config.max_objects_per_chunk}"
        try:
            if self.config.enable_caching:
                cached = self._load_cached_chunk(key)
                if cached is not None:
                    return cached
            raw = await self._fetch_chunk_data(start_date, end_date)
            processed, candidates = [], []
            for index in range(0, len(raw), self.config.batch_size):
                batch = await self._process_object_batch(raw[index:index + self.config.batch_size])
                processed.extend(batch['processed'])
                candidates.extend(batch['candidates'])
            statuses = [obj.get('orbit_acquisition', {}).get('status') for obj in processed]
            metrics = {
                'acquisition_status': getattr(raw, 'metadata', {}).get('completeness_status', 'unverified'),
                'source_metadata': getattr(raw, 'metadata', {}),
                'encounter_count': len(raw),
                'scored_count': sum(obj['screening_assessment']['screening_score'] is not None for obj in processed),
                'abstention_count': sum(obj['screening_assessment']['screening_score'] is None for obj in processed),
                'enrichment_errors': statuses.count('unavailable'),
                'screening_status': 'partial' if 'unavailable' in statuses else 'executed',
                'count_unit': 'computed_encounters_not_unique_objects_or_observations',
            }
            result = ChunkResult(start_date, end_date, len(raw), len(processed), len(candidates),
                (time.time() - start) * 1000, True, chunk_data=processed, intake_metrics=metrics)
            if self.config.enable_caching and not metrics['enrichment_errors']:
                self._save_cached_chunk(key, result)
            return result
        except Exception as exc:
            self.logger.warning('CAD chunk unavailable: %s', exc)
            return ChunkResult(start_date, end_date, 0, 0, 0, (time.time() - start) * 1000,
                False, error_message=f'{type(exc).__name__}: {exc}', chunk_data=[],
                intake_metrics={'acquisition_status': 'unavailable', 'reason': str(exc)})

    async def _fetch_chunk_data(self, start_date, end_date):
        if self.base_poller is None:
            return await self._fetch_nasa_cad_data(start_date, end_date)
        result = await self._call_enhanced_poller(start_date, end_date)
        return _EncounterRecords(result['neos'], result['metadata'])

    async def _call_enhanced_poller(self, start_date, end_date):
        """Validate the base adapter response, including zero results and truncation."""
        from ..data.sources.cad import parse_cad_payload
        method = (getattr(self.base_poller, 'fetch_cad_data', None)
                  or getattr(self.base_poller, 'fetch_cad_data_with_cache', None))
        if method is None:
            raise ValueError('Configured CAD adapter has no supported fetch method')
        limit = self.config.max_objects_per_chunk
        payload = await asyncio.to_thread(method, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), limit)
        query = {'date-min': start_date.strftime('%Y-%m-%d'), 'date-max': end_date.strftime('%Y-%m-%d'),
                 'limit': limit, 'body': 'Earth', 'dist-max': '.05', 'scope_basis': 'base_adapter_CAD_default_filters'}
        parsed = parse_cad_payload(payload, query=query)
        return {'neos': parsed['records'], 'metadata': parsed['metadata']}

    def _convert_cad_to_neo_format(self, row, fields, discovery_date):
        """Compatibility adapter: query start is not an object discovery date."""
        from ..data.sources.cad import parse_cad_row
        return parse_cad_row(row, fields, {'schema_status': 'unverified_row_only'})

    async def _fetch_nasa_cad_data(self, start_date, end_date):
        from ..data.sources.cad import parse_cad_payload
        from ..data.sources.base import _APPLICATION_HEADERS
        import requests
        query = {'date-min': start_date.strftime('%Y-%m-%d'), 'date-max': end_date.strftime('%Y-%m-%d'),
                 'body': 'Earth', 'dist-max': '.05', 'diameter': 'true', 'fullname': 'true',
                 'limit': self.config.max_objects_per_chunk}
        def request():
            response = provider_get('https://ssd-api.jpl.nasa.gov/cad.api', params=query,
                                    headers=_APPLICATION_HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        payload = await asyncio.to_thread(request)
        from datetime import timezone
        parsed = parse_cad_payload(payload, query=query, retrieved_at=datetime.now(timezone.utc).isoformat())
        return _EncounterRecords(parsed['records'], parsed['metadata'])

    @staticmethod
    def _record_to_neo(neo_obj):
        from ..data.models import NEOData, PhysicalProperties
        from ..data.sources.cad import close_approach_from_record
        from copy import deepcopy
        neo = deepcopy(neo_obj['neo_data']) if isinstance(neo_obj.get('neo_data'), NEOData) else NEOData.from_dict(neo_obj)
        if neo.physical_properties is None and any(neo_obj.get(k) is not None for k in ('diameter_km', 'absolute_magnitude_h')):
            neo.physical_properties = PhysicalProperties(diameter_km=neo_obj.get('diameter_km'),
                diameter_uncertainty=neo_obj.get('diameter_uncertainty'), absolute_magnitude_h=neo_obj.get('absolute_magnitude_h'))
        if neo_obj.get('cad_evidence'):
            approach = close_approach_from_record(neo_obj)
            if not any(ca.epoch_jd_tdb == approach.epoch_jd_tdb and ca.orbit_id == approach.orbit_id
                       and ca.native_calendar == approach.native_calendar and ca.body == approach.body for ca in neo.close_approaches):
                neo.add_close_approach(approach)
            neo.evidence_metadata.setdefault('cad_intake', {})[neo_obj['encounter_id']] = neo_obj['cad_evidence']
        return neo

    async def _enrich_orbit(self, neo_obj):
        from ..data.models import NEOData
        from copy import deepcopy
        neo = self._record_to_neo(neo_obj)
        if neo.orbital_elements is not None:
            neo_obj['orbit_acquisition'] = {'status': 'provided', 'basis': 'provided_catalogue_input_not_independent_validation'}
        else:
            try:
                if self.data_fetcher is None:
                    from ..data.fetcher import DataFetcher
                    self.data_fetcher = DataFetcher()
                if getattr(self, '_enrichment_provider_id', None) != id(self.data_fetcher):
                    self._enrichment_provider_id = id(self.data_fetcher)
                    self._orbit_cache, self._orbit_failures = {}, {}
                des = neo_obj['designation']
                failure = self._orbit_failures.get(des)
                if failure and time.monotonic() < failure['retry_after']:
                    raise RuntimeError(failure['reason'])
                cached = self._orbit_cache.get(des)
                if cached and time.monotonic() - cached[0] > min(self.config.cache_ttl_seconds, 3600):
                    cached = None
                if cached is None:
                    try:
                        value = await asyncio.to_thread(self.data_fetcher.fetch_neo_data, des)
                        if not isinstance(value, NEOData) or value.designation != des:
                            raise ValueError('Orbit acquisition returned an invalid or different object')
                    except Exception as exc:
                        self._orbit_failures[des] = {'reason': str(exc), 'retry_after': time.monotonic() + max(30, self.config.rate_limit_delay)}
                        raise
                    self._orbit_cache[des] = (time.monotonic(), value)
                    await asyncio.sleep(self.config.rate_limit_delay)
                acquired = deepcopy(self._orbit_cache[des][1])
                if not isinstance(acquired, NEOData) or acquired.designation != des:
                    raise ValueError('Orbit acquisition returned an invalid or different object')
                neo_obj['neo_data'] = acquired
                neo = self._record_to_neo(neo_obj)
                neo_obj['orbit_acquisition'] = {'status': 'available' if neo.orbital_elements else 'not_available',
                    'basis': 'separate_catalogue_orbit_solution'}
            except Exception as exc:
                neo_obj['orbit_acquisition'] = {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'}
        neo_obj['neo_data'] = neo
        neo_obj['orbital_elements'] = neo.orbital_elements.to_dict() if neo.orbital_elements else None
        return neo

    async def _process_object_batch(self, batch):
        from ..analysis.screening import ScreeningAnalysisService
        processed, candidates = [], []
        for item in batch:
            neo = await self._enrich_orbit(item)
            assessment = ScreeningAnalysisService().analyze_neo(neo)
            payload = assessment.to_payload()
            item['screening_assessment'] = payload
            item['xviii_swarm_score'] = {**payload, 'overall_score': payload['screening_score'],
                'processing_level': 'uncalibrated_canonical_screening'}
            processed.append(item)
            if assessment.candidate:
                candidates.append(item)
        return {'processed': processed, 'candidates': candidates}

    async def _apply_xviii_swarm_first_stage(self, neo_obj):
        """Legacy function name; deterministic canonical rank, never random scoring."""
        from ..analysis.screening import ScreeningAnalysisService
        payload = ScreeningAnalysisService().analyze_neo(self._record_to_neo(neo_obj)).to_payload()
        return {**payload, 'overall_score': payload['screening_score'],
                'processing_level': 'uncalibrated_canonical_screening'}

    def _simple_artificial_screening(self, neo_obj):
        """Compatibility candidate flag, not an artificial-origin classification."""
        from ..analysis.screening import ScreeningAnalysisService
        return ScreeningAnalysisService().analyze_neo(self._record_to_neo(neo_obj)).candidate

    @staticmethod
    def _cache_policy_id():
        from ..analysis.screening import ScreeningAnalysisService
        return ScreeningAnalysisService().analyze_raw('CACHE_POLICY', {}).policy_id

    def _load_cached_chunk(self, cache_key):
        from ..data.models import NEOData
        from datetime import timezone
        paths = list(self.cache_dir.glob(f'{cache_key}_*.json'))
        exact = self.cache_dir / f'{cache_key}.json'
        if exact.exists():
            paths.append(exact)
        for path in sorted(paths, reverse=True):
            try:
                data = json.loads(path.read_text())
                from ..data.sources.cad import _hash
                checksum = data.pop('content_sha256', None)
                if checksum != _hash(data):
                    continue
                if data.get('schema_version') != 'cad-intake-v1' or data.get('cache_policy_id') != self._cache_policy_id():
                    continue  # Historical legacy cache is preserved, not silently trusted.
                stamp = datetime.fromisoformat(data.pop('cached_at'))
                age = (datetime.now(timezone.utc) - stamp).total_seconds()
                if age < 0 or age > self.config.cache_ttl_seconds:
                    continue
                data.pop('cache_policy_id')
                for key in ('start_date', 'end_date'):
                    data[key] = datetime.fromisoformat(data[key])
                for record in data.get('chunk_data', []):
                    if isinstance(record.get('neo_data'), dict):
                        record['neo_data'] = NEOData.from_dict(record['neo_data'])
                return ChunkResult(**data)
            except Exception as exc:
                self.logger.warning('Unusable CAD cache %s: %s', path.name, exc)
        return None

    def _save_cached_chunk(self, cache_key, result):
        from dataclasses import asdict
        from datetime import timezone
        from uuid import uuid4
        from ..data.models import NEOData
        from copy import deepcopy
        try:
            data = asdict(result)
            data['start_date'], data['end_date'] = result.start_date.isoformat(), result.end_date.isoformat()
            data['chunk_data'] = deepcopy(result.chunk_data)
            for record in data['chunk_data'] or []:
                neo = record.get('neo_data')
                if isinstance(neo, NEOData):
                    record['neo_data'] = neo.to_dict()
                    if neo.physical_properties:
                        record['neo_data']['physical_properties'] = neo.physical_properties.to_dict()
            data.update(cache_policy_id=self._cache_policy_id(), cached_at=datetime.now(timezone.utc).isoformat())
            from ..data.sources.cad import _hash
            data['content_sha256'] = _hash(data)
            serialized = json.dumps(data, indent=2, allow_nan=False)
            path = self.cache_dir / f"{cache_key}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid4().hex[:8]}.json"
            with path.open('x') as handle:
                handle.write(serialized)
        except Exception as exc:
            self.logger.warning('Could not cache CAD intake: %s', exc)

    async def _process_chunk_with_retries(self, start_date, end_date, chunk_id):
        attempts = self.config.retry_attempts
        if type(attempts) is not int or attempts < 1:
            raise ValueError('CAD retry attempts must be a positive integer')
        for attempt in range(attempts):
            result = await self.process_single_chunk(start_date, end_date, chunk_id)
            if result.success:
                return result
            if attempt + 1 < attempts:
                await asyncio.sleep(max(1., self.config.rate_limit_delay) * 2 ** attempt)
        return result

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
                        
                        result = await self._process_chunk_with_retries(start_date, end_date, i)

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
                
                result = await self._process_chunk_with_retries(start_date, end_date, i)
                chunk_results.append(result)
                
                if result.success:
                    total_objects += result.object_count
                    total_candidates += result.candidate_count
        
        end_time = datetime.now()
        
        # Calculate summary statistics
        successful_chunks = sum(1 for r in chunk_results if r.success)
        failed_chunks = total_chunks - successful_chunks
        total_processing_time = sum(r.processing_time_ms for r in chunk_results) / 1000  # Convert to seconds
        
        distinct = self._merge_chunks([r.chunk_data or [] for r in chunk_results if r.success])
        summary_stats = {
            'count_unit': 'computed_encounter_records_before_overlap_dedup',
            'distinct_encounters': len(distinct),
            'unique_designations': len({r.get('designation') for r in distinct}),
            'scored_distinct_encounters': sum(r.get('screening_assessment', {}).get('screening_score') is not None for r in distinct),
            'source_coverage_status': 'complete' if failed_chunks == 0 else 'partial' if successful_chunks else 'unavailable',
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
    
    @staticmethod
    def _merge_chunks(chunks):
        """Deduplicate the same encounter while preserving distinct dated events."""
        from hashlib import sha256
        seen = {}
        for chunk in chunks:
            for obj in chunk:
                identity = obj.get('encounter_id')
                if identity is None:
                    identity = sha256(json.dumps({'designation': obj.get('designation'),
                        'date': obj.get('close_approach_date'), 'encounter': obj.get('close_approach')},
                        sort_keys=True, default=str).encode()).hexdigest()
                if identity not in seen:
                    seen[identity] = obj
                elif obj.get('cad_evidence') != seen[identity].get('cad_evidence'):
                    from copy import deepcopy
                    seen[identity].setdefault('duplicate_encounter_versions', []).append(deepcopy(obj.get('cad_evidence')))
        return list(seen.values())

    def _save_historical_results(self, result: HistoricalPollingResult):
        """Save complete historical polling results."""
        try:
            from uuid import uuid4
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"historical_poll_{result.total_time_span_years}y_{timestamp}_{uuid4().hex[:8]}.json"
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
                'chunk_summaries': [{'start_date': r.start_date.isoformat(), 'end_date': r.end_date.isoformat(),
                    'success': r.success, 'error_message': r.error_message, 'intake_metrics': r.intake_metrics}
                    for r in result.chunk_results],
                'evidence_storage': {'chunk_cache_enabled': self.config.enable_caching,
                    'note': 'Operational intake/cache, not a frozen scientific case dataset'},
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
            
            with open(filepath, 'x') as f:
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