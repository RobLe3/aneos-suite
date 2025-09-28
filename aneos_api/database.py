"""
Database integration for aNEOS API.

Provides database connectivity, ORM models, and data persistence
for analysis results, user data, metrics, and system state.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logging.warning("SQLAlchemy not available, database features disabled")

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False
    logging.warning("SQLite not available")

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("ANEOS_DATABASE_URL", "sqlite:///./aneos.db")
DATABASE_PATH = Path("aneos.db")

if HAS_SQLALCHEMY:
    # Create engine and session
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
    else:
        engine = create_engine(DATABASE_URL, echo=False)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    
    # Database Models
    class User(Base):
        """User model for authentication and authorization."""
        __tablename__ = "users"
        
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(String(100), unique=True, index=True)
        username = Column(String(50), unique=True, index=True)
        email = Column(String(255), unique=True, index=True)
        password_hash = Column(String(255))
        role = Column(String(20), default="viewer")
        api_keys = Column(JSON)  # Store API keys as JSON array
        created_at = Column(DateTime, default=datetime.utcnow)
        last_login = Column(DateTime)
        is_active = Column(Boolean, default=True)
    
    class AnalysisResult(Base):
        """Analysis result model for storing NEO analysis data."""
        __tablename__ = "analysis_results"
        
        id = Column(Integer, primary_key=True, index=True)
        designation = Column(String(50), index=True)
        analysis_date = Column(DateTime, default=datetime.utcnow, index=True)
        overall_score = Column(Float)
        classification = Column(String(20), index=True)
        confidence = Column(Float)
        processing_time = Column(Float)
        
        # JSON fields for complex data
        anomaly_score_data = Column(JSON)
        orbital_elements = Column(JSON)
        close_approaches = Column(JSON)
        indicator_results = Column(JSON)
        raw_neo_data = Column(JSON)
        
        # Metadata
        data_quality_score = Column(Float)
        cache_hit = Column(Boolean, default=False)
        analyzed_by = Column(String(50))
    
    class MLPrediction(Base):
        """ML prediction model for storing machine learning results."""
        __tablename__ = "ml_predictions"
        
        id = Column(Integer, primary_key=True, index=True)
        designation = Column(String(50), index=True)
        prediction_date = Column(DateTime, default=datetime.utcnow, index=True)
        model_id = Column(String(100))
        anomaly_score = Column(Float)
        anomaly_probability = Column(Float)
        is_anomaly = Column(Boolean)
        confidence = Column(Float)
        
        # JSON fields
        model_predictions = Column(JSON)  # Individual model predictions
        feature_contributions = Column(JSON)
        
        # Metadata
        feature_count = Column(Integer)
        feature_quality = Column(Float)
        prediction_time = Column(Float)
        cache_hit = Column(Boolean, default=False)
    
    class SystemMetrics(Base):
        """System metrics model for storing monitoring data."""
        __tablename__ = "system_metrics"
        
        id = Column(Integer, primary_key=True, index=True)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
        # System metrics
        cpu_percent = Column(Float)
        memory_percent = Column(Float)
        memory_used_mb = Column(Float)
        disk_usage_percent = Column(Float)
        network_bytes_sent = Column(Integer)
        network_bytes_recv = Column(Integer)
        process_count = Column(Integer)
        
        # Application metrics
        analysis_count = Column(Integer, default=0)
        prediction_count = Column(Integer, default=0)
        active_connections = Column(Integer, default=0)
        cache_hit_rate = Column(Float, default=0.0)
    
    class Alert(Base):
        """Alert model for storing system alerts."""
        __tablename__ = "alerts"
        
        id = Column(Integer, primary_key=True, index=True)
        alert_id = Column(String(100), unique=True, index=True)
        alert_type = Column(String(50), index=True)
        alert_level = Column(String(20), index=True)
        title = Column(String(255))
        message = Column(Text)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
        # Status
        acknowledged = Column(Boolean, default=False)
        acknowledged_by = Column(String(50))
        acknowledged_at = Column(DateTime)
        resolved = Column(Boolean, default=False)
        resolved_by = Column(String(50))
        resolved_at = Column(DateTime)
        
        # Additional data
        data = Column(JSON)
    
    class TrainingSession(Base):
        """Training session model for ML model training tracking."""
        __tablename__ = "training_sessions"
        
        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String(100), unique=True, index=True)
        started_at = Column(DateTime, default=datetime.utcnow)
        completed_at = Column(DateTime)
        status = Column(String(20), default="running")  # running, completed, failed, cancelled
        
        # Training parameters
        model_types = Column(JSON)
        training_size = Column(Integer)
        validation_split = Column(Float)
        hyperparameter_optimization = Column(Boolean, default=False)
        
        # Results
        training_score = Column(Float)
        validation_score = Column(Float)
        model_paths = Column(JSON)
        
        # Metadata
        started_by = Column(String(50))
        error_message = Column(Text)
    
    class APIUsage(Base):
        """API usage tracking model."""
        __tablename__ = "api_usage"
        
        id = Column(Integer, primary_key=True, index=True)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        endpoint = Column(String(255), index=True)
        method = Column(String(10))
        status_code = Column(Integer, index=True)
        response_time = Column(Float)
        
        # User information
        user_id = Column(String(100), index=True)
        api_key = Column(String(100))
        client_ip = Column(String(50))
        user_agent = Column(String(255))
        
        # Request/response size
        request_size = Column(Integer)
        response_size = Column(Integer)

    class EnrichedNEO(Base):
        """Enriched NEO model for comprehensive multi-source data storage."""
        __tablename__ = "enriched_neos"
        
        id = Column(Integer, primary_key=True, index=True)
        designation = Column(String(50), unique=True, index=True)  # Primary identifier
        
        # Core metadata
        first_discovered = Column(DateTime, index=True)
        last_updated = Column(DateTime, default=datetime.utcnow, index=True)
        data_sources = Column(JSON)  # List of all sources that provided data
        data_quality_score = Column(Float, default=0.0)
        
        # NASA CAD data
        nasa_cad_data = Column(JSON)
        nasa_cad_last_update = Column(DateTime)
        
        # NASA SBDB data  
        nasa_sbdb_data = Column(JSON)
        nasa_sbdb_last_update = Column(DateTime)
        
        # MPC data
        mpc_data = Column(JSON)
        mpc_last_update = Column(DateTime)
        
        # NEODyS data
        neodys_data = Column(JSON)
        neodys_last_update = Column(DateTime)
        
        # Consolidated orbital elements (best available from all sources)
        orbital_elements = Column(JSON)
        orbital_elements_source = Column(String(50))  # Which source provided the best data
        orbital_elements_quality = Column(Float, default=0.0)
        
        # Physical properties (consolidated)
        physical_properties = Column(JSON)  # diameter, albedo, mass, rotation period, etc.
        
        # Close approach data (consolidated from all sources)
        close_approaches = Column(JSON)
        
        # Discovery and observation history
        discovery_data = Column(JSON)
        observation_history = Column(JSON)
        
        # Classification and analysis results
        pha_status = Column(Boolean, default=False)  # Potentially Hazardous Asteroid
        neo_classification = Column(String(20))  # Atira, Aten, Apollo, Amor
        artificial_probability = Column(Float, default=0.0)
        artificial_analysis_date = Column(DateTime)
        risk_factors = Column(JSON)
        
        # Data completeness tracking
        has_orbital_elements = Column(Boolean, default=False)
        has_physical_properties = Column(Boolean, default=False)
        has_close_approaches = Column(Boolean, default=False)
        has_discovery_data = Column(Boolean, default=False)
        completeness_score = Column(Float, default=0.0)  # 0-1 score of data completeness
        
        # Polling metadata
        polling_sessions = Column(JSON)  # Track which polling sessions included this NEO
        total_detections = Column(Integer, default=1)  # How many times this NEO was detected
        
        # Analysis metadata
        analyzed_count = Column(Integer, default=0)
        last_analysis_date = Column(DateTime)
        analysis_results = Column(JSON)  # Store analysis results history
    
else:
    # Fallback models when SQLAlchemy is not available
    Base = None
    User = None
    AnalysisResult = None
    MLPrediction = None
    SystemMetrics = None
    Alert = None
    TrainingSession = None
    APIUsage = None
    EnrichedNEO = None

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self.engine = engine if HAS_SQLALCHEMY else None
        self.SessionLocal = SessionLocal if HAS_SQLALCHEMY else None
        
    def get_db(self) -> Session:
        """Get database session."""
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not available")
        
        db = self.SessionLocal()
        try:
            return db
        except Exception:
            db.close()
            raise
    
    def init_database(self):
        """Initialize database tables."""
        if not HAS_SQLALCHEMY:
            logger.warning("Cannot initialize database - SQLAlchemy not available")
            return False
        
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def close_database(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

# Database service functions
class AnalysisService:
    """Service for managing analysis results in database."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_analysis_result(self, result_data: Dict[str, Any]) -> Optional[int]:
        """Save analysis result to database."""
        if not HAS_SQLALCHEMY:
            return None
        
        try:
            analysis = AnalysisResult(
                designation=result_data['designation'],
                overall_score=result_data.get('overall_score', 0.0),
                classification=result_data.get('classification', 'unknown'),
                confidence=result_data.get('confidence', 0.0),
                processing_time=result_data.get('processing_time', 0.0),
                anomaly_score_data=result_data.get('anomaly_score_data'),
                orbital_elements=result_data.get('orbital_elements'),
                close_approaches=result_data.get('close_approaches'),
                indicator_results=result_data.get('indicator_results'),
                raw_neo_data=result_data.get('raw_neo_data'),
                data_quality_score=result_data.get('data_quality_score', 0.0),
                cache_hit=result_data.get('cache_hit', False),
                analyzed_by=result_data.get('analyzed_by', 'system')
            )
            
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            
            return analysis.id
            
        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}")
            self.db.rollback()
            return None
    
    def get_analysis_results(self, designation: Optional[str] = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis results from database."""
        if not HAS_SQLALCHEMY:
            return []
        
        try:
            query = self.db.query(AnalysisResult)
            
            if designation:
                query = query.filter(AnalysisResult.designation == designation)
            
            results = query.order_by(AnalysisResult.analysis_date.desc()).limit(limit).all()
            
            return [
                {
                    'id': r.id,
                    'designation': r.designation,
                    'analysis_date': r.analysis_date,
                    'overall_score': r.overall_score,
                    'classification': r.classification,
                    'confidence': r.confidence,
                    'processing_time': r.processing_time,
                    'anomaly_score_data': r.anomaly_score_data,
                    'orbital_elements': r.orbital_elements,
                    'close_approaches': r.close_approaches,
                    'data_quality_score': r.data_quality_score
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get analysis results: {e}")
            return []

class MLService:
    """Service for managing ML predictions in database."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[int]:
        """Save ML prediction to database."""
        if not HAS_SQLALCHEMY:
            return None
        
        try:
            prediction = MLPrediction(
                designation=prediction_data['designation'],
                model_id=prediction_data.get('model_id', 'unknown'),
                anomaly_score=prediction_data.get('anomaly_score', 0.0),
                anomaly_probability=prediction_data.get('anomaly_probability', 0.0),
                is_anomaly=prediction_data.get('is_anomaly', False),
                confidence=prediction_data.get('confidence', 0.0),
                model_predictions=prediction_data.get('model_predictions'),
                feature_contributions=prediction_data.get('feature_contributions'),
                feature_count=prediction_data.get('feature_count', 0),
                feature_quality=prediction_data.get('feature_quality', 0.0),
                prediction_time=prediction_data.get('prediction_time', 0.0),
                cache_hit=prediction_data.get('cache_hit', False)
            )
            
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            
            return prediction.id
            
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            self.db.rollback()
            return None

class MetricsService:
    """Service for managing system metrics in database."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_metrics(self, metrics_data: Dict[str, Any]) -> Optional[int]:
        """Save system metrics to database."""
        if not HAS_SQLALCHEMY:
            return None
        
        try:
            metrics = SystemMetrics(
                cpu_percent=metrics_data.get('cpu_percent', 0.0),
                memory_percent=metrics_data.get('memory_percent', 0.0),
                memory_used_mb=metrics_data.get('memory_used_mb', 0.0),
                disk_usage_percent=metrics_data.get('disk_usage_percent', 0.0),
                network_bytes_sent=metrics_data.get('network_bytes_sent', 0),
                network_bytes_recv=metrics_data.get('network_bytes_recv', 0),
                process_count=metrics_data.get('process_count', 0),
                analysis_count=metrics_data.get('analysis_count', 0),
                prediction_count=metrics_data.get('prediction_count', 0),
                active_connections=metrics_data.get('active_connections', 0),
                cache_hit_rate=metrics_data.get('cache_hit_rate', 0.0)
            )
            
            self.db.add(metrics)
            self.db.commit()
            self.db.refresh(metrics)
            
            return metrics.id
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            self.db.rollback()
            return None
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history from database."""
        if not HAS_SQLALCHEMY:
            return []
        
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            results = self.db.query(SystemMetrics).filter(
                SystemMetrics.timestamp >= start_time
            ).order_by(SystemMetrics.timestamp.desc()).all()
            
            return [
                {
                    'timestamp': r.timestamp,
                    'cpu_percent': r.cpu_percent,
                    'memory_percent': r.memory_percent,
                    'analysis_count': r.analysis_count,
                    'prediction_count': r.prediction_count,
                    'cache_hit_rate': r.cache_hit_rate
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []

class EnrichedNEOService:
    """Service for managing enriched NEO data in database."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def enrich_neo_data(self, polling_results: List[Dict[str, Any]], polling_session_id: str) -> Dict[str, Any]:
        """
        Enrich NEO database with polling results from multiple sources.
        
        Args:
            polling_results: List of NEO data from polling
            polling_session_id: Unique identifier for this polling session
            
        Returns:
            Dict with enrichment statistics
        """
        if not HAS_SQLALCHEMY:
            return {'error': 'Database not available'}
        
        stats = {
            'new_neos': 0,
            'updated_neos': 0,
            'enriched_neos': 0,
            'total_processed': len(polling_results),
            'session_id': polling_session_id
        }
        
        try:
            # Track processed NEOs in this session to avoid duplicates
            processed_in_session = {}
            
            for result in polling_results:
                designation = result.get('designation', 'Unknown')
                if designation == 'Unknown':
                    continue
                
                # Check if already processed in this session
                if designation in processed_in_session:
                    # Update the in-session NEO with additional source data
                    existing_neo = processed_in_session[designation]
                    self._update_existing_neo(existing_neo, result, polling_session_id)
                    stats['enriched_neos'] += 1
                    continue
                
                # Check if NEO already exists in database
                existing_neo = self.db.query(EnrichedNEO).filter(
                    EnrichedNEO.designation == designation
                ).first()
                
                if existing_neo:
                    # Update existing NEO with new data
                    self._update_existing_neo(existing_neo, result, polling_session_id)
                    stats['updated_neos'] += 1
                    stats['enriched_neos'] += 1
                    processed_in_session[designation] = existing_neo
                else:
                    # Create new NEO record
                    new_neo = self._create_new_neo(result, polling_session_id)
                    stats['new_neos'] += 1
                    processed_in_session[designation] = new_neo
            
            self.db.commit()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to enrich NEO data: {e}")
            self.db.rollback()
            return {'error': str(e)}
    
    def _create_new_neo(self, result: Dict[str, Any], polling_session_id: str) -> EnrichedNEO:
        """Create a new enriched NEO record."""
        designation = result['designation']
        data_source = result.get('data_source', 'Unknown')
        
        # Initialize source-specific data
        source_data = {
            'nasa_cad_data': None,
            'nasa_sbdb_data': None, 
            'mpc_data': None,
            'neodys_data': None
        }
        
        source_updates = {
            'nasa_cad_last_update': None,
            'nasa_sbdb_last_update': None,
            'mpc_last_update': None,
            'neodys_last_update': None
        }
        
        # Set data for the specific source
        now = datetime.utcnow()
        if data_source == 'NASA_CAD':
            source_data['nasa_cad_data'] = result
            source_updates['nasa_cad_last_update'] = now
        elif data_source == 'NASA_SBDB':
            source_data['nasa_sbdb_data'] = result
            source_updates['nasa_sbdb_last_update'] = now
        elif data_source == 'MPC':
            source_data['mpc_data'] = result
            source_updates['mpc_last_update'] = now
        elif data_source == 'NEODyS':
            source_data['neodys_data'] = result
            source_updates['neodys_last_update'] = now
        
        # Calculate data completeness
        completeness = self._calculate_completeness(result)
        
        new_neo = EnrichedNEO(
            designation=designation,
            first_discovered=now,
            last_updated=now,
            data_sources=[data_source],
            data_quality_score=result.get('data_quality_score', 0.5),
            orbital_elements=result.get('orbital_elements', {}),
            orbital_elements_source=data_source,
            orbital_elements_quality=self._assess_orbital_quality(result.get('orbital_elements', {})),
            physical_properties=result.get('physical_properties', {}),
            close_approaches=result.get('close_approaches', []),
            discovery_data=result.get('discovery_data', {}),
            artificial_probability=result.get('artificial_probability', 0.0),
            risk_factors=result.get('risk_factors', []),
            has_orbital_elements=bool(result.get('orbital_elements')),
            has_physical_properties=bool(result.get('physical_properties')),
            has_close_approaches=bool(result.get('close_approaches')),
            has_discovery_data=bool(result.get('discovery_data')),
            completeness_score=completeness,
            polling_sessions=[polling_session_id],
            total_detections=1,
            **source_data,
            **source_updates
        )
        
        self.db.add(new_neo)
        return new_neo
    
    def _update_existing_neo(self, existing_neo: EnrichedNEO, result: Dict[str, Any], polling_session_id: str):
        """Update existing NEO with new data from polling."""
        data_source = result.get('data_source', 'Unknown')
        now = datetime.utcnow()
        
        # Update last updated timestamp
        existing_neo.last_updated = now
        
        # Add data source if not already present
        if data_source not in existing_neo.data_sources:
            existing_neo.data_sources.append(data_source)
        
        # Update source-specific data
        if data_source == 'NASA_CAD':
            existing_neo.nasa_cad_data = result
            existing_neo.nasa_cad_last_update = now
        elif data_source == 'NASA_SBDB':
            existing_neo.nasa_sbdb_data = result
            existing_neo.nasa_sbdb_last_update = now
        elif data_source == 'MPC':
            existing_neo.mpc_data = result
            existing_neo.mpc_last_update = now
        elif data_source == 'NEODyS':
            existing_neo.neodys_data = result
            existing_neo.neodys_last_update = now
        
        # Update consolidated data with best available
        self._update_consolidated_data(existing_neo, result, data_source)
        
        # Update completeness tracking
        if result.get('orbital_elements'):
            existing_neo.has_orbital_elements = True
        if result.get('physical_properties'):
            existing_neo.has_physical_properties = True
        if result.get('close_approaches'):
            existing_neo.has_close_approaches = True
        if result.get('discovery_data'):
            existing_neo.has_discovery_data = True
        
        # Recalculate completeness score
        existing_neo.completeness_score = self._calculate_completeness_from_existing(existing_neo)
        
        # Update artificial analysis if present
        if result.get('artificial_probability', 0) > existing_neo.artificial_probability:
            existing_neo.artificial_probability = result['artificial_probability']
            existing_neo.artificial_analysis_date = now
        
        # Add risk factors
        if result.get('risk_factors'):
            existing_risk_factors = existing_neo.risk_factors or []
            for factor in result['risk_factors']:
                if factor not in existing_risk_factors:
                    existing_risk_factors.append(factor)
            existing_neo.risk_factors = existing_risk_factors
        
        # Update polling session tracking
        polling_sessions = existing_neo.polling_sessions or []
        if polling_session_id not in polling_sessions:
            polling_sessions.append(polling_session_id)
            existing_neo.polling_sessions = polling_sessions
        
        existing_neo.total_detections += 1
    
    def _update_consolidated_data(self, neo: EnrichedNEO, result: Dict[str, Any], source: str):
        """Update consolidated orbital elements with best available data."""
        result_orbitals = result.get('orbital_elements', {})
        if not result_orbitals:
            return
        
        # Assess quality of new orbital data
        new_quality = self._assess_orbital_quality(result_orbitals)
        
        # Update if this source has better quality data
        if new_quality > neo.orbital_elements_quality:
            neo.orbital_elements = result_orbitals
            neo.orbital_elements_source = source
            neo.orbital_elements_quality = new_quality
    
    def _assess_orbital_quality(self, orbital_elements: Dict[str, Any]) -> float:
        """Assess the quality of orbital elements data."""
        if not orbital_elements:
            return 0.0
        
        # Quality based on completeness and precision
        required_elements = ['e', 'i', 'a', 'q']  # eccentricity, inclination, semi-major axis, perihelion
        quality = 0.0
        
        for element in required_elements:
            if element in orbital_elements:
                quality += 0.25  # Each required element adds 25%
        
        # Bonus for additional elements
        bonus_elements = ['ma', 'om', 'w', 'epoch']  # mean anomaly, longitude of ascending node, argument of perihelion, epoch
        for element in bonus_elements:
            if element in orbital_elements:
                quality += 0.05  # Each bonus element adds 5%
        
        return min(quality, 1.0)
    
    def _calculate_completeness(self, result: Dict[str, Any]) -> float:
        """Calculate data completeness score for a new NEO."""
        score = 0.0
        
        # Orbital elements (40% of score)
        if result.get('orbital_elements'):
            score += 0.4
        
        # Physical properties (20% of score)
        if result.get('physical_properties'):
            score += 0.2
        
        # Close approaches (20% of score)
        if result.get('close_approaches'):
            score += 0.2
        
        # Discovery data (20% of score)
        if result.get('discovery_data'):
            score += 0.2
        
        return score
    
    def _calculate_completeness_from_existing(self, neo: EnrichedNEO) -> float:
        """Calculate completeness score from existing NEO record."""
        score = 0.0
        
        if neo.has_orbital_elements:
            score += 0.4
        if neo.has_physical_properties:
            score += 0.2
        if neo.has_close_approaches:
            score += 0.2
        if neo.has_discovery_data:
            score += 0.2
        
        return score
    
    def get_enriched_neo(self, designation: str) -> Optional[Dict[str, Any]]:
        """Get enriched NEO data by designation."""
        if not HAS_SQLALCHEMY:
            return None
        
        try:
            neo = self.db.query(EnrichedNEO).filter(
                EnrichedNEO.designation == designation
            ).first()
            
            if not neo:
                return None
            
            return {
                'designation': neo.designation,
                'first_discovered': neo.first_discovered,
                'last_updated': neo.last_updated,
                'data_sources': neo.data_sources,
                'data_quality_score': neo.data_quality_score,
                'orbital_elements': neo.orbital_elements,
                'orbital_elements_source': neo.orbital_elements_source,
                'physical_properties': neo.physical_properties,
                'close_approaches': neo.close_approaches,
                'discovery_data': neo.discovery_data,
                'artificial_probability': neo.artificial_probability,
                'risk_factors': neo.risk_factors,
                'completeness_score': neo.completeness_score,
                'total_detections': neo.total_detections,
                'source_data': {
                    'nasa_cad': neo.nasa_cad_data,
                    'nasa_sbdb': neo.nasa_sbdb_data,
                    'mpc': neo.mpc_data,
                    'neodys': neo.neodys_data
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get enriched NEO {designation}: {e}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the enriched NEO database."""
        if not HAS_SQLALCHEMY:
            return {'error': 'Database not available'}
        
        try:
            total_neos = self.db.query(EnrichedNEO).count()
            
            # Count by completeness
            high_completeness = self.db.query(EnrichedNEO).filter(
                EnrichedNEO.completeness_score >= 0.8
            ).count()
            
            medium_completeness = self.db.query(EnrichedNEO).filter(
                EnrichedNEO.completeness_score >= 0.5,
                EnrichedNEO.completeness_score < 0.8
            ).count()
            
            # Count by artificial probability
            high_artificial = self.db.query(EnrichedNEO).filter(
                EnrichedNEO.artificial_probability >= 0.8
            ).count()
            
            suspicious = self.db.query(EnrichedNEO).filter(
                EnrichedNEO.artificial_probability >= 0.5,
                EnrichedNEO.artificial_probability < 0.8
            ).count()
            
            # Count by data sources
            multi_source = self.db.query(EnrichedNEO).filter(
                EnrichedNEO.total_detections > 1
            ).count()
            
            return {
                'total_neos': total_neos,
                'high_completeness': high_completeness,
                'medium_completeness': medium_completeness,
                'high_artificial_probability': high_artificial,
                'suspicious_objects': suspicious,
                'multi_source_detections': multi_source,
                'database_coverage': {
                    'complete': high_completeness / total_neos if total_neos > 0 else 0,
                    'partial': medium_completeness / total_neos if total_neos > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}

# Global database manager
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_database() -> Session:
    """FastAPI dependency to get database session."""
    if not HAS_SQLALCHEMY:
        raise RuntimeError("Database not available")
    
    db = db_manager.get_db()
    try:
        yield db
    finally:
        db.close()

# Utility functions
def init_database() -> bool:
    """Initialize the database."""
    return db_manager.init_database()

def get_database_status() -> Dict[str, Any]:
    """Get database connection status."""
    if not HAS_SQLALCHEMY:
        return {
            'available': False,
            'error': 'SQLAlchemy not installed'
        }
    
    try:
        # Test database connection
        db = db_manager.get_db()
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            'available': True,
            'engine': str(db_manager.engine.url),
            'tables': len(Base.metadata.tables) if Base else 0
        }
        
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }

def cleanup_old_data(days: int = 30) -> Dict[str, int]:
    """Clean up old data from database."""
    if not HAS_SQLALCHEMY:
        return {'error': 'Database not available'}
    
    try:
        db = db_manager.get_db()
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Clean up old metrics
        metrics_deleted = db.query(SystemMetrics).filter(
            SystemMetrics.timestamp < cutoff_date
        ).delete()
        
        # Clean up old API usage
        usage_deleted = db.query(APIUsage).filter(
            APIUsage.timestamp < cutoff_date
        ).delete()
        
        db.commit()
        db.close()
        
        return {
            'metrics_deleted': metrics_deleted,
            'usage_deleted': usage_deleted,
            'cleanup_date': cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        return {'error': str(e)}

# Initialize database on import
if HAS_SQLALCHEMY:
    init_result = init_database()
    logger.info(f"Database initialization: {'success' if init_result else 'failed'}")
else:
    logger.warning("Database features disabled - SQLAlchemy not available")