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