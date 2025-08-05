"""
Real-time prediction system for aNEOS ML models.

This module provides real-time anomaly detection capabilities using trained
machine learning models, with caching, monitoring, and alerting features.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .features import FeatureEngineer, FeatureVector
from .models import AnomalyDetectionModel, ModelEnsemble, ModelConfig, PredictionResult, create_model
from .training import TrainingPipeline, TrainingSession
from ..data.models import NEOData
from ..analysis.pipeline import AnalysisPipeline
from ..analysis.indicators.base import IndicatorResult
from ..data.cache import get_cache_manager

logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """Configuration for real-time prediction."""
    
    # Model configuration
    model_path: str = "models"
    use_ensemble: bool = True
    fallback_to_indicators: bool = True
    
    # Performance configuration
    prediction_timeout: float = 30.0  # seconds
    cache_predictions: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Quality configuration
    min_confidence_threshold: float = 0.1
    feature_quality_threshold: float = 0.5
    
    # Monitoring configuration
    enable_monitoring: bool = True
    alert_threshold: float = 0.8
    batch_size: int = 10

@dataclass
class Alert:
    """Alert for high-anomaly NEO detection."""
    designation: str
    anomaly_score: float
    ml_probability: float
    confidence: float
    alert_level: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    contributing_factors: List[str] = field(default_factory=list)
    model_predictions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'designation': self.designation,
            'anomaly_score': self.anomaly_score,
            'ml_probability': self.ml_probability,
            'confidence': self.confidence,
            'alert_level': self.alert_level,
            'timestamp': self.timestamp.isoformat(),
            'contributing_factors': self.contributing_factors,
            'model_predictions': self.model_predictions
        }

class ModelManager:
    """Manages loading and caching of trained models."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.loaded_models: Dict[str, AnomalyDetectionModel] = {}
        self.loaded_ensembles: Dict[str, ModelEnsemble] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
    def load_model(self, model_id: str) -> Optional[AnomalyDetectionModel]:
        """Load a specific model by ID."""
        with self._lock:
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]
            
            # Try to find model file
            model_files = list(self.model_path.glob(f"*{model_id}*.pkl"))
            
            if not model_files:
                logger.warning(f"Model file not found for ID: {model_id}")
                return None
            
            model_file = model_files[0]
            
            try:
                # Determine model type from filename or metadata
                if "isolation_forest" in model_file.name:
                    model = create_model("isolation_forest")
                elif "one_class_svm" in model_file.name:
                    model = create_model("one_class_svm")
                elif "autoencoder" in model_file.name:
                    model = create_model("autoencoder")
                else:
                    logger.warning(f"Unknown model type for file: {model_file}")
                    return None
                
                model.load_model(str(model_file))
                self.loaded_models[model_id] = model
                
                logger.info(f"Loaded model: {model_id}")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return None
    
    def load_ensemble(self, ensemble_id: str) -> Optional[ModelEnsemble]:
        """Load an ensemble by ID."""
        with self._lock:
            if ensemble_id in self.loaded_ensembles:
                return self.loaded_ensembles[ensemble_id]
            
            # Try to find ensemble directory
            ensemble_dirs = list(self.model_path.glob(f"*{ensemble_id}*"))
            ensemble_dirs = [d for d in ensemble_dirs if d.is_dir()]
            
            if not ensemble_dirs:
                logger.warning(f"Ensemble directory not found for ID: {ensemble_id}")
                return None
            
            ensemble_dir = ensemble_dirs[0]
            
            try:
                # Load ensemble metadata to determine model configs
                metadata_file = ensemble_dir / "ensemble_metadata.pkl"
                if not metadata_file.exists():
                    logger.warning(f"Ensemble metadata not found: {metadata_file}")
                    return None
                
                import pickle
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Create model configs based on metadata
                model_configs = []
                for model_type in metadata['model_types']:
                    config = ModelConfig(model_type=model_type)
                    model_configs.append(config)
                
                # Create and load ensemble
                ensemble = ModelEnsemble([], metadata['weights'])
                ensemble.load_ensemble(str(ensemble_dir), model_configs)
                
                self.loaded_ensembles[ensemble_id] = ensemble
                
                logger.info(f"Loaded ensemble: {ensemble_id} with {len(ensemble.models)} models")
                return ensemble
                
            except Exception as e:
                logger.error(f"Failed to load ensemble {ensemble_id}: {e}")
                return None
    
    def get_latest_model(self) -> Optional[AnomalyDetectionModel]:
        """Get the most recently trained model."""
        model_files = list(self.model_path.glob("*.pkl"))
        if not model_files:
            return None
        
        # Sort by modification time
        latest_file = max(model_files, key=lambda f: f.stat().st_mtime)
        
        # Extract model ID from filename
        model_id = latest_file.stem
        
        return self.load_model(model_id)
    
    def get_latest_ensemble(self) -> Optional[ModelEnsemble]:
        """Get the most recently trained ensemble."""
        ensemble_dirs = [d for d in self.model_path.iterdir() if d.is_dir() and "ensemble" in d.name]
        
        if not ensemble_dirs:
            return None
        
        # Sort by modification time
        latest_dir = max(ensemble_dirs, key=lambda d: d.stat().st_mtime)
        
        # Extract ensemble ID from directory name
        ensemble_id = latest_dir.name
        
        return self.load_ensemble(ensemble_id)
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models and ensembles."""
        models = []
        
        # Individual models
        for model_file in self.model_path.glob("*.pkl"):
            if "ensemble" not in model_file.name:
                models.append({
                    'id': model_file.stem,
                    'type': 'model',
                    'file_path': str(model_file),
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime)
                })
        
        # Ensembles
        for ensemble_dir in self.model_path.iterdir():
            if ensemble_dir.is_dir() and "ensemble" in ensemble_dir.name:
                models.append({
                    'id': ensemble_dir.name,
                    'type': 'ensemble',
                    'file_path': str(ensemble_dir),
                    'modified': datetime.fromtimestamp(ensemble_dir.stat().st_mtime)
                })
        
        return sorted(models, key=lambda x: x['modified'], reverse=True)

class RealTimePredictor:
    """Real-time anomaly prediction system."""
    
    def __init__(self, analysis_pipeline: AnalysisPipeline, config: PredictionConfig):
        self.analysis_pipeline = analysis_pipeline
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager(config.model_path)
        self.cache_manager = get_cache_manager()
        
        # Prediction statistics
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'cache_hits': 0,
            'average_prediction_time': 0.0,
            'high_anomaly_count': 0
        }
        
        # Alert system
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Load models
        self._initialize_models()
        
        logger.info("RealTimePredictor initialized")
    
    def _initialize_models(self) -> None:
        """Initialize models for prediction."""
        try:
            if self.config.use_ensemble:
                self.primary_model = self.model_manager.get_latest_ensemble()
                if self.primary_model:
                    logger.info(f"Using ensemble model: {self.primary_model.ensemble_id}")
                else:
                    logger.warning("No ensemble found, falling back to single model")
                    self.primary_model = self.model_manager.get_latest_model()
            else:
                self.primary_model = self.model_manager.get_latest_model()
            
            if self.primary_model:
                logger.info("Primary model loaded successfully")
            else:
                logger.warning("No trained models found - predictions will use indicators only")
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.primary_model = None
    
    async def predict_anomaly(self, designation: str, neo_data: Optional[NEOData] = None) -> PredictionResult:
        """Predict anomaly for a single NEO."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.config.cache_predictions:
                cache_key = f"ml_prediction_{designation}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    self.prediction_stats['cache_hits'] += 1
                    logger.debug(f"Using cached prediction for {designation}")
                    
                    # Convert back to PredictionResult
                    return PredictionResult(
                        designation=cached_result['designation'],
                        anomaly_score=cached_result['anomaly_score'],
                        anomaly_probability=cached_result['anomaly_probability'],
                        is_anomaly=cached_result['is_anomaly'],
                        confidence=cached_result['confidence'],
                        model_id=cached_result['model_id'],
                        feature_contributions=cached_result.get('feature_contributions'),
                        predicted_at=datetime.fromisoformat(cached_result['predicted_at'])
                    )
            
            # Fetch NEO data if not provided
            if neo_data is None:
                neo_data = await self.analysis_pipeline._fetch_neo_data(designation)
                
                if neo_data is None:
                    raise ValueError(f"Could not fetch data for {designation}")
            
            # Run analysis pipeline to get indicator results
            pipeline_result = await self.analysis_pipeline.analyze_neo(designation, neo_data)
            
            if pipeline_result.errors:
                logger.warning(f"Analysis pipeline had errors for {designation}: {pipeline_result.errors}")
            
            # Extract features
            indicator_results = pipeline_result.anomaly_score.indicator_scores
            feature_vector = self.feature_engineer.extract_features(neo_data, indicator_results)
            
            # Assess feature quality
            feature_quality = self._assess_feature_quality(feature_vector)
            
            if feature_quality < self.config.feature_quality_threshold:
                logger.warning(f"Low feature quality for {designation}: {feature_quality:.2f}")
                
                # Fall back to indicator-only prediction if configured
                if self.config.fallback_to_indicators:
                    return self._create_indicator_prediction(designation, pipeline_result.anomaly_score)
            
            # Make ML prediction
            if self.primary_model:
                ml_score, ml_probability = self._predict_with_model(feature_vector)
                
                # Combine with traditional analysis
                combined_score = self._combine_predictions(
                    ml_score, 
                    pipeline_result.anomaly_score.overall_score,
                    feature_quality
                )
                
                confidence = min(feature_quality, 0.9)  # Cap confidence based on feature quality
                is_anomaly = combined_score > 0.5
                
                # Get feature contributions if available
                feature_contributions = self._calculate_feature_contributions(feature_vector)
                
                result = PredictionResult(
                    designation=designation,
                    anomaly_score=combined_score,
                    anomaly_probability=ml_probability,
                    is_anomaly=is_anomaly,
                    confidence=confidence,
                    model_id=getattr(self.primary_model, 'model_id', 'unknown'),
                    feature_contributions=feature_contributions
                )
            else:
                # No ML model available, use indicator-only prediction
                result = self._create_indicator_prediction(designation, pipeline_result.anomaly_score)
            
            # Cache result
            if self.config.cache_predictions:
                cache_key = f"ml_prediction_{designation}"
                self.cache_manager.set(cache_key, result.__dict__, ttl=self.config.cache_ttl)
            
            # Update statistics
            self.prediction_stats['successful_predictions'] += 1
            
            # Check for alerts
            if self.config.enable_monitoring:
                self._check_for_alerts(result, pipeline_result.anomaly_score)
            
            prediction_time = time.time() - start_time
            self._update_prediction_stats(prediction_time)
            
            logger.debug(f"Prediction completed for {designation}: {result.anomaly_score:.3f} (time: {prediction_time:.2f}s)")
            
            return result
            
        except Exception as e:
            self.prediction_stats['failed_predictions'] += 1
            logger.error(f"Prediction failed for {designation}: {e}")
            
            # Return default prediction
            return PredictionResult(
                designation=designation,
                anomaly_score=0.0,
                anomaly_probability=0.0,
                is_anomaly=False,
                confidence=0.0,
                model_id="error"
            )
        
        finally:
            self.prediction_stats['total_predictions'] += 1
    
    async def predict_batch(self, designations: List[str], 
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> List[PredictionResult]:
        """Predict anomalies for a batch of NEOs."""
        logger.info(f"Starting batch prediction for {len(designations)} NEOs")
        
        results = []
        completed = 0
        
        # Process in smaller batches to manage memory
        batch_size = self.config.batch_size
        
        for i in range(0, len(designations), batch_size):
            batch = designations[i:i + batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as executor:
                futures = [
                    executor.submit(asyncio.run, self.predict_anomaly(designation))
                    for designation in batch
                ]
                
                for future in futures:
                    try:
                        result = future.result(timeout=self.config.prediction_timeout)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch prediction failed for one item: {e}")
                        # Add error result
                        results.append(PredictionResult(
                            designation="unknown",
                            anomaly_score=0.0,
                            anomaly_probability=0.0,
                            is_anomaly=False,
                            confidence=0.0,
                            model_id="batch_error"
                        ))
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(designations))
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
    
    def _predict_with_model(self, feature_vector: FeatureVector) -> Tuple[float, float]:
        """Make prediction with the primary model."""
        try:
            # Prepare features
            features = feature_vector.features.reshape(1, -1)
            
            # Get predictions
            if isinstance(self.primary_model, ModelEnsemble):
                anomaly_score = self.primary_model.predict(features)[0]
                anomaly_probability = self.primary_model.predict_proba(features)[0]
            else:
                anomaly_score = self.primary_model.predict(features)[0]
                anomaly_probability = self.primary_model.predict_proba(features)[0]
            
            return float(anomaly_score), float(anomaly_probability)
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0.0, 0.0
    
    def _combine_predictions(self, ml_score: float, indicator_score: float, feature_quality: float) -> float:
        """Combine ML and indicator-based predictions."""
        # Weight ML prediction by feature quality
        ml_weight = feature_quality
        indicator_weight = 1.0 - ml_weight
        
        # Weighted average
        combined_score = ml_score * ml_weight + indicator_score * indicator_weight
        
        # Apply boost if both methods agree
        if ml_score > 0.5 and indicator_score > 0.5:
            agreement_boost = min(ml_score * indicator_score * 0.2, 0.2)
            combined_score += agreement_boost
        
        return min(combined_score, 1.0)
    
    def _create_indicator_prediction(self, designation: str, anomaly_score) -> PredictionResult:
        """Create prediction result using only indicator scores."""
        return PredictionResult(
            designation=designation,
            anomaly_score=anomaly_score.overall_score,
            anomaly_probability=anomaly_score.overall_score,  # Use score as probability
            is_anomaly=anomaly_score.classification != 'natural',
            confidence=anomaly_score.confidence,
            model_id="indicators_only"
        )
    
    def _assess_feature_quality(self, feature_vector: FeatureVector) -> float:
        """Assess quality of extracted features."""
        if 'data_quality' not in feature_vector.metadata:
            return 0.5  # Default quality
        
        quality_info = feature_vector.metadata['data_quality']
        
        # Calculate composite quality score
        completeness = quality_info.get('completeness', 0.0)
        validity = quality_info.get('validity', 0.0)
        
        # Check for too many zero features
        zero_fraction = np.mean(feature_vector.features == 0.0)
        non_zero_quality = 1.0 - min(zero_fraction, 0.8)  # Penalize if > 80% zeros
        
        # Combine quality metrics
        overall_quality = (completeness + validity + non_zero_quality) / 3.0
        
        return overall_quality
    
    def _calculate_feature_contributions(self, feature_vector: FeatureVector) -> Optional[Dict[str, float]]:
        """Calculate feature contributions to prediction (simplified)."""
        try:
            # This is a simplified version - in practice, you'd use SHAP or similar
            # For now, just return the top features by absolute value
            
            feature_values = np.abs(feature_vector.features)
            top_indices = np.argsort(feature_values)[-10:]  # Top 10 features
            
            contributions = {}
            for idx in top_indices:
                if idx < len(feature_vector.feature_names):
                    feature_name = feature_vector.feature_names[idx]
                    contributions[feature_name] = float(feature_values[idx])
            
            return contributions
            
        except Exception as e:
            logger.warning(f"Feature contribution calculation failed: {e}")
            return None
    
    def _check_for_alerts(self, ml_result: PredictionResult, analysis_result) -> None:
        """Check if prediction warrants an alert."""
        try:
            # Determine alert level
            alert_level = None
            
            if ml_result.anomaly_probability > 0.9:
                alert_level = "critical"
            elif ml_result.anomaly_probability > 0.8:
                alert_level = "high"
            elif ml_result.anomaly_probability > 0.6:
                alert_level = "medium"
            elif ml_result.anomaly_probability > 0.4:
                alert_level = "low"
            
            if alert_level:
                # Create alert
                alert = Alert(
                    designation=ml_result.designation,
                    anomaly_score=ml_result.anomaly_score,
                    ml_probability=ml_result.anomaly_probability,
                    confidence=ml_result.confidence,
                    alert_level=alert_level,
                    timestamp=datetime.now(),
                    contributing_factors=analysis_result.risk_factors,
                    model_predictions={ml_result.model_id: ml_result.anomaly_probability}
                )
                
                self.alerts.append(alert)
                
                # Trigger alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
                
                logger.info(f"Alert generated: {alert_level} level for {ml_result.designation}")
                
                if alert_level in ["high", "critical"]:
                    self.prediction_stats['high_anomaly_count'] += 1
                    
        except Exception as e:
            logger.error(f"Alert check failed: {e}")
    
    def _update_prediction_stats(self, prediction_time: float) -> None:
        """Update prediction statistics."""
        # Update average prediction time
        total = self.prediction_stats['total_predictions']
        current_avg = self.prediction_stats['average_prediction_time']
        
        self.prediction_stats['average_prediction_time'] = (
            (current_avg * (total - 1) + prediction_time) / total
        )
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from recent time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
        return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        stats = self.prediction_stats.copy()
        
        # Add derived statistics
        if stats['total_predictions'] > 0:
            stats['success_rate'] = stats['successful_predictions'] / stats['total_predictions']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_predictions']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        stats['model_info'] = {
            'primary_model_id': getattr(self.primary_model, 'model_id', None) if self.primary_model else None,
            'model_type': type(self.primary_model).__name__ if self.primary_model else None,
            'available_models': len(self.model_manager.list_available_models())
        }
        
        stats['recent_alerts'] = len(self.get_recent_alerts(24))
        
        return stats
    
    def clear_prediction_cache(self) -> None:
        """Clear prediction cache."""
        # This would clear ML prediction cache keys
        # Implementation depends on cache manager capabilities
        logger.info("Prediction cache clearing requested")
    
    def reload_models(self) -> None:
        """Reload models from disk."""
        logger.info("Reloading models...")
        
        # Clear loaded models
        self.model_manager.loaded_models.clear()
        self.model_manager.loaded_ensembles.clear()
        
        # Reinitialize
        self._initialize_models()
        
        logger.info("Models reloaded successfully")