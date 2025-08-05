"""
Training pipeline for aNEOS machine learning models.

This module provides comprehensive training capabilities including data preparation,
model training, validation, hyperparameter optimization, and model management.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from sklearn.model_selection import train_test_split, ParameterGrid
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .features import FeatureEngineer, FeatureVector
from .models import (
    AnomalyDetectionModel, ModelEnsemble, ModelConfig, TrainingResult,
    create_model, create_default_ensemble
)
from ..data.models import NEOData
from ..analysis.pipeline import AnalysisPipeline
from ..analysis.scoring import AnomalyScore

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Data configuration
    training_data_path: Optional[str] = None
    validation_split: float = 0.2
    test_split: float = 0.1
    min_samples: int = 100
    
    # Model configuration
    model_types: List[str] = field(default_factory=lambda: ["isolation_forest", "one_class_svm"])
    use_ensemble: bool = True
    ensemble_weights: Optional[List[float]] = None
    
    # Training configuration
    hyperparameter_optimization: bool = True
    cross_validation_folds: int = 5
    random_state: int = 42
    
    # Feature configuration
    feature_selection: bool = True
    feature_selection_threshold: float = 0.01
    normalize_features: bool = True
    
    # Output configuration
    model_output_dir: str = "models"
    save_training_data: bool = True
    save_feature_importance: bool = True
    
    # Performance configuration
    max_workers: int = 4
    timeout_minutes: int = 60

@dataclass
class TrainingSession:
    """Represents a training session with all metadata."""
    session_id: str
    config: TrainingConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    models_trained: List[str] = field(default_factory=list)
    training_results: List[TrainingResult] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'config': self.config.__dict__,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'models_trained': self.models_trained,
            'training_results': [result.__dict__ for result in self.training_results],
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'errors': self.errors
        }

class TrainingDataManager:
    """Manages training data collection and preparation."""
    
    def __init__(self, analysis_pipeline: AnalysisPipeline):
        self.analysis_pipeline = analysis_pipeline
        self.feature_engineer = FeatureEngineer()
        
    async def collect_training_data(self, neo_designations: List[str],
                                  progress_callback: Optional[Callable[[int, int], None]] = None) -> List[FeatureVector]:
        """Collect training data from NEO designations."""
        logger.info(f"Collecting training data for {len(neo_designations)} NEOs")
        
        feature_vectors = []
        completed = 0
        
        # Process NEOs in batches to manage memory
        batch_size = 50
        for i in range(0, len(neo_designations), batch_size):
            batch = neo_designations[i:i + batch_size]
            
            # Analyze batch
            analysis_results = await self.analysis_pipeline.analyze_batch(batch)
            
            # Extract features for each result
            for result in analysis_results:
                if not result.errors and result.anomaly_score:
                    try:
                        # Get NEO data (assuming it's cached from analysis)
                        neo_data = await self.analysis_pipeline._fetch_neo_data(result.designation)
                        
                        if neo_data:
                            # Extract features including indicator results
                            indicator_results = {
                                name: result for name, result in result.anomaly_score.indicator_scores.items()
                            }
                            
                            feature_vector = self.feature_engineer.extract_features(
                                neo_data, indicator_results
                            )
                            
                            # Add anomaly score as metadata
                            feature_vector.metadata['anomaly_score'] = result.anomaly_score.overall_score
                            feature_vector.metadata['classification'] = result.anomaly_score.classification
                            
                            feature_vectors.append(feature_vector)
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract features for {result.designation}: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(neo_designations))
        
        logger.info(f"Collected {len(feature_vectors)} feature vectors")
        return feature_vectors
    
    def prepare_training_data(self, feature_vectors: List[FeatureVector],
                            config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels for training."""
        if not feature_vectors:
            raise ValueError("No feature vectors provided")
        
        # Create feature matrix
        feature_matrix, feature_names, designations = self.feature_engineer.create_feature_matrix(feature_vectors)
        
        # Create labels based on anomaly scores (for validation purposes)
        labels = []
        for fv in feature_vectors:
            if fv.designation in designations:
                classification = fv.metadata.get('classification', 'natural')
                # Convert classification to binary (1 = anomalous, 0 = natural)
                label = 1 if classification in ['suspicious', 'highly_suspicious', 'artificial'] else 0
                labels.append(label)
        
        labels = np.array(labels)
        
        # Feature selection if enabled
        if config.feature_selection and HAS_SKLEARN:
            feature_matrix, feature_names = self._select_features(
                feature_matrix, labels, feature_names, config.feature_selection_threshold
            )
        
        logger.info(f"Prepared training data: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        return feature_matrix, labels, feature_names
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                        threshold: float) -> Tuple[np.ndarray, List[str]]:
        """Select features based on importance."""
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.ensemble import RandomForestClassifier
        
        try:
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = rf.feature_importances_
            
            # Select features above threshold
            selected_indices = importance > threshold
            selected_features = X[:, selected_indices]
            selected_names = [name for i, name in enumerate(feature_names) if selected_indices[i]]
            
            logger.info(f"Feature selection: {len(selected_names)}/{len(feature_names)} features selected")
            return selected_features, selected_names
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using all features")
            return X, feature_names
    
    def save_training_data(self, feature_vectors: List[FeatureVector], filepath: str) -> None:
        """Save training data to file."""
        data = {
            'feature_vectors': [fv.to_dict() for fv in feature_vectors],
            'saved_at': datetime.now().isoformat(),
            'count': len(feature_vectors)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Training data saved to {filepath}")
    
    def load_training_data(self, filepath: str) -> List[FeatureVector]:
        """Load training data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        feature_vectors = []
        for fv_data in data['feature_vectors']:
            fv = FeatureVector(
                designation=fv_data['designation'],
                features=np.array(fv_data['features']),
                feature_names=fv_data['feature_names'],
                metadata=fv_data['metadata'],
                created_at=datetime.fromisoformat(fv_data['created_at'])
            )
            feature_vectors.append(fv)
        
        logger.info(f"Loaded {len(feature_vectors)} feature vectors from {filepath}")
        return feature_vectors

class HyperparameterOptimizer:
    """Optimizes model hyperparameters."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Define parameter grids for different models
        self.parameter_grids = {
            'isolation_forest': {
                'n_estimators': [100, 200, 300],
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'max_features': [0.5, 0.7, 1.0]
            },
            'one_class_svm': {
                'kernel': ['rbf', 'sigmoid'],
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'autoencoder': {
                'hidden_sizes': [[64, 32, 64], [128, 64, 32, 64, 128], [256, 128, 64, 128, 256]],
                'learning_rate': [0.001, 0.01, 0.1],
                'epochs': [50, 100, 150],
                'dropout_rate': [0.1, 0.2, 0.3]
            }
        }
    
    def optimize_model(self, model_type: str, X_train: np.ndarray, X_val: np.ndarray,
                      y_val: np.ndarray) -> Tuple[ModelConfig, float]:
        """Optimize hyperparameters for a specific model type."""
        logger.info(f"Optimizing hyperparameters for {model_type}")
        
        if model_type not in self.parameter_grids:
            logger.warning(f"No parameter grid defined for {model_type}")
            return ModelConfig(model_type=model_type), 0.0
        
        best_config = None
        best_score = -np.inf
        
        # Grid search
        param_grid = self.parameter_grids[model_type]
        for params in ParameterGrid(param_grid):
            try:
                # Create and train model with these parameters
                config = ModelConfig(model_type=model_type, parameters=params)
                model = create_model(model_type, config)
                
                # Train model
                model.fit(X_train)
                
                # Evaluate on validation set
                if model.is_trained:
                    val_scores = model.predict(X_val)
                    
                    # Calculate AUC if we have labels
                    if len(np.unique(y_val)) > 1:
                        score = roc_auc_score(y_val, val_scores)
                    else:
                        # Use mean validation score for unsupervised case
                        score = np.mean(val_scores)
                    
                    if score > best_score:
                        best_score = score
                        best_config = config
                
            except Exception as e:
                logger.warning(f"Parameter combination failed for {model_type}: {e}")
                continue
        
        if best_config is None:
            logger.warning(f"No valid configuration found for {model_type}, using default")
            best_config = ModelConfig(model_type=model_type)
            best_score = 0.0
        
        logger.info(f"Best {model_type} configuration found with score: {best_score:.4f}")
        return best_config, best_score

class TrainingPipeline:
    """Main training pipeline coordinator."""
    
    def __init__(self, analysis_pipeline: AnalysisPipeline, config: TrainingConfig):
        self.analysis_pipeline = analysis_pipeline
        self.config = config
        self.data_manager = TrainingDataManager(analysis_pipeline)
        self.optimizer = HyperparameterOptimizer(config) if config.hyperparameter_optimization else None
        
        # Create output directory
        Path(config.model_output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("TrainingPipeline initialized")
    
    async def train_models(self, neo_designations: List[str],
                          progress_callback: Optional[Callable[[str, float], None]] = None) -> TrainingSession:
        """Train models on provided NEO designations."""
        session = TrainingSession(
            session_id=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config,
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"Starting training session: {session.session_id}")
            
            if progress_callback:
                progress_callback("Collecting training data", 0.1)
            
            # Collect and prepare training data
            feature_vectors = await self.data_manager.collect_training_data(
                neo_designations,
                progress_callback=lambda c, t: progress_callback(f"Collecting data ({c}/{t})", 0.1 + 0.3 * c / t) if progress_callback else None
            )
            
            if len(feature_vectors) < self.config.min_samples:
                raise ValueError(f"Insufficient training data: {len(feature_vectors)} < {self.config.min_samples}")
            
            # Prepare feature matrix
            X, y, feature_names = self.data_manager.prepare_training_data(feature_vectors, self.config)
            session.feature_names = feature_names
            
            # Split data
            if progress_callback:
                progress_callback("Splitting data", 0.4)
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.config.validation_split + self.config.test_split,
                random_state=self.config.random_state, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            if self.config.test_split > 0:
                val_size = self.config.validation_split / (self.config.validation_split + self.config.test_split)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=1-val_size,
                    random_state=self.config.random_state, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
                )
            else:
                X_val, y_val = X_temp, y_temp
                X_test, y_test = None, None
            
            # Train individual models
            if progress_callback:
                progress_callback("Training models", 0.5)
            
            trained_models = []
            
            for i, model_type in enumerate(self.config.model_types):
                try:
                    logger.info(f"Training {model_type} model")
                    
                    # Optimize hyperparameters if enabled
                    if self.optimizer:
                        config, _ = self.optimizer.optimize_model(model_type, X_train, X_val, y_val)
                    else:
                        config = ModelConfig(model_type=model_type)
                    
                    # Create and train model
                    model = create_model(model_type, config)
                    training_result = model.fit(X_train)
                    
                    # Validate model
                    if model.is_trained:
                        val_scores = model.predict(X_val)
                        
                        # Calculate validation metrics
                        if len(np.unique(y_val)) > 1:
                            val_auc = roc_auc_score(y_val, val_scores)
                            precision, recall, _ = precision_recall_curve(y_val, val_scores)
                            val_pr_auc = auc(recall, precision)
                        else:
                            val_auc = 0.0
                            val_pr_auc = 0.0
                        
                        training_result.validation_score = val_auc
                        training_result.training_metadata.update({
                            'validation_auc': val_auc,
                            'validation_pr_auc': val_pr_auc,
                            'validation_samples': len(y_val)
                        })
                        
                        # Save model
                        model_path = Path(self.config.model_output_dir) / f"{session.session_id}_{model_type}.pkl"
                        model.save_model(str(model_path))
                        
                        trained_models.append(model)
                        session.models_trained.append(model_type)
                        session.training_results.append(training_result)
                        
                        logger.info(f"Successfully trained {model_type} (validation AUC: {val_auc:.4f})")
                    
                    if progress_callback:
                        progress_callback(f"Trained {model_type}", 0.5 + 0.3 * (i + 1) / len(self.config.model_types))
                
                except Exception as e:
                    error_msg = f"Failed to train {model_type}: {e}"
                    logger.error(error_msg)
                    session.errors.append(error_msg)
            
            # Create ensemble if requested
            if self.config.use_ensemble and len(trained_models) > 1:
                if progress_callback:
                    progress_callback("Creating ensemble", 0.8)
                
                try:
                    ensemble = ModelEnsemble(trained_models, self.config.ensemble_weights)
                    
                    # Evaluate ensemble
                    ensemble_scores = ensemble.predict(X_val)
                    if len(np.unique(y_val)) > 1:
                        ensemble_auc = roc_auc_score(y_val, ensemble_scores)
                    else:
                        ensemble_auc = 0.0
                    
                    # Save ensemble
                    ensemble_dir = Path(self.config.model_output_dir) / f"{session.session_id}_ensemble"
                    ensemble.save_ensemble(str(ensemble_dir))
                    
                    # Add ensemble result
                    ensemble_result = TrainingResult(
                        model_id=ensemble.ensemble_id,
                        model_type="ensemble",
                        training_score=np.mean([r.training_score for r in session.training_results]),
                        validation_score=ensemble_auc,
                        training_metadata={
                            'model_count': len(trained_models),
                            'ensemble_auc': ensemble_auc,
                            'component_models': [m.model_id for m in trained_models]
                        }
                    )
                    session.training_results.append(ensemble_result)
                    session.models_trained.append("ensemble")
                    
                    logger.info(f"Ensemble created with {len(trained_models)} models (AUC: {ensemble_auc:.4f})")
                
                except Exception as e:
                    error_msg = f"Failed to create ensemble: {e}"
                    logger.error(error_msg)
                    session.errors.append(error_msg)
            
            # Calculate performance metrics
            session.performance_metrics = self._calculate_performance_metrics(
                session.training_results, X_test, y_test if X_test is not None else None
            )
            
            # Save training data if requested
            if self.config.save_training_data:
                data_path = Path(self.config.model_output_dir) / f"{session.session_id}_training_data.json"
                self.data_manager.save_training_data(feature_vectors, str(data_path))
            
            # Save session metadata
            session_path = Path(self.config.model_output_dir) / f"{session.session_id}_session.json"
            with open(session_path, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            
            session.end_time = datetime.now()
            training_duration = (session.end_time - session.start_time).total_seconds()
            
            logger.info(f"Training session completed in {training_duration:.1f}s")
            logger.info(f"Successfully trained {len(session.models_trained)} models")
            
            if progress_callback:
                progress_callback("Training completed", 1.0)
            
            return session
            
        except Exception as e:
            error_msg = f"Training session failed: {e}"
            logger.error(error_msg)
            session.errors.append(error_msg)
            session.end_time = datetime.now()
            return session
    
    def _calculate_performance_metrics(self, training_results: List[TrainingResult],
                                     X_test: Optional[np.ndarray], y_test: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {
            'training_summary': {
                'total_models': len(training_results),
                'successful_models': len([r for r in training_results if r.training_score > 0]),
                'mean_training_score': np.mean([r.training_score for r in training_results]),
                'mean_validation_score': np.mean([r.validation_score for r in training_results]),
                'best_model': max(training_results, key=lambda x: x.validation_score).model_id if training_results else None
            }
        }
        
        # Test set evaluation if available
        if X_test is not None and y_test is not None:
            test_metrics = {}
            
            for result in training_results:
                if result.model_type != "ensemble":  # Skip ensemble for individual model metrics
                    try:
                        # Would need to load model to evaluate on test set
                        # This is a placeholder for test evaluation
                        test_metrics[result.model_id] = {
                            'test_auc': 0.0,  # Placeholder
                            'test_samples': len(y_test)
                        }
                    except Exception as e:
                        logger.warning(f"Test evaluation failed for {result.model_id}: {e}")
            
            metrics['test_evaluation'] = test_metrics
        
        return metrics
    
    def load_training_session(self, session_id: str) -> TrainingSession:
        """Load a previous training session."""
        session_path = Path(self.config.model_output_dir) / f"{session_id}_session.json"
        
        if not session_path.exists():
            raise FileNotFoundError(f"Training session not found: {session_id}")
        
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        
        # Reconstruct TrainingResult objects
        training_results = []
        for result_data in session_data['training_results']:
            result = TrainingResult(
                model_id=result_data['model_id'],
                model_type=result_data['model_type'],
                training_score=result_data['training_score'],
                validation_score=result_data['validation_score'],
                feature_importance=result_data.get('feature_importance'),
                training_metadata=result_data['training_metadata'],
                trained_at=datetime.fromisoformat(result_data['trained_at'])
            )
            training_results.append(result)
        
        # Reconstruct TrainingConfig
        config_data = session_data['config']
        config = TrainingConfig(**config_data)
        
        # Reconstruct TrainingSession
        session = TrainingSession(
            session_id=session_data['session_id'],
            config=config,
            start_time=datetime.fromisoformat(session_data['start_time']),
            end_time=datetime.fromisoformat(session_data['end_time']) if session_data['end_time'] else None,
            models_trained=session_data['models_trained'],
            training_results=training_results,
            feature_names=session_data['feature_names'],
            performance_metrics=session_data['performance_metrics'],
            errors=session_data['errors']
        )
        
        return session
    
    def list_training_sessions(self) -> List[str]:
        """List all available training sessions."""
        model_dir = Path(self.config.model_output_dir)
        session_files = list(model_dir.glob("*_session.json"))
        
        session_ids = []
        for session_file in session_files:
            session_id = session_file.stem.replace("_session", "")
            session_ids.append(session_id)
        
        return sorted(session_ids, reverse=True)  # Most recent first