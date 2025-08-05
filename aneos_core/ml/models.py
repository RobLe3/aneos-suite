"""
Machine learning models for aNEOS anomaly detection.

This module provides various ML models for detecting artificial NEO signatures,
including unsupervised anomaly detection, ensemble methods, and deep learning approaches.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import pickle
import joblib
from pathlib import Path

# Optional ML dependencies with graceful degradation
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, GridSearchCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available, ML models will have limited functionality")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .features import FeatureVector

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_type: str = "isolation_forest"
    parameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing: List[str] = field(default_factory=lambda: ["standardize", "handle_outliers"])
    validation_split: float = 0.2
    random_state: int = 42
    
@dataclass
class TrainingResult:
    """Result from model training."""
    model_id: str
    model_type: str
    training_score: float
    validation_score: float
    feature_importance: Optional[Dict[str, float]] = None
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    trained_at: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionResult:
    """Result from model prediction."""
    designation: str
    anomaly_score: float
    anomaly_probability: float
    is_anomaly: bool
    confidence: float
    model_id: str
    feature_contributions: Optional[Dict[str, float]] = None
    predicted_at: datetime = field(default_factory=datetime.now)

class AnomalyDetectionModel(ABC):
    """Abstract base class for anomaly detection models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = []
        self.model_id = f"{config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        pass
    
    def preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess features according to configuration."""
        X_processed = X.copy()
        
        for preprocessing_step in self.config.preprocessing:
            if preprocessing_step == "standardize":
                if fit:
                    self.scaler = StandardScaler()
                    X_processed = self.scaler.fit_transform(X_processed)
                else:
                    if self.scaler is not None:
                        X_processed = self.scaler.transform(X_processed)
            
            elif preprocessing_step == "robust_scale":
                if fit:
                    self.scaler = RobustScaler()
                    X_processed = self.scaler.fit_transform(X_processed)
                else:
                    if self.scaler is not None:
                        X_processed = self.scaler.transform(X_processed)
            
            elif preprocessing_step == "handle_outliers":
                # Cap extreme values at 99th percentile
                if fit:
                    self.outlier_bounds = np.percentile(X_processed, [1, 99], axis=0)
                
                if hasattr(self, 'outlier_bounds'):
                    X_processed = np.clip(X_processed, 
                                        self.outlier_bounds[0], 
                                        self.outlier_bounds[1])
        
        # Handle NaN and infinite values
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X_processed
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'model_id': self.model_id,
            'is_trained': self.is_trained,
            'outlier_bounds': getattr(self, 'outlier_bounds', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.model_id = model_data['model_id']
        self.is_trained = model_data['is_trained']
        
        if 'outlier_bounds' in model_data:
            self.outlier_bounds = model_data['outlier_bounds']
        
        logger.info(f"Model loaded from {filepath}")

class IsolationForestModel(AnomalyDetectionModel):
    """Isolation Forest for anomaly detection."""
    
    def __init__(self, config: ModelConfig):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for IsolationForestModel")
        
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'contamination': 0.1,
            'random_state': config.random_state,
            'n_jobs': -1
        }
        default_params.update(config.parameters)
        
        self.model = IsolationForest(**default_params)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Train Isolation Forest."""
        logger.info(f"Training Isolation Forest with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Train model
        self.model.fit(X_processed)
        self.is_trained = True
        
        # Calculate training score
        train_scores = self.model.decision_function(X_processed)
        train_predictions = self.model.predict(X_processed)
        
        # Training score is the mean decision function value for normal samples
        normal_scores = train_scores[train_predictions == 1]
        training_score = np.mean(normal_scores) if len(normal_scores) > 0 else 0.0
        
        return TrainingResult(
            model_id=self.model_id,
            model_type="isolation_forest",
            training_score=training_score,
            validation_score=training_score,  # No separate validation for unsupervised
            training_metadata={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'contamination': self.model.contamination,
                'n_estimators': self.model.n_estimators
            }
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (decision function values)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_processed = self.preprocess_features(X, fit=False)
        scores = self.model.decision_function(X_processed)
        
        # Convert to [0,1] range where higher = more anomalous
        # Isolation forest returns negative values for anomalies
        normalized_scores = (0.5 - scores) / (0.5 - scores.min()) if scores.min() < 0.5 else scores
        return np.clip(normalized_scores, 0, 1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        scores = self.predict(X)
        # Convert scores to probabilities using sigmoid-like function
        probabilities = 1 / (1 + np.exp(-5 * (scores - 0.5)))
        return probabilities

class OneClassSVMModel(AnomalyDetectionModel):
    """One-Class SVM for anomaly detection."""
    
    def __init__(self, config: ModelConfig):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for OneClassSVMModel")
        
        super().__init__(config)
        
        # Default parameters
        default_params = {
            'kernel': 'rbf',
            'gamma': 'scale',
            'nu': 0.1
        }
        default_params.update(config.parameters)
        
        self.model = OneClassSVM(**default_params)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Train One-Class SVM."""
        logger.info(f"Training One-Class SVM with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Train model
        self.model.fit(X_processed)
        self.is_trained = True
        
        # Calculate training score
        train_scores = self.model.decision_function(X_processed)
        training_score = np.mean(train_scores)
        
        return TrainingResult(
            model_id=self.model_id,
            model_type="one_class_svm",
            training_score=training_score,
            validation_score=training_score,
            training_metadata={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'kernel': self.model.kernel,
                'nu': self.model.nu
            }
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_processed = self.preprocess_features(X, fit=False)
        scores = self.model.decision_function(X_processed)
        
        # Normalize scores to [0,1] range
        scores_min, scores_max = scores.min(), scores.max()
        if scores_max > scores_min:
            normalized_scores = (scores_max - scores) / (scores_max - scores_min)
        else:
            normalized_scores = np.zeros_like(scores)
        
        return normalized_scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        scores = self.predict(X)
        # Convert to probabilities
        probabilities = scores  # Already normalized to [0,1]
        return probabilities

class AutoencoderModel(AnomalyDetectionModel):
    """Neural network autoencoder for anomaly detection."""
    
    def __init__(self, config: ModelConfig):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for AutoencoderModel")
        
        super().__init__(config)
        
        # Default parameters
        self.hidden_sizes = config.parameters.get('hidden_sizes', [64, 32, 16, 32, 64])
        self.learning_rate = config.parameters.get('learning_rate', 0.001)
        self.epochs = config.parameters.get('epochs', 100)
        self.batch_size = config.parameters.get('batch_size', 32)
        self.dropout_rate = config.parameters.get('dropout_rate', 0.1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_autoencoder(self, input_size: int) -> nn.Module:
        """Create autoencoder architecture."""
        
        class Autoencoder(nn.Module):
            def __init__(self, input_size, hidden_sizes, dropout_rate):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_size = input_size
                
                for hidden_size in hidden_sizes[:len(hidden_sizes)//2 + 1]:
                    encoder_layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                
                self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout
                
                # Decoder
                decoder_layers = []
                
                for hidden_size in hidden_sizes[len(hidden_sizes)//2 + 1:]:
                    decoder_layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_size = hidden_size
                
                # Output layer
                decoder_layers.extend([
                    nn.Linear(prev_size, input_size),
                    nn.Sigmoid()  # Assuming normalized input
                ])
                
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return Autoencoder(input_size, self.hidden_sizes, self.dropout_rate)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Train autoencoder."""
        logger.info(f"Training Autoencoder with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Create model
        self.model = self._create_autoencoder(X_processed.shape[1]).to(self.device)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder: input = target
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, batch_target in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(dataloader)
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
        
        self.is_trained = True
        
        # Calculate training score (reconstruction error)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            training_score = float(torch.mean(reconstruction_errors))
        
        # Calculate threshold for anomaly detection (95th percentile of training errors)
        self.anomaly_threshold = float(torch.quantile(reconstruction_errors, 0.95))
        
        return TrainingResult(
            model_id=self.model_id,
            model_type="autoencoder",
            training_score=training_score,
            validation_score=training_score,
            training_metadata={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'epochs': self.epochs,
                'final_loss': total_loss / self.epochs,
                'anomaly_threshold': self.anomaly_threshold
            }
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (reconstruction errors)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_processed = self.preprocess_features(X, fit=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        # Normalize scores to [0,1] range
        errors = reconstruction_errors.cpu().numpy()
        normalized_scores = np.clip(errors / (2 * self.anomaly_threshold), 0, 1)
        
        return normalized_scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        scores = self.predict(X)
        # Convert reconstruction error to probability
        probabilities = 1 - np.exp(-scores)  # Higher error = higher probability
        return probabilities

class ModelEnsemble:
    """Ensemble of multiple anomaly detection models."""
    
    def __init__(self, models: List[AnomalyDetectionModel], weights: Optional[List[float]] = None):
        """Initialize ensemble with list of models."""
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"ModelEnsemble initialized with {len(models)} models")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> List[TrainingResult]:
        """Train all models in ensemble."""
        training_results = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Training ensemble model {i+1}/{len(self.models)}: {model.config.model_type}")
            
            try:
                result = model.fit(X, y)
                training_results.append(result)
            except Exception as e:
                logger.error(f"Failed to train model {i}: {e}")
                # Create dummy result for failed model
                training_results.append(TrainingResult(
                    model_id=f"failed_{i}",
                    model_type=model.config.model_type,
                    training_score=0.0,
                    validation_score=0.0,
                    training_metadata={'error': str(e)}
                ))
        
        return training_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble of models."""
        all_predictions = []
        
        for model in self.models:
            if model.is_trained:
                try:
                    predictions = model.predict(X)
                    all_predictions.append(predictions)
                except Exception as e:
                    logger.warning(f"Model {model.model_id} prediction failed: {e}")
        
        if not all_predictions:
            logger.warning("No models available for prediction")
            return np.zeros(X.shape[0])
        
        # Weighted average of predictions
        weighted_predictions = np.zeros(X.shape[0])
        total_weight = 0
        
        for predictions, weight in zip(all_predictions, self.weights[:len(all_predictions)]):
            weighted_predictions += predictions * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_predictions /= total_weight
        
        return weighted_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using ensemble."""
        all_probabilities = []
        
        for model in self.models:
            if model.is_trained:
                try:
                    probabilities = model.predict_proba(X)
                    all_probabilities.append(probabilities)
                except Exception as e:
                    logger.warning(f"Model {model.model_id} probability prediction failed: {e}")
        
        if not all_probabilities:
            return np.zeros(X.shape[0])
        
        # Weighted average of probabilities
        weighted_probabilities = np.zeros(X.shape[0])
        total_weight = 0
        
        for probabilities, weight in zip(all_probabilities, self.weights[:len(all_probabilities)]):
            weighted_probabilities += probabilities * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_probabilities /= total_weight
        
        return weighted_probabilities
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        predictions = {}
        
        for model in self.models:
            if model.is_trained:
                try:
                    model_predictions = model.predict(X)
                    predictions[model.model_id] = model_predictions
                except Exception as e:
                    logger.warning(f"Model {model.model_id} prediction failed: {e}")
        
        return predictions
    
    def save_ensemble(self, directory: str) -> None:
        """Save entire ensemble to directory."""
        ensemble_dir = Path(directory)
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for i, model in enumerate(self.models):
            model_path = ensemble_dir / f"model_{i}.pkl"
            model.save_model(str(model_path))
        
        # Save ensemble metadata
        metadata = {
            'ensemble_id': self.ensemble_id,
            'weights': self.weights,
            'model_count': len(self.models),
            'model_types': [model.config.model_type for model in self.models]
        }
        
        metadata_path = ensemble_dir / "ensemble_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Ensemble saved to {directory}")
    
    def load_ensemble(self, directory: str, model_configs: List[ModelConfig]) -> None:
        """Load ensemble from directory."""
        ensemble_dir = Path(directory)
        
        # Load metadata
        metadata_path = ensemble_dir / "ensemble_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.ensemble_id = metadata['ensemble_id']
        self.weights = metadata['weights']
        
        # Load individual models
        self.models = []
        for i in range(metadata['model_count']):
            model_path = ensemble_dir / f"model_{i}.pkl"
            
            # Create model based on type
            model_type = metadata['model_types'][i]
            config = model_configs[i] if i < len(model_configs) else ModelConfig(model_type=model_type)
            
            if model_type == "isolation_forest":
                model = IsolationForestModel(config)
            elif model_type == "one_class_svm":
                model = OneClassSVMModel(config)
            elif model_type == "autoencoder":
                model = AutoencoderModel(config)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            model.load_model(str(model_path))
            self.models.append(model)
        
        logger.info(f"Ensemble loaded from {directory}")

# Factory function for creating models
def create_model(model_type: str, config: Optional[ModelConfig] = None) -> AnomalyDetectionModel:
    """Create anomaly detection model by type."""
    if config is None:
        config = ModelConfig(model_type=model_type)
    
    if model_type == "isolation_forest":
        return IsolationForestModel(config)
    elif model_type == "one_class_svm":
        return OneClassSVMModel(config)
    elif model_type == "autoencoder":
        return AutoencoderModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_default_ensemble() -> ModelEnsemble:
    """Create ensemble with default models."""
    models = []
    
    # Isolation Forest
    if_config = ModelConfig(
        model_type="isolation_forest",
        parameters={'n_estimators': 200, 'contamination': 0.1}
    )
    models.append(create_model("isolation_forest", if_config))
    
    # One-Class SVM
    if HAS_SKLEARN:
        svm_config = ModelConfig(
            model_type="one_class_svm",
            parameters={'kernel': 'rbf', 'nu': 0.05}
        )
        models.append(create_model("one_class_svm", svm_config))
    
    # Autoencoder (if PyTorch available)
    if HAS_TORCH:
        ae_config = ModelConfig(
            model_type="autoencoder",
            parameters={'hidden_sizes': [128, 64, 32, 64, 128], 'epochs': 150}
        )
        models.append(create_model("autoencoder", ae_config))
    
    # Equal weights for all models
    weights = [1.0] * len(models)
    
    return ModelEnsemble(models, weights)