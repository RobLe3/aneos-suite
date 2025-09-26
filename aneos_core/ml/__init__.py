"""
Machine Learning Module for aNEOS - Advanced ML-based anomaly detection.

This module provides machine learning capabilities including feature engineering,
model training, ensemble methods, and real-time prediction for NEO analysis.
"""

from typing import List, Optional

# Feature Engineering
from .features import (
    FeatureEngineer, FeatureVector,
    OrbitalFeatureExtractor, VelocityFeatureExtractor,
    TemporalFeatureExtractor, GeographicFeatureExtractor,
    IndicatorFeatureExtractor
)

# ML Models
from .models import (
    AnomalyDetectionModel, ModelConfig, ModelEnsemble,
    IsolationForestModel, OneClassSVMModel, AutoencoderModel,
    create_model, create_default_ensemble
)

# Training Pipeline
from .training import (
    TrainingPipeline, TrainingConfig, TrainingSession,
    TrainingDataManager, HyperparameterOptimizer
)

# Real-time Prediction
from .prediction import (
    RealTimePredictor, PredictionConfig, PredictionResult,
    ModelManager, Alert
)

__version__ = "0.7.0"
__author__ = "aNEOS Project"

# Quick access functions
def create_feature_engineer() -> FeatureEngineer:
    """Create a feature engineer with all extractors."""
    return FeatureEngineer()

def create_training_pipeline(analysis_pipeline, config: Optional[TrainingConfig] = None) -> TrainingPipeline:
    """Create a training pipeline with default or custom configuration."""
    if config is None:
        config = TrainingConfig()
    return TrainingPipeline(analysis_pipeline, config)

def create_predictor(analysis_pipeline, config: Optional[PredictionConfig] = None) -> RealTimePredictor:
    """Create a real-time predictor with default or custom configuration."""
    if config is None:
        config = PredictionConfig()
    return RealTimePredictor(analysis_pipeline, config)

async def quick_ml_analysis(designation: str, analysis_pipeline) -> PredictionResult:
    """Quick ML analysis of a single NEO."""
    predictor = create_predictor(analysis_pipeline)
    return await predictor.predict_anomaly(designation)

__all__ = [
    # Feature Engineering
    'FeatureEngineer', 'FeatureVector',
    'OrbitalFeatureExtractor', 'VelocityFeatureExtractor',
    'TemporalFeatureExtractor', 'GeographicFeatureExtractor',
    'IndicatorFeatureExtractor',
    
    # ML Models
    'AnomalyDetectionModel', 'ModelConfig', 'ModelEnsemble',
    'IsolationForestModel', 'OneClassSVMModel', 'AutoencoderModel',
    'create_model', 'create_default_ensemble',
    
    # Training
    'TrainingPipeline', 'TrainingConfig', 'TrainingSession',
    'TrainingDataManager', 'HyperparameterOptimizer',
    
    # Prediction
    'RealTimePredictor', 'PredictionConfig', 'PredictionResult',
    'ModelManager', 'Alert',
    
    # Utility Functions
    'create_feature_engineer', 'create_training_pipeline',
    'create_predictor', 'quick_ml_analysis'
]
