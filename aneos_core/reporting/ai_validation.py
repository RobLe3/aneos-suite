#!/usr/bin/env python3
"""
AI-Driven Anomaly Validation System for aNEOS Core

Provides advanced anomaly detection, validation, and classification capabilities
with machine learning models, slingshot effect detection, and confidence scoring.
Replicates and enhances the AI validation from legacy reporting_neos_ng_v3.0.py.
"""

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import threading
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class AIAnomalyValidator:
    """
    Advanced AI-driven anomaly validation system.
    
    Provides comprehensive anomaly detection using machine learning models,
    confidence scoring, and professional validation criteria matching the
    academic rigor of the legacy system.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize AI anomaly validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.has_sklearn = HAS_SKLEARN
        self.has_pandas = HAS_PANDAS
        self.has_tqdm = HAS_TQDM
        
        # AI model and scaler
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        # Validation configuration
        self.config = {
            "verification_threshold": 10.0,  # Confidence threshold for verification
            "anomaly_threshold_multiplier": 1.5,  # Dynamic threshold multiplier
            "slingshot_threshold_multiplier": 2.0,  # Slingshot detection threshold
            "min_training_samples": 10,  # Minimum samples needed for training
            "model_confidence_threshold": 0.7,  # Model R² threshold for reliability
            "features": [
                "semi_major_axis",
                "eccentricity", 
                "inclination"
            ],
            "target": "delta_v",
            "fallback_features": [
                "Dynamic TAS",
                "Raw TAS",
                "Close Approaches"
            ]
        }
        
        # Model performance metrics
        self.model_metrics = {
            "r2_score": 0.0,
            "mse": 0.0,
            "training_samples": 0,
            "feature_importance": {},
            "training_timestamp": None
        }
    
    def run_with_spinner(self, func, *args, **kwargs):
        """
        Execute function with progress spinner (replicates legacy behavior).
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        done = [False]
        result = [None]
        exception = [None]
        
        def spin():
            """Spinner animation thread."""
            spinner_chars = ['|', '/', '-', '\\']
            idx = 0
            while not done[0]:
                sys.stdout.write(f"\r\033[33mTraining AI model... {spinner_chars[idx % len(spinner_chars)]}\033[0m")
                sys.stdout.flush()
                time.sleep(0.1)
                idx += 1
            sys.stdout.write("\r\033[32mTraining AI model... done!          \n\033[0m")
            sys.stdout.flush()
        
        def execute():
            """Execute function in separate thread."""
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
            finally:
                done[0] = True
        
        # Start threads
        spinner_thread = threading.Thread(target=spin)
        execute_thread = threading.Thread(target=execute)
        
        spinner_thread.start()
        execute_thread.start()
        
        execute_thread.join()
        spinner_thread.join()
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def prepare_training_data(self, data: List[Dict[str, Any]]) -> Tuple[Any, Any, List[str]]:
        """
        Prepare training data from NEO dataset.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        if self.has_pandas:
            df = pd.DataFrame(data)
        else:
            # Manual data preparation without pandas
            df = data
        
        # Primary features (orbital mechanics)
        primary_features = self.config["features"]
        target_field = self.config["target"]
        
        # Check feature availability
        available_features = []
        feature_data = []
        
        if self.has_pandas:
            for feature in primary_features:
                if feature in df.columns and df[feature].notna().sum() > 0:
                    available_features.append(feature)
            
            if not available_features:
                # Fallback to alternative features
                self.logger.warning("Primary orbital features not available, using fallback features")
                for feature in self.config["fallback_features"]:
                    if feature in df.columns and df[feature].notna().sum() > 0:
                        available_features.append(feature)
            
            if available_features:
                # Prepare feature matrix
                X = df[available_features].fillna(0).values
                
                # Prepare target vector
                if target_field in df.columns:
                    y = df[target_field].fillna(0).values
                else:
                    # Use Dynamic TAS as fallback target
                    y = df.get("Dynamic TAS", df.get("Raw TAS", pd.Series([0] * len(df)))).fillna(0).values
            else:
                raise ValueError("No suitable features found for training")
        
        else:
            # Manual preparation without pandas
            for feature in primary_features:
                feature_values = [item.get(feature, 0) for item in data if item.get(feature) is not None]
                if len(feature_values) > 0:
                    available_features.append(feature)
            
            if not available_features:
                for feature in self.config["fallback_features"]:
                    feature_values = [item.get(feature, 0) for item in data if item.get(feature) is not None]
                    if len(feature_values) > 0:
                        available_features.append(feature)
            
            if available_features:
                # Build feature matrix manually
                X = []
                y = []
                for item in data:
                    features = []
                    for feature in available_features:
                        features.append(item.get(feature, 0))
                    
                    target_value = item.get(target_field, item.get("Dynamic TAS", item.get("Raw TAS", 0)))
                    if target_value is None:
                        target_value = 0
                    
                    X.append(features)
                    y.append(target_value)
                
                X = np.array(X)
                y = np.array(y)
            else:
                raise ValueError("No suitable features found for training")
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Available features: {available_features}")
        
        return X, y, available_features
    
    def train_orbital_anomaly_model(self, data: List[Dict[str, Any]]) -> Optional[RandomForestRegressor]:
        """
        Train orbital anomaly detection model with professional validation.
        
        Replicates and enhances the training from legacy reporting_neos_ng_v3.0.py.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Trained RandomForestRegressor model or None if training failed
        """
        if not self.has_sklearn:
            self.logger.warning("scikit-learn not available, skipping AI model training")
            return None
        
        if len(data) < self.config["min_training_samples"]:
            self.logger.warning(f"Insufficient training data: {len(data)} samples (minimum: {self.config['min_training_samples']})")
            return None
        
        try:
            # Prepare training data
            X, y, feature_names = self.prepare_training_data(data)
            
            if len(X) < self.config["min_training_samples"]:
                self.logger.warning("Insufficient valid training samples after preparation")
                return None
            
            # Split data for validation
            if len(X) > 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                verbose=0
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Validate model performance
            y_pred = self.model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store metrics
            self.model_metrics.update({
                "r2_score": r2,
                "mse": mse,
                "training_samples": len(X_train),
                "feature_importance": dict(zip(feature_names, self.model.feature_importances_)),
                "training_timestamp": datetime.now().isoformat()
            })
            
            if r2 >= self.config["model_confidence_threshold"]:
                self.is_trained = True
                self.logger.info(f"Model trained successfully - R²: {r2:.3f}, MSE: {mse:.3f}")
            else:
                self.logger.warning(f"Model performance below threshold - R²: {r2:.3f} (threshold: {self.config['model_confidence_threshold']})")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error training orbital anomaly model: {e}")
            return None
    
    def validate_orbital_anomalies(self, data: List[Dict[str, Any]], 
                                 model: Optional[RandomForestRegressor] = None) -> List[Dict[str, Any]]:
        """
        Validate orbital anomalies using AI model with confidence scoring.
        
        Args:
            data: List of NEO data dictionaries
            model: Optional pre-trained model (uses self.model if None)
            
        Returns:
            Enhanced data with anomaly validation results
        """
        enhanced_data = []
        model_to_use = model or self.model
        
        if not model_to_use or not self.is_trained:
            self.logger.warning("No trained model available, using fallback validation")
            return self._fallback_anomaly_validation(data)
        
        try:
            # Prepare prediction data using same features as training
            X_pred, _, feature_names = self.prepare_training_data(data)
            
            if self.scaler:
                X_pred_scaled = self.scaler.transform(X_pred)
            else:
                X_pred_scaled = X_pred
            
            # Generate predictions
            expected_values = model_to_use.predict(X_pred_scaled)
            
            # Process each NEO
            for i, neo in enumerate(data):
                enhanced_neo = neo.copy()
                
                # Get actual and expected values
                actual_delta_v = neo.get("delta_v", 0)
                expected_delta_v = expected_values[i] if i < len(expected_values) else 0
                
                # Calculate anomaly confidence
                if expected_delta_v != 0:
                    anomaly_confidence = abs(actual_delta_v - expected_delta_v) / (abs(expected_delta_v) + 1e-6)
                else:
                    # Fallback to TAS-based confidence
                    dynamic_tas = neo.get("Dynamic TAS", neo.get("dynamic_tas", 0))
                    if dynamic_tas != 0:
                        tas_mean = np.mean([item.get("Dynamic TAS", item.get("dynamic_tas", 0)) for item in data])
                        tas_std = np.std([item.get("Dynamic TAS", item.get("dynamic_tas", 0)) for item in data])
                        if tas_std > 0:
                            anomaly_confidence = abs(dynamic_tas - tas_mean) / tas_std
                        else:
                            anomaly_confidence = 0
                    else:
                        anomaly_confidence = 0
                
                # Determine dynamic threshold
                all_confidences = []
                for j, other_neo in enumerate(data):
                    other_actual = other_neo.get("delta_v", 0)
                    other_expected = expected_values[j] if j < len(expected_values) else 0
                    if other_expected != 0:
                        conf = abs(other_actual - other_expected) / (abs(other_expected) + 1e-6)
                        all_confidences.append(conf)
                
                if all_confidences:
                    confidence_mean = np.mean(all_confidences)
                    confidence_std = np.std(all_confidences)
                    dynamic_threshold = confidence_mean + (confidence_std * self.config["anomaly_threshold_multiplier"])
                else:
                    dynamic_threshold = 1.0  # Default threshold
                
                # Validate anomaly
                is_anomaly = anomaly_confidence >= dynamic_threshold
                is_verified = anomaly_confidence > self.config["verification_threshold"]
                
                # Add validation results
                enhanced_neo.update({
                    "expected_delta_v": expected_delta_v,
                    "anomaly_confidence": anomaly_confidence,
                    "ai_validated_anomaly": is_anomaly,
                    "is_verified_anomaly": is_verified,
                    "verification_status": "[Verified]" if is_verified else "[Unverified]",
                    "dynamic_threshold": dynamic_threshold,
                    "validation_timestamp": datetime.now().isoformat()
                })
                
                enhanced_data.append(enhanced_neo)
            
            self.logger.info(f"Validated {len(enhanced_data)} NEOs with AI model")
            
        except Exception as e:
            self.logger.error(f"Error in AI anomaly validation: {e}")
            return self._fallback_anomaly_validation(data)
        
        return enhanced_data
    
    def _fallback_anomaly_validation(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback anomaly validation using TAS-based scoring.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Enhanced data with fallback validation results
        """
        enhanced_data = []
        
        # Calculate TAS statistics for fallback validation
        tas_values = []
        for neo in data:
            tas = neo.get("Dynamic TAS", neo.get("dynamic_tas", neo.get("Raw TAS", 0)))
            if tas is not None and tas != 0:
                tas_values.append(tas)
        
        if tas_values:
            tas_mean = np.mean(tas_values)
            tas_std = np.std(tas_values)
        else:
            tas_mean = 0
            tas_std = 1
        
        for neo in data:
            enhanced_neo = neo.copy()
            
            # Use TAS for fallback anomaly detection
            tas = neo.get("Dynamic TAS", neo.get("dynamic_tas", neo.get("Raw TAS", 0)))
            if tas is not None and tas != 0 and tas_std > 0:
                anomaly_confidence = abs(tas - tas_mean) / tas_std
            else:
                anomaly_confidence = 0
            
            is_anomaly = anomaly_confidence > 1.5  # Simple threshold
            is_verified = anomaly_confidence > self.config["verification_threshold"]
            
            enhanced_neo.update({
                "expected_delta_v": 0,  # No prediction available
                "anomaly_confidence": anomaly_confidence,
                "ai_validated_anomaly": is_anomaly,
                "is_verified_anomaly": is_verified,
                "verification_status": "[Verified]" if is_verified else "[Unverified]",
                "dynamic_threshold": 1.5,
                "validation_method": "fallback_tas",
                "validation_timestamp": datetime.now().isoformat()
            })
            
            enhanced_data.append(enhanced_neo)
        
        self.logger.info(f"Applied fallback validation to {len(enhanced_data)} NEOs")
        return enhanced_data
    
    def detect_slingshot_effect(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect and filter slingshot effects to reduce false positives.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Enhanced data with slingshot detection results
        """
        enhanced_data = []
        
        # Calculate delta_v statistics for slingshot detection
        delta_v_values = [neo.get("delta_v", 0) for neo in data if neo.get("delta_v", 0) != 0]
        
        if delta_v_values and len(delta_v_values) > 1:
            delta_v_std = np.std(delta_v_values)
            slingshot_threshold = delta_v_std * self.config["slingshot_threshold_multiplier"]
        else:
            slingshot_threshold = 5.0  # Default threshold
        
        for neo in data:
            enhanced_neo = neo.copy()
            
            # Detect slingshot effect
            delta_v = neo.get("delta_v", 0)
            if len(delta_v_values) > 1:
                # Look for sudden velocity changes
                delta_v_diff = abs(delta_v - np.mean(delta_v_values))
                is_slingshot = delta_v_diff > slingshot_threshold
            else:
                is_slingshot = False
            
            # Add slingshot detection results
            enhanced_neo.update({
                "slingshot_flag": is_slingshot,
                "slingshot_threshold": slingshot_threshold,
                "delta_v_deviation": abs(delta_v - np.mean(delta_v_values)) if delta_v_values else 0
            })
            
            # Filter anomalies affected by slingshot
            if enhanced_neo.get("ai_validated_anomaly", False) and is_slingshot:
                enhanced_neo["ai_validated_anomaly"] = False
                enhanced_neo["slingshot_filtered"] = True
            
            enhanced_data.append(enhanced_neo)
        
        slingshot_count = sum(1 for neo in enhanced_data if neo.get("slingshot_flag", False))
        if slingshot_count > 0:
            self.logger.info(f"Detected and filtered {slingshot_count} potential slingshot effects")
        
        return enhanced_data
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive model performance metrics.
        
        Returns:
            Dictionary with model performance information
        """
        return {
            "is_trained": self.is_trained,
            "has_sklearn": self.has_sklearn,
            "model_metrics": self.model_metrics.copy(),
            "config": self.config.copy(),
            "validation_criteria": {
                "verification_threshold": self.config["verification_threshold"],
                "confidence_description": "Anomaly confidence > 10 indicates significant deviation",
                "verification_explanation": "[Verified] = confidence > 10, [Unverified] = confidence ≤ 10"
            }
        }


def create_ai_validator(logger: Optional[logging.Logger] = None) -> AIAnomalyValidator:
    """Create an AI anomaly validator instance."""
    return AIAnomalyValidator(logger)