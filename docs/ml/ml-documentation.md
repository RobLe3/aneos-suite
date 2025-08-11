# aNEOS Machine Learning Documentation

Advanced Machine Learning Framework for Near Earth Object Anomaly Detection

## Table of Contents

1. [Introduction](#introduction)
2. [ML Architecture Overview](#ml-architecture-overview)
3. [Feature Engineering](#feature-engineering)
4. [Model Architectures](#model-architectures)
5. [Training Pipeline](#training-pipeline)
6. [Inference System](#inference-system)
7. [Model Management](#model-management)
8. [Performance Evaluation](#performance-evaluation)
9. [Deployment Strategies](#deployment-strategies)
10. [Advanced Topics](#advanced-topics)

---

## Introduction

### Machine Learning in aNEOS

The aNEOS machine learning framework provides advanced anomaly detection capabilities for identifying artificial Near Earth Objects. The ML system complements the rule-based scientific indicators with data-driven pattern recognition.

### Key Capabilities

- **Unsupervised Anomaly Detection**: Identify outliers without labeled training data
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Deep Learning**: Neural networks for complex pattern recognition
- **Real-time Inference**: Fast predictions for operational use
- **Continuous Learning**: Adaptive models that improve with new data
- **Interpretable Results**: Understanding model decisions and feature importance

### ML Framework Philosophy

```
Scientific Rigor + Data-Driven Learning = Robust Detection

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Physics-Based  │    │   ML-Enhanced    │    │   Hybrid        │
│  Indicators     │ +  │   Pattern        │ =  │   Anomaly       │
│  (Expert        │    │   Recognition    │    │   Detection     │
│   Knowledge)    │    │   (Data-Driven)  │    │   System        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## ML Architecture Overview

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Feature Layer   │    │  Model Layer    │
│                 │    │                  │    │                 │
│ • NEO Data      │───▶│ • Feature        │───▶│ • Isolation     │
│ • Indicators    │    │   Engineering    │    │   Forest        │
│ • Observations  │    │ • Normalization  │    │ • One-Class SVM │
│ • Metadata      │    │ • Selection      │    │ • Autoencoder   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Storage Layer  │    │ Training System  │    │ Inference API   │
│                 │    │                  │    │                 │
│ • Model Store   │    │ • Training       │    │ • Real-time     │
│ • Features      │    │   Pipeline       │    │   Prediction    │
│ • Experiments   │    │ • Validation     │    │ • Batch         │
│ • Metrics       │    │ • Optimization   │    │   Processing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Overview

#### 1. Feature Engineering Layer
- **Raw data ingestion** from scientific indicators
- **Feature extraction** and transformation
- **Dimensionality reduction** for efficiency
- **Feature selection** for relevant signals

#### 2. Model Layer
- **Unsupervised models** for anomaly detection
- **Ensemble methods** for robust predictions
- **Deep learning** for complex patterns
- **Model versioning** and management

#### 3. Training System
- **Automated training** pipelines
- **Hyperparameter optimization** 
- **Cross-validation** and testing
- **Performance monitoring**

#### 4. Inference System
- **Real-time prediction** API
- **Batch processing** capabilities
- **Model serving** infrastructure
- **Result interpretation**

---

## Feature Engineering

### Feature Vector Construction

The ML models operate on comprehensive feature vectors derived from NEO data and scientific indicator results.

#### Base Feature Categories

```python
class FeatureVector:
    """Comprehensive feature vector for ML models."""
    
    def __init__(self):
        self.features = {}
        self.feature_categories = {
            'orbital_elements': [],      # Basic orbital parameters
            'derived_orbital': [],       # Computed orbital characteristics  
            'physical_properties': [],   # Size, albedo, spectral data
            'temporal_features': [],     # Time-based patterns
            'spatial_features': [],      # Geographic distributions
            'observational_features': [], # Observation circumstances
            'indicator_scores': [],      # Scientific indicator results
            'statistical_features': []   # Statistical measures
        }
        
    def extract_features(self, neo_data, indicator_results):
        """Extract comprehensive feature set from NEO data."""
        
        # Orbital element features
        self._extract_orbital_features(neo_data.orbital_elements)
        
        # Physical property features  
        self._extract_physical_features(neo_data.physical_properties)
        
        # Temporal pattern features
        self._extract_temporal_features(neo_data.close_approaches)
        
        # Spatial distribution features
        self._extract_spatial_features(neo_data.close_approaches)
        
        # Observational features
        self._extract_observational_features(neo_data.observations)
        
        # Scientific indicator features
        self._extract_indicator_features(indicator_results)
        
        # Statistical features
        self._extract_statistical_features(neo_data)
        
        return self._combine_features()
```

#### Orbital Element Features

```python
def _extract_orbital_features(self, orbital_elements):
    """Extract features from orbital elements."""
    
    if not orbital_elements:
        return self._add_missing_features('orbital_elements', 8)
    
    # Primary orbital elements
    features = {
        'eccentricity': orbital_elements.eccentricity,
        'semi_major_axis': orbital_elements.semi_major_axis,
        'inclination': np.radians(orbital_elements.inclination),
        'longitude_ascending_node': np.radians(orbital_elements.longitude_ascending_node),
        'argument_perihelion': np.radians(orbital_elements.argument_perihelion),
        'mean_anomaly': np.radians(orbital_elements.mean_anomaly)
    }
    
    # Derived orbital characteristics
    if features['semi_major_axis'] and features['eccentricity']:
        # Aphelion and perihelion distances
        features['aphelion_distance'] = features['semi_major_axis'] * (1 + features['eccentricity'])
        features['perihelion_distance'] = features['semi_major_axis'] * (1 - features['eccentricity'])
        
        # Orbital period (Kepler's third law)
        features['orbital_period'] = np.sqrt(features['semi_major_axis']**3) * 365.25  # days
        
        # Energy and angular momentum proxies
        features['orbital_energy'] = -1 / (2 * features['semi_major_axis'])  # Specific energy
        features['angular_momentum'] = np.sqrt(features['semi_major_axis'] * (1 - features['eccentricity']**2))
    
    # Tisserand parameter (Jupiter)
    if all(k in features for k in ['semi_major_axis', 'eccentricity', 'inclination']):
        a_jup = 5.204  # Jupiter semi-major axis
        features['tisserand_jupiter'] = (
            a_jup / features['semi_major_axis'] + 
            2 * np.sqrt(features['semi_major_axis'] / a_jup * (1 - features['eccentricity']**2)) * 
            np.cos(features['inclination'])
        )
    
    self.feature_categories['orbital_elements'] = list(features.keys())
    self.features.update(features)
```

#### Physical Property Features

```python
def _extract_physical_features(self, physical_properties):
    """Extract features from physical properties."""
    
    features = {}
    
    if physical_properties:
        # Basic physical properties
        if physical_properties.diameter:
            features['log_diameter'] = np.log10(max(physical_properties.diameter, 1e-6))
            features['diameter_normalized'] = physical_properties.diameter / 1.0  # km
            
        if physical_properties.albedo:
            features['log_albedo'] = np.log10(max(physical_properties.albedo, 1e-6))
            features['albedo_normalized'] = physical_properties.albedo
            
        # Size-albedo relationship
        if physical_properties.diameter and physical_properties.albedo:
            features['size_albedo_product'] = physical_properties.diameter * physical_properties.albedo
            features['size_albedo_ratio'] = physical_properties.diameter / max(physical_properties.albedo, 1e-6)
            
        # Absolute magnitude relationships
        if physical_properties.absolute_magnitude:
            features['absolute_magnitude'] = physical_properties.absolute_magnitude
            
            # Estimated diameter from H (if not directly available)
            if not physical_properties.diameter and physical_properties.albedo:
                estimated_diameter = 1329 / np.sqrt(physical_properties.albedo) * 10**(-0.2 * physical_properties.absolute_magnitude)
                features['estimated_diameter'] = estimated_diameter
                features['diameter_estimate_ratio'] = physical_properties.diameter / estimated_diameter if physical_properties.diameter else 1.0
        
        # Spectral features (if available)
        if hasattr(physical_properties, 'spectral_type') and physical_properties.spectral_type:
            spectral_encoding = self._encode_spectral_type(physical_properties.spectral_type)
            features.update(spectral_encoding)
            
    # Fill missing features with defaults
    default_physical_features = [
        'log_diameter', 'diameter_normalized', 'log_albedo', 'albedo_normalized',
        'size_albedo_product', 'size_albedo_ratio', 'absolute_magnitude'
    ]
    
    for feature_name in default_physical_features:
        if feature_name not in features:
            features[feature_name] = 0.0  # or appropriate default
    
    self.feature_categories['physical_properties'] = list(features.keys())
    self.features.update(features)
```

#### Temporal Pattern Features

```python
def _extract_temporal_features(self, close_approaches):
    """Extract temporal pattern features from close approaches."""
    
    features = {}
    
    if close_approaches and len(close_approaches) > 1:
        # Extract approach times
        approach_times = [ca.close_approach_date for ca in close_approaches if ca.close_approach_date]
        approach_times.sort()
        
        if len(approach_times) >= 2:
            # Time intervals between approaches
            intervals = []
            for i in range(1, len(approach_times)):
                interval = (approach_times[i] - approach_times[i-1]).total_seconds() / 86400.0  # days
                intervals.append(interval)
            
            # Statistical measures of intervals
            features['mean_approach_interval'] = np.mean(intervals)
            features['std_approach_interval'] = np.std(intervals)
            features['min_approach_interval'] = np.min(intervals)
            features['max_approach_interval'] = np.max(intervals)
            
            # Regularity metrics
            cv = features['std_approach_interval'] / max(features['mean_approach_interval'], 1.0)
            features['approach_regularity'] = 1.0 / (1.0 + cv)  # 0=irregular, 1=regular
            
            # Periodicity analysis (simplified)
            if len(intervals) >= 3:
                features['interval_autocorr'] = self._calculate_autocorrelation(intervals)
            
            # Temporal clustering
            features['temporal_clustering'] = self._calculate_temporal_clustering(approach_times)
    
    # Observation timing features
    if hasattr(self, 'observation_history'):
        features.update(self._extract_observation_timing_features())
    
    # Fill missing temporal features
    default_temporal_features = [
        'mean_approach_interval', 'std_approach_interval', 'approach_regularity',
        'interval_autocorr', 'temporal_clustering'
    ]
    
    for feature_name in default_temporal_features:
        if feature_name not in features:
            features[feature_name] = 0.0
    
    self.feature_categories['temporal_features'] = list(features.keys())
    self.features.update(features)
```

#### Spatial Distribution Features

```python
def _extract_spatial_features(self, close_approaches):
    """Extract spatial distribution features."""
    
    features = {}
    
    if close_approaches:
        # Extract valid subpoints
        subpoints = []
        distances = []
        
        for ca in close_approaches:
            if ca.subpoint and len(ca.subpoint) == 2:
                lat, lon = ca.subpoint
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    subpoints.append((lat, lon))
                    if ca.distance_au:
                        distances.append(ca.distance_au)
        
        if subpoints:
            # Geographic distribution statistics
            lats = [sp[0] for sp in subpoints]
            lons = [sp[1] for sp in subpoints]
            
            features['lat_mean'] = np.mean(lats)
            features['lat_std'] = np.std(lats)
            features['lon_mean'] = np.mean(lons)  # Note: circular statistics would be better
            features['lon_std'] = np.std(lons)
            
            # Geographic spread
            features['geographic_spread'] = self._calculate_geographic_spread(subpoints)
            
            # Clustering metrics
            if len(subpoints) >= 3:
                features['geographic_clustering'] = self._calculate_geographic_clustering(subpoints)
                features['strategic_correlation'] = self._calculate_strategic_correlation(subpoints)
            
            # Distance statistics
            if distances:
                features['mean_approach_distance'] = np.mean(distances)
                features['std_approach_distance'] = np.std(distances)
                features['min_approach_distance'] = np.min(distances)
    
    # Default spatial features
    default_spatial_features = [
        'lat_mean', 'lat_std', 'lon_mean', 'lon_std', 
        'geographic_spread', 'geographic_clustering', 'strategic_correlation',
        'mean_approach_distance', 'std_approach_distance'
    ]
    
    for feature_name in default_spatial_features:
        if feature_name not in features:
            features[feature_name] = 0.0
    
    self.feature_categories['spatial_features'] = list(features.keys())
    self.features.update(features)
```

#### Scientific Indicator Features

```python
def _extract_indicator_features(self, indicator_results):
    """Extract features from scientific indicator results."""
    
    features = {}
    
    if indicator_results:
        for result in indicator_results:
            # Individual indicator scores
            features[f'{result.indicator_name}_raw_score'] = result.raw_score
            features[f'{result.indicator_name}_weighted_score'] = result.weighted_score  
            features[f'{result.indicator_name}_confidence'] = result.confidence
            
            # Metadata features (if available)
            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, (int, float)):
                        features[f'{result.indicator_name}_{key}'] = float(value)
        
        # Aggregate indicator statistics
        raw_scores = [r.raw_score for r in indicator_results]
        weighted_scores = [r.weighted_score for r in indicator_results]
        confidences = [r.confidence for r in indicator_results]
        
        features['indicator_mean_raw'] = np.mean(raw_scores)
        features['indicator_std_raw'] = np.std(raw_scores)
        features['indicator_max_raw'] = np.max(raw_scores)
        features['indicator_mean_weighted'] = np.mean(weighted_scores)
        features['indicator_std_weighted'] = np.std(weighted_scores)
        features['indicator_mean_confidence'] = np.mean(confidences)
        
        # High-score indicator count
        features['high_score_indicators'] = sum(1 for score in raw_scores if score > 0.7)
        
    self.feature_categories['indicator_scores'] = list(features.keys())
    self.features.update(features)
```

### Feature Preprocessing

#### Normalization and Scaling

```python
class FeaturePreprocessor:
    """Feature preprocessing pipeline for ML models."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
        self.preprocessing_config = {
            'normalization_method': 'standardize',  # 'standardize', 'minmax', 'robust'
            'handle_missing': 'impute',             # 'impute', 'drop', 'flag'
            'outlier_treatment': 'clip',            # 'clip', 'transform', 'flag'
            'feature_selection': True
        }
        
    def fit_transform(self, feature_vectors, feature_names=None):
        """Fit preprocessor and transform features."""
        
        # Convert to numpy array if needed
        if isinstance(feature_vectors, list):
            feature_vectors = np.array([fv.to_array() for fv in feature_vectors])
        
        # Handle missing values
        feature_vectors = self._handle_missing_values(feature_vectors, fit=True)
        
        # Outlier treatment
        feature_vectors = self._handle_outliers(feature_vectors, fit=True)
        
        # Normalization
        feature_vectors = self._normalize_features(feature_vectors, fit=True)
        
        # Feature selection
        if self.preprocessing_config['feature_selection']:
            feature_vectors = self._select_features(feature_vectors, fit=True)
        
        return feature_vectors
    
    def transform(self, feature_vectors):
        """Transform features using fitted preprocessor."""
        
        # Convert to numpy array if needed
        if isinstance(feature_vectors, list):
            feature_vectors = np.array([fv.to_array() for fv in feature_vectors])
        
        # Apply same preprocessing steps (without fitting)
        feature_vectors = self._handle_missing_values(feature_vectors, fit=False)
        feature_vectors = self._handle_outliers(feature_vectors, fit=False)
        feature_vectors = self._normalize_features(feature_vectors, fit=False)
        
        if self.preprocessing_config['feature_selection']:
            feature_vectors = self._select_features(feature_vectors, fit=False)
        
        return feature_vectors
    
    def _normalize_features(self, X, fit=False):
        """Normalize features according to configuration."""
        
        method = self.preprocessing_config['normalization_method']
        
        if method == 'standardize':
            from sklearn.preprocessing import StandardScaler
            if fit:
                self.scalers['standard'] = StandardScaler()
                X_scaled = self.scalers['standard'].fit_transform(X)
            else:
                X_scaled = self.scalers['standard'].transform(X)
                
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            if fit:
                self.scalers['minmax'] = MinMaxScaler()
                X_scaled = self.scalers['minmax'].fit_transform(X)
            else:
                X_scaled = self.scalers['minmax'].transform(X)
                
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            if fit:
                self.scalers['robust'] = RobustScaler()
                X_scaled = self.scalers['robust'].fit_transform(X)
            else:
                X_scaled = self.scalers['robust'].transform(X)
        else:
            X_scaled = X
        
        return X_scaled
```

---

## Model Architectures

### Unsupervised Anomaly Detection Models

#### 1. Isolation Forest

**Theoretical Foundation**: Isolation Forest isolates anomalies by randomly selecting features and split values. Anomalous points require fewer splits to isolate.

```python
class IsolationForestModel(AnomalyDetectionModel):
    """Isolation Forest implementation optimized for NEO data."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Optimized parameters for NEO data
        self.model_params = {
            'n_estimators': config.parameters.get('n_estimators', 200),
            'contamination': config.parameters.get('contamination', 0.1),
            'max_samples': config.parameters.get('max_samples', 'auto'),
            'max_features': config.parameters.get('max_features', 1.0),
            'random_state': config.random_state,
            'n_jobs': -1  # Use all CPU cores
        }
        
        self.model = IsolationForest(**self.model_params)
        
    def fit(self, X, y=None):
        """Train Isolation Forest on feature vectors."""
        
        logger.info(f"Training Isolation Forest: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Fit model
        self.model.fit(X_processed)
        self.is_trained = True
        
        # Calculate training metrics
        decision_scores = self.model.decision_function(X_processed)
        predictions = self.model.predict(X_processed)
        
        # Anomaly threshold (95th percentile of decision scores)
        self.anomaly_threshold = np.percentile(decision_scores, 95)
        
        training_result = TrainingResult(
            model_id=self.model_id,
            model_type='isolation_forest',
            training_score=float(np.mean(decision_scores)),
            validation_score=float(np.mean(decision_scores)),  # No separate validation for unsupervised
            training_metadata={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'contamination': self.model_params['contamination'],
                'anomaly_threshold': self.anomaly_threshold,
                'anomalies_detected': int(np.sum(predictions == -1))
            }
        )
        
        return training_result
    
    def predict(self, X):
        """Predict anomaly scores using decision function."""
        
        X_processed = self.preprocess_features(X, fit=False)
        
        # Get decision scores (higher = more normal, lower = more anomalous)
        decision_scores = self.model.decision_function(X_processed)
        
        # Convert to anomaly scores (0-1, higher = more anomalous)
        # Normalize using training statistics
        normalized_scores = (self.anomaly_threshold - decision_scores) / (self.anomaly_threshold - decision_scores.min())
        anomaly_scores = np.clip(normalized_scores, 0, 1)
        
        return anomaly_scores
    
    def get_feature_importance(self, feature_names=None):
        """Calculate feature importance for Isolation Forest."""
        
        # Isolation Forest doesn't provide direct feature importance
        # Use permutation importance as approximation
        if not self.is_trained:
            return None
        
        # This would require a validation set for proper implementation
        # Simplified version for demonstration
        return {
            'method': 'permutation_importance',
            'note': 'Requires validation set for accurate computation'
        }
```

#### 2. One-Class SVM

**Theoretical Foundation**: One-Class SVM learns a decision boundary around the normal data in feature space, treating anything outside as anomalous.

```python
class OneClassSVMModel(AnomalyDetectionModel):
    """One-Class SVM optimized for high-dimensional NEO features."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model_params = {
            'kernel': config.parameters.get('kernel', 'rbf'),
            'nu': config.parameters.get('nu', 0.05),  # Expected fraction of anomalies
            'gamma': config.parameters.get('gamma', 'scale'),
            'degree': config.parameters.get('degree', 3),  # For polynomial kernel
            'coef0': config.parameters.get('coef0', 0.0),  # For polynomial/sigmoid kernels
            'tol': config.parameters.get('tol', 1e-3),
            'shrinking': config.parameters.get('shrinking', True),
            'cache_size': config.parameters.get('cache_size', 200),  # MB
            'max_iter': config.parameters.get('max_iter', -1)
        }
        
        self.model = OneClassSVM(**self.model_params)
        
    def fit(self, X, y=None):
        """Train One-Class SVM."""
        
        logger.info(f"Training One-Class SVM: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess features  
        X_processed = self.preprocess_features(X, fit=True)
        
        # Fit model
        self.model.fit(X_processed)
        self.is_trained = True
        
        # Calculate training metrics
        decision_scores = self.model.decision_function(X_processed)
        predictions = self.model.predict(X_processed)
        
        # Decision boundary (0 for SVM)
        self.decision_boundary = 0.0
        
        training_result = TrainingResult(
            model_id=self.model_id,
            model_type='one_class_svm',
            training_score=float(np.mean(decision_scores)),
            validation_score=float(np.mean(decision_scores)),
            training_metadata={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'kernel': self.model_params['kernel'],
                'nu': self.model_params['nu'],
                'n_support_vectors': self.model.support_.shape[0],
                'anomalies_detected': int(np.sum(predictions == -1))
            }
        )
        
        return training_result
    
    def predict(self, X):
        """Predict anomaly scores using decision function."""
        
        X_processed = self.preprocess_features(X, fit=False)
        
        # Get decision scores
        decision_scores = self.model.decision_function(X_processed)
        
        # Convert to anomaly scores (0-1, higher = more anomalous)
        # Negative scores indicate anomalies in One-Class SVM
        anomaly_scores = np.where(decision_scores < 0, 
                                 np.abs(decision_scores) / (np.abs(decision_scores.min()) + 1e-6),
                                 0.0)
        
        return np.clip(anomaly_scores, 0, 1)
```

#### 3. Autoencoder Neural Networks

**Theoretical Foundation**: Autoencoders learn to compress and reconstruct normal data. High reconstruction error indicates anomalies.

```python
class AutoencoderModel(AnomalyDetectionModel):
    """Deep autoencoder for complex pattern detection in NEO features."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Architecture parameters
        self.encoder_layers = config.parameters.get('encoder_layers', [128, 64, 32])
        self.decoder_layers = config.parameters.get('decoder_layers', [32, 64, 128])
        self.latent_dim = config.parameters.get('latent_dim', 16)
        self.activation = config.parameters.get('activation', 'relu')
        self.dropout_rate = config.parameters.get('dropout_rate', 0.1)
        
        # Training parameters
        self.learning_rate = config.parameters.get('learning_rate', 0.001)
        self.batch_size = config.parameters.get('batch_size', 32)
        self.epochs = config.parameters.get('epochs', 100)
        self.patience = config.parameters.get('patience', 10)
        
        # Advanced features
        self.use_batch_norm = config.parameters.get('use_batch_norm', True)
        self.use_residual = config.parameters.get('use_residual', False)
        self.use_attention = config.parameters.get('use_attention', False)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Autoencoder using device: {self.device}")
        
    def _build_autoencoder(self, input_dim):
        """Build autoencoder architecture."""
        
        class AdvancedAutoencoder(nn.Module):
            def __init__(self, input_dim, encoder_layers, decoder_layers, latent_dim, 
                        activation, dropout_rate, use_batch_norm, use_residual):
                super().__init__()
                
                self.input_dim = input_dim
                self.latent_dim = latent_dim
                self.use_residual = use_residual
                
                # Activation function
                if activation == 'relu':
                    self.activation = nn.ReLU()
                elif activation == 'leaky_relu':
                    self.activation = nn.LeakyReLU(0.1)
                elif activation == 'elu':
                    self.activation = nn.ELU()
                else:
                    self.activation = nn.ReLU()
                
                # Encoder
                encoder_dims = [input_dim] + encoder_layers + [latent_dim]
                encoder_modules = []
                
                for i in range(len(encoder_dims) - 1):
                    encoder_modules.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
                    
                    if i < len(encoder_dims) - 2:  # No activation/dropout on latent layer
                        if use_batch_norm:
                            encoder_modules.append(nn.BatchNorm1d(encoder_dims[i+1]))
                        encoder_modules.append(self.activation)
                        encoder_modules.append(nn.Dropout(dropout_rate))
                
                self.encoder = nn.Sequential(*encoder_modules)
                
                # Decoder  
                decoder_dims = [latent_dim] + decoder_layers + [input_dim]
                decoder_modules = []
                
                for i in range(len(decoder_dims) - 1):
                    decoder_modules.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
                    
                    if i < len(decoder_dims) - 2:  # No activation on output layer
                        if use_batch_norm:
                            decoder_modules.append(nn.BatchNorm1d(decoder_dims[i+1]))
                        decoder_modules.append(self.activation)
                        decoder_modules.append(nn.Dropout(dropout_rate))
                    else:
                        # Output layer - sigmoid for normalized features
                        decoder_modules.append(nn.Sigmoid())
                
                self.decoder = nn.Sequential(*decoder_modules)
                
                # Residual connections (if enabled)
                if use_residual and input_dim == decoder_dims[-1]:
                    self.residual_weight = nn.Parameter(torch.tensor(0.1))
            
            def forward(self, x):
                # Store input for residual connection
                input_x = x
                
                # Encode
                latent = self.encoder(x)
                
                # Decode
                reconstructed = self.decoder(latent)
                
                # Residual connection (if enabled and dimensions match)
                if self.use_residual and input_x.shape == reconstructed.shape:
                    reconstructed = reconstructed + self.residual_weight * input_x
                
                return reconstructed, latent
            
            def encode(self, x):
                return self.encoder(x)
            
            def decode(self, z):
                return self.decoder(z)
        
        return AdvancedAutoencoder(
            input_dim, self.encoder_layers, self.decoder_layers, self.latent_dim,
            self.activation, self.dropout_rate, self.use_batch_norm, self.use_residual
        )
    
    def fit(self, X, y=None):
        """Train the autoencoder model."""
        
        logger.info(f"Training Autoencoder: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        # Create model
        self.model = self._build_autoencoder(X_processed.shape[1]).to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder: input = target
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        training_losses = []
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_x, batch_target in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, latent = self.model(batch_x)
                loss = criterion(reconstructed, batch_target)
                
                # Add regularization terms
                if self.use_residual:
                    # L2 regularization on residual weight
                    loss += 0.001 * self.model.residual_weight.pow(2)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_epoch_loss)
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.is_trained = True
        
        # Calculate anomaly threshold from training data
        self.model.eval()
        with torch.no_grad():
            reconstructed, _ = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            
            # Set threshold at 95th percentile of training errors
            self.anomaly_threshold = float(torch.quantile(reconstruction_errors, 0.95))
        
        training_result = TrainingResult(
            model_id=self.model_id,
            model_type='autoencoder',
            training_score=best_loss,
            validation_score=best_loss,
            training_metadata={
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'epochs_trained': epoch + 1,
                'best_loss': best_loss,
                'anomaly_threshold': self.anomaly_threshold,
                'architecture': {
                    'encoder_layers': self.encoder_layers,
                    'latent_dim': self.latent_dim,
                    'decoder_layers': self.decoder_layers
                }
            }
        )
        
        return training_result
    
    def predict(self, X):
        """Predict anomaly scores based on reconstruction error."""
        
        X_processed = self.preprocess_features(X, fit=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed, _ = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        # Convert to anomaly scores (0-1 range)
        errors = reconstruction_errors.cpu().numpy()
        anomaly_scores = np.clip(errors / (2 * self.anomaly_threshold), 0, 1)
        
        return anomaly_scores
    
    def get_latent_representation(self, X):
        """Get latent space representation of data."""
        
        X_processed = self.preprocess_features(X, fit=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_tensor)
        
        return latent.cpu().numpy()
```

### Model Ensemble Framework

```python
class AdvancedModelEnsemble:
    """Advanced ensemble combining multiple anomaly detection models."""
    
    def __init__(self, models, weights=None, combination_method='weighted_average'):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.combination_method = combination_method
        self.ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Performance tracking
        self.model_performances = {}
        
    def fit(self, X, y=None):
        """Train all models in ensemble."""
        
        training_results = []
        
        for i, model in enumerate(self.models):
            logger.info(f"Training ensemble model {i+1}/{len(self.models)}: {model.config.model_type}")
            
            try:
                # Train individual model
                result = model.fit(X, y)
                training_results.append(result)
                
                # Track performance
                self.model_performances[model.model_id] = {
                    'training_score': result.training_score,
                    'model_type': result.model_type,
                    'weight': self.weights[i]
                }
                
            except Exception as e:
                logger.error(f"Failed to train model {i}: {e}")
                training_results.append(None)
        
        return training_results
    
    def predict(self, X):
        """Generate ensemble predictions."""
        
        if self.combination_method == 'weighted_average':
            return self._weighted_average_prediction(X)
        elif self.combination_method == 'voting':
            return self._voting_prediction(X)
        elif self.combination_method == 'stacking':
            return self._stacking_prediction(X)
        elif self.combination_method == 'dynamic_weighting':
            return self._dynamic_weighted_prediction(X)
        else:
            return self._weighted_average_prediction(X)
    
    def _weighted_average_prediction(self, X):
        """Weighted average of model predictions."""
        
        predictions = []
        active_weights = []
        
        for i, model in enumerate(self.models):
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    active_weights.append(self.weights[i])
                except Exception as e:
                    logger.warning(f"Model {model.model_id} prediction failed: {e}")
        
        if not predictions:
            return np.zeros(X.shape[0])
        
        # Weighted combination
        predictions = np.array(predictions)
        active_weights = np.array(active_weights)
        active_weights = active_weights / active_weights.sum()  # Renormalize
        
        ensemble_prediction = np.average(predictions, axis=0, weights=active_weights)
        
        return ensemble_prediction
    
    def _dynamic_weighted_prediction(self, X):
        """Dynamic weighting based on model confidence."""
        
        predictions = []
        confidences = []
        
        for model in self.models:
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    
                    # Calculate prediction confidence (simplified)
                    pred_confidence = self._calculate_prediction_confidence(model, X, pred)
                    
                    predictions.append(pred)
                    confidences.append(pred_confidence)
                    
                except Exception as e:
                    logger.warning(f"Model {model.model_id} prediction failed: {e}")
        
        if not predictions:
            return np.zeros(X.shape[0])
        
        # Dynamic weighting based on confidence
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Weight by confidence for each prediction
        ensemble_prediction = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            sample_confidences = confidences[:, i]
            sample_weights = sample_confidences / (sample_confidences.sum() + 1e-8)
            
            ensemble_prediction[i] = np.sum(predictions[:, i] * sample_weights)
        
        return ensemble_prediction
    
    def _calculate_prediction_confidence(self, model, X, predictions):
        """Calculate confidence in model predictions (simplified)."""
        
        # This is a simplified confidence estimation
        # More sophisticated methods could use:
        # - Prediction intervals
        # - Model uncertainty quantification
        # - Ensemble disagreement measures
        
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X)
                # Confidence based on probability values
                confidences = np.max(probabilities, axis=1) if probabilities.ndim > 1 else probabilities
            except:
                # Fallback to score-based confidence
                confidences = 1.0 - np.abs(predictions - 0.5) * 2  # Higher confidence for extreme scores
        else:
            # Score-based confidence
            confidences = 1.0 - np.abs(predictions - 0.5) * 2
        
        return np.clip(confidences, 0.1, 1.0)  # Minimum confidence of 0.1
    
    def get_model_contributions(self, X):
        """Get individual model contributions to ensemble prediction."""
        
        contributions = {}
        
        for i, model in enumerate(self.models):
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    contributions[model.model_id] = {
                        'predictions': pred,
                        'weight': self.weights[i],
                        'weighted_contribution': pred * self.weights[i]
                    }
                except Exception as e:
                    logger.warning(f"Failed to get contribution from {model.model_id}: {e}")
        
        return contributions
```

---

## Training Pipeline

### Automated Training System

```python
class MLTrainingPipeline:
    """Automated ML training pipeline for aNEOS models."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.experiment_tracker = ExperimentTracker()
        self.model_registry = ModelRegistry()
        
    def execute_training_pipeline(self, training_config):
        """Execute complete training pipeline."""
        
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting ML training pipeline: {pipeline_id}")
        
        try:
            # 1. Data preparation
            training_data = self._prepare_training_data(training_config)
            
            # 2. Feature engineering
            features, feature_metadata = self._engineer_features(training_data)
            
            # 3. Model training
            trained_models = self._train_models(features, training_config)
            
            # 4. Model evaluation
            evaluation_results = self._evaluate_models(trained_models, features)
            
            # 5. Model selection
            best_models = self._select_best_models(evaluation_results, training_config)
            
            # 6. Ensemble creation
            ensemble = self._create_ensemble(best_models, training_config)
            
            # 7. Final validation
            final_validation = self._final_validation(ensemble, features)
            
            # 8. Model registration
            self._register_models(ensemble, final_validation, pipeline_id)
            
            return {
                'pipeline_id': pipeline_id,
                'status': 'success',
                'trained_models': len(trained_models),
                'best_models': len(best_models),
                'ensemble_performance': final_validation,
                'model_ids': [model.model_id for model in ensemble.models]
            }
            
        except Exception as e:
            logger.error(f"Training pipeline {pipeline_id} failed: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _prepare_training_data(self, config):
        """Prepare training data from various sources."""
        
        # Load NEO data
        neo_dataset = self._load_neo_dataset(config['data_sources'])
        
        # Apply quality filters
        filtered_dataset = self._apply_quality_filters(neo_dataset, config['quality_filters'])
        
        # Create train/validation split
        train_data, val_data = self._create_train_val_split(filtered_dataset, config['validation_split'])
        
        return {
            'train': train_data,
            'validation': val_data,
            'metadata': {
                'total_samples': len(neo_dataset),
                'filtered_samples': len(filtered_dataset),
                'train_samples': len(train_data),
                'validation_samples': len(val_data)
            }
        }
    
    def _engineer_features(self, training_data):
        """Feature engineering pipeline."""
        
        feature_engineer = FeatureEngineer()
        
        # Extract features from training data
        train_features = []
        for neo_data in training_data['train']:
            # Run scientific indicators
            indicator_results = self._run_scientific_indicators(neo_data)
            
            # Create feature vector
            feature_vector = FeatureVector()
            feature_vector.extract_features(neo_data, indicator_results)
            
            train_features.append(feature_vector)
        
        # Similarly for validation data
        val_features = []
        for neo_data in training_data['validation']:
            indicator_results = self._run_scientific_indicators(neo_data)
            feature_vector = FeatureVector()
            feature_vector.extract_features(neo_data, indicator_results)
            val_features.append(feature_vector)
        
        # Feature preprocessing
        preprocessor = FeaturePreprocessor()
        
        # Fit on training data
        train_features_processed = preprocessor.fit_transform(train_features)
        val_features_processed = preprocessor.transform(val_features)
        
        return {
            'train': train_features_processed,
            'validation': val_features_processed,
            'preprocessor': preprocessor,
            'feature_names': feature_engineer.get_feature_names(),
            'feature_stats': feature_engineer.get_feature_statistics()
        }, {
            'n_features': train_features_processed.shape[1],
            'preprocessing_config': preprocessor.preprocessing_config
        }
    
    def _train_models(self, features, config):
        """Train multiple models with different configurations."""
        
        trained_models = []
        
        # Model configurations to try
        model_configs = self._generate_model_configs(config)
        
        for model_config in model_configs:
            logger.info(f"Training {model_config.model_type} model")
            
            try:
                # Create model
                model = create_model(model_config.model_type, model_config)
                
                # Train model
                training_result = model.fit(features['train'])
                
                # Store training result
                model.training_result = training_result
                trained_models.append(model)
                
                # Track experiment
                self.experiment_tracker.log_experiment(model, training_result)
                
            except Exception as e:
                logger.error(f"Failed to train {model_config.model_type}: {e}")
        
        return trained_models
    
    def _generate_model_configs(self, config):
        """Generate model configurations for training."""
        
        configs = []
        
        # Isolation Forest configurations
        for contamination in [0.05, 0.1, 0.15]:
            for n_estimators in [100, 200, 300]:
                configs.append(ModelConfig(
                    model_type='isolation_forest',
                    parameters={
                        'contamination': contamination,
                        'n_estimators': n_estimators,
                        'max_features': 1.0
                    }
                ))
        
        # One-Class SVM configurations
        for nu in [0.05, 0.1, 0.15]:
            for gamma in ['scale', 'auto', 0.1, 1.0]:
                configs.append(ModelConfig(
                    model_type='one_class_svm',
                    parameters={
                        'nu': nu,
                        'gamma': gamma,
                        'kernel': 'rbf'
                    }
                ))
        
        # Autoencoder configurations
        for latent_dim in [8, 16, 32]:
            for learning_rate in [0.001, 0.01]:
                configs.append(ModelConfig(
                    model_type='autoencoder',
                    parameters={
                        'latent_dim': latent_dim,
                        'learning_rate': learning_rate,
                        'encoder_layers': [128, 64, 32],
                        'decoder_layers': [32, 64, 128],
                        'epochs': 100,
                        'patience': 10
                    }
                ))
        
        return configs
```

### Hyperparameter Optimization

```python
class HyperparameterOptimizer:
    """Automated hyperparameter optimization for ML models."""
    
    def __init__(self, optimization_method='bayesian'):
        self.optimization_method = optimization_method
        self.optimization_history = []
        
    def optimize_model(self, model_type, X_train, X_val, optimization_config):
        """Optimize hyperparameters for a given model type."""
        
        if self.optimization_method == 'grid_search':
            return self._grid_search_optimization(model_type, X_train, X_val, optimization_config)
        elif self.optimization_method == 'random_search':
            return self._random_search_optimization(model_type, X_train, X_val, optimization_config)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(model_type, X_train, X_val, optimization_config)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _bayesian_optimization(self, model_type, X_train, X_val, config):
        """Bayesian optimization using Gaussian Process."""
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to random search")
            return self._random_search_optimization(model_type, X_train, X_val, config)
        
        # Define search space based on model type
        search_space = self._get_search_space(model_type)
        
        def objective_function(params):
            """Objective function for optimization."""
            
            # Create model config with current parameters
            param_dict = self._params_to_dict(model_type, params)
            model_config = ModelConfig(model_type=model_type, parameters=param_dict)
            
            try:
                # Create and train model
                model = create_model(model_type, model_config)
                training_result = model.fit(X_train)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                
                # Use appropriate metric (to be minimized)
                score = self._calculate_optimization_score(val_predictions, model_type)
                
                return score
                
            except Exception as e:
                logger.warning(f"Optimization trial failed: {e}")
                return 1.0  # Return worst possible score
        
        # Run optimization
        result = gp_minimize(
            func=objective_function,
            dimensions=search_space,
            n_calls=config.get('n_trials', 50),
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )
        
        # Convert best parameters back to dict
        best_params = self._params_to_dict(model_type, result.x)
        
        return {
            'best_parameters': best_params,
            'best_score': result.fun,
            'optimization_history': result.func_vals,
            'n_trials': len(result.func_vals)
        }
    
    def _get_search_space(self, model_type):
        """Define search space for each model type."""
        
        if model_type == 'isolation_forest':
            return [
                Integer(50, 500, name='n_estimators'),
                Real(0.01, 0.2, name='contamination'),
                Real(0.5, 1.0, name='max_features')
            ]
        
        elif model_type == 'one_class_svm':
            return [
                Real(0.01, 0.2, name='nu'),
                Categorical(['scale', 'auto'], name='gamma'),
                Categorical(['rbf', 'poly', 'sigmoid'], name='kernel')
            ]
        
        elif model_type == 'autoencoder':
            return [
                Integer(8, 64, name='latent_dim'),
                Real(0.0001, 0.01, name='learning_rate'),
                Real(0.0, 0.3, name='dropout_rate'),
                Integer(50, 200, name='epochs')
            ]
        
        else:
            raise ValueError(f"No search space defined for {model_type}")
```

---

## Inference System

### Real-time Prediction API

```python
class MLInferenceEngine:
    """Real-time ML inference engine for aNEOS."""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_cache = {}
        self.prediction_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.inference_stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'average_latency': 0.0
        }
        
    def load_model(self, model_id, model_path=None):
        """Load a trained model for inference."""
        
        if model_id in self.loaded_models:
            logger.debug(f"Model {model_id} already loaded")
            return
        
        try:
            if model_path is None:
                model_path = self._get_model_path(model_id)
            
            # Load model based on type
            model_metadata = self._load_model_metadata(model_path)
            model_type = model_metadata['model_type']
            
            if model_type == 'ensemble':
                model = self._load_ensemble_model(model_path)
            else:
                model = self._load_single_model(model_path, model_type)
            
            self.loaded_models[model_id] = model
            logger.info(f"Loaded model {model_id} ({model_type})")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def predict(self, neo_data, model_id=None, use_cache=True):
        """Make ML prediction for NEO data."""
        
        start_time = time.time()
        
        try:
            # Use default model if none specified
            if model_id is None:
                model_id = self._get_default_model_id()
            
            # Check cache
            cache_key = self._generate_cache_key(neo_data, model_id)
            if use_cache and cache_key in self.prediction_cache:
                self.inference_stats['cache_hits'] += 1
                return self.prediction_cache[cache_key]
            
            # Load model if needed
            if model_id not in self.loaded_models:
                self.load_model(model_id)
            
            model = self.loaded_models[model_id]
            
            # Prepare features
            feature_vector = self._prepare_features(neo_data)
            
            # Make prediction
            if isinstance(model, ModelEnsemble):
                prediction_result = self._predict_with_ensemble(model, feature_vector)
            else:
                prediction_result = self._predict_with_single_model(model, feature_vector)
            
            # Cache result
            if use_cache:
                self.prediction_cache[cache_key] = prediction_result
            
            # Update statistics
            latency = time.time() - start_time
            self._update_inference_stats(latency)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed for {neo_data.designation}: {e}")
            raise
    
    def _prepare_features(self, neo_data):
        """Prepare feature vector from NEO data."""
        
        # Run scientific indicators
        from aneos_core.analysis.pipeline import create_analysis_pipeline
        pipeline = create_analysis_pipeline()
        
        # Get indicator results
        indicator_results = []
        for indicator in pipeline.indicators:
            try:
                result = indicator.safe_evaluate(neo_data)
                indicator_results.append(result)
            except Exception as e:
                logger.warning(f"Indicator {indicator.name} failed: {e}")
        
        # Create feature vector
        feature_vector = FeatureVector()
        feature_vector.extract_features(neo_data, indicator_results)
        
        return feature_vector
    
    def _predict_with_ensemble(self, ensemble, feature_vector):
        """Make prediction with ensemble model."""
        
        # Convert feature vector to array
        features = np.array([feature_vector.to_array()])
        
        # Get ensemble prediction
        anomaly_score = float(ensemble.predict(features)[0])
        
        # Get individual model contributions
        model_contributions = ensemble.get_model_contributions(features)
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(ensemble, features, anomaly_score)
        
        return PredictionResult(
            designation=feature_vector.designation,
            anomaly_score=anomaly_score,
            anomaly_probability=self._score_to_probability(anomaly_score),
            is_anomaly=anomaly_score > 0.5,
            confidence=confidence,
            model_id=ensemble.ensemble_id,
            feature_contributions=self._calculate_feature_contributions(ensemble, features),
            predicted_at=datetime.now()
        )
    
    def _predict_with_single_model(self, model, feature_vector):
        """Make prediction with single model."""
        
        # Convert feature vector to array
        features = np.array([feature_vector.to_array()])
        
        # Make prediction
        anomaly_score = float(model.predict(features)[0])
        
        # Calculate probability and confidence
        anomaly_probability = float(model.predict_proba(features)[0]) if hasattr(model, 'predict_proba') else self._score_to_probability(anomaly_score)
        confidence = self._calculate_single_model_confidence(model, features, anomaly_score)
        
        return PredictionResult(
            designation=feature_vector.designation,
            anomaly_score=anomaly_score,
            anomaly_probability=anomaly_probability,
            is_anomaly=anomaly_score > 0.5,
            confidence=confidence,
            model_id=model.model_id,
            predicted_at=datetime.now()
        )
    
    def batch_predict(self, neo_data_list, model_id=None, batch_size=32):
        """Batch prediction for multiple NEOs."""
        
        results = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(neo_data_list), batch_size):
            batch = neo_data_list[i:i+batch_size]
            
            batch_results = []
            for neo_data in batch:
                try:
                    result = self.predict(neo_data, model_id)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch prediction failed for {neo_data.designation}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def get_inference_statistics(self):
        """Get inference engine statistics."""
        
        return {
            'loaded_models': len(self.loaded_models),
            'cache_size': len(self.prediction_cache),
            'total_predictions': self.inference_stats['total_predictions'],
            'cache_hit_rate': self.inference_stats['cache_hits'] / max(self.inference_stats['total_predictions'], 1),
            'average_latency_ms': self.inference_stats['average_latency'] * 1000
        }
```

### Streaming Inference

```python
class StreamingMLProcessor:
    """Real-time streaming ML processor for continuous NEO analysis."""
    
    def __init__(self, inference_engine, stream_config):
        self.inference_engine = inference_engine
        self.stream_config = stream_config
        self.processing_queue = asyncio.Queue(maxsize=stream_config.get('queue_size', 1000))
        self.result_handlers = []
        self.is_running = False
        
    async def start_streaming(self):
        """Start streaming processing."""
        
        self.is_running = True
        
        # Start processing tasks
        tasks = []
        
        # Input processing task
        tasks.append(asyncio.create_task(self._input_processor()))
        
        # ML processing workers
        n_workers = self.stream_config.get('n_workers', 4)
        for i in range(n_workers):
            tasks.append(asyncio.create_task(self._ml_worker(f"worker_{i}")))
        
        # Output processing task
        tasks.append(asyncio.create_task(self._output_processor()))
        
        # Run all tasks
        await asyncio.gather(*tasks)
    
    async def _input_processor(self):
        """Process incoming NEO data stream."""
        
        while self.is_running:
            try:
                # Get NEO data from input stream
                neo_data = await self._get_next_neo_data()
                
                if neo_data:
                    # Add to processing queue
                    await self.processing_queue.put({
                        'neo_data': neo_data,
                        'timestamp': datetime.now(),
                        'processing_id': self._generate_processing_id()
                    })
                
                # Rate limiting
                await asyncio.sleep(self.stream_config.get('input_interval', 0.1))
                
            except Exception as e:
                logger.error(f"Input processor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _ml_worker(self, worker_id):
        """ML processing worker."""
        
        while self.is_running:
            try:
                # Get item from queue
                item = await asyncio.wait_for(self.processing_queue.get(), timeout=5.0)
                
                # Make ML prediction
                prediction_result = await self._async_predict(item['neo_data'])
                
                # Add metadata
                prediction_result.processing_metadata = {
                    'worker_id': worker_id,
                    'processing_id': item['processing_id'],
                    'queue_time': (datetime.now() - item['timestamp']).total_seconds()
                }
                
                # Send to output handlers
                await self._handle_prediction_result(prediction_result)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # No items in queue, continue
                continue
            except Exception as e:
                logger.error(f"ML worker {worker_id} error: {e}")
    
    async def _async_predict(self, neo_data):
        """Asynchronous ML prediction wrapper."""
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def sync_predict():
            return self.inference_engine.predict(neo_data)
        
        result = await loop.run_in_executor(None, sync_predict)
        return result
    
    async def _handle_prediction_result(self, result):
        """Handle prediction result through registered handlers."""
        
        for handler in self.result_handlers:
            try:
                await handler(result)
            except Exception as e:
                logger.error(f"Result handler error: {e}")
    
    def add_result_handler(self, handler):
        """Add result handler for streaming predictions."""
        self.result_handlers.append(handler)
    
    async def stop_streaming(self):
        """Stop streaming processing."""
        self.is_running = False
        
        # Wait for queue to be processed
        await self.processing_queue.join()
```

---

This completes the comprehensive ML Documentation for aNEOS. The framework provides advanced machine learning capabilities for anomaly detection while maintaining scientific rigor and operational reliability.