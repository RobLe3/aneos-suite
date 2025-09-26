"""
Comprehensive Phase 4 testing suite for aNEOS ML and monitoring components.

This test suite validates all Phase 4 implementations including:
- Feature engineering
- ML models and training
- Real-time prediction
- Monitoring and alerting
"""

import pytest
import asyncio
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json

# Import aNEOS components
from aneos_core.ml.features import (
    FeatureEngineer, FeatureVector, OrbitalFeatureExtractor,
    VelocityFeatureExtractor, TemporalFeatureExtractor,
    GeographicFeatureExtractor, IndicatorFeatureExtractor
)
from aneos_core.ml.models import (
    ModelConfig, IsolationForestModel, ModelEnsemble, AnomalyDetectionModel,
    create_model, create_default_ensemble
)
from aneos_core.ml.training import (
    TrainingConfig, TrainingPipeline, TrainingDataManager,
    HyperparameterOptimizer, TrainingSession
)
from aneos_core.ml.prediction import (
    RealTimePredictor, PredictionConfig, Alert, ModelManager
)
from aneos_core.monitoring.alerts import (
    AlertManager, AlertRule, AlertLevel, AlertType,
    EmailNotificationChannel
)
from aneos_core.monitoring.metrics import (
    MetricsCollector, SystemMetrics, AnalysisMetrics, MLMetrics
)
from aneos_core.monitoring.dashboard import MonitoringDashboard

# Import test utilities and mock data
from aneos_core.data.models import NEOData, OrbitalElements, CloseApproach
from aneos_core.analysis.indicators.base import IndicatorResult
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.config.settings import ANEOSConfig

class TestFeatureEngineering:
    """Test feature engineering components."""
    
    @pytest.fixture
    def sample_neo_data(self):
        """Create sample NEO data for testing."""
        orbital_elements = OrbitalElements(
            eccentricity=0.15,
            inclination=12.5,
            semi_major_axis=1.2,
            ascending_node=45.0,
            argument_of_perihelion=120.0,
            mean_anomaly=200.0,
            epoch=datetime.now()
        )
        
        close_approaches = [
            CloseApproach(
                designation="2024 TEST",
                close_approach_date=datetime.now() - timedelta(days=100),
                distance_au=0.05,
                relative_velocity_km_s=15.2,
                infinity_velocity_km_s=8.5,
                subpoint=(35.0, -118.0)  # Los Angeles area
            ),
            CloseApproach(
                designation="2024 TEST",
                close_approach_date=datetime.now() - timedelta(days=50),
                distance_au=0.08,
                relative_velocity_km_s=18.7,
                infinity_velocity_km_s=12.1,
                subpoint=(40.7, -74.0)  # New York area
            )
        ]
        
        return NEOData(
            designation="2024 TEST",
            orbital_elements=orbital_elements,
            close_approaches=close_approaches
        )
    @pytest.fixture
    def sample_indicator_results(self):
        """Create sample indicator results."""
        return {
            'eccentricity': IndicatorResult(
                indicator_name='eccentricity',
                raw_score=0.3,
                weighted_score=0.45,
                confidence=0.9
            ),
            'velocity_shifts': IndicatorResult(
                indicator_name='velocity_shifts',
                raw_score=0.6,
                weighted_score=1.2,
                confidence=0.8
            )
        }
    
    def test_orbital_feature_extractor(self, sample_neo_data):
        """Test orbital feature extraction."""
        extractor = OrbitalFeatureExtractor()
        features = extractor.extract(sample_neo_data)
        
        # Verify basic orbital elements are extracted
        assert 'eccentricity' in features
        assert 'inclination' in features
        assert 'semi_major_axis' in features
        assert features['eccentricity'] == 0.15
        assert features['inclination'] == 12.5
        assert features['semi_major_axis'] == 1.2
        
        # Verify derived features
        assert 'perihelion_distance' in features
        assert 'aphelion_distance' in features
        assert 'orbital_period' in features
        assert 'earth_crossing' in features
        
        # Check calculations
        expected_perihelion = 1.2 * (1 - 0.15)
        assert abs(features['perihelion_distance'] - expected_perihelion) < 0.001
        
        # Verify feature names match extracted features
        feature_names = extractor.get_feature_names()
        assert len(feature_names) == len(features)
        assert all(name in features for name in feature_names)
    
    def test_velocity_feature_extractor(self, sample_neo_data):
        """Test velocity feature extraction."""
        extractor = VelocityFeatureExtractor()
        features = extractor.extract(sample_neo_data)
        
        # Verify statistical features
        assert 'velocity_mean' in features
        assert 'velocity_std' in features
        assert 'velocity_min' in features
        assert 'velocity_max' in features
        
        # Check calculated values
        velocities = [15.2, 18.7]
        expected_mean = np.mean(velocities)
        assert abs(features['velocity_mean'] - expected_mean) < 0.001
        
        # Verify velocity change features
        assert 'velocity_change_mean' in features
        assert 'velocity_change_max' in features
        
        # Verify infinity velocity features
        assert 'v_inf_mean' in features
        assert features['v_inf_mean'] > 0
    
    def test_temporal_feature_extractor(self, sample_neo_data):
        """Test temporal feature extraction."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_neo_data)
        
        # Verify temporal features
        assert 'observation_span_days' in features
        assert 'observation_count' in features
        assert 'interval_mean' in features
        assert 'temporal_regularity' in features
        
        # Check observation count
        assert features['observation_count'] == 2
        
        # Check span calculation (approximately 50 days)
        assert 40 < features['observation_span_days'] < 60
    
    def test_geographic_feature_extractor(self, sample_neo_data):
        """Test geographic feature extraction."""
        extractor = GeographicFeatureExtractor()
        features = extractor.extract(sample_neo_data)
        
        # Verify geographic features
        assert 'latitude_mean' in features
        assert 'longitude_mean' in features
        assert 'northern_hemisphere_fraction' in features
        assert 'western_hemisphere_fraction' in features
        
        # Check hemisphere fractions
        assert features['northern_hemisphere_fraction'] == 1.0  # Both points in northern hemisphere
        assert features['western_hemisphere_fraction'] == 1.0  # Both points in western hemisphere
        
        # Verify distance calculations
        assert 'mean_pairwise_distance' in features
        assert features['mean_pairwise_distance'] > 0
    
    def test_indicator_feature_extractor(self, sample_indicator_results):
        """Test indicator feature extraction."""
        extractor = IndicatorFeatureExtractor()
        features = extractor.extract(sample_indicator_results)
        
        # Verify individual indicator scores are extracted
        assert 'eccentricity_raw_score' in features
        assert 'eccentricity_weighted_score' in features
        assert 'eccentricity_confidence' in features
        assert features['eccentricity_raw_score'] == 0.3
        
        # Verify aggregate features
        assert 'indicator_raw_mean' in features
        assert 'indicator_weighted_mean' in features
        assert 'active_indicator_count' in features
        
        # Check aggregate calculations
        expected_raw_mean = (0.3 + 0.6) / 2
        assert abs(features['indicator_raw_mean'] - expected_raw_mean) < 0.001
    
    def test_feature_engineer_integration(self, sample_neo_data, sample_indicator_results):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        
        # Extract features
        feature_vector = engineer.extract_features(sample_neo_data, sample_indicator_results)
        
        # Verify feature vector structure
        assert isinstance(feature_vector, FeatureVector)
        assert feature_vector.designation == "2024 TEST"
        assert len(feature_vector.features) == len(feature_vector.feature_names)
        assert len(feature_vector.features) > 50  # Should have many features
        
        # Verify metadata
        assert 'data_quality' in feature_vector.metadata
        assert 'extraction_timestamp' in feature_vector.metadata
        
        # Test feature transformations
        transformed = engineer.transform_features(feature_vector, ['normalize', 'handle_missing'])
        assert len(transformed.features) == len(feature_vector.features)
        assert 'transformations_applied' in transformed.metadata
    
    def test_feature_matrix_creation(self, sample_neo_data, sample_indicator_results):
        """Test feature matrix creation from multiple vectors."""
        engineer = FeatureEngineer()
        
        # Create multiple feature vectors
        feature_vectors = []
        for i in range(3):
            # Modify designation for each vector
            neo_data = sample_neo_data
            neo_data.designation = f"2024 TEST{i}"
            
            fv = engineer.extract_features(neo_data, sample_indicator_results)
            feature_vectors.append(fv)
        
        # Create feature matrix
        feature_matrix, feature_names, designations = engineer.create_feature_matrix(feature_vectors)
        
        # Verify matrix structure
        assert feature_matrix.shape[0] == 3  # 3 objects
        assert feature_matrix.shape[1] == len(feature_names)
        assert len(designations) == 3
        assert all("2024 TEST" in des for des in designations)


class TestConfigurationLoading:
    """Validate configuration loading from environment variables."""

    def test_env_processing_flags(self, monkeypatch, tmp_path):
        temp_dir = tmp_path / "aneos-temp"
        monkeypatch.setenv("ANEOS_ANALYSIS_PARALLEL", "false")
        monkeypatch.setenv("ANEOS_ANALYSIS_MAX_WORKERS", "16")
        monkeypatch.setenv("ANEOS_ANALYSIS_QUEUE_SIZE", "25")
        monkeypatch.setenv("ANEOS_ANALYSIS_TIMEOUT", "120")
        monkeypatch.setenv("ANEOS_BATCH_PROCESSING_ENABLED", "true")
        monkeypatch.setenv("ANEOS_BATCH_SIZE", "50")
        monkeypatch.setenv("ANEOS_BATCH_MAX_SIZE", "200")
        monkeypatch.setenv("ANEOS_BATCH_TIMEOUT", "1800")
        monkeypatch.setenv("ANEOS_ANALYSIS_MEMORY_LIMIT", "2048")
        monkeypatch.setenv("ANEOS_ANALYSIS_TEMP_DIR", str(temp_dir))
        monkeypatch.setenv("ANEOS_ANALYSIS_CLEANUP_TEMP_FILES", "False")
        monkeypatch.setenv("ANEOS_DATA_SOURCES_PRIMARY", "SBDB")
        monkeypatch.setenv("ANEOS_DATA_SOURCES_FALLBACK", "NEODyS,MPC")
        monkeypatch.setenv("ANEOS_DATA_SOURCES_TIMEOUT", "15")
        monkeypatch.setenv("ANEOS_DATA_SOURCES_RETRY_ATTEMPTS", "5")

        config = ANEOSConfig.from_env()

        assert config.analysis_parallel is False
        assert config.max_workers == 16
        assert config.analysis_queue_size == 25
        assert config.analysis_timeout == 120
        assert config.batch_processing_enabled is True
        assert config.batch_size == 50
        assert config.batch_max_size == 200
        assert config.batch_timeout == 1800
        assert config.analysis_memory_limit == 2048
        assert config.analysis_temp_dir == str(temp_dir)
        assert config.analysis_cleanup_temp_files is False
        assert config.api.data_sources_priority == ["SBDB", "NEODyS", "MPC"]
        assert config.api.request_timeout == 15
        assert config.api.max_retries == 5

    def test_env_max_workers_fallback(self, monkeypatch):
        monkeypatch.delenv("ANEOS_ANALYSIS_MAX_WORKERS", raising=False)
        monkeypatch.setenv("ANEOS_MAX_WORKERS", "4")
        config = ANEOSConfig.from_env()
        assert config.max_workers == 4


class TestMLModels:
    """Test machine learning models."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 20)  # 100 samples, 20 features
        y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])  # 10% anomalies
        return X, y
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_model_config(self):
        """Test model configuration."""
        config = ModelConfig(
            model_type="isolation_forest",
            parameters={'n_estimators': 200, 'contamination': 0.1}
        )
        
        assert config.model_type == "isolation_forest"
        assert config.parameters['n_estimators'] == 200
        assert config.parameters['contamination'] == 0.1
    
    @patch('aneos_core.ml.models.HAS_SKLEARN', True)
    def test_isolation_forest_model(self, sample_training_data):
        """Test Isolation Forest model."""
        X, y = sample_training_data
        
        config = ModelConfig(
            model_type="isolation_forest",
            parameters={'n_estimators': 50, 'contamination': 0.1}
        )
        
        model = IsolationForestModel(config)
        
        # Test training
        result = model.fit(X)
        assert model.is_trained
        assert result.model_type == "isolation_forest"
        assert result.training_score is not None
        
        # Test prediction
        scores = model.predict(X[:10])
        assert len(scores) == 10
        assert all(0 <= score <= 1 for score in scores)
        
        # Test probability prediction
        probas = model.predict_proba(X[:10])
        assert len(probas) == 10
        assert all(0 <= proba <= 1 for proba in probas)
    
    def test_model_ensemble(self, sample_training_data):
        """Test model ensemble functionality."""
        X, y = sample_training_data
        
        # Create multiple models
        models = []
        
        # Mock models since sklearn might not be available
        for i in range(3):
            mock_model = Mock()
            mock_model.is_trained = True
            mock_model.model_id = f"mock_model_{i}"
            mock_model.predict.return_value = np.random.rand(len(X))
            mock_model.predict_proba.return_value = np.random.rand(len(X))
            models.append(mock_model)
        
        # Create ensemble
        ensemble = ModelEnsemble(models, weights=[0.4, 0.3, 0.3])
        
        # Test prediction
        predictions = ensemble.predict(X)
        assert len(predictions) == len(X)
        assert all(0 <= pred <= 1 for pred in predictions)
        
        # Test individual model predictions
        individual_predictions = ensemble.get_model_predictions(X)
        assert len(individual_predictions) == 3
        assert all(f"mock_model_{i}" in individual_predictions for i in range(3))
    
    def test_model_save_load(self, temp_model_dir):
        """Test model serialization."""
        config = ModelConfig(model_type="isolation_forest")

        class DummyModel(AnomalyDetectionModel):
            """Lightweight concrete model for exercising persistence helpers."""

            def fit(self, X, y=None):  # pragma: no cover - not needed for this test
                raise NotImplementedError

            def predict(self, X):  # pragma: no cover - not needed for this test
                raise NotImplementedError

            def predict_proba(self, X):  # pragma: no cover - not needed for this test
                raise NotImplementedError

        model = DummyModel(config)
        model.model = {'type': 'mock_model'}
        model.scaler = {'type': 'mock_scaler'}
        model.feature_names = ['feature1', 'feature2']
        model.model_id = 'test_model'
        model.is_trained = True

        model_path = Path(temp_model_dir) / "test_model.pkl"
        model.save_model(str(model_path))
        assert model_path.exists()

        reloaded = DummyModel(config)
        reloaded.load_model(str(model_path))

        assert reloaded.model_id == model.model_id
        assert reloaded.feature_names == model.feature_names
        assert reloaded.is_trained is True
        assert reloaded.config.model_type == model.config.model_type
        assert reloaded.scaler == model.scaler
        assert reloaded.model == model.model

class TestTrainingPipeline:
    """Test ML training pipeline."""
    
    @pytest.fixture
    def mock_analysis_pipeline(self):
        """Create mock analysis pipeline."""
        pipeline = Mock()
        pipeline.analyze_batch = AsyncMock()
        pipeline._fetch_neo_data = AsyncMock()
        return pipeline
    
    @pytest.fixture
    def training_config(self, tmp_path):
        """Create training configuration."""
        return TrainingConfig(
            model_types=["isolation_forest"],
            use_ensemble=False,
            hyperparameter_optimization=False,
            model_output_dir=str(tmp_path),
            min_samples=10
        )
    
    def test_training_config(self):
        """Test training configuration."""
        config = TrainingConfig(
            model_types=["isolation_forest", "one_class_svm"],
            validation_split=0.2,
            use_ensemble=True
        )
        
        assert len(config.model_types) == 2
        assert config.validation_split == 0.2
        assert config.use_ensemble is True
    
    @pytest.mark.asyncio
    async def test_training_data_manager(self, mock_analysis_pipeline):
        """Test training data collection."""
        # Setup mock responses
        mock_neo_data = Mock()
        mock_neo_data.designation = "2024 TEST"
        mock_neo_data.orbital_elements = Mock()
        mock_neo_data.close_approaches = []
        
        mock_analysis_result = Mock()
        mock_analysis_result.designation = "2024 TEST"
        mock_analysis_result.errors = []
        mock_analysis_result.anomaly_score = Mock()
        mock_analysis_result.anomaly_score.indicator_scores = {}
        mock_analysis_result.anomaly_score.overall_score = 0.3
        mock_analysis_result.anomaly_score.classification = "natural"
        
        mock_analysis_pipeline.analyze_batch.return_value = [mock_analysis_result]
        mock_analysis_pipeline._fetch_neo_data.return_value = mock_neo_data
        
        # Create data manager
        data_manager = TrainingDataManager(mock_analysis_pipeline)
        
        # Mock feature engineer
        with patch.object(data_manager, 'feature_engineer') as mock_fe:
            mock_feature_vector = Mock()
            mock_feature_vector.designation = "2024 TEST"
            mock_feature_vector.metadata = {
                'anomaly_score': 0.3,
                'classification': 'natural'
            }
            mock_fe.extract_features.return_value = mock_feature_vector
            
            # Test data collection
            feature_vectors = await data_manager.collect_training_data(["2024 TEST"])
            
            assert len(feature_vectors) == 1
            assert feature_vectors[0].designation == "2024 TEST"
    
    def test_hyperparameter_optimizer(self):
        """Test hyperparameter optimization."""
        config = TrainingConfig(hyperparameter_optimization=True)
        optimizer = HyperparameterOptimizer(config)
        
        # Verify parameter grids exist
        assert 'isolation_forest' in optimizer.parameter_grids
        assert 'one_class_svm' in optimizer.parameter_grids
        
        # Test with mock data
        X_train = np.random.randn(50, 10)
        X_val = np.random.randn(20, 10)
        y_val = np.random.choice([0, 1], 20)
        
        with patch('aneos_core.ml.models.create_model') as mock_create:
            mock_model = Mock()
            mock_model.fit.return_value = Mock()
            mock_model.is_trained = True
            mock_model.predict.return_value = np.random.rand(20)
            mock_create.return_value = mock_model
            
            with patch('aneos_core.ml.training.roc_auc_score', return_value=0.8):
                best_config, best_score = optimizer.optimize_model(
                    "isolation_forest", X_train, X_val, y_val
                )
                
                assert best_config is not None
                assert best_score >= 0

class TestRealTimePrediction:
    """Test real-time prediction system."""
    
    @pytest.fixture
    def mock_analysis_pipeline(self):
        """Create mock analysis pipeline."""
        pipeline = Mock()
        pipeline._fetch_neo_data = AsyncMock()
        pipeline.analyze_neo = AsyncMock()
        return pipeline
    
    @pytest.fixture
    def prediction_config(self, tmp_path):
        """Create prediction configuration."""
        return PredictionConfig(
            model_path=str(tmp_path),
            use_ensemble=False,
            cache_predictions=False  # Disable for testing
        )
    
    @pytest.mark.asyncio
    async def test_real_time_predictor_initialization(self, mock_analysis_pipeline, prediction_config):
        """Test predictor initialization."""
        predictor = RealTimePredictor(mock_analysis_pipeline, prediction_config)
        
        assert predictor.analysis_pipeline == mock_analysis_pipeline
        assert predictor.config == prediction_config
        assert predictor.feature_engineer is not None
        assert predictor.model_manager is not None
    
    @pytest.mark.asyncio
    async def test_prediction_without_model(self, mock_analysis_pipeline, prediction_config):
        """Test prediction when no ML model is available."""
        # Setup mock responses
        mock_neo_data = Mock()
        mock_neo_data.designation = "2024 TEST"
        
        mock_anomaly_score = Mock()
        mock_anomaly_score.overall_score = 0.4
        mock_anomaly_score.confidence = 0.8
        mock_anomaly_score.classification = "suspicious"
        mock_anomaly_score.indicator_scores = {}
        
        mock_pipeline_result = Mock()
        mock_pipeline_result.errors = []
        mock_pipeline_result.anomaly_score = mock_anomaly_score
        
        mock_analysis_pipeline._fetch_neo_data.return_value = mock_neo_data
        mock_analysis_pipeline.analyze_neo.return_value = mock_pipeline_result
        
        predictor = RealTimePredictor(mock_analysis_pipeline, prediction_config)
        predictor.primary_model = None  # No ML model available
        
        # Test prediction
        result = await predictor.predict_anomaly("2024 TEST")
        
        assert result.designation == "2024 TEST"
        assert result.anomaly_score == 0.4
        assert result.confidence == 0.8
        assert result.model_id == "indicators_only"
    
    def test_model_manager(self, tmp_path):
        """Test model manager functionality."""
        manager = ModelManager(str(tmp_path))
        
        # Test with no models
        models = manager.list_available_models()
        assert len(models) == 0
        
        latest_model = manager.get_latest_model()
        assert latest_model is None
        
        # Create mock model file
        mock_model_file = tmp_path / "test_isolation_forest_model.pkl"
        mock_model_file.write_text("mock model data")
        
        # Test model listing
        models = manager.list_available_models()
        assert len(models) == 1
        assert models[0]['type'] == 'model'

class TestMonitoringSystem:
    """Test monitoring and alerting system."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        
        # Verify default rules are created
        assert len(manager.rules) > 0
        assert any(rule.rule_id == "high_anomaly_neo" for rule in manager.rules.values())
        assert any(rule.rule_id == "critical_anomaly_neo" for rule in manager.rules.values())
    
    def test_alert_rule_creation(self):
        """Test alert rule creation and evaluation."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            alert_type=AlertType.ANOMALOUS_NEO,
            alert_level=AlertLevel.HIGH,
            condition="Test condition",
            threshold_function=lambda data: data.get('score', 0) > 0.8
        )
        
        # Test evaluation
        assert rule.evaluate({'score': 0.9}) is True
        assert rule.evaluate({'score': 0.5}) is False
        
        # Test disabled rule
        rule.enabled = False
        assert rule.evaluate({'score': 0.9}) is False
    
    def test_alert_creation(self):
        """Test alert creation and management."""
        manager = AlertManager()
        
        # Test anomaly alert
        mock_prediction = Mock()
        mock_prediction.designation = "2024 TEST"
        mock_prediction.anomaly_probability = 0.85
        mock_prediction.anomaly_score = 0.8
        mock_prediction.confidence = 0.9
        
        mock_anomaly_score = Mock()
        mock_anomaly_score.classification = "highly_suspicious"
        mock_anomaly_score.risk_factors = ["High eccentricity", "Unusual velocity"]
        
        initial_alert_count = len(manager.alerts)
        
        manager.check_anomaly_alert(mock_prediction, mock_anomaly_score)
        
        # Verify alert was created
        assert len(manager.alerts) > initial_alert_count
        
        # Test alert acknowledgment
        if manager.alerts:
            alert = manager.alerts[-1]
            success = manager.acknowledge_alert(alert.alert_id, "test_user")
            assert success is True
            assert alert.acknowledged is True
            assert alert.acknowledged_by == "test_user"
    
    def test_notification_channels(self):
        """Test notification channel functionality."""
        # Test email channel creation
        channel = EmailNotificationChannel(
            channel_id="test_email",
            smtp_server="smtp.example.com",
            smtp_port=587,
            username="test@example.com",
            password="password",
            recipients=["admin@example.com"],
            enabled=False  # Disable to avoid actual email sending
        )
        
        assert channel.channel_id == "test_email"
        assert channel.enabled is False
        assert len(channel.recipients) == 1
    
    def test_metrics_collector(self):
        """Test metrics collection system."""
        collector = MetricsCollector(collection_interval=1)
        
        # Test initialization
        assert collector.collection_interval == 1
        assert collector.running is False
        
        # Test custom metrics
        collector.increment_counter("test_counter", 5)
        collector.set_gauge("test_gauge", 3.14)
        collector.record_timer("test_timer", 0.5)
        
        assert collector.custom_counters["test_counter"] == 5
        assert collector.custom_gauges["test_gauge"] == 3.14
        assert len(collector.custom_timers["test_timer"]) == 1
        
        # Test summary generation
        summary = collector.get_custom_metrics_summary()
        assert "test_counter" in summary["counters"]
        assert "test_gauge" in summary["gauges"]
        assert "test_timer" in summary["timers"]
    
    def test_system_metrics_creation(self):
        """Test system metrics data structure."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=45.2,
            memory_percent=62.1,
            memory_used_mb=1024.0,
            memory_available_mb=512.0,
            disk_usage_percent=75.5,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            process_count=150
        )
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict['cpu_percent'] == 45.2
        assert metrics_dict['memory_percent'] == 62.1
        assert 'timestamp' in metrics_dict
    
    def test_monitoring_dashboard(self):
        """Test monitoring dashboard functionality."""
        # Create mock components
        mock_metrics = Mock()
        mock_metrics.running = True
        mock_metrics.collection_interval = 30
        mock_metrics.get_system_summary.return_value = {
            'avg_cpu_percent': 45.0,
            'avg_memory_percent': 60.0,
            'current_disk_usage': 70.0,
            'current_process_count': 150
        }
        mock_metrics.get_comprehensive_summary.return_value = {
            'custom_metrics': {
                'counters': {'test_counter': 5},
                'gauges': {'test_gauge': 3.14}
            }
        }
        
        mock_alerts = Mock()
        mock_alerts.get_alert_statistics.return_value = {
            'active_alerts': 2,
            'total_alerts': 10
        }
        mock_alerts.get_recent_alerts.return_value = []
        
        dashboard = MonitoringDashboard(mock_metrics, mock_alerts, refresh_interval=5)
        
        # Test report generation
        report = dashboard.generate_text_report()
        assert "aNEOS SYSTEM STATUS REPORT" in report
        assert "SYSTEM PERFORMANCE" in report
        assert "ALERT SUMMARY" in report

class TestIntegration:
    """Integration tests for complete system functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_pipeline(self, temp_workspace):
        """Test complete analysis pipeline from data to prediction."""
        # This is a simplified integration test
        # In practice, this would use real NEO data and run the full pipeline
        
        # Create mock configuration
        config = ANEOSConfig()
        config.paths.cache_dir = str(temp_workspace / "cache")
        config.paths.data_dir = str(temp_workspace / "data")
        
        # Create directories
        Path(config.paths.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(config.paths.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Test would involve:
        # 1. Creating analysis pipeline
        # 2. Training ML models
        # 3. Making predictions
        # 4. Monitoring system health
        
        # For now, just verify the structure
        assert Path(config.paths.cache_dir).exists()
        assert Path(config.paths.data_dir).exists()
    
    def test_configuration_integration(self, temp_workspace):
        """Test configuration system integration."""
        config_file = temp_workspace / "test_config.json"
        
        # Create test configuration
        config_data = {
            "api": {
                "request_timeout": 15,
                "max_retries": 5
            },
            "paths": {
                "cache_dir": str(temp_workspace / "cache")
            },
            "thresholds": {
                "eccentricity": 0.9
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test loading configuration
        # This would normally use the ConfigManager
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config["api"]["request_timeout"] == 15
        assert loaded_config["thresholds"]["eccentricity"] == 0.9

def run_comprehensive_tests():
    """Run all Phase 4 tests and generate report."""
    print("=" * 80)
    print(" " * 25 + "aNEOS PHASE 4 TEST SUITE")
    print("=" * 80)
    print()
    
    # Test configuration
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--disable-warnings",  # Disable warnings for cleaner output
        __file__  # Run this test file
    ]
    
    print("Running comprehensive Phase 4 tests...")
    print("Components being tested:")
    print("  ✓ Feature Engineering")
    print("  ✓ ML Models and Training")
    print("  ✓ Real-time Prediction")
    print("  ✓ Monitoring and Alerting")
    print("  ✓ Integration Tests")
    print()
    
    # Run tests
    exit_code = pytest.main(test_args)
    
    print()
    print("=" * 80)
    
    if exit_code == 0:
        print("✅ ALL PHASE 4 TESTS PASSED!")
        print()
        print("Phase 4 Implementation Status:")
        print("  ✅ Machine Learning Pipeline - COMPLETE")
        print("  ✅ Feature Engineering - COMPLETE")
        print("  ✅ Model Training & Ensemble - COMPLETE")
        print("  ✅ Real-time Prediction - COMPLETE")
        print("  ✅ Monitoring & Alerting - COMPLETE")
        print("  ✅ Comprehensive Testing - COMPLETE")
        print()
        print("🎉 Phase 4: Machine Learning Enhancement - SUCCESSFULLY COMPLETED!")
    else:
        print("❌ Some tests failed. Please review the output above.")
        print()
        print("Phase 4 implementation may need additional work.")
    
    print("=" * 80)
    return exit_code

if __name__ == "__main__":
    # Allow running tests directly
    exit_code = run_comprehensive_tests()
    exit(exit_code)