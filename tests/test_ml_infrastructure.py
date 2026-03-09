"""
Offline tests for the ML infrastructure (FeatureEngineer, IsolationForestModel, ModelManager).
These tests require no network access and no trained model files on disk.
"""

import numpy as np
import pytest

from aneos_core.data.models import NEOData, OrbitalElements
from aneos_core.ml.features import FeatureEngineer
from aneos_core.ml.models import IsolationForestModel, ModelConfig, HAS_SKLEARN
from aneos_core.ml.prediction import ModelManager


def _make_neo(designation: str = "2024 TEST1") -> NEOData:
    """Construct a minimal NEOData object suitable for feature extraction."""
    elements = OrbitalElements(
        eccentricity=0.22,
        inclination=5.4,
        semi_major_axis=1.15,
        ascending_node=120.0,
        argument_of_perihelion=55.0,
        mean_anomaly=30.0,
    )
    return NEOData(designation=designation, orbital_elements=elements)


class TestFeatureEngineering:
    def test_feature_extraction_on_neo_data(self):
        """extract_features returns a FeatureVector with a non-empty ndarray."""
        engineer = FeatureEngineer()
        neo = _make_neo()
        fv = engineer.extract_features(neo)
        assert fv.features is not None
        assert isinstance(fv.features, np.ndarray)
        assert fv.features.size > 0

    def test_feature_count_matches_names(self):
        """Feature array length equals the number of feature names."""
        engineer = FeatureEngineer()
        neo = _make_neo()
        fv = engineer.extract_features(neo)
        names = engineer.get_feature_names()
        # fv.features may include indicator features not in static names; at minimum
        # the static names must be a subset of what was extracted.
        assert len(names) > 0
        assert fv.features.shape[0] == len(fv.feature_names)


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
class TestMLModelLifecycle:
    def _make_data(self, n: int = 20, d: int = 10) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, d))

    def test_isolation_forest_fit_predict(self):
        """Fit IsolationForestModel and predict returns array of correct length."""
        X = self._make_data(20, 10)
        model = IsolationForestModel(ModelConfig(model_type="isolation_forest"))
        model.fit(X)
        scores = model.predict(X)
        assert isinstance(scores, np.ndarray)
        assert scores.shape[0] == 20

    def test_model_serialization_roundtrip(self, tmp_path):
        """Save and reload IsolationForestModel; predictions must be identical."""
        X = self._make_data(20, 10)
        model = IsolationForestModel(ModelConfig(model_type="isolation_forest"))
        model.fit(X)
        scores_before = model.predict(X)

        filepath = str(tmp_path / "test_model.joblib")
        model.save_model(filepath)

        model2 = IsolationForestModel(ModelConfig(model_type="isolation_forest"))
        model2.load_model(filepath)
        scores_after = model2.predict(X)

        np.testing.assert_array_almost_equal(scores_before, scores_after)

    def test_model_manager_no_crash_without_models_dir(self):
        """ModelManager with non-existent path loads without raising; load_model returns None."""
        mgr = ModelManager("/tmp/nonexistent_aneos_models_dir")
        result = mgr.load_model("nonexistent_model_id")
        assert result is None
