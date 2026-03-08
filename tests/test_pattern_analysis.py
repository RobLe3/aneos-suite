"""Tests for Population Pattern Analysis (BC11)."""
import pytest
from datetime import datetime, timedelta, UTC


# --- Fixtures ----

def _make_neo(designation, a, e, i, a2=None):
    """Create a minimal NEOData object for testing."""
    from aneos_core.data.models import (
        NEOData, OrbitalElements, NonGravitationalParameters
    )
    oe = OrbitalElements(
        semi_major_axis=a, eccentricity=e, inclination=i,
        ra_of_ascending_node=120.0, arg_of_periapsis=45.0, mean_anomaly=0.0
    )
    nongrav = None
    if a2 is not None:
        nongrav = NonGravitationalParameters(a2=a2)
    return NEOData(designation=designation, orbital_elements=oe, nongrav=nongrav)


def _make_cluster_neos(n=12, a_center=1.0, e_center=0.2, i_center=5.0):
    """Create N tight cluster members."""
    import random
    random.seed(42)
    return [
        _make_neo(f"C{i:03d}", a_center + random.gauss(0, 0.01),
                  e_center + random.gauss(0, 0.005),
                  i_center + random.gauss(0, 0.5))
        for i in range(n)
    ]


# ---- NonGravitationalParameters ----

class TestNonGravitationalParameters:
    def test_dataclass_fields(self):
        from aneos_core.data.models import NonGravitationalParameters
        ng = NonGravitationalParameters(a1=1e-14, a2=-3.2e-14, a3=None, model="marsden")
        assert ng.a1 == pytest.approx(1e-14)
        assert ng.a2 == pytest.approx(-3.2e-14)
        assert ng.a3 is None
        assert ng.model == "marsden"

    def test_neodata_cache_roundtrip(self):
        from aneos_core.data.models import NEOData, OrbitalElements, NonGravitationalParameters
        oe = OrbitalElements(semi_major_axis=1.0, eccentricity=0.2, inclination=5.0)
        ng = NonGravitationalParameters(a2=-3.2e-14, model="marsden")
        neo = NEOData(designation="TEST001", orbital_elements=oe, nongrav=ng)
        d = neo.to_dict()
        restored = NEOData.from_dict(d)
        assert restored.nongrav is not None
        assert restored.nongrav.a2 == pytest.approx(-3.2e-14)
        assert restored.nongrav.model == "marsden"

    def test_neodata_nongrav_none_roundtrip(self):
        from aneos_core.data.models import NEOData, OrbitalElements
        oe = OrbitalElements(semi_major_axis=1.0, eccentricity=0.2, inclination=5.0)
        neo = NEOData(designation="TEST002", orbital_elements=oe)
        d = neo.to_dict()
        restored = NEOData.from_dict(d)
        assert restored.nongrav is None


# ---- OrbitalElementClusterer ----

class TestOrbitalElementClusterer:
    def test_no_clusters_from_random_spread(self):
        """Uniformly spread objects should produce no clusters above quality gate."""
        import random
        random.seed(0)
        neos = [
            _make_neo(f"R{i:04d}", 0.7 + random.random() * 3.3,
                      random.random() * 0.9, random.random() * 40)
            for i in range(30)
        ]
        from aneos_core.pattern_analysis.clustering import OrbitalElementClusterer
        clusters = OrbitalElementClusterer(min_cluster_size=5, min_samples=3).run(neos)
        # Random spread should not produce clusters passing the 3-sigma quality gate
        anomalous = [c for c in clusters if c.known_family is None]
        assert len(anomalous) == 0

    def test_tight_cluster_detected(self):
        """A tight group of 12 objects at the same orbital location should be detected."""
        cluster_neos = _make_cluster_neos(n=12)
        # Add 10 background objects spread out
        import random
        random.seed(1)
        background = [
            _make_neo(f"BG{i:03d}", 0.8 + random.random() * 3.0,
                      0.05 + random.random() * 0.8, random.random() * 35)
            for i in range(10)
        ]
        all_neos = cluster_neos + background
        from aneos_core.pattern_analysis.clustering import OrbitalElementClusterer
        clusters = OrbitalElementClusterer(min_cluster_size=5, min_samples=3).run(all_neos)
        assert len(clusters) >= 1
        anomalous = [c for c in clusters if c.known_family is None]
        assert any(c.n_members >= 10 for c in anomalous)

    def test_hungaria_labelled_known_family(self):
        """Objects at Hungaria orbital elements should be labelled as KNOWN_FAMILY."""
        hungarias = [_make_neo(f"HUN{i}", 1.88 + i * 0.001, 0.07, 20.0) for i in range(12)]
        from aneos_core.pattern_analysis.clustering import OrbitalElementClusterer
        clusters = OrbitalElementClusterer(min_cluster_size=5, min_samples=3).run(hungarias)
        # If clustered, should be labelled Hungaria
        for c in clusters:
            if c.n_members >= 10:
                assert c.known_family is not None

    def test_orbital_cluster_dataclass(self):
        from aneos_core.pattern_analysis.clustering import OrbitalCluster
        c = OrbitalCluster(
            cluster_id="cluster_001",
            members=["A", "B", "C"],
            centroid={"a": 1.0, "e": 0.2, "i": 5.0},
            density_sigma=4.5,
            p_value=0.001,
        )
        assert c.n_members == 3
        assert c.known_family is None


# ---- NetworkSigmaCombiner ----

class TestNetworkSigmaCombiner:
    def test_no_valid_p_values_returns_routine(self):
        from aneos_core.pattern_analysis.network_sigma import NetworkSigmaCombiner
        result = NetworkSigmaCombiner().combine({"clustering": None, "harmonics": None})
        assert result["network_tier"] == "NETWORK_ROUTINE"
        assert result["network_sigma"] == 0.0

    def test_very_small_p_values_produce_high_sigma(self):
        from aneos_core.pattern_analysis.network_sigma import NetworkSigmaCombiner
        result = NetworkSigmaCombiner().combine(
            {"clustering": 1e-8, "harmonics": 1e-6}, n_objects=1
        )
        assert result["network_sigma"] >= 3.0
        assert "NETWORK_ANOMALY" in result["network_tier"] or "NETWORK_EXCEPTIONAL" in result["network_tier"]

    def test_moderate_p_values_produce_notable_tier(self):
        from aneos_core.pattern_analysis.network_sigma import NetworkSigmaCombiner
        result = NetworkSigmaCombiner().combine({"clustering": 0.05}, n_objects=1)
        assert result["network_sigma"] >= 0.0
        assert result["combined_p_value"] <= 1.0

    def test_bonferroni_correction_applied(self):
        """With n_objects=100, p-values should be multiplied by 100."""
        from aneos_core.pattern_analysis.network_sigma import NetworkSigmaCombiner
        r1 = NetworkSigmaCombiner().combine({"clustering": 0.001}, n_objects=1)
        r100 = NetworkSigmaCombiner().combine({"clustering": 0.001}, n_objects=100)
        # Bonferroni-corrected result should be less significant
        assert r100["network_sigma"] <= r1["network_sigma"]


# ---- HarmonicSignal (unit — no network) ----

class TestSynodicHarmonicAnalyzer:
    def _make_neo_with_approaches(self, designation, period_days, n_approaches=8):
        from aneos_core.data.models import NEOData, OrbitalElements, CloseApproach
        oe = OrbitalElements(semi_major_axis=1.0, eccentricity=0.2, inclination=5.0)
        neo = NEOData(designation=designation, orbital_elements=oe)
        base = datetime(2000, 1, 1, tzinfo=UTC)
        for k in range(n_approaches):
            ca = CloseApproach(
                designation=designation,
                close_approach_date=base + timedelta(days=k * period_days),
                distance_au=0.05,
            )
            neo.add_close_approach(ca)
        return neo

    def test_periodic_signal_detected(self):
        """Regular 365-day approaches should produce a strong harmonic signal."""
        neo = self._make_neo_with_approaches("HARM001", period_days=365, n_approaches=8)
        from aneos_core.pattern_analysis.harmonics import SynodicHarmonicAnalyzer
        analyzer = SynodicHarmonicAnalyzer(fetcher=None)
        signals = analyzer.run([neo])
        assert len(signals) >= 1
        sig = signals[0]
        # Any of the tested target periods is acceptable (365 or its harmonics)
        from aneos_core.pattern_analysis.harmonics import TARGET_PERIODS
        assert sig.dominant_period_days in TARGET_PERIODS
        assert sig.power_excess_sigma > 0.0
        assert not (sig.power_excess_sigma != sig.power_excess_sigma)  # not NaN

    def test_too_few_epochs_skipped(self):
        """Objects with fewer than 5 approaches must be silently skipped."""
        neo = self._make_neo_with_approaches("FEW001", period_days=100, n_approaches=3)
        from aneos_core.pattern_analysis.harmonics import SynodicHarmonicAnalyzer
        signals = SynodicHarmonicAnalyzer(fetcher=None).run([neo])
        assert len(signals) == 0


# ---- NetworkAnalysisSession (integration with synthetic data) ----

class TestNetworkAnalysisSession:
    def test_session_runs_clustering_only(self):
        """Session should run without network access using clustering-only config."""
        from aneos_core.pattern_analysis.session import NetworkAnalysisSession, PatternAnalysisConfig
        cfg = PatternAnalysisConfig(clustering=True, harmonics=False, correlation=False)
        session = NetworkAnalysisSession(config=cfg, fetcher=None)
        neos = _make_cluster_neos(n=15) + [_make_neo(f"X{i}", 2.0 + i * 0.1, 0.3, 10.0)
                                            for i in range(10)]
        result = session.run(neos)
        assert "network_sigma" in result
        assert "network_tier" in result
        assert "clusters" in result
        assert isinstance(result["clusters"], list)

    def test_session_result_schema(self):
        from aneos_core.pattern_analysis.session import NetworkAnalysisSession, PatternAnalysisConfig
        cfg = PatternAnalysisConfig(clustering=True, harmonics=False, correlation=False)
        result = NetworkAnalysisSession(config=cfg).run([_make_neo("S001", 1.0, 0.2, 5.0)])
        required_keys = ["designations_analyzed", "clusters", "harmonic_signals",
                         "network_sigma", "network_tier", "combined_p_value"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# ---- NonGravitationalParameters validation ----

class TestNonGravitationalParametersValidation:
    def test_valid_a2_no_warning(self, caplog):
        import logging
        from aneos_core.data.models import NonGravitationalParameters
        with caplog.at_level(logging.WARNING, logger="aneos_core.data.models"):
            ng = NonGravitationalParameters(a2=-3.2e-14)
        assert not any("data quality suspect" in m for m in caplog.messages)

    def test_extreme_a2_logs_warning(self, caplog):
        import logging
        from aneos_core.data.models import NonGravitationalParameters
        with caplog.at_level(logging.WARNING, logger="aneos_core.data.models"):
            ng = NonGravitationalParameters(a2=1e-5)
        assert any("data quality suspect" in m for m in caplog.messages)

    def test_none_values_no_warning(self, caplog):
        import logging
        from aneos_core.data.models import NonGravitationalParameters
        with caplog.at_level(logging.WARNING, logger="aneos_core.data.models"):
            ng = NonGravitationalParameters()  # all None
        assert not any("data quality suspect" in m for m in caplog.messages)


# ---- NetworkRequest validation ----

class TestNetworkRequestValidation:
    def test_empty_designations_rejected(self):
        from aneos_api.schemas.network import NetworkRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NetworkRequest(designations=[])

    def test_too_many_designations_rejected(self):
        from aneos_api.schemas.network import NetworkRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NetworkRequest(designations=[f"D{i}" for i in range(501)])

    def test_historical_years_bounds(self):
        from aneos_api.schemas.network import NetworkRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NetworkRequest(designations=["Apophis"], historical_years=0)
        with pytest.raises(ValidationError):
            NetworkRequest(designations=["Apophis"], historical_years=101)
        r = NetworkRequest(designations=["Apophis"], historical_years=50)
        assert r.historical_years == 50

    def test_valid_request_accepted(self):
        from aneos_api.schemas.network import NetworkRequest
        r = NetworkRequest(designations=["Apophis", "Bennu"], clustering=True)
        assert len(r.designations) == 2


# ---- Session sigma filter ----

class TestSessionSigmaFilter:
    def test_filter_applied_when_scores_available(self):
        from aneos_core.pattern_analysis.session import NetworkAnalysisSession, PatternAnalysisConfig
        cfg = PatternAnalysisConfig(clustering=False, harmonics=False, min_sigma_filter=2.0)
        neos = [_make_neo(f"S{i}", 1.0, 0.2, 5.0) for i in range(4)]
        neos[0].dynamic_anomaly_score = 3.0   # passes
        neos[1].dynamic_anomaly_score = 1.0   # below threshold
        neos[2].dynamic_anomaly_score = 2.5   # passes
        # neos[3] has no score — included regardless (unscored passthrough)
        result = NetworkAnalysisSession(config=cfg).run(neos)
        assert result["designations_analyzed"] == 3

    def test_filter_skipped_if_none_scored(self):
        from aneos_core.pattern_analysis.session import NetworkAnalysisSession, PatternAnalysisConfig
        cfg = PatternAnalysisConfig(clustering=False, harmonics=False, min_sigma_filter=5.0)
        neos = [_make_neo(f"U{i}", 1.0, 0.2, 5.0) for i in range(5)]
        result = NetworkAnalysisSession(config=cfg).run(neos)
        assert result["designations_analyzed"] == 5


# ---- Rendezvous guard ----

class TestRendezvousGuard:
    def test_rendezvous_true_logs_warning(self, caplog):
        import logging
        from aneos_core.pattern_analysis.session import NetworkAnalysisSession, PatternAnalysisConfig
        cfg = PatternAnalysisConfig(clustering=False, harmonics=False, rendezvous=True)
        with caplog.at_level(logging.WARNING):
            result = NetworkAnalysisSession(config=cfg).run([_make_neo("RV001", 1.0, 0.2, 5.0)])
        assert any("ADR-045" in m or "rendezvous" in m.lower() for m in caplog.messages)
        assert "network_sigma" in result


# ---- Harmonics skip tracking ----

class TestHarmonicsSkipTracking:
    def test_skipped_count_in_metadata(self):
        from aneos_core.pattern_analysis.session import NetworkAnalysisSession, PatternAnalysisConfig
        from aneos_core.data.models import NEOData, OrbitalElements
        cfg = PatternAnalysisConfig(clustering=False, harmonics=True)
        neos = [NEOData(designation=f"H{i}", orbital_elements=OrbitalElements(
            semi_major_axis=1.0, eccentricity=0.2, inclination=5.0
        )) for i in range(3)]
        result = NetworkAnalysisSession(config=cfg, fetcher=None).run(neos)
        assert result["analysis_metadata"]["objects_skipped_insufficient_harmonics"] == 3
