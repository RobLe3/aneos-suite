"""
Comprehensive test suite for the validated sigma 5 artificial NEO detector.

Tests the scientifically rigorous implementation against known artificial
and natural objects to verify proper sigma confidence calculation.
"""

import pytest
import numpy as np
from datetime import datetime
from aneos_core.detection.validated_sigma5_artificial_neo_detector import (
    ValidatedSigma5ArtificialNEODetector,
    EvidenceType,
    EvidenceSource
)

@pytest.fixture
def detector():
    """Provide validated detector instance for testing."""
    return ValidatedSigma5ArtificialNEODetector()

class TestValidatedDetector:
    """Test suite for validated sigma 5 detector."""
    
    def test_tesla_roadster_detection(self, detector):
        """Test detection of Tesla Roadster as confirmed artificial object."""
        
        # Tesla Roadster orbital elements (JPL Horizons data)
        tesla_orbital = {
            'a': 1.325,  # AU
            'e': 0.256,  # 
            'i': 1.077   # degrees - very low inclination (launch signature)
        }
        
        tesla_physical = {
            'mass_estimate': 1350,  # kg (vehicle + upper stage)
            'diameter': 12,  # m (approximate vehicle size)
            'absolute_magnitude': 28.0,  # Very faint
            'radar_signature': {
                'radar_cross_section': 15.0,  # Enhanced due to metal structure
                'polarization_ratio': 0.4
            }
        }
        
        result = detector.analyze_neo_validated(tesla_orbital, tesla_physical)
        
        # Validated detector should achieve sigma 5 for known artificial object
        assert result.is_artificial == True, f"Tesla Roadster not detected as artificial (σ={result.sigma_confidence:.2f})"
        assert result.sigma_confidence >= 5.0, f"Tesla Roadster sigma confidence {result.sigma_confidence:.2f} < 5.0"
        # With base prior 0.001 (0.1% artificial rate) and LR≈10 at σ≥5, Bayesian posterior
        # is mathematically bounded to ~1–5%.  Smoking-gun evidence (course corrections,
        # propulsion signatures) is required to push posterior above 50%.  Asserting > 0.01
        # verifies the prior is updated meaningfully without requiring unachievable values
        # from orbital/physical evidence alone.
        assert result.bayesian_probability > 0.01, f"Bayesian probability not meaningfully updated: {result.bayesian_probability:.6f}"
        assert len(result.evidence_sources) >= 2, "Insufficient evidence sources"
        
        # Check that orbital evidence shows strong anomaly
        orbital_evidence = next((e for e in result.evidence_sources if e.evidence_type == EvidenceType.ORBITAL_DYNAMICS), None)
        assert orbital_evidence is not None, "Missing orbital evidence"
        assert orbital_evidence.effect_size > 2.0, f"Weak orbital anomaly: {orbital_evidence.effect_size:.2f}"
        
        print(f"✅ Tesla Roadster: σ={result.sigma_confidence:.2f}, P(artificial)={result.bayesian_probability:.6f}")
    
    def test_apophis_natural_classification(self, detector):
        """Test that Apophis (confirmed natural NEO) is correctly classified."""
        
        # 99942 Apophis orbital elements
        apophis_orbital = {
            'a': 0.922,  # AU
            'e': 0.191,  #
            'i': 3.331   # degrees
        }
        
        apophis_physical = {
            'mass_estimate': 2.7e10,  # kg (much more massive than artificial objects)
            'diameter': 370,  # m (large natural asteroid)
            'absolute_magnitude': 19.7,  # Relatively bright
            'density_estimate': 3200  # kg/m³ (typical rocky asteroid)
        }
        
        result = detector.analyze_neo_validated(apophis_orbital, apophis_physical)
        
        # Should be classified as natural
        assert result.is_artificial == False, f"Apophis incorrectly classified as artificial (σ={result.sigma_confidence:.2f})"
        assert result.sigma_confidence < 5.0, f"Apophis sigma confidence too high: {result.sigma_confidence:.2f}"
        assert result.bayesian_probability < 0.5, f"High artificial probability for natural object: {result.bayesian_probability:.6f}"
        
        print(f"✅ Apophis: σ={result.sigma_confidence:.2f}, P(artificial)={result.bayesian_probability:.6f}")
    
    def test_bennu_natural_classification(self, detector):
        """Test that Bennu (OSIRIS-REx target) is correctly classified as natural."""
        
        # 101955 Bennu orbital elements
        bennu_orbital = {
            'a': 1.126,  # AU
            'e': 0.204,  #
            'i': 6.035   # degrees
        }
        
        bennu_physical = {
            'mass_estimate': 7.8e10,  # kg (OSIRIS-REx measurement)
            'diameter': 492,  # m (high precision shape model)
            'absolute_magnitude': 20.9,
            'density_estimate': 1190,  # kg/m³ (rubble pile structure)
            'spectral_type': 'B'  # Carbonaceous asteroid
        }
        
        result = detector.analyze_neo_validated(bennu_orbital, bennu_physical)
        
        # Should be clearly natural
        assert result.is_artificial == False, "Bennu incorrectly classified as artificial"
        assert result.sigma_confidence < 3.0, f"Bennu shows unexpectedly high anomaly: σ={result.sigma_confidence:.2f}"
        
        print(f"✅ Bennu: σ={result.sigma_confidence:.2f}, P(artificial)={result.bayesian_probability:.6f}")
    
    def test_validation_metrics_available(self, detector):
        """Test that validation metrics are properly calculated."""
        
        validation_report = detector.get_validation_report()
        
        assert validation_report['validation_performed'] == True, "Validation not performed"
        assert 'performance_metrics' in validation_report, "Missing performance metrics"
        assert 'confusion_matrix' in validation_report, "Missing confusion matrix"
        
        metrics = validation_report['performance_metrics']
        assert 0.0 <= metrics['sensitivity'] <= 1.0, f"Invalid sensitivity: {metrics['sensitivity']}"
        assert 0.0 <= metrics['specificity'] <= 1.0, f"Invalid specificity: {metrics['specificity']}"
        assert 0.0 <= metrics['f1_score'] <= 1.0, f"Invalid F1 score: {metrics['f1_score']}"
        
        # For reliable sigma 5 detection, we expect high performance
        assert metrics['f1_score'] > 0.8, f"Low F1 score: {metrics['f1_score']:.3f}"
        
        print(f"✅ Validation metrics: F1={metrics['f1_score']:.3f}, Sensitivity={metrics['sensitivity']:.3f}, Specificity={metrics['specificity']:.3f}")
    
    def test_ground_truth_database_loaded(self, detector):
        """Test that ground truth databases are properly loaded."""
        
        assert len(detector.artificial_objects_db) > 0, "No artificial objects in database"
        assert len(detector.natural_objects_db) > 0, "No natural objects in database"
        
        # Check that Tesla Roadster is in artificial database
        tesla_found = any('Tesla' in obj['name'] for obj in detector.artificial_objects_db)
        assert tesla_found, "Tesla Roadster not found in artificial objects database"
        
        # Check that Apophis is in natural database  
        apophis_found = any('Apophis' in obj['name'] for obj in detector.natural_objects_db)
        assert apophis_found, "Apophis not found in natural objects database"
        
        print(f"✅ Ground truth: {len(detector.artificial_objects_db)} artificial, {len(detector.natural_objects_db)} natural objects")
    
    def test_evidence_quality_scoring(self, detector):
        """Test that evidence quality is properly assessed."""
        
        # Complete data should get high quality score
        complete_orbital = {'a': 1.5, 'e': 0.3, 'i': 10.0}
        evidence = detector._calculate_orbital_anomaly_score(complete_orbital)
        assert evidence.quality_score >= 0.9, f"Complete data quality too low: {evidence.quality_score}"
        
        # Incomplete data should get lower quality score
        incomplete_orbital = {'a': 1.5, 'e': 0.0, 'i': 0.0}  # Missing e and i
        evidence = detector._calculate_orbital_anomaly_score(incomplete_orbital)
        assert evidence.quality_score < 0.9, f"Incomplete data quality too high: {evidence.quality_score}"
        
        print(f"✅ Evidence quality: Complete={evidence.quality_score:.2f}")
    
    def test_bayesian_fusion_conservatism(self, detector):
        """Test that Bayesian fusion is appropriately conservative."""
        
        # Test with marginal evidence - should not trigger sigma 5
        marginal_orbital = {
            'a': 1.8,   # Somewhat unusual but not extreme
            'e': 0.3,   # Moderate eccentricity
            'i': 12.0   # Typical inclination
        }
        
        result = detector.analyze_neo_validated(marginal_orbital)
        
        # Should not achieve sigma 5 with marginal evidence
        assert result.sigma_confidence < 5.0, f"Marginal evidence achieved sigma 5: {result.sigma_confidence:.2f}"
        assert result.bayesian_probability < 0.95, f"Too high probability for marginal evidence: {result.bayesian_probability:.6f}"
        
        print(f"✅ Conservative fusion: σ={result.sigma_confidence:.2f} for marginal evidence")
    
    def test_statistical_significance_calculation(self, detector):
        """Test that statistical significance is properly calculated."""
        
        # Test with known Tesla Roadster data
        tesla_orbital = {'a': 1.325, 'e': 0.256, 'i': 1.077}
        
        result = detector.analyze_neo_validated(tesla_orbital)
        
        # P-value should be very small for sigma 5 detection
        if result.is_artificial:
            assert result.combined_p_value < 1e-6, f"P-value too large for sigma 5: {result.combined_p_value:.2e}"
        
        # Check that confidence intervals are reasonable
        for evidence in result.evidence_sources:
            ci_lower, ci_upper = evidence.confidence_interval
            assert ci_lower <= evidence.anomaly_score <= ci_upper, "Anomaly score outside confidence interval"
            assert ci_upper > ci_lower, "Invalid confidence interval"
        
        print(f"✅ Statistical significance: p-value={result.combined_p_value:.2e}")
    
    def test_false_discovery_rate_control(self, detector):
        """Test that false discovery rate is properly controlled."""
        
        validation_report = detector.get_validation_report()
        
        if validation_report['validation_performed']:
            cm = validation_report['confusion_matrix']
            
            # Calculate empirical false discovery rate
            if cm['true_positives'] + cm['false_positives'] > 0:
                empirical_fdr = cm['false_positives'] / (cm['true_positives'] + cm['false_positives'])
                
                # For sigma 5 detection, FDR should be very low
                assert empirical_fdr < 0.1, f"False discovery rate too high: {empirical_fdr:.3f}"
                
                print(f"✅ FDR control: {empirical_fdr:.3f}")
    
    def test_reproducibility(self, detector):
        """Test that results are reproducible."""
        
        test_orbital = {'a': 1.325, 'e': 0.256, 'i': 1.077}
        test_physical = {'mass_estimate': 1350, 'diameter': 12}
        
        # Run analysis multiple times
        results = []
        for _ in range(5):
            result = detector.analyze_neo_validated(test_orbital, test_physical)
            results.append((result.sigma_confidence, result.bayesian_probability))
        
        # Results should be identical (deterministic algorithm)
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10, "Non-reproducible sigma confidence"
            assert abs(results[i][1] - results[0][1]) < 1e-10, "Non-reproducible Bayesian probability"
        
        print(f"✅ Reproducibility: σ={results[0][0]:.6f} consistent across runs")

@pytest.mark.integration
class TestIntegrationValidation:
    """Integration tests for end-to-end validation."""
    
    def test_cross_validation_performance(self, detector):
        """Test that cross-validation achieves acceptable performance."""
        
        validation_report = detector.get_validation_report()
        
        if validation_report['validation_performed']:
            metrics = validation_report['performance_metrics']
            
            # For a reliable detector, expect good performance
            assert metrics['balanced_accuracy'] > 0.8, f"Low balanced accuracy: {metrics['balanced_accuracy']:.3f}"
            assert metrics['f1_score'] > 0.7, f"Low F1 score: {metrics['f1_score']:.3f}"
            
            print(f"✅ Cross-validation: Balanced accuracy={metrics['balanced_accuracy']:.3f}")
    
    def test_sigma_5_threshold_justified(self, detector):
        """Test that sigma 5 threshold is scientifically justified."""
        
        # Sigma 5 corresponds to p < 5.7e-7
        from scipy import stats as _scipy_stats
        expected_p_value = 2 * (1 - _scipy_stats.norm.cdf(5.0))
        
        assert abs(detector.SIGMA_5_P_VALUE - expected_p_value) < 1e-10, "Incorrect sigma 5 p-value"
        assert detector.SIGMA_5_CONFIDENCE == 5.0, "Incorrect sigma 5 threshold"
        
        print(f"✅ Sigma 5 threshold: p-value={detector.SIGMA_5_P_VALUE:.2e}")

def _sbdb_reachable() -> bool:
    try:
        import requests
        requests.head("https://ssd-api.jpl.nasa.gov/sbdb.api", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _sbdb_reachable(), reason="JPL SBDB not reachable")
def test_detector_accuracy_on_ground_truth():
    from aneos_core.datasets.ground_truth_dataset_preparation import GroundTruthDatasetBuilder
    from aneos_core.datasets.ground_truth_validator import GroundTruthValidator
    from aneos_core.detection.detection_manager import DetectionManager, DetectorType

    builder = GroundTruthDatasetBuilder()
    builder.compile_verified_artificial_objects()
    builder.query_jpl_sbdb_natural_neos(limit=30)

    objects = builder.artificial_objects + builder.natural_objects
    detector = DetectionManager(preferred_detector=DetectorType.VALIDATED)
    report = GroundTruthValidator().calibrated_run(objects, detector)

    assert report.specificity >= 0.80, f"FPR too high: specificity={report.specificity:.2f}"
    assert report.sensitivity >= 0.33, f"TPR too low: sensitivity={report.sensitivity:.2f}"
    print(f"Ground truth (threshold={report.threshold:.5f}): "
          f"sens={report.sensitivity:.2f}, spec={report.specificity:.2f}, "
          f"F1={report.f1:.2f}, ROC-AUC={report.roc_auc:.2f} "
          f"on {report.n_artificials}+{report.n_naturals} objects")


if __name__ == "__main__":
    # Run basic validation tests
    detector = ValidatedSigma5ArtificialNEODetector()
    
    print("🧪 Running validation tests...\n")
    
    # Test Tesla Roadster detection
    tesla_orbital = {'a': 1.325, 'e': 0.256, 'i': 1.077}
    tesla_physical = {'mass_estimate': 1350, 'diameter': 12, 'absolute_magnitude': 28.0}
    
    tesla_result = detector.analyze_neo_validated(tesla_orbital, tesla_physical)
    print(f"Tesla Roadster: σ={tesla_result.sigma_confidence:.2f}, P(artificial)={tesla_result.bayesian_probability:.6f}")
    
    # Test Apophis classification  
    apophis_orbital = {'a': 0.922, 'e': 0.191, 'i': 3.331}
    apophis_physical = {'mass_estimate': 2.7e10, 'diameter': 370, 'absolute_magnitude': 19.7}
    
    apophis_result = detector.analyze_neo_validated(apophis_orbital, apophis_physical)
    print(f"Apophis: σ={apophis_result.sigma_confidence:.2f}, P(artificial)={apophis_result.bayesian_probability:.6f}")
    
    # Validation report
    validation_report = detector.get_validation_report()
    if validation_report['validation_performed']:
        metrics = validation_report['performance_metrics']
        print(f"\nValidation metrics:")
        print(f"- F1 Score: {metrics['f1_score']:.3f}")
        print(f"- Sensitivity: {metrics['sensitivity']:.3f}")  
        print(f"- Specificity: {metrics['specificity']:.3f}")
        print(f"- Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    
    print("\n✅ Validation tests complete!")