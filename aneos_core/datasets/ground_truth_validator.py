"""
Ground Truth Validator for aNEOS artificial NEO detector evaluation.

Runs binary classification metrics against a labelled dataset of
GroundTruthObject instances produced by GroundTruthDatasetBuilder.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Binary classification metrics from ground truth evaluation."""
    n_artificials: int
    n_naturals: int
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float   # tp / (tp + fn)
    specificity: float   # tn / (tn + fp)
    ppv: float           # tp / (tp + fp)
    npv: float           # tn / (tn + fn)
    f1: float
    roc_auc: float
    threshold: float
    per_object: List[Dict[str, Any]] = field(default_factory=list)
    dataset_source: str = ""


class GroundTruthValidator:
    """Evaluates a detector against a labelled ground truth dataset."""

    def run(self, objects, detector, threshold: float = 0.5) -> ValidationReport:
        """
        Run binary classification evaluation.

        Parameters
        ----------
        objects : list of GroundTruthObject
        detector : any object with .analyze_neo(orbital_elements, physical_params)
                   returning a DetectionResult with .artificial_probability
        threshold : float
            Decision boundary for classifying an object as artificial.
        """
        per_object = []

        for obj in objects:
            score = self._score_object(obj, detector)
            predicted = score >= threshold
            per_object.append({
                'id': obj.object_id,
                'true_label': obj.is_artificial,
                'predicted': predicted,
                'score': score,
            })

        artificials = [o for o in per_object if o['true_label']]
        naturals = [o for o in per_object if not o['true_label']]

        tp = sum(1 for o in artificials if o['predicted'])
        fn = sum(1 for o in artificials if not o['predicted'])
        fp = sum(1 for o in naturals if o['predicted'])
        tn = sum(1 for o in naturals if not o['predicted'])

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0.0

        roc_auc = self._compute_roc_auc(per_object)

        all_sources = {getattr(obj, 'source', '') for obj in objects}
        dataset_source = '; '.join(sorted(s for s in all_sources if s))

        return ValidationReport(
            n_artificials=len(artificials),
            n_naturals=len(naturals),
            tp=tp, fp=fp, tn=tn, fn=fn,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            f1=f1,
            roc_auc=roc_auc,
            threshold=threshold,
            per_object=per_object,
            dataset_source=dataset_source,
        )

    def calibrated_run(self, objects, detector, min_sensitivity: float = 0.33) -> ValidationReport:
        """
        Run with the highest threshold that still achieves min_sensitivity.

        This "sensitivity-constrained max-specificity" calibration is appropriate
        when false positives are costly (astronomical claims require few FPs) and
        the detector's output range is far from 0.5.  Scores are computed once.
        The chosen threshold is recorded in ValidationReport.threshold.
        """
        # Score every object once
        scored = [(obj, self._score_object(obj, detector)) for obj in objects]

        # Sort candidate thresholds descending (from strictest to most permissive)
        candidates = sorted({s for _, s in scored}, reverse=True)
        best_t = candidates[-1] if candidates else 0.5
        n_pos = sum(1 for obj, _ in scored if obj.is_artificial)

        if n_pos > 0:
            for t in candidates:
                tp = sum(1 for obj, s in scored if obj.is_artificial and s >= t)
                fn = sum(1 for obj, s in scored if obj.is_artificial and s < t)
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if sens >= min_sensitivity:
                    best_t = t
                    break  # highest threshold satisfying the sensitivity target

        # Build per_object entries using the calibrated threshold
        per_object = []
        for obj, score in scored:
            per_object.append({
                'id': obj.object_id,
                'true_label': obj.is_artificial,
                'predicted': score >= best_t,
                'score': score,
            })

        artificials = [o for o in per_object if o['true_label']]
        naturals = [o for o in per_object if not o['true_label']]
        tp = sum(1 for o in artificials if o['predicted'])
        fn = sum(1 for o in artificials if not o['predicted'])
        fp = sum(1 for o in naturals if o['predicted'])
        tn = sum(1 for o in naturals if not o['predicted'])

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0.0

        all_sources = {getattr(obj, 'source', '') for obj, _ in scored}
        dataset_source = '; '.join(sorted(s for s in all_sources if s))

        return ValidationReport(
            n_artificials=len(artificials),
            n_naturals=len(naturals),
            tp=tp, fp=fp, tn=tn, fn=fn,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            f1=f1,
            roc_auc=self._compute_roc_auc(per_object),
            threshold=best_t,
            per_object=per_object,
            dataset_source=dataset_source,
        )

    def _score_object(self, obj, detector) -> float:
        """Returns artificial_probability in [0, 1]."""
        try:
            result = detector.analyze_neo(
                obj.orbital_elements,
                obj.physical_params or {},
            )
            return float(result.artificial_probability)
        except Exception as exc:
            logger.warning(f"Scoring failed for {obj.object_id}: {exc}")
            return 0.0

    def _compute_roc_auc(self, per_object: List[Dict[str, Any]]) -> float:
        """Compute ROC AUC; falls back to trapezoidal approximation if sklearn unavailable."""
        if not per_object:
            return 0.5

        try:
            from sklearn.metrics import roc_auc_score
            y_true = [int(o['true_label']) for o in per_object]
            y_score = [o['score'] for o in per_object]
            if len(set(y_true)) < 2:
                return 0.5
            return float(roc_auc_score(y_true, y_score))
        except ImportError:
            pass
        except Exception:
            pass

        return self._trapz_roc_auc(per_object)

    def _trapz_roc_auc(self, per_object: List[Dict[str, Any]]) -> float:
        """Trapezoidal AUC approximation from sorted scores."""
        n_pos = sum(1 for o in per_object if o['true_label'])
        n_neg = len(per_object) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        sorted_items = sorted(per_object, key=lambda x: x['score'], reverse=True)
        tpr_list = [0.0]
        fpr_list = [0.0]
        tp = fp = 0
        for item in sorted_items:
            if item['true_label']:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        auc = 0.0
        for i in range(1, len(tpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
        return auc
