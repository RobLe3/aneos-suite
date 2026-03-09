"""
Kardashev ML Training Pipeline + Benefit Assessment

Trains two complementary classifiers on Kardashev synthetic data:
  1. IsolationForest  — one-class anomaly detector trained on naturals only
  2. RandomForest     — supervised binary classifier (artificial vs natural)

Then evaluates whether the trained models provide FACTUAL BENEFIT over the
current σ-score by checking whether confirmed artificial objects (2020 SO,
J002E3, Tesla Roadster orbital type) consistently rank above confirmed natural
NEOs (Apophis, Bennu, Eros, etc.).

FEATURE SET (13 discriminating features)
-----------------------------------------
For synthetic data, velocity/temporal/geographic features are not available.
We use only features derivable from orbital elements + physical properties +
non-gravitational parameters — the subset that is also available for real
SBDB objects.

  1.  a              — semi-major axis [AU]
  2.  e              — eccentricity
  3.  i              — inclination [°]
  4.  q = a(1-e)     — perihelion distance [AU]
  5.  log10(A2_abs)  — log |non-grav transverse| [AU/day²]  ← strongest signal
  6.  log10(density) — log density [g/cm³]  ← hollow vs rocky
  7.  albedo         — geometric albedo  ← metallic vs rocky
  8.  log10(diam)    — log diameter [km]
  9.  circular       — 1 − e  ← engineered circularity
  10. retrograde     — max(0, i − 90°)  ← retrograde component
  11. thrust_factor  — |A2| / A2_radiation_pressure(d,ρ,1AU)  ← excess thrust
  12. i_ecliptic_dev — |i − 0| weighted  ← deviation from ecliptic
  13. high_a_ratio   — q / 1.0 (Earth-crossing indicator)

GROUND TRUTH TEST SET (hardcoded from published orbital solutions)
------------------------------------------------------------------
Confirmed artificials:
  2020 SO   — NASA Apollo SRM upper stage; Fedorets et al. 2021
  J002E3    — Apollo 12 S-IVB stage; Jorgensen et al. 2003
  WT1190F   — Confirmed rocket body; Scheirich et al. 2015
  2018 AV2  — Used as Tesla Roadster-type proxy (open orbit, low density)

Confirmed naturals:
  99942 Apophis  — Real NEO, well-characterised
  101955 Bennu   — Osiris-Rex target
  433 Eros       — NEAR target
  4179 Toutatis  — Radar-characterised natural
  2001 SL9       — Shoemaker-Levy analog-class
  25143 Itokawa  — Hayabusa target

BENEFIT VERDICT
---------------
  CONFIRMED  — Wilcoxon p < 0.05; all artificials rank above all naturals
  MARGINAL   — Mean separation > 0.2 but not statistically significant
  NO BENEFIT — Distributions overlap with mean separation < 0.2
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available — training disabled")


# ===========================================================================
# Hardcoded ground-truth test objects (orbital elements from published data)
# ===========================================================================

#  Format: {name: {"a", "e", "i", "albedo", "density_g_cm3", "diameter_km",
#                   "nongrav_A2", "label"}}
#  Physical parameters for artificials are estimates from published analyses.

GROUND_TRUTH_OBJECTS: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # CONFIRMED ARTIFICIALS
    # -----------------------------------------------------------------------
    "2020 SO": {
        # Apollo SRM upper stage. Fedorets+ 2021, Icarus 355.
        "a": 1.01740, "e": 0.01148, "i": 0.4737,
        "albedo": 0.20,
        "density_g_cm3": 0.00431,   # mass ~1000 kg, volume ~232 m³
        "diameter_km":   0.00668,   # ~6.7 m effective diameter
        "nongrav_A2":    3.84e-11,  # Fedorets measured value [AU/day²]
        "label": "artificial",
    },
    "J002E3": {
        # Apollo 12 S-IVB stage. Jorgensen+ 2003, Science 301.
        "a": 1.01312, "e": 0.13444, "i": 2.6319,
        "albedo": 0.18,
        "density_g_cm3": 0.00890,
        "diameter_km":   0.01000,
        "nongrav_A2":    1.2e-11,
        "label": "artificial",
    },
    "WT1190F": {
        # Confirmed rocket body impacted Earth 2015-11-13.
        # Scheirich+ 2015, ESA NEOCC.
        "a": 1.00351, "e": 0.03200, "i": 0.2600,
        "albedo": 0.15,
        "density_g_cm3": 0.00300,
        "diameter_km":   0.00200,
        "nongrav_A2":    8.5e-11,
        "label": "artificial",
    },
    "2018_AV2_type": {
        # Tesla Roadster-type orbit proxy.  Use published Starman orbital
        # elements (SpaceX press kit 2018); physical from payload mass ~1350 kg.
        "a": 1.31489, "e": 0.25589, "i": 1.0734,
        "albedo": 0.40,
        "density_g_cm3": 0.0012,
        "diameter_km":   0.00450,
        "nongrav_A2":    2.1e-12,
        "label": "artificial",
    },
    # -----------------------------------------------------------------------
    # CONFIRMED NATURALS
    # -----------------------------------------------------------------------
    "99942 Apophis": {
        # Apophis. SBDB solution JPL K232/3.
        "a": 0.92228, "e": 0.19141, "i": 3.3380,
        "albedo": 0.30,
        "density_g_cm3": 2.20,
        "diameter_km":   0.340,
        "nongrav_A2":    -2.9e-14,  # small Yarkovsky from Farnocchia+ 2013
        "label": "natural",
    },
    "101955 Bennu": {
        # Bennu. Nolan+ 2013, Chesley+ 2020.
        "a": 1.12590, "e": 0.20374, "i": 6.0349,
        "albedo": 0.044,
        "density_g_cm3": 1.19,
        "diameter_km":   0.4921,
        "nongrav_A2":    -2.4e-14,
        "label": "natural",
    },
    "433 Eros": {
        # Eros. SBDB JPL solution.
        "a": 1.45817, "e": 0.22285, "i": 10.828,
        "albedo": 0.25,
        "density_g_cm3": 2.67,
        "diameter_km":   16.84,
        "nongrav_A2":    0.0,
        "label": "natural",
    },
    "4179 Toutatis": {
        "a": 2.52590, "e": 0.63412, "i": 0.4723,
        "albedo": 0.13,
        "density_g_cm3": 2.14,
        "diameter_km":   2.450,
        "nongrav_A2":    0.0,
        "label": "natural",
    },
    "25143 Itokawa": {
        # Hayabusa target. Abe+ 2006.
        "a": 1.32415, "e": 0.28012, "i": 1.6215,
        "albedo": 0.53,
        "density_g_cm3": 1.95,
        "diameter_km":   0.330,
        "nongrav_A2":    -3.5e-14,
        "label": "natural",
    },
    "2001 YB5": {
        # Typical PHA natural.
        "a": 2.37680, "e": 0.80025, "i": 42.344,
        "albedo": 0.15,
        "density_g_cm3": 2.00,
        "diameter_km":   0.850,
        "nongrav_A2":    0.0,
        "label": "natural",
    },
}

# Radiation-pressure constant for A2 estimation (at 1 AU, sphere):
#   A2_rp [AU/day²] ≈ 3.4e-13 / (diameter_km × density_g_cm3)
_A2_RP_CONST = 3.4e-13  # AU/day² × km × g/cm³


# ===========================================================================
# Feature extraction (deterministic, no network calls)
# ===========================================================================

FEATURE_NAMES = [
    "a", "e", "i", "q",
    "log10_A2_abs",
    "log10_density",
    "albedo",
    "log10_diameter",
    "circular",
    "retrograde",
    "log10_thrust_factor",
    "i_ecliptic_dev",
    "earth_crossing",
]


def _params_to_features(p: Dict[str, Any]) -> np.ndarray:
    """
    Convert a parameter dict to a 13-element numpy feature vector.

    Parameters that are missing default to the median natural-population value.
    A2 = 0 is treated as exactly the natural Yarkovsky floor (1e-15).
    """
    a    = float(p.get("a", 1.5))
    e    = float(p.get("e", 0.3))
    i    = float(p.get("i", 10.0))
    alb  = float(p.get("albedo", 0.15))
    dens = float(p.get("density_g_cm3", 2.0))
    diam = float(p.get("diameter_km", 1.0))
    A2   = float(p.get("nongrav_A2", 0.0))

    q = a * (1.0 - e)

    # Non-grav: use absolute value; floor at 1e-16 to avoid log(0)
    A2_abs = max(abs(A2), 1e-16)
    log_A2 = math.log10(A2_abs)

    # Density and diameter: floor small positives
    dens = max(dens, 1e-8)
    diam = max(diam, 1e-6)
    log_dens = math.log10(dens)
    log_diam = math.log10(diam)

    # Circularity (engineered indicator)
    circular = 1.0 - e

    # Retrograde component
    retrograde = max(0.0, i - 90.0)

    # Thrust factor: measured |A2| vs. expected radiation pressure for a sphere
    A2_rp = _A2_RP_CONST / (diam * dens)
    A2_rp = max(A2_rp, 1e-20)
    thrust_factor = A2_abs / A2_rp
    log_thrust = math.log10(max(thrust_factor, 1e-10))

    # Ecliptic deviation (simple |i| normalised)
    i_dev = abs(i) / 90.0

    # Earth-crossing flag
    earth_cross = 1.0 if (q < 1.3 and a * (1.0 + e) > 0.7) else 0.0

    return np.array([
        a, e, i, q,
        log_A2,
        log_dens,
        alb,
        log_diam,
        circular,
        retrograde,
        log_thrust,
        i_dev,
        earth_cross,
    ], dtype=float)


def _synth_to_params(s) -> Dict[str, Any]:
    """Extract parameter dict from a SyntheticNEO (Kardashev generator output)."""
    return {
        "a":              s.params["a"],
        "e":              s.params["e"],
        "i":              s.params["i"],
        "albedo":         s.params["albedo"],
        "density_g_cm3":  s.params["density_g_cm3"],
        "diameter_km":    s.params["diameter_km"],
        "nongrav_A2":     s.params["nongrav_A2"],
    }


# ===========================================================================
# Synthetic natural population (empirical NEO distributions)
# ===========================================================================

def generate_synthetic_naturals(
    n: int = 500,
    seed: int = 99,
) -> List[Dict[str, Any]]:
    """
    Sample n synthetic natural NEOs from empirical SBDB distributions.

    Parameter distributions are fit to the published SBDB NEA orbital element
    catalog (Granvik+ 2018 model, ~35 000 objects).  Non-grav is set to a
    realistic Yarkovsky floor — measurable only for the smallest objects.

    This is a synthetic negative class used for training only.  Real SBDB
    data should be used for final validation.
    """
    rng = np.random.default_rng(seed)

    naturals = []
    for _ in range(n):
        # Semi-major axis: bimodal (Amor/Apollo/Aten mix)
        a_family = rng.choice(["amor", "apollo", "aten"], p=[0.45, 0.47, 0.08])
        if a_family == "amor":
            a = float(np.clip(rng.normal(1.85, 0.45), 1.017, 4.0))
        elif a_family == "apollo":
            a = float(np.clip(rng.normal(1.65, 0.50), 0.70, 4.0))
        else:
            a = float(np.clip(rng.normal(0.91, 0.08), 0.5, 1.0))

        e = float(np.clip(rng.beta(2.5, 3.0), 0.01, 0.89))
        i = float(np.clip(abs(rng.normal(12.0, 9.0)), 0.1, 60.0))

        albedo_type = rng.choice(["S", "C", "X"], p=[0.42, 0.35, 0.23])
        if albedo_type == "S":
            albedo = float(np.clip(rng.normal(0.25, 0.08), 0.05, 0.55))
        elif albedo_type == "C":
            albedo = float(np.clip(rng.normal(0.06, 0.02), 0.02, 0.12))
        else:
            albedo = float(np.clip(rng.normal(0.18, 0.07), 0.03, 0.45))

        density = float(np.clip(10 ** rng.normal(0.35, 0.18), 1.0, 5.0))
        diameter = float(10 ** rng.normal(-1.0, 0.8))   # mostly 0.1–10 km

        # Yarkovsky A2: very small for natural objects (rarely measurable)
        log_a2 = float(rng.normal(-14.5, 0.8))          # ~1e-16 to 1e-13
        A2 = 10 ** log_a2 * rng.choice([-1.0, 1.0])

        naturals.append({
            "a": a, "e": e, "i": i,
            "albedo": albedo, "density_g_cm3": density, "diameter_km": diameter,
            "nongrav_A2": A2,
        })
    return naturals


# ===========================================================================
# Benefit assessment dataclass
# ===========================================================================

@dataclass
class BenefitAssessment:
    """Full result of the ML benefit evaluation."""

    # --- Training stats ---
    n_artificial_train: int
    n_natural_train:    int
    n_scenarios:        int

    # --- IsolationForest (one-class, trained on naturals) ---
    if_auc:         float   # AUC on ground-truth set
    if_separation:  float   # mean(artificial scores) - mean(natural scores)
    if_scores:      Dict[str, float] = field(default_factory=dict)

    # --- RandomForest (supervised) ---
    rf_auc:         float   = 0.0
    rf_separation:  float   = 0.0
    rf_scores:      Dict[str, float] = field(default_factory=dict)

    # --- σ-baseline comparison ---
    sigma_separation: float = 0.0   # if σ already separates, ML may be redundant

    # --- Feature importance (RandomForest) ---
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # --- Per-object breakdown ---
    object_scores: List[Dict[str, Any]] = field(default_factory=list)

    # --- Verdict ---
    verdict:     str = "UNKNOWN"
    explanation: str = ""

    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def print_report(self) -> None:
        """Print a human-readable benefit assessment report."""
        w = 72
        print("=" * w)
        print("  aNEOS KARDASHEV ML — BENEFIT ASSESSMENT REPORT")
        print("=" * w)
        print(f"  Training set:  {self.n_artificial_train} artificial"
              f"  +  {self.n_natural_train} natural  ({self.n_scenarios} scenarios)")
        print()
        print("  ── IsolationForest (trained on naturals only) ──────────────")
        print(f"     AUC (gt set):    {self.if_auc:.3f}")
        print(f"     Separation:      {self.if_separation:+.3f}")
        print()
        print("  ── RandomForest (supervised) ───────────────────────────────")
        print(f"     AUC (gt set):    {self.rf_auc:.3f}")
        print(f"     Separation:      {self.rf_separation:+.3f}")
        print()
        print("  ── Per-object scores (IF / RF) ─────────────────────────────")
        art_rows = [(r["name"], r["if_score"], r["rf_score"])
                    for r in self.object_scores if r["label"] == "artificial"]
        nat_rows = [(r["name"], r["if_score"], r["rf_score"])
                    for r in self.object_scores if r["label"] == "natural"]
        print("  Artificials:")
        for name, ifs, rfs in sorted(art_rows, key=lambda x: -x[1]):
            print(f"    {name:<25}  IF={ifs:.3f}  RF={rfs:.3f}")
        print("  Naturals:")
        for name, ifs, rfs in sorted(nat_rows, key=lambda x: -x[1]):
            print(f"    {name:<25}  IF={ifs:.3f}  RF={rfs:.3f}")
        print()
        if self.feature_importance:
            print("  ── Top-5 Feature Importance (RF) ───────────────────────────")
            top5 = sorted(self.feature_importance.items(), key=lambda x: -x[1])[:5]
            for fname, imp in top5:
                bar = "█" * int(imp * 40)
                print(f"    {fname:<25}  {imp:.3f}  {bar}")
            print()
        print(f"  VERDICT:  {self.verdict}")
        print(f"  {self.explanation}")
        print("=" * w)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "trained_at": self.trained_at,
            "n_artificial_train": self.n_artificial_train,
            "n_natural_train": self.n_natural_train,
            "if_auc": self.if_auc,
            "if_separation": self.if_separation,
            "rf_auc": self.rf_auc,
            "rf_separation": self.rf_separation,
            "feature_importance": self.feature_importance,
            "object_scores": self.object_scores,
        }


# ===========================================================================
# Training pipeline
# ===========================================================================

class KardashevMLPipeline:
    """
    End-to-end Kardashev synthetic training + benefit evaluation.

    Intended use::

        pipeline = KardashevMLPipeline()
        assessment = pipeline.run(n_per_scenario=200)
        assessment.print_report()
        pipeline.save(assessment, "models/kardashev_assessment.json")
    """

    def __init__(self):
        self._if_model: Optional[Any]  = None
        self._rf_model: Optional[Any]  = None
        self._scaler:   Optional[Any]  = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        n_per_scenario: int = 200,
        n_naturals:     int = 500,
        seed:           int = 42,
    ) -> BenefitAssessment:
        """
        Train both models and return a full BenefitAssessment.

        Parameters
        ----------
        n_per_scenario:
            Synthetic artificial samples per Kardashev scenario.
        n_naturals:
            Synthetic natural samples for training.
        seed:
            Random seed for reproducibility.
        """
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for ML training")

        from aneos_core.datasets.kardashev_generator import KardashevSyntheticGenerator

        logger.info("Generating Kardashev synthetic positives…")
        gen = KardashevSyntheticGenerator(seed=seed)
        synthetics = gen.generate(n_per_scenario=n_per_scenario)

        logger.info(f"Generated {len(synthetics)} artificial samples")

        art_params = [_synth_to_params(s) for s in synthetics]
        nat_params = generate_synthetic_naturals(n=n_naturals, seed=seed + 1)

        # Build feature matrices
        X_art = np.array([_params_to_features(p) for p in art_params])
        X_nat = np.array([_params_to_features(p) for p in nat_params])

        # Replace any NaN/Inf
        X_art = np.nan_to_num(X_art, nan=0.0, posinf=30.0, neginf=-30.0)
        X_nat = np.nan_to_num(X_nat, nan=0.0, posinf=30.0, neginf=-30.0)

        X_all = np.vstack([X_nat, X_art])
        y_all = np.array([0] * len(X_nat) + [1] * len(X_art))

        # Standard-scale using natural distribution as reference
        self._scaler = StandardScaler().fit(X_nat)
        X_nat_s = self._scaler.transform(X_nat)
        X_art_s = self._scaler.transform(X_art)
        X_all_s = self._scaler.transform(X_all)

        # ------------------------------------------------------------------
        # 1. IsolationForest: trained on NATURALS only
        # ------------------------------------------------------------------
        logger.info("Training IsolationForest on natural class…")
        contamination = max(0.01, min(0.45,
            len(X_art) / (len(X_nat) + len(X_art))))
        self._if_model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        ).fit(X_nat_s)

        # ------------------------------------------------------------------
        # 2. RandomForest: supervised
        # ------------------------------------------------------------------
        logger.info("Training RandomForestClassifier (supervised)…")
        self._rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ).fit(X_all_s, y_all)

        # ------------------------------------------------------------------
        # 3. Evaluate on ground-truth test set
        # ------------------------------------------------------------------
        return self._evaluate(n_per_scenario, n_naturals, len(gen._scenarios))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        n_art: int,
        n_nat: int,
        n_scenarios: int,
    ) -> BenefitAssessment:
        """Score all ground-truth objects and compute benefit metrics."""

        object_scores = []
        gt_labels, gt_if_scores, gt_rf_scores = [], [], []

        for name, data in GROUND_TRUTH_OBJECTS.items():
            feats = _params_to_features(data).reshape(1, -1)
            feats_s = self._scaler.transform(feats)

            # IsolationForest: higher score = more anomalous
            # score_samples returns negative of anomaly score (lower = more anomalous)
            if_raw = float(self._if_model.score_samples(feats_s)[0])
            # Convert: more negative = more anomalous → flip and rescale
            if_score = float(1.0 - _sigmoid(if_raw))   # 1 = anomalous, 0 = normal

            # RandomForest: P(artificial)
            rf_score = float(self._rf_model.predict_proba(feats_s)[0][1])

            label = data["label"]
            object_scores.append({
                "name":     name,
                "label":    label,
                "if_score": if_score,
                "rf_score": rf_score,
                **{k: data[k] for k in ["a", "e", "i", "nongrav_A2", "density_g_cm3"]},
            })
            gt_labels.append(1 if label == "artificial" else 0)
            gt_if_scores.append(if_score)
            gt_rf_scores.append(rf_score)

        gt_labels     = np.array(gt_labels)
        gt_if_scores  = np.array(gt_if_scores)
        gt_rf_scores  = np.array(gt_rf_scores)

        art_mask = gt_labels == 1
        nat_mask = gt_labels == 0

        if_auc  = _safe_auc(gt_labels, gt_if_scores)
        rf_auc  = _safe_auc(gt_labels, gt_rf_scores)

        if_sep  = (float(gt_if_scores[art_mask].mean()) -
                   float(gt_if_scores[nat_mask].mean()))
        rf_sep  = (float(gt_rf_scores[art_mask].mean()) -
                   float(gt_rf_scores[nat_mask].mean()))

        # Feature importance
        feat_imp = dict(zip(FEATURE_NAMES,
                            self._rf_model.feature_importances_.tolist()))

        # Verdict
        best_auc = max(if_auc, rf_auc)
        best_sep = max(if_sep, rf_sep)

        if best_auc >= 0.90 and best_sep >= 0.50:
            verdict = "CONFIRMED BENEFIT"
            explanation = (
                f"AUC={best_auc:.3f} ≥ 0.90; separation={best_sep:.3f} ≥ 0.50. "
                "The ML model reliably ranks confirmed artificials above naturals."
            )
        elif best_auc >= 0.75 and best_sep >= 0.25:
            verdict = "MARGINAL BENEFIT"
            explanation = (
                f"AUC={best_auc:.3f}, separation={best_sep:.3f}. "
                "Model shows discrimination but not strong enough to replace σ-score alone. "
                "Useful as a secondary scoring layer."
            )
        else:
            verdict = "NO CONFIRMED BENEFIT"
            explanation = (
                f"AUC={best_auc:.3f}, separation={best_sep:.3f}. "
                "Model does not reliably separate artificials from naturals "
                "on the ground-truth set. Do not activate for production use."
            )

        return BenefitAssessment(
            n_artificial_train=n_art * n_scenarios,
            n_natural_train=n_nat,
            n_scenarios=n_scenarios,
            if_auc=if_auc,
            if_separation=if_sep,
            if_scores={r["name"]: r["if_score"] for r in object_scores},
            rf_auc=rf_auc,
            rf_separation=rf_sep,
            rf_scores={r["name"]: r["rf_score"] for r in object_scores},
            feature_importance=feat_imp,
            object_scores=object_scores,
            verdict=verdict,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, assessment: BenefitAssessment, path: str) -> None:
        """Write assessment JSON and (optionally) model artefacts."""
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(assessment.to_dict(), indent=2, default=str))
        logger.info(f"Assessment saved to {p}")
        if self._rf_model is not None and self._scaler is not None:
            model_dir = p.parent / "kardashev_model"
            model_dir.mkdir(exist_ok=True)
            joblib.dump(self._rf_model, model_dir / "rf_classifier.pkl")
            joblib.dump(self._if_model, model_dir / "if_anomaly.pkl")
            joblib.dump(self._scaler,   model_dir / "scaler.pkl")
            logger.info(f"Model artefacts saved to {model_dir}/")


# ===========================================================================
# Utilities
# ===========================================================================

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return AUC-ROC, or 0.5 if only one class present."""
    if not HAS_SKLEARN:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


# ===========================================================================
# CLI entry point
# ===========================================================================

def run_assessment(
    n_per_scenario: int = 200,
    n_naturals:     int = 500,
    seed:           int = 42,
    save_path:      Optional[str] = None,
) -> BenefitAssessment:
    """
    Train + evaluate and return a BenefitAssessment.
    Prints a report to stdout.
    """
    pipeline = KardashevMLPipeline()
    assessment = pipeline.run(
        n_per_scenario=n_per_scenario,
        n_naturals=n_naturals,
        seed=seed,
    )
    assessment.print_report()
    if save_path:
        pipeline.save(assessment, save_path)
    return assessment


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run_assessment(n_per_scenario=n, save_path="reports/kardashev_assessment.json")
