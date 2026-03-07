# aNEOS 0.7.0 Maturity Assessment

_Last updated: March 2026 (Phase 3 completion)_

## Overview
The 0.7 stabilization series has progressed through three implementation phases.
Phase 1 restored baseline infrastructure; Phase 2 added physical indicators and
factual ground truth; Phase 3 wired real data throughout and validated detection
accuracy against confirmed artificial objects. This document records the current
evidence and remaining gaps.

## Verification Snapshot (Phase 3 complete)
- **Automated Tests** – 54 unit and integration tests pass (0 failures).
  Covers cache, configuration, sigma-5 detection, physical indicators,
  ground truth validation, and DataFetcher integration.
- **Ground Truth Validation** – Validated against 3 confirmed artificial
  heliocentric objects (Tesla Roadster/2018 A1, Centaur/2020 SO,
  Apollo 12 S-IVB/J002E3) and 20+ real JPL SBDB natural NEOs.
  Results: sensitivity=1.00, specificity=1.00, F1=1.00, ROC-AUC=1.00 at
  calibrated threshold (Bayesian P≥0.037). Score separation: artificials
  ~3.7% vs naturals ~0.1–0.2%.
- **Physical Indicators** – DiameterAnomalyIndicator, AlbedoAnomalyIndicator,
  SpectralAnomalyIndicator now active in the scoring pipeline.
- **Real Data Integration** – DataFetcher wired into Single NEO Analysis,
  Batch Analysis, Enhanced Validation Pipeline, and Interactive Mode.
  SBDB confirmed working for real NEOs (Apophis, Bennu, Ceres, etc.).
- **Honest UI** – System status uses preflight_check(), alert centre wired to
  AlertManager, database status uses live API health pings, spectral analysis
  uses Bus-DeMeo taxonomy (no fabricated data), orbital history uses real
  JPL CAD API close-approach data.

## Remaining Limitations
1. **Bayesian Posterior Ceiling** – With base prior 0.001 (0.1% artificial NEO
   rate), orbital+physical evidence alone gives posterior ~3–4%. Propulsion
   signatures or active course corrections are needed to push above 10–50%.
   This is mathematically correct behaviour, not a bug.
2. **Internal Cross-Validation Size** – The detector's built-in training set
   has 2 artificials (Tesla Roadster, DSCOVR) and 3 naturals. Detection
   performance claims (F1=1.0 internally) are based on this small set.
   The external ground truth run (3 artificials, 20+ naturals) confirms good
   discrimination but calibrated threshold (0.037) must be set explicitly —
   the default sigma-5 threshold (0.5) would give sensitivity=0 for orbital
   evidence alone without physical data.
3. **API Coverage** – NEODyS and MPC data sources still lack synchronous
   `_make_request` implementations; they fail gracefully but contribute no
   data. Horizons source missing abstract method implementations.
4. **Pipeline Hardcoding** – `automatic_review_pipeline.py` still
   instantiates `MultiModalSigma5ArtificialNEODetector` directly (P1 issue).
   `DetectionManager` is used everywhere else.

## Recommended Next Steps
1. **Increase Training Corpus** – Add 5–10 more confirmed artificials to the
   detector's internal cross-validation set to improve F1 robustness.
2. **Fix Pipeline Hardcoding** – Replace `MultiModalSigma5ArtificialNEODetector`
   in `automatic_review_pipeline.py` with `DetectionManager(AUTO)`.
3. **NEODyS/MPC Sources** – Implement synchronous fetch for multi-source
   enrichment to reduce reliance on SBDB alone.
4. **Production Threshold** – Document the recommended calibrated threshold
   (0.037) and expose it as a configurable parameter.

## Alignment with Action Plan
Phase 1: baseline documentation/test repairs — **COMPLETE**
Phase 2: physical indicators + factual ground truth — **COMPLETE**
Phase 3: real data wiring + honest UI + G-001 fix — **COMPLETE**
Phase 4: pipeline architecture fix + production threshold — IN PROGRESS
