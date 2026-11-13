# Architecture Decision Records — aNEOS Suite

_Historical design basis: 2026-03-06–10 / v1.1.0. Current-status review: 2026-11-13 / v2.2.0b2 plus main 3dd47f1._
_Concept document: README.md + docs/scientific/scientific-documentation.md_

**Current-status note (2026-11-13):** this cumulative record includes historical
claims, not universal implementation or scientific-validation evidence. ADR-065/068/069
and later acquisition/persistence decisions govern installed screening. The
[discovery/integrity review](../engineering/discovery-integrity-roadmap-review-2026-11-13.md)
proposes #86/#87 within existing boundaries; it accepts no incompatible contract or
new bounded context. Review material contracts before implementation; do not reuse
a concurrent PR's ADR number.

ADRs capture every significant architectural decision found in the aNEOS codebase,
including implicit decisions and identified risks. Status values:
**Accepted** | **Superseded** | **Risk** | **Deferred** | **Open** | **Concept-Misaligned**

---

## Data Acquisition

### ADR-001: Four External APIs as Sole Authoritative Data Sources

**Status**: Accepted / Risk

**Context**
aNEOS has no native sensors. All orbital, physical, spectral, and close-approach
data must come from published catalogs.

**Decision**
Use four external sources with a configurable priority order
(`APIConfig.data_sources_priority: ["SBDB", "NEODyS", "MPC", "Horizons"]`),
each implemented as its own `DataSourceBase` subclass:
- `SBDBSource` (`data/sources/sbdb.py`) — JPL Small Body Database
- `NEODySSource` (`data/sources/neodys.py`) — ESA NEO Dynamics Site
- `MPCSource` (`data/sources/mpc.py`) — Minor Planet Center
- `HorizonsSource` (`data/sources/horizons.py`) — JPL Horizons System

**Consequences**
- (+) Leverages peer-reviewed orbital solutions; no data collection cost
- (-) **Risk (ADR-012)**: When all four APIs are unavailable the system silently
  falls back to simulation. The maturity assessment (August 2025) named this
  the highest operational risk.
- (-) No local bulk-download path; every run requires live network access

**ADR-001 Update (2026-03-08 / Phase 10)**
A fifth data path — the SBDB CAD API (`cad.api`) — is now called by
`DataFetcher._fetch_close_approaches()` as a supplemental (non-blocking) source
exclusively for close-approach data. It is not part of the priority-ordered
source list for orbital elements; it runs after the primary fetch completes.

**Files**: `aneos_core/data/sources/`, `aneos_core/config/settings.py:APIConfig`,
`aneos_core/data/fetcher.py:_fetch_close_approaches()`

---

### ADR-002: DataFetcher — Multi-Source Orchestrator with Concurrent Fetching

**Status**: Accepted

**Context**
Individual source calls are slow; needing multiple sources per object multiplies
latency. Data quality varies by source; a single-source strategy misses fields.

**Decision**
`DataFetcher` (`data/fetcher.py`) uses a `ThreadPoolExecutor` (default 4 workers)
to query sources in parallel, merges results using data-quality scoring, and
writes to `CacheManager`. Priority is SBDB → NEODyS → MPC → Horizons; richer
results override sparse ones field-by-field.

**Consequences**
- (+) Reduces per-object latency; maximises field completeness
- (-) Thread pool + cache writes require careful lock discipline; the cache
  `RLock` was specifically introduced to fix thread-safety regressions
- (-) Concurrent API hits may trigger rate-limiting on some sources

**Files**: `aneos_core/data/fetcher.py`, `aneos_core/data/cache.py`

---

### ADR-003: Thread-Safe LRU Cache with TTL and Optional Disk Persistence

**Status**: Accepted

**Context**
Repeated analysis runs re-query the same objects. Redundant API calls waste quota
and time. The original feature-rich cache broke the test suite after a refactor.

**Decision**
`CacheManager` (`data/cache.py`) uses `RLock` for thread safety, implements
best-effort LRU eviction, TTL expiry, and optional JSON/pickle disk persistence.
A legacy-compatible `CacheEntry(key, value)` / `CacheStats` surface was
deliberately re-introduced to restore the 61-test regression suite.

**Consequences**
- (+) 85%+ cache hit rate on repeated analysis; restores test compatibility
- (-) Dual surface area (legacy + new) increases cognitive overhead
- (-) Pickle persistence is a latent deserialization security risk if cache
  files come from untrusted sources

**Files**: `aneos_core/data/cache.py`

---

### ADR-004: Time-Chunked Historical Polling (5-Year Windows)

**Status**: Accepted

**Context**
200-year historical NASA CAD datasets contain millions of close-approach records
that cannot fit in memory simultaneously.

**Decision**
`HistoricalChunkedPoller` (`polling/historical_chunked_poller.py`) processes
data in configurable time windows (`ChunkConfig`):
- `chunk_size_years: 5` — time window per batch
- `overlap_days: 7` — prevents boundary-straddle losses
- `max_objects_per_chunk: 50,000` — memory ceiling per chunk
- `batch_size: 1,000` — parallel processing unit
- `retry_attempts: 3` — failure resilience

**Consequences**
- (+) Memory-bounded; restartable at chunk boundaries
- (-) Boundary overlap logic is not independently validated; potential for
  duplicates or missed objects at 7-day seams
- (-) 200 years ÷ 5-year chunks = 40 sequential API calls minimum per run

**Files**: `aneos_core/polling/historical_chunked_poller.py`

---

## Configuration & Settings

### ADR-005: Structured Dataclass Configuration with Environment Variable Override

**Status**: Accepted

**Context**
The original monolith used a global `CONFIG` dictionary. This made testing
difficult and prevented environment-specific configuration.

**Decision**
`settings.py` provides five typed dataclasses:
- `ThresholdConfig` — per-indicator detection thresholds
- `WeightConfig` — per-category scoring weights
- `APIConfig` — endpoint URLs, timeouts, retry parameters
- `ANEOSConfig` — root configuration combining all sub-configs
- `ConfigManager` — loads JSON/YAML files and overlays environment variables

Boolean env vars are parsed via `_get_bool_env()` accepting `1/true/yes/on`.
The `advanced_scoring_weights.json` file (`config/`) allows runtime weight tuning
without code changes.

**Consequences**
- (+) Type-safe; testable; environment-overridable
- (+) `advanced_scoring_weights.json` enables tuning without a code deploy
- (-) YAML support is optional (`try/except import yaml`); silent degradation
  to JSON-only if PyYAML missing
- (-) No schema validation; a malformed config file logs a warning but may
  silently use defaults

**Files**: `aneos_core/config/settings.py`,
`aneos_core/config/advanced_scoring_weights.json`

---

## Core Domain Models

### ADR-006: Typed Dataclass Models with Alias Synchronization

**Status**: Accepted

**Context**
The original codebase passed raw dictionaries between components. Fields were
named differently across sources (e.g., `ascending_node` vs `ra_of_ascending_node`,
`arg_of_periapsis` vs `argument_of_perihelion`).

**Decision**
`models.py` defines typed dataclasses: `OrbitalElements`, `PhysicalProperties`,
`NEOData`, `AnalysisResult`, `CloseApproach`. `OrbitalElements.__post_init__`
calls `_synchronize_aliases()` to keep dual-naming conventions in sync and
`_validate()` to enforce physical bounds (eccentricity in [0,1), inclination
in [0°,180°]).

**Consequences**
- (+) Self-documenting; catches data quality issues at ingestion time
- (+) Alias sync enables backward compatibility across API source format differences
- (-) Validation raises on construction; callers must catch `ValueError`
- (-) `OrbitalElements` mixes orbital mechanics fields with physical properties
  (diameter, albedo, rotation period) — a domain model violation that
  conflates two distinct concepts

**Files**: `aneos_core/data/models.py`

---

## Analysis & Indicator System

### ADR-007: Pluggable Indicator Architecture with Abstract Base

**Status**: Accepted

**Context**
The concept document specifies 5 indicator categories with 11 named indicators.
The system needs to support adding new indicators without touching the scoring pipeline.

**Decision**
`AnomalyIndicator` (`analysis/indicators/base.py`) is an ABC with:
- Abstract `evaluate(neo_data: NEOData) -> IndicatorResult`
- `NumericRangeIndicator` — for threshold-based detection
- `StatisticalIndicator` — for population-deviation scoring
- `IndicatorConfig(weight, enabled, confidence_threshold)` per instance
- `IndicatorResult(raw_score, weighted_score, confidence, metadata,
  contributing_factors)` as the universal output contract

Concrete indicator modules:
- `indicators/orbital.py` — Eccentricity, Inclination, SemiMajorAxis,
  OrbitalResonance, OrbitalStability (5 indicators)
- `indicators/velocity.py` — VelocityShift, Acceleration, VelocityConsistency,
  InfinityVelocity (4 indicators)
- `indicators/temporal.py` — CloseApproachRegularity, ObservationGap,
  Periodicity, TemporalInertia (4 indicators)
- `indicators/geographic.py` — SubpointClustering, GeographicBias (2 indicators)

**Concept alignment — PARTIAL MISALIGNMENT**
The concept document specifies a **Physical Indicators** category
(`diameter_anomalies`, `albedo_anomalies`, `spectral_anomalies`). The scoring
module (`scoring.py`) maps these names, but **no `indicators/physical.py` file
exists**. Physical anomaly detection is absent from the indicator pipeline.

**Consequences**
- (+) 15 implemented indicators; easily extensible
- (-) Physical indicator category is mapped in scoring but never evaluated —
  a gap between concept and implementation
- (-) Total: 15 indicators implemented vs 11 specified in concept doc (the
  extra indicators are not documented in the scientific methodology)

**Files**: `aneos_core/analysis/indicators/`

---

### ADR-008: Dual Scoring Systems — Standard vs Advanced (ATLAS)

**Status**: Accepted — Keep Both (Phase 23): scoring.py = σ gate; advanced_scoring.py = explanation layer. Call-site contract documented.

**Context**
Two independent scoring systems exist in `aneos_core/analysis/`:
1. `scoring.py` — `ScoreCalculator` using 6 category maps, 4-tier classification
   (`natural/suspicious/highly_suspicious/artificial`), thresholds at
   0.0 / 0.30 / 0.60 / 0.80
2. `advanced_scoring.py` — `AdvancedScoreCalculator` (ATLAS) using 6 weighted
   clue categories with continuous [0,1] scoring, debris penalty system, and
   configurable via `advanced_scoring_weights.json`

**Decision (implicit)**
Both systems were developed independently. `advanced_scoring.py` is used by
`EnhancedAnalysisPipeline` and labelled the ATLAS implementation. `scoring.py`
is the base layer used by `AnalysisPipeline`.

**ATLAS Clue Categories** (advanced_scoring.py):
| Category | Weight | Purpose |
|----------|--------|---------|
| Encounter Geometry | 0.15 | Distance + relative speed |
| Orbit Behavior | 0.25 | Repeat passes, non-gravitational acceleration |
| Physical Traits | 0.20 | Area-to-mass ratio, radar, thermal |
| Spectral Identity | 0.20 | Color curve anomalies |
| Dynamical Sanity | 0.15 | Yarkovsky drift check |
| Human Origin | 0.05 | Space debris correlation (veto) |

**Concept alignment — PARTIAL MISALIGNMENT**
The scientific documentation defines 5 indicator categories; ATLAS uses 6. The
Encounter Geometry category has no explicit counterpart in the concept document.

**Consequences**
- (-) Two scoring philosophies coexist with no documented decision on which is
  authoritative for production use
- (-) `advanced_scoring.py` disables configuration warnings via a comment
  labelled `# EMERGENCY` — a code smell indicating rushed workaround

**Files**: `aneos_core/analysis/scoring.py`,
`aneos_core/analysis/advanced_scoring.py`

### ADR-008 Update (2026-03-07)
**Decision**: `ValidatedSigma5ArtificialNEODetector` designated as the canonical
production detector (DetectorType.VALIDATED, priority 0 in DetectionManager).
Four superseded detectors archived to `detection/_archive/`. ATLAS
(`AdvancedScoreCalculator`) designated canonical for full-data pipeline scenarios
where close-approach geometry and physical traits are available; Standard
`ScoreCalculator` retired from new production use. EMERGENCY suppressions removed
from `advanced_scoring.py`; configuration now logs at DEBUG level.
**Empirical basis**: Ground truth validation (Phase 3) confirms ValidatedSigma5 achieves
sensitivity=1.00, specificity=1.00 on 3 artificials + 20 natural JPL NEOs.
**⚠ Caveat (VALIDATION_INTEGRITY audit, 2026-03-08)**: Corpus is small (N=3 artificials,
N≈20 naturals); thresholds hand-tuned to the same objects; F1=1.000 is spurious on this
dataset. Blind-test set not yet built. Performance on unseen objects unverified. See
`docs/scientific/VALIDATION_INTEGRITY.md` for full remediation roadmap.

---

### ADR-009: Analysis Pipeline with ThreadPoolExecutor Concurrency

**Status**: Accepted

**Context**
Running 15 independent indicators sequentially per object is too slow for
50,000-object batches.

**Decision**
`AnalysisPipeline` (`analysis/pipeline.py`) uses `ThreadPoolExecutor(max_workers=10)`
with `as_completed()` to evaluate all indicators concurrently per object. Async
`asyncio` patterns are also used in the pipeline's entry points for
non-blocking orchestration.

**Consequences**
- (+) Near-linear speedup for indicator evaluation up to 10 workers
- (-) Python GIL limits true CPU parallelism; numpy-heavy indicators may not
  benefit fully
- (-) Mixed async/thread paradigms create complexity; callers must be aware
  of the execution context

**Files**: `aneos_core/analysis/pipeline.py`

---

### ADR-010: Impact Probability as Secondary Mission Component

**Status**: Accepted — Concept-Aligned

**Context**
The concept document (README.md) defines a **dual mission**: (1) Artificial NEO
detection and (2) Comprehensive Planetary Defense including Earth and Moon
impact probability assessment.

**Decision**
`ImpactProbabilityCalculator` (`analysis/impact_probability.py`) implements:
- Collision cross-section with gravitational focusing (Earth and Moon)
- Monte Carlo uncertainty propagation (`scipy.integrate.solve_ivp`)
- Gravitational keyhole analysis for resonant returns
- Time-dependent impact probability evolution
- Impact energy (MT TNT) and crater diameter estimation
- Earth vs Moon impact ratio comparison

`ImpactEnhancedPipeline` (`analysis/impact_enhanced_pipeline.py`) wires impact
assessment into the standard analysis pipeline output.

`ImpactProbability` dataclass carries: `collision_probability`,
`collision_probability_per_year`, `time_to_impact_years`,
`probability_uncertainty (95% CI)`, `calculation_confidence`, `data_arc_years`,
and full Moon-vs-Earth comparative metrics.

**Concept alignment**: Full. README example results match the code's output
structure (e.g., Earth: 8.52×10⁻¹⁵, Moon: 6.65×10⁻⁹ for 2024 YR4).

**Consequences**
- (+) Planetary defense mission is fully implemented and concept-aligned
- (-) Impact calculator depends on orbital uncertainty data that may not always
  be present; graceful degradation path unclear

**Files**: `aneos_core/analysis/impact_probability.py`,
`aneos_core/analysis/impact_enhanced_pipeline.py`

---

## Detection System

### ADR-011: DetectionManager — Priority-Based Detector Registry

**Status**: Accepted — Pipeline bypass fixed (Phase 19)

**Context**
Multiple detector versions exist (see ADR-018). Something must choose which
detector runs for a given request.

**Decision**
`DetectionManager` (`detection/detection_manager.py`) maintains a `DetectorType`
enum and loads detectors in priority order:

| Priority | Detector | Type |
|----------|----------|------|
| 0 (highest) | ValidatedSigma5ArtificialNEODetector | VALIDATED |
| 1 | MultiModalSigma5ArtificialNEODetector | MULTIMODAL |
| 2 | ProductionArtificialNEODetector | PRODUCTION |
| 3 | CorrectedSigma5ArtificialNEODetector | CORRECTED |
| 4 (lowest) | Sigma5ArtificialNEODetector | BASIC |

`DetectorType.AUTO` selects the highest-priority detector that loads
successfully. Each detector is wrapped through a type-specific adapter method
to normalize output to `DetectionResult`.

**Previously: Critical misalignment (RESOLVED — Phase 19)**
`automatic_review_pipeline.py` previously hardcoded `MultiModalSigma5ArtificialNEODetector`
at line 52, bypassing `DetectionManager` entirely. Fixed in Phase 19: the pipeline
now uses `DetectionManager(preferred_detector=DetectorType.AUTO)` at line 175,
correctly routing through the priority-0 VALIDATED detector.

**Consequences**
- (+) `DetectionManager` provides a clean abstraction for all callers
- (+) Pipeline now honours `DetectorType.AUTO` priority order (VALIDATED → MULTIMODAL → …)
- (+) **Phase 19 fix**: `automatic_review_pipeline.py` line 175 uses `DetectionManager(AUTO)`
- (-) Manual wrappers per detector type create maintenance burden

**Files**: `aneos_core/detection/detection_manager.py`,
`aneos_core/pipeline/automatic_review_pipeline.py`

---

### ADR-012: Unified Detection Interface and OrbitalElementsNormalizer

**Status**: Accepted

**Context**
Each detector was developed independently with different input format expectations
(`a`/`e`/`i` vs `semi_major_axis`/`eccentricity`/`inclination`, various nested
structures).

**Decision**
`interfaces/detection.py` defines:
- `DetectionResult` — canonical output: `is_artificial`, `confidence`,
  `sigma_level`, `artificial_probability`, `classification`, `analysis`,
  `risk_factors`, `metadata`
- `OrbitalElementsNormalizer.normalize()` — resolves dual-naming conventions
  to a single dict supporting both short and long key forms simultaneously
- `ArtificialNEODetector`, `EnhancedDetector`, `MultiModalDetector` — ABCs
  defining the required interface for each detector tier

`interfaces/unified_analysis.py` adds `aNEOSAnalysisResult` and
`AnalysisCapability` enum for the system's external-facing contract.

**Consequences**
- (+) New detectors can integrate without touching calling code
- (-) `DetectionResult._determine_classification()` re-implements classification
  logic that also exists in `scoring.py` — duplicated logic

**Files**: `aneos_core/interfaces/detection.py`,
`aneos_core/interfaces/unified_analysis.py`

---

### ADR-013: Five Detector Variants as Successive Calibration Iterations

**Status**: Superseded — Canonical Is VALIDATED, Others Are Archives (archive completed Phase 4C.1)

**Context**
The sigma-5 detector was iteratively calibrated to address false-positive and
false-negative feedback from internal review cycles.

**Decision (implicit, across multiple commits)**
Each calibration iteration produced a new file rather than modifying the previous:

| File | Purpose |
|------|---------|
| `sigma5_artificial_neo_detector.py` | Original basic implementation |
| `corrected_sigma5_artificial_neo_detector.py` | First calibration pass |
| `production_artificial_neo_detector.py` | Calibrated for real-world thresholds |
| `multimodal_sigma5_artificial_neo_detector.py` | Multi-modal evidence fusion |
| `validated_sigma5_artificial_neo_detector.py` | Scientifically validated version |
| `sigma5_corrected_statistical_framework.py` | Statistical framework refactor |

`ProductionArtificialNEODetector` uses hard-coded production parameters
(`semi_major_axis_threshold: 1.5 AU`, `eccentricity_threshold: 0.6`,
`inclination_low: 50°`, `confidence_threshold: 0.60`).

**Concept alignment**: The concept document calls for a single multi-modal
sigma-5 detector. The file proliferation is an engineering artifact, not a
concept requirement.

**Recommendation**: Designate `validated_sigma5_artificial_neo_detector.py`
as the sole canonical file; move others to `detection/_archive/`.

**Files**: `aneos_core/detection/`

---

### ADR-014: Bayesian Base Rate Correction (1-in-1000 Prior)

**Status**: Accepted — Concept-Aligned

**Context**
Raw sigma scores (e.g., 2.8σ = 99.5% statistical rarity) were previously
reported as detection confidence, misleading users into thinking 99.5%
probability of being artificial.

**Decision**
Implemented Bayesian inference with a prior `P(artificial) = 0.001` (0.1%
base rate). The calibrated artificial probability is computed as:

```
P(A|evidence) = P(evidence|A) * P(A) / P(evidence)
```

This separates "statistical rarity" from "artificial probability", matching
the scientific methodology requirement in the concept document.

**Concept alignment**: Full. README example explicitly shows the separation:
"Statistical Significance: 99.5% (2.8σ rarity)" vs "Calibrated Artificial
Probability: 0.1% (realistic assessment)".

**Consequences**
- (+) Scientifically honest; prevents users from over-interpreting sigma scores
- (-) The 0.1% prior is not empirically derived; it is an assumed value

**Files**: `aneos_core/detection/validated_sigma5_artificial_neo_detector.py`

---

## Validation System

### ADR-015: Five-Stage Multi-Stage Validation Pipeline

**Status**: Accepted — Concept-Aligned

**Context**
The concept document requires multi-stage validation with progressively higher
confidence thresholds and >90% false-positive rejection.

**Decision**
`MultiStageValidator` (`validation/multi_stage_validator.py`) implements a
sequential 5-stage pipeline:

| Stage | Name | FP Reduction Target |
|-------|------|---------------------|
| 1 | Data Quality Filter | 60% |
| 2 | Known Object Cross-Match | 80% |
| 3 | Physical Plausibility | 90% |
| 4 | Statistical Significance | 95% |
| 5 | Expert Review Threshold | >98% |

Each stage produces a `ValidationStageResult(stage_number, stage_name, passed,
score, confidence, false_positive_reduction, details, processing_time_ms)`.
`EnhancedAnalysisResult` wraps the original result with validation data via
`__getattr__` proxy for backward compatibility.

**Consequences**
- (+) Additive architecture preserves original pipeline; fail-safe wrapping
- (-) FP reduction targets are theoretical; not validated against real
  ground truth data

**Files**: `aneos_core/validation/multi_stage_validator.py`

---

### ADR-016: KAPPA SWARM — Radar Polarization Analysis

**Status**: Accepted

**Context**
Natural rocky asteroids have characteristic radar SC/OC circular polarization
ratios. Artificial objects (metallic shells, manufactured surfaces) would exhibit
systematically different ratios.

**Decision**
`RadarPolarizationAnalyzer` (`validation/radar_polarization_analysis.py`,
~2,036 lines) implements Arecibo/Goldstone/Green Bank radar data integration,
SC/OC ratio calculation against natural asteroid population distributions,
and sigma-deviation scoring.

**Concept alignment**: Aligned with the physical indicators category in the
scientific document. This SWARM implements the "non-natural spectral signatures"
and "geometric shapes" artificial object signatures.

**Files**: `aneos_core/validation/radar_polarization_analysis.py`

---

### ADR-017: LAMBDA SWARM — Thermal-IR Analysis

**Status**: Accepted

**Context**
Natural asteroids follow NEATM (Near Earth Asteroid Thermal Model) with
characteristic beaming parameter η and emissivity. Artificial objects with
different material properties would deviate.

**Decision**
`ThermalIRAnalyzer` (`validation/thermal_ir_analysis.py`, ~1,598 lines)
computes the beaming parameter deviation from the natural population distribution
and scores the thermal signature anomaly.

**Files**: `aneos_core/validation/thermal_ir_analysis.py`

---

### ADR-018: MU SWARM — Gaia Astrometric Calibration

**Status**: Accepted

**Context**
Non-gravitational accelerations (A1, A2 in Marsden parameterization) beyond the
Yarkovsky-effect model indicate active propulsion or anomalous forces.

**Decision**
`GaiaAstrometricCalibrator` (`validation/gaia_astrometric_calibration.py`,
~1,048 lines) integrates Gaia EDR3/DR3 via `astroquery.gaia` for sub-mas
proper motion validation, cross-matches in ICRS, and detects anomalous kinematics.
Includes `astropy` coordinate transforms and Gaia catalog TAP queries.

**Consequences**
- (+) Gaia EDR3 provides ~1.8 billion reference sources for cross-matching
- (-) Requires `astroquery.gaia` and network access to the Gaia TAP service;
  offline operation is not possible for this SWARM

**Files**: `aneos_core/validation/gaia_astrometric_calibration.py`

---

### ADR-019: CLAUDETTE SWARM — Statistical Testing & False Positive Prevention

**Status**: Accepted — Concept-Aligned

**Context**
The concept document requires multiple testing correction to prevent spurious
detections when analyzing thousands of indicators across thousands of objects.

**Decision**
Two components implement the CLAUDETTE SWARM:
- `StatisticalTesting` (`statistical_testing.py`, ~484 lines) — formal
  hypothesis testing per indicator: H₀ = natural, H₁ = artificial; Bonferroni
  and Benjamini-Hochberg multiple testing corrections; p-value calculation
- `FalsePositivePrevention` (`false_positive_prevention.py`) —
  space debris cross-matching (`SpaceDebrisMatch`), known artificial object
  exclusion, confusion matrix tracking

**Concept alignment**: Full. Scientific doc explicitly requires multiple testing
correction.

**Files**: `aneos_core/validation/statistical_testing.py`,
`aneos_core/validation/false_positive_prevention.py`

---

### ADR-020: THETA SWARM — Human Hardware Analysis

**Status**: Accepted

**Context**
Known human-made objects (dead satellites, rocket bodies, space probes) must be
identified and excluded or penalized before classifying objects as potentially
alien-artificial.

**Decision**
`HumanHardwareAnalyzer` (`validation/human_hardware_analysis.py`, ~1,403 lines)
cross-references objects against TLE databases and known spacecraft catalogs,
producing `HumanHardwareMatch` results with confidence scoring. A veto
mechanism can block artificial classification if human origin probability > 0.8.

`AdvancedScoringConfig.human_origin_weight = 0.05` in ATLAS applies a
`debris_penalty = 0.4` when human origin confidence exceeds `debris_confidence_threshold = 0.8`.

**Files**: `aneos_core/validation/human_hardware_analysis.py`

---

### ADR-021: Delta-BIC Model Comparison

**Status**: Accepted — Concept-Aligned

**Context**
The concept document requires formal model comparison between natural and
artificial orbital dynamics models, not just threshold-based scoring.

**Decision**
`DeltaBICAnalyzer` (`validation/delta_bic_analysis.py`, ~299 lines) implements
Bayesian Information Criterion model comparison. `DeltaBICResult` reports
`delta_bic`, `preferred_model`, `model_confidence`, `natural_model_bic`,
`artificial_model_bic`. Negative delta_bic favors the natural model; positive
favors artificial. Stage 3 of the multi-stage pipeline is enhanced with Delta-BIC
via `enhance_stage3_with_delta_bic()`.

**Files**: `aneos_core/validation/delta_bic_analysis.py`

---

### ADR-022: Uncertainty Analysis — Monte Carlo + Bootstrap Propagation

**Status**: Accepted — Concept-Aligned

**Context**
The concept document requires uncertainty quantification throughout the pipeline.
Single-point estimates without confidence intervals are insufficient for
peer-reviewed publication.

**Decision**
`UncertaintyAnalysis` (`validation/uncertainty_analysis.py`, ~622 lines)
implements:
- Monte Carlo propagation through the full indicator pipeline
- Bootstrap confidence estimation
- Sensitivity analysis (parameter perturbation)
- `UncertaintyResult` with `confidence_interval_95`, `confidence_interval_99`,
  `indicator_uncertainties`, `sensitivity_analysis`

**Files**: `aneos_core/validation/uncertainty_analysis.py`

---

### ADR-023: Large-Scale Monte Carlo False Positive Validation

**Status**: Accepted

**Context**
The claimed FPR of 5.7×10⁻⁷ cannot be validated on real data alone (no ground
truth). Synthetic population tests are required.

**Decision**
`LargeScaleMonteCarlo` (`validation/large_scale_monte_carlo.py`, ~488 lines)
generates synthetic natural NEO populations with realistic orbital distributions
and runs the full detection pipeline to measure empirical FPR.
`MonteCarloFalsePositiveValidation` (`monte_carlo_false_positive_validation.py`,
~410 lines) wraps this into a validation report format.

**Concept alignment**: Full. Scientific document explicitly requires synthetic
population validation.

**Consequences**
- (+) Enables FPR claims to be empirically supported rather than just theoretical
- (-) Synthetic populations may not perfectly represent the true natural NEO
  distribution; results are only as good as the generative model

**Files**: `aneos_core/validation/large_scale_monte_carlo.py`,
`aneos_core/validation/monte_carlo_false_positive_validation.py`

---

### ADR-024: Spectral Outlier Analysis

**Status**: Accepted

**Context**
Natural asteroid spectral classes (C, S, M, V, etc.) have known color/albedo
distributions. Objects outside these distributions warrant investigation.

**Decision**
`SpectralOutlierAnalyzer` (`validation/spectral_outlier_analysis.py`, ~432 lines)
classifies objects against known taxonomic classes and produces
`SpectralOutlierResult` with outlier significance and closest matching class.

**Files**: `aneos_core/validation/spectral_outlier_analysis.py`

---

### ADR-025: Physical Sanity Validator (Calibration Plan v1.2)

**Status**: Accepted — Concept-Aligned

**Context**
Output results must be physically self-consistent. Previous versions produced
contradictory outputs (e.g., "high risk" with P_impact = 0; sub-km crater for
a 500m object).

**Decision**
`PhysicalSanityValidator` (`validation/physical_sanity.py`, ~409 lines)
implements Calibration Plan v1.2 checks:
- Energy consistency: E = ½mv²
- Crater scaling laws verification (pi-scaling for specific size regimes)
- Risk label vs probability logical consistency
- Unit consistency across all output fields

`ValidationResult` enum: `PASS | WARNING | FAIL`. `PhysicalValidationResult`
carries `issues`, `warnings`, `corrected_values`, and `validation_notes`.

**Concept alignment**: Full. Physical plausibility is required by the scientific
methodology.

**Files**: `aneos_core/validation/physical_sanity.py`

---

### ADR-026: Consistency Validator — Contradiction Blocking

**Status**: Accepted

**Context**
Analysis outputs could be individually valid but mutually contradictory
(e.g., impact risk HIGH but probability = 0; artificial flag set but
classification says "natural").

**Decision**
`ConsistencyValidator` (`validation/consistency_validator.py`, ~303 lines)
defines three `ConsistencyViolation` types:
- `RISK_PROBABILITY_CONTRADICTION`
- `ARTIFICIAL_CLASSIFICATION_MISMATCH`
- `PHYSICS_VIOLATION`
- `SEVERE_INCONSISTENCY`

`ConsistencyResult.blocked_report = True` prevents report output when severe
contradictions are found.

**Concept alignment**: Full. Interim assessment compliance required.

**Files**: `aneos_core/validation/consistency_validator.py`

---

## Pipeline Orchestration

### ADR-027: Four-Stage ATLAS Automatic Review Pipeline

**Status**: Accepted — Concept-Aligned

**Context**
The concept document specifies a progressive filtering funnel from 50,000 raw
objects to ~50 expert-review candidates.

**Decision**
`AutomaticReviewPipeline` (`pipeline/automatic_review_pipeline.py`) implements:

| Stage | Max Objects | Threshold | Component |
|-------|-------------|-----------|-----------|
| RAW_OBJECTS | unlimited | — | HistoricalChunkedPoller |
| FIRST_STAGE_REVIEW | 50,000 | configurable | `DetectionManager(DetectorType.AUTO)` |
| MULTI_STAGE_VALIDATION | 500 | 0.60 | MultiStageValidator |
| EXPERT_REVIEW_QUEUE | 50 | 0.80 | Expert Queue |

`PipelineConfig` carries `StageConfig` per stage, each with
`max_candidates`, `score_threshold`, `processing_timeout_seconds`,
`retry_attempts`, `parallel_workers`.

**Previously critical issue (RESOLVED — Phase 19)**: Pipeline hardcoded
`MultiModalSigma5ArtificialNEODetector` instead of routing through `DetectionManager`.
Fixed: pipeline now uses `DetectionManager(AUTO)` (see ADR-011 for registry design).

**Concept alignment**: Full in structure and implementation.

**Files**: `aneos_core/pipeline/automatic_review_pipeline.py`

---

### ADR-028: Pipeline Integration Layer — Menu-to-Pipeline Bridge

**Status**: Accepted / Risk

**Context**
The 10,400-line menu system needs to launch the automatic pipeline without
directly importing all pipeline components (to avoid cascading import failures).

**Decision**
`PipelineIntegration` (`integration/pipeline_integration.py`) wraps all pipeline
imports in `try/except` with a `HAS_PIPELINE_COMPONENTS` flag. When components
are missing, it falls back gracefully — implementing the silent simulation
anti-pattern documented in ADR-030.

**Consequences**
- (+) Menu system stays operational even when optional pipeline packages absent
- (-) Same silent fallback risk as ADR-030: analysis may run on mock data
  without user awareness

**Files**: `aneos_core/integration/pipeline_integration.py`

---

## Utilities & Cross-Cutting Concerns

### ADR-029: Circuit Breaker Pattern for External API Calls

**Status**: Accepted

**Context**
Repeated calls to unavailable external APIs waste time and can trigger
rate-limiting bans. Failing fast is better than queuing 40 identical timeouts.

**Decision**
`CircuitBreaker` (`utils/patterns.py`) implements the standard three-state
pattern (CLOSED → OPEN → HALF_OPEN → CLOSED):
- Opens after `failure_threshold: 5` consecutive failures
- Waits `timeout_seconds: 60` before attempting HALF_OPEN
- Closes after `success_threshold: 3` consecutive successes in HALF_OPEN
- `monitoring_window: 300s` for failure rate tracking

**Consequences**
- (+) Prevents cascade failures when NASA/ESA APIs are degraded
- (-) Circuit breaker is implemented but its wiring to `DataFetcher` / individual
  source clients is not verified in the codebase scan

**Files**: `aneos_core/utils/patterns.py`

---

### ADR-030: Statistical Utilities as Centralized Library

**Status**: Accepted

**Context**
Sigma-to-p-value conversions, multiple testing corrections, and confidence
interval calculations were previously duplicated across modules.

**Decision**
`statistical_utils.py` (`utils/`) provides:
- `sigma_to_p_value(sigma)` — two-sided via `scipy.stats.norm.cdf`
- `sigma_to_confidence_level(sigma)` — returns percentage
- Multiple testing correction utilities

**Concept alignment**: Full. Centralized, reproducible sigma calculation is a
core scientific methodology requirement.

**Files**: `aneos_core/utils/statistical_utils.py`

---

## User Interface

### ADR-031: Monolithic Rich-Based Interactive Menu

**Status**: Accepted / Technical Debt

**Context**
25+ distinct analytical workflows need to be discoverable and accessible from
a single entry point for researchers without CLI expertise.

**Decision**
`aneos_menu.py` (~10,400 lines) is a single-file Rich-based terminal UI that
directly invokes all pipeline components, reporting modules, API launchers, and
diagnostic tools. `aneos.py` (~567 lines) is a thin CLI wrapper that maps
named commands to menu functions.

**Consequences**
- (+) Zero setup for interactive use; all features discoverable from one surface
- (-) 10,400-line file makes automated testing difficult; menu logic and business
  logic are mixed
- (-) Inline `try/except` throughout produces the silent-fallback risk (ADR-032)

**Files**: `aneos_menu.py`, `aneos.py`

---

### ADR-032: Silent Simulation Fallback on Integration Failure

**Status**: Mitigated (Phase 23) — All invocation-time failures now either raise `ImportError` with install instructions (PHAMoidScanner — Phase 21E) or log clearly and return safely (WebhookNotificationChannel, AlertManager — Phase 23C).

**Context**
Optional pipeline components and external API dependencies cause initialization
failures at runtime. The choice was made to continue silently rather than fail.

**Decision (implicit)**
`HAS_PIPELINE_COMPONENTS`, `HAS_ANALYSIS`, `HAS_ML`, `HAS_RICH`, `HAS_FASTAPI`
guard flags throughout the codebase silently substitute mock results when
real data paths fail. The maturity assessment (August 2025) named this a key risk.

**Concept alignment — MISALIGNED**
The concept document claims "Real Data Integration: NASA/JPL data with
comprehensive explanations." Silent simulation directly contradicts this.

**Consequences**
- (-) Operational runs may analyse synthetic data while appearing to use real NEO data
- (-) No user warning when falling back to simulation mode
- (-) Undermines any detection claim made from a run that silently fell back

**Recommendation**: Replace silent fallbacks with explicit
`IntegrationError` exceptions and a pre-flight health check that validates
required dependencies before any analysis run starts.

**Partial remediation (Phase 16 — ADR-050)**: Silent fallback eliminated in the menu
orbital data path (`_get_test_data()` now raises `ValueError` for unknown designations).
Silent simulation risk remains in `PipelineIntegration` and API `HAS_*` guard paths.

**Files**: `aneos_core/integration/pipeline_integration.py`,
`aneos_api/app.py`, `aneos_menu.py`

---

## ML System

### ADR-033: Machine Learning Module — Scaffolded, Partially Integrated

**Status**: Partially Active (core dependency for monitoring) — Training deferred (no ground truth dataset)

**Context**
The concept document mentions ML as a future methodology component. A full ML
scaffold was built in `aneos_core/ml/` during Phase 6 planning.

**Decision**
`aneos_core/ml/` contains:
- `features.py` — `FeatureVector` extraction from NEO observables
- `models.py` — `IsolationForest`, `RandomForest`, `OneClassSVM`, `DBSCAN`,
  `PCA`, optional `torch` neural network; all behind `HAS_SKLEARN`/`HAS_TORCH`
  guards
- `training.py` — `TrainingPipeline` for supervised/unsupervised training
- `prediction.py` — `RealTimePredictor` for live inference; defines `Alert` class

**Partial integration found**: `monitoring/alerts.py` imports `Alert` from
`ml.prediction`, meaning the ML module IS partially wired into the monitoring
alert system — it is not completely isolated.

**Concept alignment**: Aligned with Phase 6 plan; explicitly deferred per
CLAUDE.md (no ground truth dataset yet).

**Consequences**
- (+) ML infrastructure ready when ground truth becomes available
- (-) `monitoring/alerts.py` hard dependency on `ml.prediction.Alert` means
  ML unavailability breaks alerting
- (-) `torch` dependency is optional but its absence silently disables neural
  network models with no user notification

**Files**: `aneos_core/ml/`, `aneos_core/monitoring/alerts.py`

---

## API & Web Layer

### ADR-034: FastAPI REST API with 52 Endpoints

**Status**: Accepted

**Context**
Researchers need programmatic access to analysis capabilities beyond the
interactive CLI.

**Decision**
`aneos_api/app.py` uses FastAPI with:
- `CORSMiddleware` and `GZipMiddleware`
- Lifespan-managed startup/shutdown
- Six endpoint groups: `analysis`, `enhanced_analysis`, `monitoring`,
  `prediction`, `streaming`, `admin`
- JWT authentication via `aneos_api/auth.py`
- SQLAlchemy 2.0+ models via `aneos_api/database.py`
- Rich dashboard at `aneos_api/dashboard.py`

**ADR-034 Update (2026-03-08 / Phase 10)**
`make spec` target now generates `docs/api/openapi.json` from live FastAPI
introspection. CI checks for spec drift (`git diff --stat docs/api/openapi.json`).
OpenAPI spec is no longer hand-maintained — drift risk resolved.

**Concept alignment**: Full. 52+ endpoints confirmed; OpenAPI spec autogenerated
and committed at every functional change.

**Consequences**
- (+) Production-grade framework; async handlers; auto-generated OpenAPI (when FastAPI available)
- (-) All FastAPI imports wrapped in `try/except HAS_FASTAPI` — API silently
  unavailable if FastAPI not installed (ADR-032 pattern)
- (-) Direct domain type usage in endpoint responses; no DTO layer

**Files**: `aneos_api/app.py`, `aneos_api/endpoints/`, `aneos_api/auth.py`

---

## Data Persistence

### ADR-035: SQLite Development / PostgreSQL Production Database Strategy

**Status**: Accepted

**Context**
The system needs persistent storage for analysis results, candidate queues,
and audit trails. Deployment ranges from a researcher's laptop to Kubernetes.

**Decision**
SQLAlchemy 2.0+ ORM abstracts the database backend:
- Local/dev: SQLite (`aneos.db` at repo root)
- Production: PostgreSQL 15 (Docker Compose env var `ANEOS_DATABASE_URL`)

**Consequences**
- (+) Zero-config local development; production-grade in containers
- (-) `aneos.db` at repo root is tracked/committed and could contain sensitive
  analysis results
- (-) SQLite and PostgreSQL behavioural differences (concurrent writes, JSON
  fields, full-text search) may expose bugs not caught in development

**Files**: `aneos_api/database.py`, `docker-compose.yml`

---

## Infrastructure & Deployment

### ADR-036: Container-First Deployment with Full Observability Stack

**Status**: Accepted / Partially Broken

**Context**
Reproducible deployment across environments requires isolation. Production
monitoring requires metrics and alerting.

**Decision**
Docker Compose (`docker-compose.yml`) defines six services:
`aneos-api`, `postgres`, `redis`, `nginx`, `prometheus`, `grafana`.

Kubernetes manifests (`k8s/`) provide `deployment.yml`, `postgres.yml`,
`redis.yml`.

Prometheus (`prometheus.yml`) scrapes the API; Grafana dashboards consume
Prometheus metrics.

**Broken elements found**:
- `docker-compose.yml` references `./init.sql` (does not exist in repo)
- `docker-compose.yml` references `./ssl/` for Nginx (does not exist in repo)
- No CI/CD pipeline exists to validate the container build on commit

**Consequences**
- (+) Full observability stack wired from the start
- (-) `docker-compose up` fails without manual creation of `init.sql` and SSL
  certificates — deployment is broken out-of-the-box

**Files**: `docker-compose.yml`, `Dockerfile`, `k8s/`, `prometheus.yml`,
`nginx.conf`

---

### ADR-037: Redis for Caching and Real-Time Streaming

**Status**: Accepted / Unverified

**Context**
Production API requires distributed caching (beyond the in-process `CacheManager`)
and real-time event streaming for the `/streaming` endpoint.

**Decision**
Redis 7 is included in Docker Compose with AOF persistence
(`--appendonly yes`). The `aneos_api/endpoints/streaming.py` endpoint is
intended to use Redis pub/sub for live alert delivery.

**Consequences**
- (+) Redis infrastructure defined
- (-) Redis is listed as a service dependency but no code path in the current
  codebase verifies it is actively used by the Python application; the
  integration may be stub-only

**Files**: `docker-compose.yml:redis`, `aneos_api/endpoints/streaming.py`

---

## Ground Truth & Validation

### ADR-038: Ground Truth Dataset — Operational (Phase 3/4 Complete)

**Status**: Accepted (Phase 3/4 complete; blind-test set still needed)

**Context**
Phase 3/4 completed the ground truth dataset and ran external validation.
The `GroundTruthDatasetBuilder` is now operational with a labelled corpus of 9
confirmed artificial heliocentric objects and up to 250 natural NEOs from JPL SBDB.

**Decision**
`GroundTruthDatasetBuilder` (`datasets/ground_truth_dataset_preparation.py`)
compiles confirmed artificial objects — 3 directly from SBDB (Tesla Roadster,
2020 SO, J002E3) plus 6 hardcoded Horizons spacecraft with physical fallbacks
(DSCOVR, OSIRIS-REx, STEREO-A/B, Spitzer, Hayabusa2) — and natural NEOs from
SBDB bulk query. `GroundTruthValidator` runs the canonical detector and
measures sensitivity, specificity, F1, ROC-AUC at calibrated threshold 0.037.

**Empirical results (Phase 3 external validation):**

| Object | True Label | Sigma | P(artificial) | Classified |
|---|---|---|---|---|
| Tesla Roadster (2018 A1) | ARTIFICIAL | 5.76 | ~3.7% | ARTIFICIAL ✅ |
| 2020 SO (Centaur) | ARTIFICIAL | 6.97 | ~3.7% | ARTIFICIAL ✅ |
| J002E3 (Apollo 12 S-IVB) | ARTIFICIAL | 5.76 | ~3.7% | ARTIFICIAL ✅ |
| 20+ JPL natural NEOs | NATURAL | < 3.0 | ~0.1–0.2% | NATURAL ✅ |

Sensitivity=1.00, specificity=1.00, F1=1.00, ROC-AUC=1.00 at threshold 0.037.

**Remaining gaps:**
- Blind-test set (withheld labels) not yet built — current validation is internal
- Corpus size remains small (3 artificials); sensitivity on unseen artificials unverified
- FPR claim of 5.7×10⁻⁷ remains theoretical; Monte Carlo synthetic validation exists
  but is not equivalent to empirical validation on a large real corpus

**Concept alignment**: Substantially improved. Validated exemplars now exist.
Critical misalignment resolved for the current corpus size.

**Files**: `aneos_core/datasets/ground_truth_dataset_preparation.py`,
`docs/engineering/sigma5_success_criteria.md`

---

## Reporting

### ADR-039: Multi-Format Professional Report Generator

**Status**: Accepted

**Context**
Research outputs must be suitable for academic publication and data sharing.

**Decision**
`ReportGenerator` (`reporting/generators.py`) produces:
- Rich terminal output (if available)
- JSON export via `exporters.py`
- CSV and FITS format exports
- AI-generated academic-style validation notes via `ai_validation.py`
- Progress tracking via `progress.py`
- Visualizations via `visualizers.py` (matplotlib/plotly)
- Analytics summaries via `analytics.py`
- Professional suite wrapper via `professional_suite.py`

**Consequences**
- (+) Publication-ready output in multiple formats
- (-) `ai_validation.py` generates validation language — risk that AI-generated
  text overstates confidence of results (matches the documentation drift
  risk noted in maturity_assessment.md)

**Files**: `aneos_core/reporting/`

---

## Concept Alignment Summary (v1.0.0)

| Concept Claim | Implementation Status | Gap |
|---------------|----------------------|-----|
| Dual mission: detection + planetary defense | Implemented | None |
| Multi-modal Sigma-5 detection | Implemented + ground-truth validated (3 artificials, 20+ naturals) | Blind-test set missing |
| 5 indicator categories, 11 indicators | 4 categories, 15 indicators | Physical category missing |
| Bayesian base rate correction | Implemented | None |
| Statistical significance ≠ artificial probability | Implemented | None |
| Moon impact assessment | Implemented; 16-field ImpactResponse | None |
| Real NASA/JPL data integration | Implemented (SBDB, Horizons, CAD API, NEODyS, MPC); silent fallback risk remains | Silent fallback (ADR-032) |
| 99.99994% confidence / FPR 5.7×10⁻⁷ | Theoretically modeled; Monte Carlo FPR validator exists | Empirically unverified at scale |
| OpenAPI spec | Auto-generated via `make spec`; committed at every change | None |
| Population-level pattern analysis | Stage 1 implemented (Phase 11–19): PA-1 clustering, PA-3 harmonics, PA-5 correlation, PA-6 Stage 1 rendezvous (Option 15) | BC11; PA-6 Stage 2 (REBOUND) deferred |

---

## Population-Level Pattern Analysis — Design ADRs

_Added: 2026-03-08 | These ADRs define the architectural decisions for the
proposed Population Pattern Analysis extension (Bounded Context 11). They are
in "Concept" status — meaning the design is agreed but no code has been written.
All must be reviewed against existing data structures before implementation begins._

---

### ADR-040: Non-Gravitational Parameter Prerequisite

**Status**: Implemented (Phase 11) — `NonGravitationalParameters` VO in `aneos_core/models.py`; SBDB `nongrav=1` optional fetch; A2 bounds validation in `__post_init__`

**Context**
The Cross-Object Residual Correlation analysis (PA-5 in the implementation plan)
requires Yarkovsky / Marsden non-gravitational acceleration parameters (A1, A2, A3)
per object. These are available in SBDB (`sbdb.api?nongrav=1`) but are not currently
parsed or stored. Neither `OrbitalElements` nor `PhysicalProperties` carry
non-gravitational fields. Without this, the correlation analysis is blocked.

**Decision**
Add a `NonGravitationalParameters` value object to `aneos_core/data/models.py`:
```python
@dataclass
class NonGravitationalParameters:
    a1: Optional[float] = None   # Radial component (AU/day²)
    a2: Optional[float] = None   # Transverse (Yarkovsky dominant)
    a3: Optional[float] = None   # Normal component
    model: Optional[str] = None  # "marsden", "chebyshev", etc.
    epoch: Optional[datetime] = None
```
Extend `NEOData` with `nongrav: Optional[NonGravitationalParameters] = None`.
Extend `SBDBSource.fetch_orbital_elements()` to parse the `nongrav_params` list
from SBDB response and populate this field.
Extend `NEOData.to_dict()`/`from_dict()` to preserve the field across cache.

**Consequences**
- (+) Unlocks ADR-046 (Cross-Object Correlation) and enriches ATLAS Orbit Behavior clue
- (-) SBDB `nongrav_params` is present for only ~3% of NEOs (those with measured Yarkovsky)
- (-) Adds a field that will be `None` for 97% of objects; consumers must handle absence

**Files** (Phase 11): `aneos_core/data/models.py` (`NonGravitationalParameters` VO),
`aneos_core/data/sources/sbdb.py`

---

### ADR-041: Historical Close-Approach Fetch Strategy

**Status**: Implemented (Phase 17) — `DataFetcher.fetch_historical_approaches()` using CAD API; 7-day TTL cache; `first_observation_date`/`last_observation_date` wired into `NEOData`

**Context**
`DataFetcher._fetch_close_approaches()` (Phase 10) uses `date-min=now`, returning
only future approaches. The Synodic Period Harmonic Analysis (PA-3 / ADR-044)
requires historical approach epochs for each object — a time-series spanning
decades — to run a Lomb-Scargle periodogram. A minimum of 5 historical epochs
is needed for a reliable period estimate.

**Decision**
Add a `date_min: str = "now"` parameter to `_fetch_close_approaches()`. Add a
companion public method `fetch_historical_approaches(designation, years_back=30)`
that calls with `date-min` set to the ISO date `years_back` years before today.
This method is NOT wired into the standard `fetch_neo_data()` path — it is
called explicitly by the population pattern analysis module to avoid adding a
30-year historical query to every single-object detection call.

**Scope control**: historical approach fetch is performed once per object per
`NetworkAnalysisSession` and cached separately with a 7-day TTL using cache key
`"neo_cad_hist:{designation}:{years_back}"`.

**Consequences**
- (+) Does not slow down existing per-object detection path
- (+) Cached — repeated pattern analysis runs do not re-query SBDB
- (-) Adds a second SBDB API call pattern that must be documented separately
- (-) Objects with fewer than 5 historical approaches are skipped for harmonic analysis

**Files** (Phase 17): `aneos_core/data/fetcher.py`

---

### ADR-042: Population Pattern Analysis — New Bounded Context Architecture

**Status**: Concept / Design Agreed

**Context**
v1.0 detects anomalies in individual objects. The proposed extension detects
anomalies at the population level: non-random groupings, temporal resonances,
correlated non-gravitational accelerations, and repeated mutual close approaches.
These are structurally different from per-object detection — they require a
collection of `NEOData` objects as input, not a single object.

**Decision**
Create `aneos_core/pattern_analysis/` as a new, self-contained module:

```
aneos_core/pattern_analysis/
├── __init__.py
├── session.py          # NetworkAnalysisSession (aggregate root)
├── clustering.py       # OrbitalElementClusterer (PA-1)
├── harmonics.py        # SynodicHarmonicAnalyzer (PA-3)
├── rendezvous.py       # RendezvousDetector (PA-6, HAS_REBOUND guard)
├── correlation.py      # NonGravCorrelator (PA-5, requires ADR-040)
└── network_sigma.py    # NetworkSigmaCombiner (Fisher's method)
```

**Data flow:**
```
Batch of NEOData objects
        │
        ▼
NetworkAnalysisSession
        │
        ├─▶ OrbitalElementClusterer  ──→ List[OrbitalCluster]
        │
        ├─▶ SynodicHarmonicAnalyzer  ──→ List[HarmonicSignal]
        │
        ├─▶ NonGravCorrelator         ──→ CorrelationMatrix  [requires ADR-040]
        │
        ├─▶ RendezvousDetector        ──→ List[RendezvousPair]  [optional, HAS_REBOUND]
        │
        └─▶ NetworkSigmaCombiner      ──→ NetworkReport
```

**Context boundary rules:**
1. Population Pattern Analysis READS `NEOData` from the Data Acquisition context
   but NEVER modifies individual `NEOData` records.
2. Per-object `DetectionResult` from BC5 is used as an input filter
   (only analyze objects that scored NOTABLE or above, σ ≥ 1.0).
3. Output is a `NetworkReport` consumed by BC7 (Reporting) and BC8 (API).
4. Each sub-module is independently runnable and can be enabled/disabled via
   a `PatternAnalysisConfig` flag.

**Dependency strategy:**
- Hard dependencies: `numpy`, `scipy` (already in requirements)
- Soft dependency: `hdbscan` — behind `HAS_HDBSCAN` guard; fallback to `sklearn.cluster.DBSCAN`
- Optional dependency: `rebound` — behind `HAS_REBOUND` guard; rendezvous module
  disabled if absent

**Consequences**
- (+) Fully modular; existing v1.0 pipeline untouched
- (+) Each sub-module independently testable with synthetic data
- (-) Requires ADR-040 (non-grav) and ADR-041 (historical CAD) before PA-5 and PA-3
  respectively; clustering (PA-1) can start immediately
- (-) `NetworkAnalysisSession` holding a batch of `NEOData` objects is memory-intensive;
  needs a configurable object cap (`max_objects: int = 500`)

**Files** (Phase 11): `aneos_core/pattern_analysis/`

---

### ADR-043: Orbital Element Clustering (PA-1)

**Status**: Concept / Ready to Implement

**Context**
NEO families (Hungaria, Phocaea, Alinda, etc.) occupy statistically non-random
regions of (a, e, i) element space. Clusters not matching known families may
indicate a shared dynamical origin, injection mechanism, or — for the primary
mission — coordinated orbital placement. This is the only PA sub-module with
no external prerequisites.

**Decision**
Normalize (a, e, i, ω, Ω) to [0,1] using population min-max bounds from the
Granvik et al. 2018 debiased NEO model (reference distribution hardcoded as
constants from the published Table 1). Apply HDBSCAN (`min_cluster_size=5,
min_samples=3`) as the primary algorithm; fall back to `sklearn.cluster.DBSCAN
(eps=0.1, min_samples=5)` if `hdbscan` is unavailable.

**Cluster quality gate**: Report only clusters with N ≥ 10 members and
mean-element density > 3σ above the background density predicted by Granvik 2018.
Clusters matching known major families (Hungaria: i > 16°, a ≈ 1.88 AU;
Alinda: 3:1 resonance at a ≈ 2.50 AU) are labelled KNOWN_FAMILY and excluded
from anomaly scoring.

**Output**: `List[OrbitalCluster]` each carrying:
- `members: List[str]` — designations
- `centroid: Dict[str, float]` — (a, e, i, ω, Ω) mean
- `density_sigma: float` — sigma above Granvik background
- `known_family: Optional[str]` — if matched
- `p_value: float` — probability of N or more objects in this volume by chance

**Consequences**
- (+) No new hard dependencies; scipy KD-tree used for density estimation
- (+) Immediately implementable
- (-) Granvik 2018 reference distribution is for the complete NEO population;
  it does not account for discovery survey biases in the aNEOS working set
- (-) ω and Ω are periodic (0°–360°); wraparound distance metric needed

**Files** (Phase 11): `aneos_core/pattern_analysis/clustering.py`

---

### ADR-044: Synodic Period Harmonic Analysis (PA-3)

**Status**: Implemented (Phase 11) — `SynodicHarmonicAnalyzer` in `aneos_core/pattern_analysis/harmonics.py`; Lomb-Scargle on dense binary time grid; skipped-object count tracked in `analysis_metadata`

**Context**
Objects with repeated Earth close approaches at predictable intervals may be in
orbital resonance with Earth (natural) or exhibiting station-keeping behavior
(potentially artificial). Lomb-Scargle is the standard technique for unevenly
spaced time-series periodicity — appropriate here because approach epochs are
not evenly sampled.

**Decision**
Use `astropy.timeseries.LombScargle` (preferred; already in requirements) or
`scipy.signal.lombscargle` (fallback) applied to the time-series of Earth
close-approach epochs for each object.

Test target periods: `[10, 30, 90, 182, 365, 730, 1825, 3650, 36525]` days,
corresponding to major Earth–synodic resonances.

Scoring: For each target period P, compute the Lomb-Scargle power at P versus
the mean background power. Excess power > 3σ above Poisson null is flagged.
The Poisson null assumes random, Poisson-distributed approach epochs consistent
with the background flux density (i.e., no intrinsic periodicity).

**Minimum data requirement**: ≥ 5 historical approach epochs per object.
Objects with fewer are silently skipped and logged at DEBUG level.

**Output per object**: `Optional[HarmonicSignal]` carrying:
- `designation: str`
- `dominant_period_days: float`
- `power_excess_sigma: float`
- `target_periods_tested: List[float]`
- `p_value: float`

**Consequences**
- (+) `astropy.timeseries.LombScargle` is already available
- (-) Blocked until historical CAD data is fetched (ADR-041)
- (-) Most NEOs have fewer than 5 recorded Earth approaches in any 30-year window;
  the majority of objects will be silently skipped

**Files** (Phase 11): `aneos_core/pattern_analysis/harmonics.py`

---

### ADR-045: Pairwise Rendezvous Detection (PA-6)

**Status**: Stage 1 Implemented (Phase 19) — Stage 2 (REBOUND) deferred

**Context**
Temporary gravitational capture or repeated co-orbital encounters between two
NEOs are dynamically rare but scientifically significant. Detection requires
propagating both objects' orbits simultaneously — expensive at O(n²) scale.

**Decision**
**Stage 1 — MOID pre-filter** (no REBOUND required):
Compute the analytical Minimum Orbital Intersection Distance (MOID) for every
candidate pair. SBDB provides `moid_ld` (lunar distance) in the `sbdb.api`
response. Pairs with MOID > 0.05 AU are discarded without propagation.

**Stage 2 — REBOUND propagation** (requires `HAS_REBOUND` guard):
For surviving pairs, use REBOUND (`ias15` integrator, 10-year arc) to propagate
both objects and compute mutual distance at each timestep (Δt = 7 days).
Flag pairs where mutual distance < 0.05 AU occurs ≥ 3 times in the arc.

**Scope restriction**: Analysis is restricted to Potentially Hazardous Asteroids
(PHA: MOID ≤ 0.05 AU and H ≤ 22.0) to limit the working set to ~2,400 objects,
making O(n²) tractable: 2,400² / 2 ≈ 2.9 million pairs pre-MOID-filter.

**Output**: `List[RendezvousPair]` each carrying:
- `designation_a, designation_b: str`
- `n_encounters: int` — count of approaches within threshold
- `min_distance_au: float`
- `encounter_epochs: List[datetime]`
- `p_value: float` — probability of this many encounters by chance

**Consequences**
- (+) MOID pre-filter eliminates > 99% of pairs before expensive propagation
- (+) PHAMoidScanner deployed in `aneos_core/pattern_analysis/rendezvous.py`; exposed as Option 15 in `aneos_menu_v2.py`
- (-) REBOUND is a compiled C dependency; `HAS_REBOUND` guard required;
  graceful no-op if absent
- (-) Even with PHA restriction, 2.9M MOID evaluations + propagation for
  surviving pairs is a multi-hour computation on a single machine
- (-) Stage 2 (REBOUND propagation) deferred until dependency adopted

**Files**: `aneos_core/pattern_analysis/rendezvous.py`

---

### ADR-046: Cross-Object Non-Gravitational Correlation (PA-5)

**Status**: Implemented (Phase 11) — `NonGravCorrelator` in `aneos_core/pattern_analysis/correlation.py`; A2 Pearson correlation within clusters; single-epoch NaN logged; Bonferroni-corrected p-values

**Context**
Synchronized non-gravitational accelerations across multiple spatially proximate
objects could indicate a shared environmental forcing mechanism (e.g., common
cometary origin, mass-shedding event) or — for the primary mission — coordinated
active propulsion. Pearson correlation of A2 (transverse Yarkovsky) time-series
across objects in the same orbital cluster is the proposed signal.

**Decision**
Operate on objects that: (a) are members of the same `OrbitalCluster` from PA-1;
and (b) have SBDB-derived `NonGravitationalParameters.a2` values (ADR-040).

For each cluster with ≥ 5 objects having A2 measurements, build an n×n Pearson
correlation matrix using A2 values projected onto shared epoch coverage.
Flag pairs with |r| > 0.7 and p < 0.01 (Bonferroni-corrected for number of pairs).

**Output**: `CorrelationMatrix` carrying:
- `cluster_id: str`
- `designations: List[str]`
- `matrix: List[List[float]]` — Pearson r values
- `flagged_pairs: List[Tuple[str, str, float]]` — (desig_a, desig_b, r)
- `p_values: List[List[float]]` — uncorrected
- `bonferroni_threshold: float`

**Consequences**
- (+) Scientifically motivated; shared Yarkovsky signatures can confirm common
  composition or parent body
- (-) ADR-040 now implemented (Phase 11); A2 prerequisite resolved
- (-) A2 data available for ≈ 3% of NEOs; most clusters will have zero objects
  with A2 — module is implemented but rarely exercises (silent no-output for ~97% of clusters)
- (-) A2 values have large individual uncertainties; correlation p-values may
  be unreliable without full covariance matrix treatment

**Files** (Phase 11): `aneos_core/pattern_analysis/correlation.py`

---

### ADR-047: Network Sigma Combination (Fisher's Method)

**Status**: Implemented (Phase 11) — `NetworkSigmaCombiner` live in `aneos_core/pattern_analysis/network_sigma.py`

**Context**
Individual pattern analysis sub-modules each produce a p-value for their
respective anomaly. A combined network-level significance score is needed to
report "how anomalous is this population as a whole" — consistent with the
per-object Fisher combination already implemented in `ValidatedSigma5ArtificialNEODetector`.

**Decision**
`NetworkSigmaCombiner` applies Fisher's method across the enabled sub-modules:

```
χ² = -2 Σ ln(p_i)   for i in {clustering, harmonics, rendezvous, correlation}
```

Under H₀ (no population-level anomaly), χ² follows χ²(2k) where k = number of
enabled sub-modules producing a valid p-value.

Network sigma is derived from the combined p-value via the same
`sigma_to_p_value()` utility used by the per-object detector (`utils/statistical_utils.py`).

**Reporting threshold**: Network sigma ≥ 3.0 triggers a NETWORK_ANOMALY alert;
≥ 5.0 triggers a NETWORK_EXCEPTIONAL alert.

**Multiple-testing correction**: When analyzing N objects in batch, Bonferroni
correction is applied per sub-module across N to control the family-wise error rate.

**Consequences**
- (+) Methodologically consistent with existing detection framework
- (+) Reuses `utils/statistical_utils.py`; no new statistical code needed
- (-) Fisher's method assumes p-values from independent sub-modules; clustering
  and harmonic analysis are correlated (related to the same orbital structure),
  violating independence. The combined sigma should be interpreted as indicative,
  not strictly rigorous.
- (-) Bonferroni correction for large N batches (> 100 objects) will suppress
  all but the most extreme signals; users must be informed

**Phase 21 extension**: Stouffer's weighted z-score method added as an alternative to Fisher's. `NetworkSigmaCombiner.combine()` now accepts `method='stouffer'` and per-key `weights`. `PatternAnalysisConfig` exposes a `combination_method` field (default `'fisher'`) that is threaded through `NetworkAnalysisSession.run()`. This partially addresses the independence assumption noted above — Stouffer's method handles heterogeneous evidence confidence better, though correlation between sub-modules still applies.

**Files** (Phase 11): `aneos_core/pattern_analysis/network_sigma.py`,
`aneos_core/utils/statistical_utils.py` (reused)
**Files** (Phase 21): `aneos_core/pattern_analysis/network_sigma.py`, `aneos_core/pattern_analysis/session.py`

---

### ADR-048: `/analyze/network` API Endpoint

**Status**: Implemented (Phase 11) — `POST /analyze/network` + `GET /analyze/network/{job_id}/status`; `NetworkAnalysisSession` aggregate root; `aneos_api/schemas/network.py`

**Context**
Population pattern analysis results must be accessible via the REST API
consistent with the existing endpoint design (async background task + status
poll, same as `/analyze/batch`).

**Decision**
New endpoint `POST /api/v1/analysis/network` accepts:

```json
{
  "designations": ["designation1", "designation2", ...],
  "historical_years": 30,
  "clustering": true,
  "harmonics": true,
  "correlation": false,
  "rendezvous": false
}
```

Response immediately returns a job ID:
```json
{"job_id": "net_20260308_...", "status": "processing", "status_url": "..."}
```

Status polling via `GET /api/v1/analysis/network/{job_id}/status` returns a
`NetworkReport` when complete.

**`NetworkReport` schema** (new Pydantic model in `aneos_api/schemas/network.py`):
```
NetworkReport:
  job_id: str
  status: str
  designations_analyzed: int
  clusters: List[OrbitalClusterResult]
  harmonic_signals: List[HarmonicSignalResult]
  rendezvous_pairs: List[RendezvousPairResult]
  correlation_matrix: Optional[CorrelationMatrixResult]
  network_sigma: float
  network_tier: str              # same ROUTINE/NOTABLE/.../EXCEPTIONAL scale
  combined_p_value: float
  sub_module_p_values: Dict[str, float]
  analysis_metadata: Dict[str, Any]
```

**Backward compatibility**: New endpoint only; all v1.0 endpoints unchanged.

**Consequences**
- (+) Consistent with existing batch pattern; no new architectural patterns
- (-) Blocked until at least PA-1 (clustering) is implemented; PA-3 (harmonics)
  requires ADR-041 first
- (-) `NetworkReport` schema must be designed carefully: cluster members reference
  designations, not embedded NEOData — keep response size manageable

**Files** (Phase 11): `aneos_api/endpoints/analysis.py` (route added),
`aneos_api/schemas/network.py` (schema defined)

---

### ADR-049: Menu Scope — Six Real Workflows Replace 121-Option God Class

**Status**: Accepted — Phase 16

**Context**
`aneos_menu.py` grew to 10,800+ lines / 266 methods in a single `ANEOSMenu` class over
14 development phases. The menu presented 121 options across 8+ nesting levels for a system
with exactly two real use-cases: (1) classify an object as artificial or natural, and
(2) assess its impact probability. Over-delivery created maintainability and honesty risks:
- 4 separate paths to the same `ValidatedSigma5ArtificialNEODetector`
- Theater (fake progress bars, military framing, animated spinners with hardcoded results)
- "Coming soon!" stubs presented alongside implemented features
- No Impact Assessment option despite `ImpactProbabilityCalculator` existing and being wired to the API

**Decision**
Introduce `ANEOSMenuV2` (`aneos_menu_v2.py`) starting with 6 core options (Phase 16),
expanded to 15 options by Phase 19:

| Option | Label | Backend |
|--------|-------|---------|
| 1 | Detect NEO (single, σ-5) | `DetectionManager(AUTO)` + `DataFetcher` |
| 2 | Multi-Evidence Analysis | `DetectionManager(AUTO)` + physical/temporal data |
| 3 | Batch Detection | `DataFetcher.fetch_multiple()` + `DetectionManager` |
| 4 | Orbital History Analysis | `HorizonsSource.fetch_orbital_history()` |
| 5 | Impact Probability | `ImpactProbabilityCalculator.calculate_comprehensive_impact_probability()` |
| 6 | Close Approach History | CAD API via `DataFetcher` |
| 7 | Live Pipeline Dashboard | `PipelineIntegration` (200-year CAD polling) |
| 8 | Population Pattern Analysis | `NetworkAnalysisSession` (BC11) |
| 9 | Browse Results | in-session cache + `AnalysisService` DB fallback |
| 10 | Export Results | `Exporter` (JSON/CSV) |
| 11 | System Health | 8-component import + API probe |
| 12 | Start API Server | `uvicorn aneos_api.app:app` |
| 13 | Detection Analytics | σ-tier + ATLAS-tier session statistics |
| 14 | Scientific Help | concept docs + sigma table |
| 15 | Rendezvous Scan | `PHAMoidScanner` (PA-6 Stage 1, ADR-052) |

`aneos_menu.py` (legacy) is preserved and accessible via `python aneos.py --legacy-menu`.
`ANEOSMenuV2` inherits from `ANEOSMenuBase` (ADR-049 companion, implemented Phase 15E).
`aneos.py` default launch switches from legacy menu to v2 menu.

**Out-of-scope (removed without replacement)**
- Docker/K8s management options (just printed CLI instructions)
- "User Management" stub (single-user tool)
- ML training menu (no trained models exist)
- Streaming/web dashboard option (feature not built)
- `generate_statistical_reports()` fake animated progress stub

**Consequences**
- (+) Every menu option invokes real, tested backend code
- (+) `ANEOSMenuV2` is independently testable (8 unit tests in `tests/test_menu_v2.py`)
- (+) Legacy 121-option menu remains accessible; no functionality deleted
- (-) Users familiar with legacy menu must use `--legacy-menu` flag
- (-) Phase 16E (theater removal from legacy menu) still needed for production readiness

**Files**: `aneos_menu_v2.py` (created), `aneos.py` (default changed), `aneos_menu_base.py` (inherited)

---

### ADR-050: No Silent Fallback for Unresolvable Orbital Data

**Status**: Accepted — Phase 16

**Context**
`ANEOSMenu._get_test_data()` had a "last resort" path that silently returned generic
orbital elements (`a=1.5, e=0.3, i=5.0`) when live data fetch failed and the designation
was not in the known alias table. Detection results produced from generic parameters are
meaningless but were presented to users without warning. This violated the honesty principle
established in the scientific integrity audit (2026-03-08).

Similarly, `_fetch_real_orbital_data()` was reading `oe.diameter`, `oe.albedo`,
`oe.spectral_type`, `oe.rot_per` — fields removed from `OrbitalElements` in Phase 6 (ADR
G-011). Physical data was silently lost, reducing detection evidence.

**Decision**
1. `_get_test_data()` raises `ValueError` with a descriptive message when the designation
   cannot be resolved. All call sites wrap in `try/except ValueError` and display the error
   to the user before returning — no analysis runs on fabricated data.

2. `_fetch_real_orbital_data()` reads physical data from `neo_data.physical_properties`
   (`PhysicalProperties.diameter_km`, `.albedo`, `.spectral_type`, `.rotation_period_hours`)
   consistent with the canonical data model established in Phase 6.

3. `ValidatedWrapper.analyze_neo()` in `DetectionManager` forwards `orbital_history`,
   `close_approach_history`, and `observation_data` from `additional_data` to
   `analyze_neo_validated()` — they were previously silently dropped.

4. `_map_validated_classification()` returns `"❔ INCONCLUSIVE (σ<2)"` for sigma below 2.0,
   replacing the scientifically incorrect `"🌍 NATURAL"` label. An INCONCLUSIVE result is
   not evidence of natural origin.

**Consequences**
- (+) Detection runs only on data that was actually fetched
- (+) σ < 2 is no longer labelled as definitively natural
- (+) Orbital history evidence reaches the detector when fetched
- (-) Some menu workflows now show an error where they previously silently continued;
  this is the correct behaviour — user must investigate the missing data

**Files**: `aneos_core/detection/detection_manager.py`, `aneos_menu.py`

---

### ADR-051: Full Object Data Panel (Options 1 & 2)

**Status**: Implemented — Phase 19

**Context**
Before a detection result is shown, users benefit from seeing the raw object
data that was fetched, so they can assess data quality and completeness before
interpreting the sigma score.

**Decision**
`_display_neo_data()` panel (style=blue) is shown before the detection result
in `_detect_single` and `_detect_multi_evidence`. Option 1 defaults `verbose=True`.
The panel shows: orbital elements, physical properties, observation arc,
completeness score, closest approach summary, sources used, and non-grav A2
if present.

**Consequences**
- (+) Users see data quality context alongside the detection verdict
- (+) Missing data is visible before the result, not hidden in metadata
- (-) Slightly longer output for single-object queries

**Files**: `aneos_menu_v2.py`

---

### ADR-052: Standalone Rendezvous Scan as Option 15 (PA-6 Stage 1)

**Status**: Stage 1 Implemented (Phase 19) — Stage 2 (REBOUND) deferred

**Context**
Option 7 pipeline is Earth-centric (CAD API). Objects on rendezvous paths to
other NEOs never appear in Earth close-approach data and therefore bypass the
main polling pipeline entirely.

**Decision**
Deploy PA-6 Stage 1 (Drummond D + period resonance + A2 correlation) as Option 15
without REBOUND. SBDB bulk query provides ~2,400 PHAs with orbital elements and
A2 parameters. Drummond D metric pairs objects by orbital similarity; period
resonance (P_a/P_b ratio) and A2 sign correlation provide supporting signals.
Stage 2 (REBOUND N-body propagation) is deferred until the dependency is adopted.

**Consequences**
- (+) Earth-independent rendezvous signal available without compiled dependencies
- (+) ~2,400 PHAs covered using SBDB bulk query
- (-) No temporal confirmation — Stage 1 flags geometric proximity only
- (-) Stage 2 (actual encounter epochs) requires REBOUND and is deferred

**Files**: `aneos_core/pattern_analysis/rendezvous.py`, `aneos_menu_v2.py`

---

### ADR-053: Physical Indicators Architecture (BC3 Gap)

**Status**: Implemented (Phase 21) — `DiameterAnomalyIndicator` and `AlbedoAnomalyIndicator` wired into single-object detection path; pipeline stays 0.0 per ADR-059.

**Context**
`scoring.py` maps a `physical` category with weight 0.20 but `aneos_core/indicators/physical.py`
does not exist. Three intended indicators (`DiameterAnomalyIndicator`, `AlbedoAnomalyIndicator`,
`SpectralAnomalyIndicator`) are planned in DDD BC3 but never implemented. The physical category
always contributes 0 to every composite score.

**Decision (original)**
Defer implementation until `PhysicalProperties` data (diameter, albedo, spectral_type) is
reliably populated from SBDB for a sufficient fraction of NEOs. Current coverage: diameter
available for ~30% of NEOs; albedo ~25%; spectral type ~15%.

**Phase 21 Implementation**
`DiameterAnomalyIndicator` and `AlbedoAnomalyIndicator` are now computed in
`detection_manager.py` `ValidatedWrapper.analyze_neo()` when `physical_data` contains
real `diameter_km` or `albedo` values from SBDB. Results are forwarded as
`diameter_anomaly` / `albedo_anomaly` keys and consumed by ATLAS
`_process_physical_traits()` to produce `ClueContribution` entries with the `μ` and `α`
flags respectively. CAD-API pipeline continues to score 0.0 for these categories (ADR-059) —
the pipeline has no per-object SBDB physical data.

**Consequences**
- (+) Physical indicators now active for single-object detection (Options 1–3) when SBDB provides diameter/albedo
- (+) μ (diameter) and α (albedo) flags appear in ATLAS output for anomalous objects
- (-) Coverage remains sparse: ~30% diameter, ~25% albedo from SBDB — most objects still score 0
- (-) Pipeline (Option 8) unchanged: physical physical categories score 0 per ADR-059
- (+) No false positives introduced: sub-weights 0.25 each; guard `weighted_score > 0` prevents activation when data is absent

**Files**: `aneos_core/analysis/indicators/physical.py` (pre-existing),
`aneos_core/analysis/advanced_scoring.py`, `aneos_core/detection/detection_manager.py`

---

### ADR-054: Analysis Result Persistence & Retrieval

**Status**: Implemented (Phase 15) — Documented retroactively

**Context**
Detection results need to persist across menu sessions so that Option 9 (Results Browser)
can show historical analyses, and the REST API can serve prior results without re-running
the detector. SQLAlchemy is already used by the API layer.

**Decision**
`aneos_api/endpoints/analysis.py` fires `_persist_detection_result()` in a background
thread after each detection run. Results are stored in `DetectionResult` SQLAlchemy table
via `AnalysisService`. `GET /results/{designation}` queries `AnalysisService.get_analysis_results()`
with a `limit=20` ceiling. The menu `_show_db_result_for(designation)` filters client-side
by designation match.

**Consequences**
- (+) Results survive menu restarts; historical comparison possible
- (+) REST API can replay results without re-analysis
- (-) `limit=20` ceiling means older results may not surface
- (-) Background thread persistence has no failure notification

**Files**: `aneos_api/endpoints/analysis.py`, `aneos_api/database.py`, `aneos_menu_v2.py`

---

### ADR-055: Menu Base Class — Separation of Concerns

**Status**: Implemented (Phase 15E) — Documented retroactively

**Context**
`ANEOSMenu` (`aneos_menu.py`, 10,800+ lines) mixed business logic with presentation.
`ANEOSMenuV2` was designed to inherit a thin UI helper layer rather than re-implementing
terminal I/O primitives.

**Decision**
`aneos_menu_base.py` (`ANEOSMenuBase`) provides 12 UI helpers:
`show_error`, `show_info`, `show_success`, `wait_for_input`, `display_table`, `display_panel`,
`format_probability`, `format_sigma`, `format_designation`, `_get_terminal_width`, `_truncate`,
`_color_for_score`. All Rich calls are guarded by `HAS_RICH`; plain-text fallbacks ensure
usability in minimal environments. `ANEOSMenuV2` and `ANEOSMenu` both inherit `ANEOSMenuBase`.

**Consequences**
- (+) UI primitives testable independently of menu logic
- (+) Rich unavailability degrades gracefully to plain text
- (-) `display_panel(expand=False)` required to prevent wide-terminal truncation (Phase 20 fix)

**Files**: `aneos_menu_base.py`, `aneos_menu_v2.py`, `aneos_menu.py`

---

### ADR-056: ML-Monitoring Integration Pattern

**Status**: Partially Active — Core Dependency Wired (Phase 14+)

**Context**
`monitoring/alerts.py` imports `Alert` from `ml.prediction` unconditionally. This creates
a hard dependency: if `aneos_core/ml/` is unavailable or broken, the monitoring alert
system fails at import. ML training is explicitly deferred (CLAUDE.md), but the monitoring
bridge is live.

**Decision**
Accept the hard dependency for now. `ml/prediction.py` defines `Alert`, `AlertManager`,
and `RealTimePredictor`. `monitoring/alerts.py` uses `Alert` as a carrier class only —
no model inference occurs at alert time. The `HAS_SKLEARN`/`HAS_TORCH` guards in `ml/models.py`
prevent model loading failures from propagating to alert dispatch.

**Consequences**
- (+) Alert infrastructure works without trained models
- (-) `from aneos_core.ml.prediction import Alert` will fail if `ml/` is removed or renamed
- (-) Circular concern: monitoring depends on ML, but ML should be independent of monitoring
- (-) No ADR previously captured this design decision

**Files**: `aneos_core/ml/prediction.py`, `aneos_core/monitoring/alerts.py`

---

### ADR-057: Progressive UI Enhancement — Progress Bars & File Browser

**Status**: Implemented (Phase 19)

**Context**
Long-running operations (batch detection, health check, population analysis) give no
progress feedback. File path entry for batch inputs requires users to type absolute paths,
which is error-prone.

**Decision**
`ANEOSMenuBase.track_progress(items, description)` wraps any iterable in a Rich
`Progress` context manager, yielding items with a live bar. Fallback: numbered print
statements when Rich unavailable. `ANEOSMenuBase.browse_files(search_dirs, extensions, prompt)`
scans common directories, shows a numbered table, and accepts either a `#` selection
or a literal path. Options 3 (batch), 8 (population analysis), and 11 (health check)
use these helpers.

**Consequences**
- (+) Users can see progress on operations taking > 2 seconds
- (+) File selection reduces invalid-path errors
- (-) Rich is a soft dependency; progress bars silently absent if not installed

**Files**: `aneos_menu_base.py`, `aneos_menu_v2.py`

---

### ADR-058: ATLAS Clue-Based Scoring & Runtime Weight Configuration

**Status**: Implemented (Phase 1+) — Documented retroactively

**Context**
The original `scoring.py` used hard-coded integer weights in category maps.
`advanced_scoring.py` (ATLAS) introduced continuous [0,1] clue scoring with weights
loadable from `advanced_scoring_weights.json` at runtime, enabling tuning without
code changes.

**Decision**
`AdvancedScoreCalculator` reads `advanced_scoring_weights.json` at construction time.
If the file is missing, default weights are used silently. Six clue categories
(Encounter Geometry 0.15, Orbit Behavior 0.25, Physical Traits 0.20, Spectral Identity 0.20,
Dynamical Sanity 0.15, Human Origin 0.05) are evaluated per object. Per-clue contribution,
score, confidence, and explanation are stored in `ClueContribution` for full traceability.
`AdvancedScoreCalculator` is designated canonical for full-data pipeline scenarios (ADR-008 Update).

**Consequences**
- (+) Weights tunable without code changes
- (+) Full per-clue audit trail in `ClueContribution`
- (-) Physical Traits and Spectral Identity clues contribute 0 when physical data absent
  (ADR-053 gap)
- (-) No validation of `advanced_scoring_weights.json` schema on load

**Pipeline proxy fix (Phase 20)**: CAD-API pipeline previously computed synthetic
radar/thermal/spectral scores from eccentricity. These are zeroed in
`automatic_review_pipeline.py`; real values require dedicated observations (ADR-053).
Δ flag threshold raised from `> 0.4` to `> 0.65` so velocity-proxy scores (capped at 0.5)
cannot trigger it. Approach and ΔBIC proxy confidence lowered to 0.35; explanations are
now confidence-conditional (proxy path shows "proximity signal / velocity anomaly signal"
rather than "Suspiciously regular" / "Non-gravitational acceleration detected").

**Files**: `aneos_core/analysis/advanced_scoring.py`, `advanced_scoring_weights.json`,
`aneos_core/pipeline/automatic_review_pipeline.py`

---

### ADR-059: Pipeline Proxy Score Discipline (Phase 20)

**Status**: Accepted — Phase 20

**Context**
`automatic_review_pipeline.py` computed synthetic scores for `radar_score`,
`thermal_score`, and `spectral_score` from orbital elements (eccentricity, inclination).
These fed ATLAS which displayed misleading explanations ("Radar polarization indicates
artificial surface properties", "Thermal emission consistent with engineered material")
when no actual observational data existed. The ΔBIC score was computed from approach
velocity and could trigger the Δ flag (threshold 0.4) even for natural slow-approaching
objects, creating false-positive pipeline candidates.

**Decision**
1. Zero out `radar_score`, `thermal_score`, `spectral_score` in the pipeline — set to
   `neo_obj.get('X_score', 0.0)` so they remain non-zero only if a caller explicitly
   provides a measured value.
2. Lower `approach_confidence` to 0.35 and `delta_bic_confidence` to 0.35 for
   CAD-sourced pipeline candidates (signalling proxy nature).
3. Raise ATLAS Δ flag threshold in `advanced_scoring.py` from `> 0.4` to `> 0.65`.
   Pipeline delta_bic_score is capped at 0.5, so no proxy score can trigger the flag.
4. Make ATLAS explanation text confidence-conditional:
   - confidence < 0.6 + `repeat_approaches` → "Close-approach proximity signal (proxy)"
   - confidence < 0.6 + `non_gravitational_acceleration` → "Velocity anomaly signal (proxy)"
5. Add data-source disclaimer and dark-comet context to `_display_pipeline_candidate_detail`
   in `aneos_menu_v2.py`.

**Consequences**
- (+) Pipeline candidates no longer display fabricated physical/spectral evidence
- (+) Δ flag reserved for objects with genuine A2 measurements or high-confidence signals
- (+) Dark comet candidacy context surfaced to analyst when Δ flag present
- (-) Physical/spectral/radar/thermal ATLAS categories will score 0 for all pipeline
  candidates until ADR-053 observational data gap is closed
- (-) Orbit-behaviour category scores are lower (proxy confidence 0.35 vs 0.5/0.7),
  reducing overall ATLAS scores and potentially pushing borderline objects below threshold

**Files**: `aneos_core/pipeline/automatic_review_pipeline.py`,
`aneos_core/analysis/advanced_scoring.py`, `aneos_menu_v2.py`

---

### ADR-060: Property-Based Testing with Hypothesis

**Status**: Implemented (Phase 21) — `tests/test_property_based.py`; `hypothesis>=6.92.0` in `requirements-core.txt`

**Context**
`hypothesis` was added to `requirements-core.txt` in Phase 20C but zero `@given` tests existed.
Coverage at 33% provides weak assurance for pure mathematical functions (orbital element
invariants, sigma/p-value round-trips, ATLAS score bounds) that are straightforward to
property-test.

**Decision**
Add a dedicated `tests/test_property_based.py` with `@given` tests covering:
1. `OrbitalElements` invariants: eccentricity ≥ 0, inclination ∈ [0, 180]
2. `ClueContribution` math: `contribution == normalized_score × weight`
3. `statistical_utils` round-trip: `p_value_to_sigma(sigma_to_p_value(σ)) ≈ σ` and inverse
4. ATLAS score bounds: `calculate_score(...).overall_score ∈ [0, 1]` for any valid orbital input
5. Targeted unit tests: Bonferroni correction, Fisher method key, boundary p-values

No production code changes are required.

**Consequences**
- (+) Invariants are verified over thousands of generated inputs on every CI run
- (+) Covers pure-math code paths that example-based tests miss
- (+) Realistic coverage push from ~33% toward ~40%
- (-) `max_examples=50` applied to ATLAS test to limit wall-time on CI

**Files**: `tests/test_property_based.py` (new)

---

_End of Architecture Decision Records_

## Scientific evidence correction — 2026-11-12

### ADR-064: Labelled non-gravitational fit evidence and atomic orbit solutions

**Status**: Implemented with offline contract and path tests; observational
physical-model comparison and scientific calibration are not validated.

**Decision**: Parse SBDB 1.3 `orbit.model_pars`, using documented `full-prec` and
`cov=mat` options. Preserve A1/A2/A3 values, sigmas, units, estimation kind,
parameter index/raw literals and force-model constants in typed records.
`NonGravitationalParameters` retains its scalar accessors and legacy cache reader;
new serialization preserves the full evidence, including labelled covariance.

Covariance remains at its own TDB solution epoch with native coordinate order
and solution-epoch elements. Validate finite values, dimensions, unique labels,
normalized symmetry and positive semidefiniteness; do not use a universal absolute
tolerance that hides invalid small acceleration variances. SI conversion is an
explicit linear coefficient/unit transformation, never a thrust determination.
Unknown units, missing sigmas and absent covariance remain explicit.

Select whole orbit solutions by configured source authority, not response timing.
Never splice non-gravitational coefficients from another source/epoch into the
selected orbit. Independent physical fields may fill gaps with field provenance
and retained conflicts. Source metadata and native fit evidence survive cache
replay. Unsupported provider schema versions or malformed fit evidence fail the
fetch rather than silently dropping acceleration data.

Expose all three axes through the existing Rich panel, detector metadata and
GET/batch REST sidecar. Do not pass fitted AU/day² A2 directly to the legacy
m/s² propulsion heuristic. This is a correction of a documented units/meaning
error, not score tuning for KY26. No new classifier, endpoint or bounded context
is introduced. Existing five-sigma interpretation remains unresolved in issue #6.

**Evidence**: `tests/test_nongrav_evidence.py` includes provider-shaped synthetic
controls, covariance rejection/round-trip/unit tests, source timing permutations,
cache→menu→detector→API paths and positive A3 preservation. The optional live KY26
contract test is network-marked and makes no collection-time requests.

**Limits**: No covariance propagation between epochs, observation residual fit,
physical-origin inference or independent calibration is supplied by this ADR.
Provider-wide multi-worker request serialization/backoff remains issue #16; an
identifying application User-Agent does not establish full fair-use compliance.

---

## ADR-065: Separate screening rank from unavailable scientific inference

**Date:** 2026-11-12
**Status:** Implemented on review branch; independent calibration not available
**Related:** ADR-064; issues #2, #6, #10, #11, #12, #15, #17

**Context:** The old weighted effect-size rank, assumed posterior and constructor
validation metrics were presented as scientific significance/origin evidence.
Entry-point-specific decisions and silent fallback could change that contract.

**Decision:** Centralize ranking in `ScreeningAnalysisService`. Keep the legacy
rank for comparison, but represent unsupported significance, posterior, FDR and
origin decisions as null with explicit statuses. Follow-up candidacy is not an
origin verdict. Content-address ranking inputs and policy configuration. Carry
typed non-gravitational evidence descriptively, without thrust inference.
Unavailable probability must stop probability-based evaluation, not become zero.

Keep existing facade names, with controlled deprecation of defective private
fusion/toy-validation functions. Do not select archived models as fallback.
Constructor examples cannot install a calibrated model. Stable tail arithmetic
remains a mathematical utility, not permission to claim significance.

**Consequences and evidence:** Existing menu, REST, batch and automatic screening
helper share the service; positive/negative controls and actual adapter parity
tests exercise it. Frozen KY26 replay remains separate. See
`docs/engineering/screening-result-contract.md` for test-contract migration,
limits and remaining legacy surfaces. No independent calibration, end-to-end
validation gate, restart durability or origin inference is established by this
decision. Existing coverage requirement remains 30%; scientific performance
acceptance remains open rather than waived.

---

## ADR-066: Required evidence-bound review gate, never rank-based validation

**Date:** 2026-11-12
**Status:** Implemented on review branch; scientific validation unavailable
**Related:** ADR-065; issues #2, #21, #22, #23, #15, #17

**Context:** Automatic review bypassed its injected validator, used ranking
thresholds as validation and fabricated a probability complement. Enhanced
fallback assumed a pass when validation was unavailable.

**Decision:** Invoke required validation for each eligible canonical assessment.
Require an evidence/policy-matched, explicit five-stage contract with consistent
boolean/status outcomes; no score override, missing-stage pass or silent fallback.
Fingerprint validator configuration and full review inputs (not just ranking
inputs), reject mid-run changes, and recheck evidence
and policy at expert-queue admission. Retain each object's status/reason and
capacity deferral. Partial execution failures cannot erase another verified
positive control, nor be hidden as complete successful validation.

The installed canonical validator records unavailable/inconclusive scientific
prerequisites without running legacy placeholder helpers. Passing a synthetic
contract fixture demonstrates usable control flow, not scientific accuracy.
Legacy helper/noncanonical remediation remains #23. No scientific pass is
installed by this decision; origin remains inconclusive and FP probability null.

**Consequences:** Existing validation/analysis boundaries remain; no new bounded
context. Enhanced unavailable fallback no longer assumes success. JSON reports
retain non-advancing outcomes; exclusive unique filenames preserve prior normal
writes, but durable/crash-safe execution remains #15. Rich presentation accepts
null inference. Tests and exact limits: `docs/engineering/required-validation-gate.md`.

---

## ADR-067: Preserve computed CAD encounters separately from fitted orbits

**Date:** 2026-11-12
**Status:** Implemented on review branch; scientific validation unavailable
**Related:** ADR-064–066; issues #2, #22, #25, #16, #15, #17

**Context:** Historical polling fabricated orbital/default data, guessed column
positions, randomized screening and hid acquisition failures as empty surveys.
Cache adapters lost physical fields and collapsed distinct encounters.

**Decision:** Parse CAD1.5 by returned labels; retain native TDB, units, raw row,
query/source metadata and explicit missingness. CAD is computed encounter evidence,
not an orbit/observation source. Obtain separate orbit data through DataFetcher or
abstain; never invent a historical discovery/observation arc. Reuse the canonical
ranking service rather than another classifier. Preserve every encountered record
and distinguish query failure, partial coverage, abstention and candidate status.

Use encounter identity for overlap deduplication and preserve source versions.
Version/checksum/expire new caches, keep legacy caches unmodified but untrusted,
and preserve complete physical fields through model serialization. Propagate
failure/coverage through workflow/reporting; exclusive normal report creation
must not overwrite prior history. Defer optional Gaia service import until an
explicit query so importing required validation cannot contact that service.

**Evidence and limits:** Network-denied controls, fresh-process import and query
positive controls, optional live KY26 contract, typed cache/source propagation,
first-stage→required-gate integration and failure reporting are documented in
`docs/engineering/cad-evidence-integrity.md`. No new bounded context/UI. No fitted
likelihood, calibrated significance, source authentication, complete historical
population, provider-wide scheduling or restart-safe job execution is established.

---

## ADR-068: Withdraw placeholder science from legacy validation and consumers

**Date:** 2026-11-12
**Status:** Implemented on review branch; calibrated scientific models not installed
**Related:** ADR-065–067; issues #2, #23, #6, #12, #15, #17

**Context:** Direct legacy validation still fabricated source/orbit/uncertainty
inputs and could pass on errors. Consumers could turn unavailable probability
into a perfect score, double-count transported results, or alert on thresholds.

**Decision:** Preserve function names but withdraw defective numerical/validation
contracts. Use actual input inventories or explicit unavailable/inconclusive
outcomes. No default null, calibrated source-quality scalar, unitless mixed-orbit
uncertainty, invented category count or asserted FP-reduction probability is
installed. Require explicit complete outcomes even for review eligibility.

Adapt legacy validation inputs through canonical screening while retaining the
original separately. Guard scientific claim aliases against legacy promotion.
Replace mock integration scoring with a real acquisition/shared-service adapter.
Keep stand-alone research analyzers available, but do not register them as
validated evidence or fuse unavailable stages into measured clues.

Propagate missingness through the existing dashboard/REST/WebSocket/JavaScript
paths; distinguish report availability, workflow outcomes and scientific
inference. No `1-FPR` origin/score interpretation or threshold-only origin alert
is supported. In-process transport deduplication is not durable exactly-once
execution. No new bounded context or UI is introduced.

**Evidence/limits:** `docs/engineering/legacy-validation-integrity.md` records
controlled deprecations, executable positive/negative/null controls and remaining
core AnalysisPipeline/API/CLI migration. Scientific capability remains unavailable;
these engineering tests do not satisfy independent accuracy or fit/calibration
requirements. No existing scientific/CI acceptance gate is lowered.

---

## ADR-069: Make core analysis a shared-screening adapter, not a second classifier

**Date:** 2026-11-12
**Status:** Implemented on review branch; scientific calibration unavailable
**Related:** ADR-065–068; issues #2, #6, #11, #12, #15–17

**Context:** AnalysisPipeline independently scored legacy indicators, returned
`natural`/zero on acquisition or processing failure, reconstructed nested cached
objects incorrectly, and populated an uncalibrated population analyzer. `/analyze`
expected missing result attributes and treated a dict as an object. Its
identity-only response cache could hide changed evidence. ML consumers assumed
bounded score/probability units and generated labels from screening decisions.

**Decision:** Core acquisition delegates to typed DataFetcher; ranking delegates
to ScreeningAnalysisService. `canonical-analysis-v1` results carry the full
assessment, policy/evidence IDs, actual NEO data, nullable compatibility score
view and explicit execution status. `overall_score` means uncalibrated ranking
units. Missing source/invalid evidence/internal error abstain rather than produce
a natural-origin result. Completeness counts presence, not reliability.

Retain standalone legacy indicators/calculators for research, but do not invoke
them or accumulate their statistical interpretation on this deployed path.
Deprecate the separate source manager (including mock source fallbacks). Use
bounded async batch scheduling with ordered per-object outcomes; callback errors
cannot discard results. Cancellation does not terminate a blocking provider worker.

`/analyze` consumes the typed result and existing flags, returns 503/422 for
unavailable/invalid source evidence, and does not read a designation-only result
cache. Existing retrieval/export uses evidence/policy-keyed in-memory snapshots;
this is not durable case history. JSON retains replay metadata; CSV names ranking
units explicitly. Existing CLI core-result displays preserve null values.

Reject legacy ML fusion/cache use with the new core score contract before any
prediction/dispatch. Require independently supplied binary labels and provenance
rather than deriving supervised truth from screening classifications. This guard
does not install a new model, validate supplied labels, or certify retained ML
research utilities. Enhanced wrapper/validator accepts the typed core assessment;
the separately mocked enhanced REST adapter still requires remediation.

**Consequences and evidence:** No new bounded context, route or UI. Contracts and
network-denied controls: `docs/engineering/core-analysis-contract.md`. Independent
calibration, observational fits, complete interface migration, global provider
scheduling and restart-safe persistence remain unfinished. No scientific or CI
quality gate is reduced; no five-sigma claim is enabled.

---

## ADR-070: Connect enhanced REST to actual screening and explicit review status

**Date:** 2026-11-12
**Status:** Implemented on review branch; no calibrated scientific model installed
**Related:** ADR-065–069; issues #2, #6, #12, #15–17

**Context:** Enhanced REST used MockNEOData and a MockPipeline returning fixed
score/confidence/artificial classification. Its Pydantic models could not import
in the installed runtime, required unavailable inference as numbers, and its
routes were not registered in the main app. Batch flags/outcomes were lost. Its
summary interpreted mixed legacy ranks as false-positive probabilities and
accepted natural/inconclusive labels as validation outcomes.

**Decision:** Register the existing declared routes under the existing analysis
prefix and authentication dependency. Acquire/analyze once through the actual
application AnalysisPipeline, bind the review wrapper to that current service,
and expose the canonical assessment and required-review outcomes separately.
Refresh reaches acquisition. Disabling optional review is `not_requested`, never
a pass, and cannot mutate another request's validator. Retain method preferences
as requested-but-unperformed where models are unavailable; no fast/statistical-only
bypass is installed. Unknown inference and uninstrumented diagnostic timings are
null; only actual top-level elapsed time is reported numerically.

Keep every ordered batch outcome, including duplicates, abstentions, invalid
measurements, provider failures and timeouts. Operational counts are not accuracy.
Read-only historical summaries replay supported current-policy assessments;
exclude and count legacy, modified and unsupported-policy records without
rewriting them. Never infer FDR, origin probability or acceptance from rank or
classification. Report the row-based unit and restricted summary scope.

**Consequences:** Existing routes become reachable in the app; no new engine,
UI or bounded context. Updated request/nullable-response/OpenAPI contracts and
network-denied actual HTTP/SQLite controls are recorded in
`docs/engineering/enhanced-rest-contract.md`. Review transport has synthetic
positive controls, not independent scientific certification. Persistence of new
enhanced reports, restart recovery, global provider control and wider legacy
menu/model/network reporting remain separate unfinished work. No five-sigma
permission, observational fit or origin conclusion is introduced.

---

## ADR-071: Unify primary legacy menu/launcher screening and reproducible reports

**Date:** 2026-11-12
**Status:** Implemented on review branch; scientific calibration remains unavailable
**Related:** ADR-065–070; issues #2, #6, #12, #13, #15–17, #30

**Context:** Rich/plain legacy menu paths differed, substituted name-based demo
orbits/history, labeled null origin as natural and formatted unavailable sigma as
a number. Workflow/report helpers manufactured probabilities/readiness. The direct
launcher interpolated a designation into executable Python and preferred another
analyzer. Its raw physical adapter used kilometres under the metres-based ranking
key. Entering the legacy menu silently cleaned caches and changed environment.

**Decision:** Primary single/batch/interactive, scientific-suite and predefined
workflow entrypoints delegate to the actual shared core pipeline. Keep native
computed Horizons history as a separate, non-observational sidecar, without a
fabricated observed epoch-zero fallback. Preserve function names; controlled-
deprecate defective prototype inference/workflow/sigma-conversion contracts.
Thresholds or detector agreement cannot establish publication readiness.

Use a replay-checked `scientific-screening-report-v1` presentation with canonical
assessment, full normalized NEO record, review-input/report fingerprints, source
basis/dependencies and explicit unavailable inference. Include run/source-file
metadata with stated limits: hashes are not authentication, a loaded-bytecode
proof, environment lock or preservation of dirty source. JSON and Markdown share
the same assessment. Keep the existing Rich UI, using neutral rendering.

Route launcher analysis/simple/detect aliases directly to the shared CLI service,
not interpolated Python or another detection engine. Document/test refresh,
JSON/Markdown and exclusive report-file output; failed acquisition/report writes
return nonzero. Exclusive creation prevents normal overwrite, not crash-safe
persistence. Entering a menu performs no implicit cleanup, database initialization
or environment changes; explicit maintenance remains separately available.

**Evidence/consequences:** `docs/engineering/legacy-menu-contract.md` records
regressions and the controlled migration of a test that required fabricated alias
data plus typed/unit fixture corrections. No gate is weakened. No new engine, UI
or bounded context; core planetary-defense modules are unchanged, not newly
validated. Remaining standalone analyzer/prototype paths are #30 and the wider
#6/#12 audit. No independent performance, physical fitting, durability or
five-sigma permission is introduced.

---

## ADR-072: Withdraw standalone fabricated inputs and unify advanced analysis contracts

**Date:** 2026-11-12
**Status:** Implemented on review branch; no calibrated scientific/impact model installed
**Related:** ADR-065–071; issues #2, #6, #12, #17, #30, #32

**Context:** SimpleNEOAnalyzer selected prototype fallback detectors, generated
hash/name-based orbits and fixed probabilities, synthesized impact inputs and
reported assumed-prior calculations as calibrated. Advanced analysis misread the
DataFetcher return contract, fabricated data on failure, used another heuristic
score and turned origin scores into threat levels. Capability counts/imports
claimed scientific maturity or statistical validation without supporting artifacts.

**Decision:** Simple/advanced synchronous and asynchronous facades use actual typed
core acquisition/shared screening and the same replayable report. Supplied raw
inputs remain a separate caller-supplied mode: missing fields are not supplemented
by fetched defaults, aliases/units/time conventions are checked, and the caller
record is retained with its verification limits. Invalid/non-JSON/nonfinite inputs
are rejected, not converted to zeros or hash-derived orbits. Unsupported per-call
source filtering raises explicitly instead of claiming the requested selection.

Keep public names, but controlled-deprecate scalar origin probabilities, mock
input generation, fixed calibration and prototype enrichment/inference. Nullable
unified-result fields preserve unavailable probability, confidence, quality and
significance. Return structured screening, not a best-detector cascade. A running
async event loop must use the async method rather than a nested sync fallback.

Impact bridges inventory actual orbit/covariance/dated-arc inputs and identify
missing observational/propagation/encounter-model prerequisites. They do not call
unfitted probability heuristics, assign negligible risk or infer maneuvers from
origin rank. Core impact/scenario calculators remain available and unchanged;
remaining REST/menu impact claims are explicitly #32, not certified by this change.

Capabilities distinguish implemented candidate screening/acquisition from provider
health and scientific validation. Import counts cannot produce maturity, coverage
or validation probabilities. Existing script command shapes delegate to the shared
CLI, with no implicit `test` designation. Source failures are nonzero outcomes.

**Evidence/limits:** `docs/engineering/standalone-analysis-contract.md` records
network-denied library and actual-process controls, source/epoch/missing cases,
positive candidates, native A3/covariance transport and unavailable impact input
inventories. No new detector, scientific engine, bounded context or UI. No new
observational fit, independent calibration/performance, durable case history or
five-sigma permission. No prior tests or acceptance gates are weakened.

### ADR-073: Shared JPL SSD transport coordination

**Status:** Implemented and integrated on draft remediation branch; bounded offline
regression evidence, not production certification. 2026-11-12. Issues #16/#17;
impact-source dependency #32. Supersedes ADR-032's optional-aiohttp requirement
only for PHA query transport, not unrelated clients or scientific algorithms.

**Context:** SBDB, Horizons, CAD and SBDB Query used separate sessions and direct
requests. Per-source limiters did not serialize the provider family. Session retries
could hide additional wire attempts. Every DataFetcher object request began with
a redundant health query. Returned source failures reset the circuit breaker.
JPL's current [API policy](https://ssd-api.jpl.nasa.gov/) requires serialized
requests, application identification and appropriate error backoff. It also calls
for avoiding redundant traffic and checking output versions; this change does not
complete those broader obligations across every source.

**Decision:** Put a shared buffered-GET adapter in the existing Data Acquisition
anti-corruption layer. Inventoried core/legacy/polling/research-tool JPL paths use
explicit wrappers; other providers retain their transport. A private POSIX file
lock spans the actual request, buffering and closure. All participating processes
must share the same coordination directory. Persist cooldowns and an in-flight
quiet-window marker; fail closed on invalid/unsafe state instead of forgetting an
outage. Do not retry inside the adapter. Disable hidden requests-Session retries
for JPL only. Preserve numeric/date Retry-After, including delays beyond the wait
budget. Remove per-object health probes; keep explicit diagnostics guarded.

Async clients offload the blocking transport. Cancelling the awaiter does not
release a still-running thread's lease. Coordination waits have a separate bound;
wire timeouts are inactivity limits, not hard end-to-end deadlines. A killed
process releases the OS lock but leaves a conservative quiet window, not proof
that JPL stopped processing. Corrupt state requires deliberate operator recovery.

**Consequences/limits:** No new bounded context, schema authority, scientific
model, distributed lock, job queue or result store. Same-host cooperative processes
with working shared-filesystem flock are covered by tests. Separate coordination
roots, other applications, multiple hosts and unusual filesystem semantics are
not covered. Wider cache reuse, endpoint version enforcement, configuration
isolation, pagination, overload control and durable jobs remain open. See
`docs/engineering/jpl-provider-transport.md` for exact controls and operations.

### ADR-074: Scoped authoritative impact evidence at the existing REST boundary

**Status:** Implemented/integrated on draft remediation branch; offline engineering
controls plus one real KY26 source transaction. Not local impact-model validation.
2026-11-12; issues #32/#16/#12/#17 and case register #7.

**Context:** `/analysis/impact` called a heuristic calculator using catalogue
elements, no residual fit/covariance propagation and an implicit 30-day arc. Its
mandatory probability output and numeric intervals/confidence implied evidence
that had not been established. Source transport coordination alone (ADR-073)
did not repair this scientific contract. Legacy menu/library paths remain separate
confirmed findings under #32.

**Decision:** Reuse the existing Data Acquisition boundary to retrieve Sentry
Mode O results through shared provider transport. Do not implement a competing
impact engine. Require supported signature/version, matching primary designation,
known method, compatible event records/counts, finite values and valid native dates.
Preserve exact bytes/hash, source computation/retrieval dates, method-specific
fit-coordinate metrics, listed-event scope and substantial source uncertainty.
These are external model predictions, not measured impacts or aNEOS validation.

The existing route returns the same replayable evidence as the source/report
service. Unknown names require identity resolution rather than trusting a possibly
unrelated result. Removed/unlisted outcomes remain distinct from transport/schema
failure and never imply zero risk. No guessed intervals, local confidence, lunar/
annual risk, damage model, local Torino class, origin or discovery significance.
Cache valid source outcomes per process with bounded TTL/size and copy isolation;
single-flight waiters share the actual attempt's result or failure, not a stale
success after a failed refresh. Async routing offloads the blocking source service.

**Evidence/limits:** See `docs/engineering/sentry-impact-contract.md`. The live KY26
transaction returned not listed and is frozen separately from historical screening.
Replay hashes are consistency checks, not source authentication. Names/aliases,
cross-process cache reuse, durable source/case/job history, independent local fits
and the remaining legacy menu/library/impact-pipeline claims remain open. No new
bounded context, UI, scientific solver or five-sigma permission is created.

### ADR-075: Impact entry-point separation and verified local evidence archive

**Status:** Implemented/integrated on draft remediation branch; bounded offline
engineering/scenario/restart controls. Not fitted impact/origin validation or a
complete durable job/case system. 2026-11-12; issues #32/#6/#12/#13/#15/#17.

**Context:** REST's Sentry correction did not repair V2 menu default arcs, null-to-zero
browser/export paths, estimated Torino classes, daemon writes of impact probability
into origin-score DB columns, or enhanced-impact quality/origin proxies. The direct
calculator still described hand-set geometric/arc factors as scientific probability,
confidence and confidence intervals. Empty domain records claimed negligible risk.

**Decision:** Menu and enhanced-impact wrapper consume the same replay/identity-bound
Sentry report as REST. Explicit requested lookups are not suppressed or promoted by
origin scores, geometric overlap or quality-to-duration proxies. Source failure,
invalid evidence, not-requested state and failed persistence remain separate.
Screening results are preserved; source evidence is not fused into origin confidence.

Keep the former probability entry-point name, but fail explicitly with a controlled
deprecation. Retain private formulas and an explicitly named legacy diagnostic replay
with supplied scenario arc and labelled nonprobability scalars. Preserve useful
conditional kinetic-energy calculation with explicit supplied mass/speed, units,
nonrelativistic assumptions and no guessed density, encounter, risk or damage model.
Missing domain values become null/not-assessed, not zero/negligible. No local Torino
estimate is produced from arbitrary probability/energy inputs.

Synchronous local archive confirmation replaces V2 impact daemon DB writes. The
existing Reporting boundary owns append-only content-addressed report snapshots:
verify/replay, write/fsync a private staging file, atomic no-replace link, fsync the
directory, then read/replay before confirmation. Existing files are never overwritten;
corruption blocks rather than triggers destructive repair. Same-ID repeat is idempotent;
new dates yield separate snapshots. Browser/JSON/CSV retain source/report identity and
nulls; menu export uses exclusive creation and has no overwriting fallback.

**Verification boundary:** whole selected offline suite now denies unmarked DNS/socket
attempts and fails even when application code catches them. A child-pytest denial and
explicit local-network positive control verify this gate. Old mock seams are migrated,
not skipped. The API enqueue tests test enqueue/status transport only; real background
network analysis and durable execution remain #6/#15, not certified by those tests.

No new UI, solver or bounded context. See `docs/engineering/impact-entrypoint-contract.md`
for archive filesystem/rollback limits, changed tests and exact scoped evidence.

### ADR-076: Transactional durable-job storage before worker/API activation

**Status:** Implemented and regression-tested storage boundary; **not integrated**
with the current network/automatic/polling API workers. 2026-11-12; #15/#6/#17.
No claim that existing in-memory jobs are now recoverable or scientifically valid.

**Context:** Population jobs use daemon threads and in-memory state; queued records
already contain numeric sigma/p-value defaults. There was no reusable persistent
execution store. File SQLite used StaticPool, sharing one connection across thread
transactions. Importing the API package assembled optional web/ML services; importing
database models also initialized schema. Storage tools therefore had unrelated
runtime and write side effects. A targeted run encountered a native Torch import
fault before storage tests; importing persistence must not depend on that backend.

**Decision:** Extend the existing SQLAlchemy Base/database, not introduce a new
queue service or competing scientific engine. Add persistent admission/job/event/
checkpoint tables with owner-scoped reads/cancellation, bounded atomic admission,
request idempotency, leased claims, fencing tokens, bounded retries, immutable input
checkpoints and atomic terminal result/event/admission commits. Checksums bind
requests/results and checkpoint/event envelopes; they are not cryptographic source
authentication or domain/scientific validation. Backend owner scopes are trusted
caller inputs, not a substitute for authenticating API principals.

For file SQLite, use independent pooled connections; retain shared connections only
for legacy explicitly in-memory databases, which JobStore rejects as nondurable.
Short job write transactions use explicit BEGIN IMMEDIATE (reads use BEGIN), disk journaling and FULL/EXTRA
synchronization; require SQLite >=3.35 for RETURNING. No provider/calculation work
runs under these transactions. Recheck lease expiry after acquiring the write lock:
a timestamp sampled before a lock wait must not authorize an expired worker.

API package exports become lazy with names retained. Storage import starts no
services and creates no database. Schema creation is explicit via existing
init_database/JobStore.initialize_schema; the full API lifespan confirms schema
setup before services and fails rather than yielding on unconfirmed setup.
Existing columns/keys and admission accounting are checked without destructive repair.
Generic database initialization registers the tables, but queue admission settings
still require explicit initialization. No automatic migration/deletion/requeue.

**Required next integration:** a versioned evidence-bound network report and domain
validator; trusted authenticated owner mapping; real worker/checkpoint/heartbeat/
retry/cancellation lifecycle; persistent REST statuses and correct status URL; startup
recovery and cross-owner/restart/provider-outage positive/negative controls. Existing
network numeric placeholders and uncalibrated feature fusion remain #6/#15, not fixed
by this storage layer. See `docs/engineering/durable-job-store.md` for exact evidence,
limits, operational setup and integration acceptance gates. No new bounded context.

### ADR-077: Evidence-bound population reports, not heuristic network significance

**Status:** Implemented/regression-tested normal session and menu/API transport;
not calibrated population inference or durable worker integration. 2026-11-13;
#6/#10/#12/#15/#17. Supersedes the scientific interpretation of ADR-042–048
population prototypes, not their preserved historical diagnostic implementations.

**Context:** The session selected objects using a mutable heuristic called sigma,
selected minimum cluster/period p-values, and fused dependent selected streams as
if calibrated. Missing orbital angles became zero; optional HDBSCAN installation
changed grouping. Computed encounter epochs became invented daily zero/one samples
and a normal-tail claim. A fitted scalar per object cannot support pairwise temporal
correlation. These paths do not establish search-wide significance or origin.

**Decision:** Existing BC11 remains the aggregate boundary. `NetworkAnalysisSession`
freezes complete NEOData inputs, runs the same ScreeningAnalysisService/policy as
single-object screening, retains ordered duplicate assessments and explicitly
records cap exclusions. It never selects on stored dynamic scores or acquires data.
`network-evidence-report-v1` binds snapshots, configuration/policy and complete
results by hashes and full deterministic replay. JSON and a verified Markdown
renderer retain evidence, native uncertainty/covariance and limitations. Hashes
are not authentication, a complete environment lock or scientific calibration.

Exploratory grouping uses one deterministic scaled circular connected-components
algorithm with documented engineering scales/radius/minimum size, no clipping,
missing-angle imputation, family identification, background population or p-value.
Duplicate designations do not increase population membership; conflicting duplicate
snapshots are rejected. Epoch/frame/uncertainty compatibility, angular degeneracy,
survey bias and physical association are explicitly outside this descriptor.

The harmonics request returns sorted unique Earth computed-encounter native TDB
Julian dates/gaps, exclusions and unestablished coverage, not a spectral detection.
There are no manufactured daily observations. Historical acquisition/window is
explicitly not performed in this pure session. Correlation exposes all A1/A2/A3
and covariance evidence but reports paired time series required; it does not turn
undefined correlation into r=0/p=1. Rendezvous remains explicitly unimplemented in
session. Disabled, requested-descriptive and unavailable are distinct states.

Legacy public prototype run/combine methods require `legacy_diagnostic=True` for
historical reproduction, otherwise controlled deprecation directs callers here.
Old numerical tests are retained under that explicit diagnostic opt-in; they do not
validate science. Legacy sigma filtering/fusion configuration is rejected, not
silently ignored. An unrelated deserialization defect found by regression testing
is corrected: angular zero must not fall back to a contradictory alias.

Existing REST polling retains a complete report, nullable inference at queued and
complete states, per-request acquisition outcomes and the correct status URL. It
rejects mismatched returned identities and redacts public exception details.
Menu display verifies replay, distinguishes ranking candidates and descriptive
groups, and reports unavailable acquisitions. **Execution remains daemon/in-memory,
not durable or owner-isolated**; ADR-076 is a storage foundation, not automatic
integration. Next work must register a resumable authenticated worker and validate
this report/request binding before JobStore.finish. No new bounded context, UI,
physical solver, calibrated probability, fitted origin model or five-sigma claim.

### ADR-078: Registered resumable network execution with committed evidence binding

**Status:** Implemented/tested worker library and fixed-window acquisition adapter;
**not integrated with authenticated REST or API lifecycle**. 2026-11-13; #6/#15/#16/#17.
Extends ADR-076 storage and ADR-077 report boundaries, not a new bounded context.

A persistent execution request freezes ordered designations, exact pattern policy,
canonical screening reference/policy ID, report engine, source configuration
fingerprint and explicit CAD calendar query bounds. Unknown/changed contracts are
rejected, not silently upgraded. The source fingerprint is not source authenticity,
plaintext credential storage or a full dependency/environment lock.

Register `network-evidence-job-v1` as the worker's sole claimable kind. Acquire each
unique identity once, validate returned typed identity and freeze its complete input
before historical acquisition or dependent work. Freeze successful bounded historical
responses separately with exact requested bounds and count/coverage evidence. Never
replace a committed measurement on retry; failure is not an immutable zero measurement.
Historical CAD events replace supplemental current/future encounters in the derived
report input; the original full NEO snapshot remains separately accessible.

Only completed versioned requests and evidence reports are eligible for terminal
storage. Full report replay is necessary but insufficient: validate reported
acquisitions against the actual immutable SQL checkpoints and request before finish.
An internally consistent rehashed report cannot substitute another acquisition.
Checkpointed final reports can recover without constructing a provider. Scientific
policy/version drift requires explicit handling, not silently recomputing old jobs.

Provider and numerical work run outside SQL transactions. Lease heartbeats renew
only live attempts; stop/deadline ends renewal even when a provider remains blocked.
SQL fencing denies late checkpoint/final writes after cancellation/expiry/reclaim.
Attempt limits and retry deadlines are persisted; honor exposed provider cooldown
hints without retrying earlier. Existing shared JPL transport coordination remains
authoritative where adapters wrap exceptions. Read-only requests may repeat after
an uncommitted interruption; no exactly-once external effects are promised.

`serve` is an explicit blocking/cooperatively stopped loop, not import-time or daemon
API activation. Driver/provider calls cannot be forcibly interrupted in-process;
lease recovery is distinct from terminating the old thread. Database/driver timeouts,
process supervision, authenticated principal mapping, managed API worker health/
startup/shutdown and end-to-end REST recovery remain required. Current network REST
still uses its old process-memory worker until that migration is complete.

See `docs/engineering/resumable-network-worker.md` for actual SQL/process controls,
limitations and commands. No probability, force-model fit, calibration or origin
claim is introduced; execution completion remains separate from scientific validation.

### ADR-079: Authenticated persistent network REST and managed execution lifecycle

**Status:** Integrated/tested for existing network routes with injected source/SQL
controls; not whole-suite authentication or scientific-readiness certification.
2026-11-13; #15/#17/#6. Composes ADR-076–078, no new bounded context.

Replace the normal network route's daemon/dictionary scheduling with app-owned
NetworkRuntime/SQL JobStore and the registered worker. Startup confirms schema/queue
and starts a non-daemon supervised worker; shutdown stops admission, signals stop and
joins within a declared budget. Unconfirmed shutdown raises instead of claiming
cleanup. Worker failure stops admission. Health reads observed worker state, never
starts services from GET, reports degraded failed checks and redacts inspection errors.
Explicitly disabled workers permit owned reads/cancellation, not new admission.

Persistent jobs require fresh verified configured/SQL API keys, or explicitly scoped
JWTs from existing `/auth/token?audience=network-jobs-v1`. Reject placeholders, legacy
mock tokens, mixed/duplicate credentials and ambiguous active-key assignments. Scoped
JWT verification pins algorithm/issuer/audience/required time claims and rechecks
active principal, credential and current role on every request. No cached mock map or
role claim grants access. Administrative authority, including credential provisioning,
uses the same fresh principal so a legacy mock administrator cannot mint a bypass.
Provisioning/listing use the same runtime database; user IDs are UUID based.
Subject/credential binding keeps key revocation effective;
key rotation retains stable ownership. Configured role keys represent shared service
principals, not individual human accounts. General legacy auth remains #17 work.

Keep existing submission/status paths; add tested/documented owned cancellation.
Derive owner scope from verified identity; no client owner field or admin cross-owner
bypass. Commit before acknowledgement; reuse scoped idempotency identity and original
query window across retries, reject conflicting intent, expose bounded backpressure.
Return actual persisted execution state and separately verified complete report, not
queued/completed default significance. Verify domain result before serving it.

Retain old direct helper names as controlled deprecations; do not consume or delete
legacy process-memory research automatically. No scientific validation, full source
enrichment, complete dataset, human-origin exclusion or five-sigma claim follows from
successful execution. See `durable-network-api.md` and the additional user-quality
requirements register (#40–#46) before claiming whole-project readiness.

### ADR-080: Replay-bound object explanations and explicit coverage profiles

**Status**: Implemented for canonical V2 detail/export; broader integration pending.
**Date**: 2026-11-13

The presentation layer must explain the computation it actually displays. Record a
calculation trace during canonical screening without changing historical v1 payloads,
weights, score order or thresholds. Build the object review from a frozen typed NEO
snapshot, policy and canonical assessment. Verify by recomputing the whole review,
not merely checking a self-supplied hash. Hashes establish consistency, not source
authenticity. Recorded application identity remains metadata, not a scientific gate.

Use the existing analysis and presentation boundaries; no second origin detector or
new bounded context. Preserve source alternatives, missing values, native epochs and
unrounded JSON. Explicitly distinguish the fixed 15-field catalogue-presence profile,
three screening inputs, source acquisition coverage and scientific readiness. Neither
profile means complete scientific knowledge. Trace graphics show shares of squared
heuristic rank, never probabilities or independent evidence.

Canonical V2 detailed views verify source/result binding; legacy assessment-only
views disclose the absent full snapshot. Exports preserve the review, remove unsupported
flat significance/origin claims and use exclusive creation. Legacy exports remain
labeled unverified. Pipeline-specific views, historical SQL playback, all other
entrypoints, durable evidence enrichment and physical-model inference are not certified
by this change. See `docs/engineering/evidence-bound-object-detail.md` and #40–#43.

### ADR-081: Atomic physical catalogue assertions and paired source selection

**Status**: Implemented for SBDB acquisition, common merge/cache/analysis and canonical
object details; broader measurement modelling/enrichment pending.
**Date**: 2026-11-13

SBDB physical values previously lost their uncertainties in the typed adapter. The
common merge could independently select a value from one source and an uncertainty
from another, including when rounded values happened to match. A scalar global
reference could then imply one provenance for a mixed-source object.

Use `physical-evidence-v1` assertions in existing NEO evidence metadata: unmodified
native parameter, units, sigma syntax, reference/notes and response context plus an
explicit normalized projection. JPL documents phys_par sigma as1-sigma. Only supported
numeric/unit forms are normalized. Missing sigma is not zero; unsupported intervals
stay native with an explicit issue. H sigma remains in the assertion even though the
legacy scalar model has no H-uncertainty field. Retrieval and container orbit dates
must not become physical measurement epochs. Estimation status remains not provided.

Select each value/uncertainty pair atomically by existing provider priority. Preserve
all source projections, references and alternatives. Do not fill a missing primary
uncertainty from a secondary source, or imply that equal rounded values prove common
measurement identity. Duplicate native assertions are marked, not independent evidence;
nonidentical same-parameter alternatives cause scalar abstention within that source.
No averaging or uncertainty-weighted origin inference is introduced.

Replay normalization and projection on reconstruction/analysis; hashes are consistency
checks, not independent source authentication. Version the fetch-cache key so new
queries cannot silently reuse old unpaired projections; do not delete, migrate or
relabel historical cache/research data. Legacy snapshots remain available but unverified.
Canonical details expose normalized assertions with their uncertainty convention.
Existing historical reviews without the optional new evidence inventory remain valid.

Geometric albedo is not a probability and may exceed one. Preserve finite nonnegative
catalogue albedo rather than clipping it to[0,1]; Bond-albedo/reflectivity conventions
must remain distinct in future physical models. This correction changes admissible
input domain, not the ranking formula, thresholds or calibration claims. See the
primary sources and execution limits in `docs/engineering/atomic-physical-evidence.md`.

### ADR-082: Native orbital epochs, explicit time scales and bound computed history

**Status**: Implemented for SBDB/Horizons/NEODyS native epoch transport and canonical
views; physical propagation/time-scale conversion and full source verification pending.
**Date**: 2026-11-13

A datetime alone cannot express a native TDB orbit epoch or establish its relationship
to the covariance epoch. SBDB supplied an epoch retained only in solution metadata;
its absence in the old typed field made the view misleading. Real Horizons history
headers contain `JD = calendar TDB`; treating every equals-sign line as an element
also discarded valid computed histories. Sub-float-precision differences could be
incorrectly declared equal by the old covariance/display epoch comparison.

Add `NativeEpoch` / `native-epoch-v1`, preserving native literal, JD/MJD/calendar
representation, explicit source time scale, role and optional header. MJD→JD is an
exact decimal representation offset on the same time scale, not a conversion to UTC.
Unknown scales remain UNKNOWN; calendar/era conventions are not guessed. Numeric epoch
equality requires compatible declared scales and is not covariance/frame/solution
compatibility. Compare retained covariance/display literals without float rounding.

Store the optional record in existing OrbitalElements; omit it from legacy serialization
when absent. Do not retrofit historical source data or attach TDB to legacy datetimes.
Bind new SBDB/Horizons typed epochs to retained source solutions at reconstruction and
analysis. Keep source-priority orbit solutions atomic; never borrow a secondary epoch
or covariance. Use a new fetch-cache key suffix, leaving earlier cache/history intact.

Parse real Horizons headers before element lines; keep complete rows bound to their
native computed epoch, not an observation. Reject unbound, duplicate or nonfinite
values. Preserve older calendar-only headers without invented Julian dates. Carry
NEODyS MJD/time-scale labels literally; no TDT/TT/TDB inference or conversion.

Views/report exports expose both orbital and covariance epochs, compatible-scale
numeric comparison, missing scales and non-propagation limits. The15-field presence
profile becomes v2 only when using a native epoch; old v1 inventories remain replayable.
Neither profile establishes scientific completeness. No new bounded context/detector,
calendar conversion service, fitted likelihood, origin conclusion or calibrated sigma
is introduced. See `docs/engineering/native-orbital-epochs.md` and issues5/40/41/43.

### ADR-083: HTTP evidence parity and explicit computed-history outcomes

**Status**: Implemented for existing history/raw-screening routes and application
request-error handling; complete acquisition ledger/auth/operational QA remain pending.
**Date**: 2026-11-13

A schema adapter must not discard a native epoch or change scientific admissibility
relative to the canonical service. Real HTTP tests reproduced native-epoch loss and
rejection of geometric albedo above one, plus a NaN validation error becoming HTTP500
when the framework tried to echo non-JSON-safe input in its error response.

Preserve native epochs in a replay-checked history DTO, with header/source/role binding,
finite strict numeric fields and explicit computed-not-observed basis. Expose bounded
sample count/query scope and distinguish a same-declared-scale numeric axis from
unresolved, mixed-scale or legacy-unbound epochs. Neither status is physical validity,
complete observations, model calibration or covariance propagation.

Use a list-compatible history query envelope inside existing acquisition boundaries:
request parameters, attempt/retrieval times, schema status and response-text hash;
available/empty/unavailable/invalid outcomes are distinct. Existing list callers stay
available, but must inspect metadata to distinguish outage from empty samples. The
HTTP route polls outside the event loop and maps invalid evidence to502, outage to503,
and an explicitly empty series to404. Do not expose raw provider exceptions. Pin
requested ELEMENTS frame/units/time-scale parameters and retain returned native labels.
No new endpoint or independent detection engine is added.

Align raw-input numeric domains with the canonical service: strict finite numbers,
no bool/string coercion, no arbitrary a/e/diameter caps overriding canonical abstention,
nonnegative catalogue albedo without probability clipping, and consistent conic checks.
Remove arbitrary adapter bounds, not scientific validity gates. Invalid conics remain
invalid; supported hyperbolic/out-of-domain inputs abstain rather than become candidates.
A request-validation handler emits bounded error type/location without raw inputs or
exception context, preserving422 rather than crashing on NaN/Infinity.

Retain existing route names and response fields, document stricter input validation,
new history-v2 metadata and changed outage semantics. Regenerate the real registered
OpenAPI snapshot and test actual HTTP plus canonical parity. No auth eligibility,
source authenticity, durable-job completeness, scientific calibration or all-menu
readiness follows from transport correctness. See `http-evidence-parity.md` and #12/40.

### ADR-084: Transactional local source assertions and replayable projections

**Status**: Implemented for explicit file-SQLite enrichment and the existing legacy
polling enrichment/detail paths. Automatic persistence of every ordinary acquisition
query and a complete eligible-source plan remain pending (#40).
**Date**: 2026-11-13

Fresh-session SQL tests reproduced lost source/session list updates, discarded physical
fields and rollback when an uncalibrated origin output was null (#54). Replace mutable
four-provider consolidation with two additive tables on the existing SQLAlchemy Base:
immutable source assertions and versioned object projections. Keep the old EnrichedNEO
row as a compatibility projection, not the scientific authority. Share deterministic
atomic source reconciliation with DataFetcher; do not create a second detector.

An enrichment call owns a fresh transaction. Serialize writers with BEGIN IMMEDIATE;
commit assertions, projection and compatibility fields together, and acknowledge only
after commit. Exact same-session/source/payload repeats are idempotent. New receipts
remain dated history; equal payloads are labelled repeats, not independent measurements.
Query outcomes can be stored without manufacturing an object or a completeness score.
Source priority and assertion IDs are inputs to each replayable projection. The latest
receipt per source is a version-selection policy, not a claim about the newest scientific
orbit solution. Preserve alternatives and select physical value/uncertainty pairs atomically.

Schema initialization is explicit and additive; incompatible evidence tables fail with
migration required. File-SQLite is the validated backend; memory and other database
backends are not accepted by this initializer. UPDATE/DELETE triggers and SHA-256 replay
checks detect ordinary mutation/inconsistency, not authenticity, administrative tamper
resistance, external signatures, backups or compliance certification. Before first ledger
write to a legacy object, archive its row values and source slots. Later writes update its
compatibility row: neither original database bytes nor a backup are promised. Legacy
reads alone do not migrate or reinterpret their stored heuristic outputs.

Do not maximize legacy origin scores into a current posterior. Current origin probability
and scientific/source completeness remain unavailable. Named 15-field catalogue presence
is separate. Direct mutable private helper names remain but raise explicit controlled-
deprecation errors. Existing public service calls retain their names. See the engineering
report for tests, recovery behavior, integration limits and reproducible examples. No new
bounded context is needed; acquisition, persistence, analysis and presentation retain their
existing responsibilities.

Compatibility note: DatabaseManager retains existing initialization for other backends,
excluding the new ledger tables. That preserves the old startup path; it is not a new
PostgreSQL runtime validation. Enrichment remains explicitly unavailable there.

### ADR-085: Honest policy inspection and validated configuration publication

**Status**: Implemented for the existing legacy analysis-policy menu and the bounded
ConfigManager loading/update contracts. Whole-menu, all-field/consumer and operational
readiness remain pending (#41/#16/#17).
**Date**: 2026-11-13

The configuration menu advertised Gaussian discovery confidence and natural/artificial
origin thresholds that the canonical service does not provide. Its alpha/BIC settings
were temporary display objects, its threshold action did not apply changes, and import
was a demonstration. Remove these false success paths, not scientific quality gates.

Keep existing helper names as controlled compatibility entries. Alpha/BIC/threshold
mutation explicitly reports unsupported/not applied. The menu inspects the actual default
ScreeningAnalysisService policy, reference parameters and unchanged policy fingerprint;
explicit custom service instances and running jobs can have different recorded IDs.
Do not introduce hidden mutable global policy files or tune the KY26 score. A candidate
is a follow-up priority, not origin proof; a noncandidate is not confirmed natural.
Required-review-v1 remains separate and is not exercised/certified by this view.

Add finite, bounded, versioned read-only inspection snapshots. Exclusive-create export
cannot overwrite an existing file. The old import helper now verifies checksum/structure
and compares an archive against the current default; it never installs configuration.
Duplicate JSON keys, invalid capability claims, invalid policy domains and legacy files
are rejected. Consistent checksums are not authenticity or independent validation.
Three actual positive/ordinary/missing synthetic controls are explicitly diagnostic.
Actual Rich/plain and 80/120/160-column output is tested/captured.

The same review reproduced explicit-file parse fallback to defaults and rejected updates
leaving invalid manager state (#57). Explicit missing/malformed/unknown-key files now
fail rather than silently activate defaults. Existing file-first versus environment-only
loading remains; there is no environment-over-file merge. Updates/reloads validate a
candidate copy before publishing it; unknown update fields fail. Existing checked
threshold/API/worker fields reject bool/string/nonfinite/domain-invalid values. This
is bounded instance-level state preservation, not validation of every execution field,
thread/process synchronization, persistent hot-reload or protection against direct mutation
of returned config objects. Existing Config.save still overwrites its explicit target;
the new menu export is a different, exclusive-create operation. No new bounded context
or independently calibrated scientific model is introduced.

## ADR-086 — Commit fixed-profile acquisition evidence before returning query results

**Date:** 2026-11-13. **Status:** implemented in draft stack; file-SQLite validated.

Ordinary queries previously merged available source values without committing a
per-query source plan/history; failed initialization could shrink apparent coverage.
MPC discarded available physical detail, allowed an incorrect provisional-year
fallback and selected the first ambiguous match. Cache filenames could collide.

Extend existing Acquisition → Reconciliation → Evidence Repository boundaries.
Freeze the installed source profile before polling, retain every outcome and native
packet where available, and commit the query, assertions and projection in one owned
SQL transaction before returning/caching a result. Persistence failure is distinct
from source failure. Reuse the existing Base, immutable ledger and detail/export;
no second engine or bounded context. Lazy repository construction avoids importing
the web application. Explicit additive schema preparation is a query prerequisite,
not an import-time side effect or a backup. Unsupported stores fail closed.

Keep source/response coverage separate from field presence and scientific readiness.
Store latest-received-per-source projections with query bindings, not fabricated
freshness. Exact request identity is not canonical alias resolution. MPC requires
native identity binding and rejects ambiguous results. Native calendar labels do
not gain an invented time scale. Whole-key cache hashes and frozen envelopes prevent
known key/plan collisions; old cache bytes are ignored, not migrated as evidence.
Partial cache records remain reusable until TTL or explicit refresh; automatic
backfill and additional scientific source roles remain pending (#40/#16).

See [acquisition contract and reproducible QA](../engineering/object-acquisition-ledger.md)
for persistence, preservation, UI and scientific limits. No calibrated sigma/origin
claim follows from a complete source profile. Existing evidence and scientific
quality gates remain unchanged; two new acquisition/repository coverage gates add
90% component thresholds. Draft delivery is not whole-application certification.

## ADR-087 — Source-native NEODyS identity and photometry, without matrix relabelling

**Date:** 2026-11-13. **Status:** implemented in draft stack; verification scoped by
[NEODyS guide](../engineering/neodys-evidence.md).

Extend the existing acquisition/normalization and atomic physical-evidence contracts.
Use documented numbered/provisional OEF files directly; only aliases need bounded
native MPC resolution. Bind native OEF identity and supported header/frame/state.
Retain source outcomes and packets instead of first-match, last-record-wins or
fabricated zeroes. Keep existing helper names; legacy nominal-HTTP projection now
uses the validated OEF path. Directory health is not object-query success.

Add a source-specific physical assertion adapter for the real MAG line and a
replay dispatcher, preserving the existing SBDB format unchanged. Normalize H
atomically; preserve G natively because the domain has no typed phase-law model.
No invented uncertainties/measurement dates/estimation status. Retain alternatives,
duplicates and non-normalized native records. Account for each data record, but do
not relabel EQU covariance as Keplerian or LSP as measured accelerations.

Partition new acquisition caches by the normalization revision without rewriting
old evidence. Reuse the same SQL ledger, menu/detail, Markdown and export paths;
no new bounded context or engine. Scientific covariance transformations, global
aliases, all-source field accounting and observational fits remain open. Add a
90% NEODyS component gate; expand the existing physical-evidence gate's test inputs,
not its threshold. The whole scientific goal is not certified by these tests.

## ADR-088 — Separate astronomical date roles from local intake and compatibility dates

**Date:** 2026-11-13. **Status:** implemented in draft stack; see
[chronology contract and QA](../engineering/object-chronology.md).

A reproduced SQL-backed export mislabeled the local append clock as discovery and
lost its UTC offset through a compatibility DateTime column. Stop populating new
`first_discovered` rows from intake. Deprecate ambiguous exported timestamp aliases;
retain old values/history but do not reassign their semantics or blanket-retag them UTC.

Derive a replayable role-separated chronology from the existing verified ledger and
frozen source snapshots. Use native observation labels/partial dates, JD/MJD scales,
covariance epochs, documented fit-production basis, adapter receipts and UTC ledger
append receipts separately. Sequence order is authoritative for ledger order; report
clock regression and unverified precision. There is no inferred common UTC axis.

Extend the existing SBDB query with documented discovery circumstances and a source-
native day-label assertion. Preserve provenance, unavailable/malformed records and
conflicting claims. State provider limits: discovery circumstances only for numbered
objects; fit-production dates are Pacific local; partial observation dates are possible.
Do not invent discovery, offset/DST, midnight UTC, physical-measurement dates or fits.

The repository continues to own transactions; the new chronology is a pure read
projection, not a new bounded context/lifecycle store. Existing review-v1 assessments
remain replayable; additional rendered chronology derives from their frozen inputs.
Partition new acquisition caches by normalization revision without rewriting history.
Add a 90% chronology component gate; retain every existing quality gate. The separate
legacy freshness-polling defects remain #66/#40, and scientific readiness remains open.

ADR-088 clarification (#68): current date-event pointers resolve inside preserved
chronology inputs, while original snapshot locators remain context references.
Reconciliation metadata is not assumed to exist at the snapshot root. Old context-
reference reports retain replay/IDs with a limitation label; no silent history rewrite
or external path resolution. This does not add another context or scientific model.

## ADR-089 — Review adapter receipt age and refresh through existing acquisition

**Date:** 2026-11-13. **Status:** implemented in draft stack, integration tests and
exact-head CI required before delivery; see [receipt review](../engineering/source-freshness.md).

The legacy stale-source query guessed compatibility column names, excluded missing
values, compared SQLite naive dates with aware dates and leaked sessions on errors.
Its re-poll path only queried SBDB and could announce updates without replayable
commit evidence. These paths cannot establish source freshness or completeness.

Use the shared fixed installed-source plan and replay-checked acquisition/assertion
history. Derive receipt age from original explicit-offset adapter reads, not local
ledger ingestion, compatibility DateTime columns or astronomical epochs. Keep legacy
clocks unknown and retain their literal values. Validated empty responses differ
from data contributions. Report clock anomalies and source failures explicitly.

A read-only policy uses a 30-day adapter-read review interval and one-hour failed-
attempt cooldown. These are operational defaults, not scientific validity bounds or
provider-mandated limits. A bounded explicit refresh rechecks stored eligibility,
then invokes the existing DataFetcher and SQL evidence repository with query-cache
bypass. It does not bypass transport pacing/cooldowns or promise fresh wire data;
an underlying SDK may cache its response. Block the whole-object refresh when a
planned installed source is deferred, rather than silently shrinking the denominator.

Only a new replay-verified committed query is reported as recorded; distinguish full
source contributions, partial contributions, no data, deferred and failed outcomes.
Preserve prior measurements and history. Inspection writes no schema or records;
refresh writes remain repository-owned. Session/resource closure is explicit.

Extend acquisition, persistence and presentation responsibilities; no new bounded
context, engine, endpoint or scheduler. Automatic selective partial-cache backfill,
provider eligibility discovery, distributed refresh admission and broad legacy polling
are not completed by this change (#40/#16). Add a 90% component gate, preserve all
existing gates and capture the actual SQL-backed receipt menu at 80/120/160 columns.

ADR-089 QA clarification (#69): requested width is not measured width. The shared
capture helper now supplies both dimensions and verifies the Rich console width;
a TERM=dumb regression checks actual SVG geometry. Historical bundles are retained
with a public limitation note, not silently re-captured or reclassified.

## ADR-090 — Replace dashboard origin thresholds with replayable screening views

**Date:** 2026-11-13. **Status:** implemented in the draft stack; engineering tests
and exact-head CI are required. Scientific calibration remains unavailable. See
[polling dashboard contracts and examples](../engineering/polling-dashboard-screening.md).

The root polling dashboard and its legacy browser accepted arbitrary probability
fields, invented confidence/completeness and converted absent scores into natural
origin. Explicit null scores crashed. The browser independently categorized, sorted
and formatted these values, including unsupported sigma filters. This contradicted
the deployed screening contract even after earlier core/REST corrections.

Replace that second origin engine with adapters to ScreeningAnalysisService and the
existing verified object-review read model. Accept complete serialized NEO snapshots,
replayable canonical assessments and frozen reviews. Preserve additional evidence,
identity, units, policy versions and source assertions. Do not promote unschematized
legacy fields into measurements; retain them as unverified input. Explicit synthetic
provenance must not become an observational assessment through a different container.

Use follow-up candidate, not flagged and inconclusive outcomes. All origin confidence,
Gaussian significance, p-values and FDR remain unavailable. Not flagged is not natural.
Separate record counts, distinct exact designation labels and typed field-inventory
availability; do not invent an aggregate scientific quality/completeness score.

Keep existing entry points and navigation. Replace faulty categorization, sorting,
details and exports with shared adapters/views. Document controlled deprecation of
origin-threshold, name-based default, temporal-origin and unitless physical-origin
helpers; sigma filter options explain their unavailability rather than crashing.
Source-listed impact filtering still requires the existing replayable Sentry contract
and object identity. It does not convert screening into a local impact-risk model.

Versioned record/dashboard exports retain finite raw input, source snapshots, policy
identities and reproducibility metadata. Re-import verifies content and preserves each
frozen policy, including heterogeneous historical policies; it does not re-rank all
records under the current default. New export directories are exclusive, with JSON,
Markdown and a completion/hash marker. Historical files are not rewritten.

These responsibilities fit the existing acquisition, analysis and presentation
boundaries; no new engine, database, REST route or bounded context is introduced.
Rendering does not query providers. Root legacy source polling and its first-payload
loss remain separate confirmed defects (#73/#71). Add a 90% dashboard/browser
component gate and actual 80/120/160 captures, retaining every existing gate. Neither
these tests nor source replay establish independent scientific readiness.

ADR-090 release clarification — 2.2.0b1: the root legacy candidate-polling/inference
methods with confirmed schema, zero-default and synthetic-score defects (#73) are
controlled deprecations during the preview. They fail before provider/inference work;
no confidence or empty-success replacement is emitted. Native retrieval and the typed
per-object service remain available. This quarantine is not completion of discovery
polling or source-retention#71. Restoring those capabilities needs shared-adapter,
identity/unit/provenance and actual-path regression evidence. The full goal remains.

## ADR-091: Preserve polling assertions before object-level presentation

**Date:** 2026-11-13. **Status:** Implemented; verification recorded in the linked PR for #71.

The legacy workflow deduplicated by designation before calling the existing
transactional enrichment service. It retained the first payload and appended other
source names, dropping complementary or conflicting measurements. A source-name
union is not retained multi-source evidence.

Pass every returned payload, unchanged, to the existing assertion ledger in one
transaction. The ledger owns exact-repeat idempotency, retained source/session
history and reproducible object projections. The workflow must verify the commit
acknowledgement before publishing a dashboard or starting receipt review. A failed
transaction cannot yield a success dashboard.

The dashboard receives the original records, not a synthetic merged measurement.
Its existing distinct-label count is separate from record count. These labels are
not an alias registry, independent observations or a completeness claim. Source
outcomes distinguish returned, empty and unavailable; chunked outcomes remain
unverified when the existing chunked boundary supplies no per-source receipt.
Neither returned rows nor a configured source name prove live retrieval/freshness.

No new storage engine, bounded context, API or inference path is needed. This fixes
the workflow's lossy handoff, not the quarantined root provider implementation
(#73), persistent discovery-attempt receipts, automatic selective backfill or full
source completeness. Old results and source payloads are not rewritten.

ADR-090 root-entrypoint follow-up — 2026-11-13 / #81: the root `NEOPoller`
summary, explorer and detail helpers now delegate through a thin presentation
adapter to the existing screening read models/browser. They do not maintain origin
categories, probability thresholds or confidence formatting. Display/navigation do
not poll or persist. Export is an explicit action using the existing exclusive
JSON/Markdown writer; unavailable/empty results remain representable. A failed
export has no success acknowledgement, and direct CLI export failure returns a
nonzero status. Root acquisition remains quarantined under #73.

An optional finite-JSON `invocation_context` records requested source/period or
page context in the existing dashboard export. It is covered by the report hash,
cloned from caller state and labeled as requested context, not verified scientific
coverage. Markdown escapes executable delimiters. Existing exports without this
field keep their original format/hash; historical files are not rewritten. This is
an additive presentation/provenance extension, not a new scoring engine or bounded
context. File import preserves the original file but is not a migration or backup.
