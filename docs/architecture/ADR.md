# Architecture Decision Records — aNEOS Suite

_Derived: 2026-03-06 | Last updated: 2026-03-08 | Codebase Version: 1.0.0_
_Concept document: README.md + docs/scientific/scientific-documentation.md_

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

**Status**: Risk / Needs Consolidation

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

**Status**: Accepted / Partially Bypassed

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

**Critical misalignment found**
`automatic_review_pipeline.py` (line 52) hardcodes:
```python
from ..detection.multimodal_sigma5_artificial_neo_detector import MultiModalSigma5ArtificialNEODetector
```
This bypasses `DetectionManager` entirely. The pipeline always runs MULTIMODAL
regardless of manager priority settings.

**Consequences**
- (+) `DetectionManager` provides a clean abstraction for all other callers
- (-) The main production pipeline ignores the manager — `DetectorType.AUTO`
  has no effect on the most-used code path
- (-) Manual wrappers per detector type create maintenance burden

**Files**: `aneos_core/detection/detection_manager.py`,
`aneos_core/pipeline/automatic_review_pipeline.py:52`

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
- `FalsePositivePrevention` (`false_positive_prevention.py`, ~769 lines) —
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
| FIRST_STAGE_REVIEW | 50,000 | configurable | MultiModalSigma5Detector (hardcoded) |
| MULTI_STAGE_VALIDATION | 500 | 0.60 | MultiStageValidator |
| EXPERT_REVIEW_QUEUE | 50 | 0.80 | Expert Queue |

`PipelineConfig` carries `StageConfig` per stage, each with
`max_candidates`, `score_threshold`, `processing_timeout_seconds`,
`retry_attempts`, `parallel_workers`.

**Critical issue**: Pipeline hardcodes `MultiModalSigma5ArtificialNEODetector`
instead of routing through `DetectionManager` (see ADR-011).

**Concept alignment**: Full in structure; partial in implementation (hardcoded
detector bypass).

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

**Status**: Risk — Needs Reversal

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

**Files**: `aneos_core/integration/pipeline_integration.py`,
`aneos_api/app.py`, `aneos_menu.py`

---

## ML System

### ADR-033: Machine Learning Module — Scaffolded, Partially Integrated

**Status**: Deferred / Partially Active

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
| Population-level pattern analysis | Not started | New bounded context required (ADR-040 through ADR-048) |

---

## Population-Level Pattern Analysis — Design ADRs

_Added: 2026-03-08 | These ADRs define the architectural decisions for the
proposed Population Pattern Analysis extension (Bounded Context 11). They are
in "Concept" status — meaning the design is agreed but no code has been written.
All must be reviewed against existing data structures before implementation begins._

---

### ADR-040: Non-Gravitational Parameter Prerequisite

**Status**: Open — Prerequisite for ADR-046

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

**Files (when implemented)**: `aneos_core/data/models.py`,
`aneos_core/data/sources/sbdb.py`

---

### ADR-041: Historical Close-Approach Fetch Strategy

**Status**: Open — Prerequisite for ADR-044

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

**Files (when implemented)**: `aneos_core/data/fetcher.py`

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

**Files (when implemented)**: `aneos_core/pattern_analysis/`

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

**Files (when implemented)**: `aneos_core/pattern_analysis/clustering.py`

---

### ADR-044: Synodic Period Harmonic Analysis (PA-3)

**Status**: Concept / Blocked on ADR-041

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

**Files (when implemented)**: `aneos_core/pattern_analysis/harmonics.py`

---

### ADR-045: Pairwise Rendezvous Detection (PA-6)

**Status**: Concept / Deferred — Implement After PA-1 through PA-5

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
- (-) REBOUND is a compiled C dependency; `HAS_REBOUND` guard required;
  graceful no-op if absent
- (-) Even with PHA restriction, 2.9M MOID evaluations + propagation for
  surviving pairs is a multi-hour computation on a single machine
- (-) Deferred: implement only after PA-1 through PA-5 are operational

**Files (when implemented)**: `aneos_core/pattern_analysis/rendezvous.py`

---

### ADR-046: Cross-Object Non-Gravitational Correlation (PA-5)

**Status**: Concept / Blocked on ADR-040

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
- (-) Blocked until ADR-040 (non-grav parameter parsing) is implemented
- (-) A2 data available for ≈ 3% of NEOs; most clusters will have zero objects
  with A2 — module will silently produce no output for 97%+ of clusters
- (-) A2 values have large individual uncertainties; correlation p-values may
  be unreliable without full covariance matrix treatment

**Files (when implemented)**: `aneos_core/pattern_analysis/correlation.py`

---

### ADR-047: Network Sigma Combination (Fisher's Method)

**Status**: Concept / Design Agreed

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

**Files (when implemented)**: `aneos_core/pattern_analysis/network_sigma.py`,
`aneos_core/utils/statistical_utils.py` (reused, not modified)

---

### ADR-048: `/analyze/network` API Endpoint

**Status**: Concept / Blocked on ADR-042 through ADR-044

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

**Files (when implemented)**: `aneos_api/endpoints/analysis.py` (new route),
`aneos_api/schemas/network.py` (new file)

---

_End of Architecture Decision Records_
