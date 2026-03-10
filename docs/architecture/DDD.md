# Domain-Driven Design — aNEOS Suite

_Derived: 2026-03-06 | Last updated: 2026-03-10 | Codebase Version: 1.1.0_
_Concept documents: README.md, docs/scientific/scientific-documentation.md,
docs/engineering/sigma5_success_criteria.md, Calibration Plan v1.2_

---

## Ubiquitous Language

The following terms have precise, agreed meaning across all bounded contexts.
All developers, scientists, and stakeholders must use these terms consistently.

| Term | Definition |
|------|-----------|
| **NEO** | Near Earth Object — any asteroid or comet with perihelion < 1.3 AU |
| **Artificial NEO** | An NEO exhibiting multi-indicator anomalies consistent with external engineering at sigma >= 5 confidence |
| **Natural NEO** | An NEO whose orbital and physical properties are consistent with solar system formation and gravitational evolution |
| **Sigma score** | Standard deviations from the mean of the reference natural NEO population for a given observable |
| **Clue** | A single measurable anomaly contributing to the composite sigma score |
| **SWARM** | A named, specialized analytical agent responsible for one physical domain of evidence |
| **ATLAS** | Advanced automated first-stage screening system (used in `advanced_scoring.py`) |
| **Candidate** | An NEO that has passed at least one stage of filtering and is under active investigation |
| **Classification** | One of: `INCONCLUSIVE (σ<2) | EDGE CASE (σ≥2) | SUSPICIOUS (σ≥3) | ARTIFICIAL VALIDATED (σ≥5)` — INCONCLUSIVE replaces "NATURAL"; σ<2 is not evidence of natural origin |
| **Calibrated probability** | Bayesian posterior P(artificial | evidence) using realistic 0.1% prior |
| **Ground truth** | A verified dataset of objects with known true labels (artificial vs natural) |
| **FPR** | False Positive Rate — probability of classifying a natural object as artificial |
| **Keyhole** | A small region of orbital space whose traversal leads to a resonant Earth-collision return |
| **Non-Gravitational Parameter** | A1/A2/A3 Marsden coefficients encoding accelerations beyond gravity (Yarkovsky thermal drift or active propulsion) |
| **MOID** | Minimum Orbital Intersection Distance — the minimum possible distance between two orbits regardless of timing; used as a pre-filter before expensive trajectory propagation |
| **NEO Population** | The complete set of NEO objects submitted to a single pattern analysis session |
| **Orbital Family** | A cluster of NEOs occupying a statistically non-random volume of (a, e, i, ω, Ω) element space, potentially sharing a dynamical origin |
| **Dynamical Pair** | Two NEOs exhibiting repeated mutual close approaches below the MOID threshold; potentially a temporary association or co-orbital object |
| **Resonant Group** | A subset of NEOs sharing a common Earth-synodic period within measurement uncertainty, as detected by Lomb-Scargle periodogram |
| **Debiased Model** | A statistical NEO population model corrected for discovery survey selection effects; the Granvik et al. 2018 model is the reference for anomaly benchmarking |
| **Network Report** | Population-level analysis output from a single `NetworkAnalysisSession`, covering all enabled sub-modules (clustering, harmonics, rendezvous, correlation) |
| **Network Sigma** | Fisher-combined significance level across all population-level evidence streams; threshold ≥ 3.0 for NETWORK_ANOMALY, ≥ 5.0 for NETWORK_EXCEPTIONAL |

---

## Context Map

```
  [External]             [aNEOS Bounded Contexts]
  NASA/ESA APIs  ──ACL──▶  Data Acquisition
  Gaia Catalog   ──ACL──▶  Multi-Modal Validation (MU SWARM)
  TLE Databases  ──ACL──▶  Multi-Modal Validation (THETA SWARM)
  Gaia TAP       ──ACL──▶  Multi-Modal Validation (MU SWARM)
  REBOUND        ──ACL──▶  Population Pattern Analysis (PA-6, optional)

  Data Acquisition ──────▶ NEO Core Domain (Core)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
             Anomaly Scoring  Multi-Modal      Ground Truth
                              Validation         Context
                    │              │               │
                    └──────────────┴───────────────┘
                                   │
                                   ▼
                         Detection & Classification
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
             Reporting &                  API & External
             Visualization               Integration
                    ▲                             ▲
                    │      (concept — BC11)        │
                    └─── Population Pattern ───────┘
                             Analysis
                    ▲              ▲
                    │              │
            Data Acquisition   Detection &
            (NEOData batch)    Classification
                               (filter: σ ≥ 1.0)
```

**BC11 context rules (live as of Phase 11; Stage 1 fully implemented):**
- BC11 READS `NEOData` from BC1 and `DetectionResult` from BC5
- BC11 NEVER modifies individual `NEOData` records (read-only consumer)
- BC11 WRITES `NetworkReport` consumed by BC7 (Reporting) and BC8 (API)
- Each sub-module is independently enabled via `PatternAnalysisConfig`

ACL = Anti-Corruption Layer (each SWARM translates external formats to domain types)

---

## Bounded Context 1: Data Acquisition

**Purpose**: Source, cache, and deliver clean NEO data objects to the domain.
Owns all interaction with external APIs.

**Code**: `aneos_core/data/`, `aneos_core/polling/`

---

### Aggregate Root: `DataFetchSession`
Owns the lifecycle of a single data retrieval operation from query
parameters through cache write to domain model delivery.

### Entities

| Entity | Code Location | Description |
|--------|--------------|-------------|
| `DataFetchSession` | `data/fetcher.py:DataFetcher` | Orchestrates multi-source parallel fetch |
| `DataChunk` | `polling/historical_chunked_poller.py` | A time-bounded (5-year) slice of historical NEO data |
| `CachedResult` | `data/cache.py:CacheEntry` | Persisted fetch result keyed by query fingerprint |
| `SBDBSource` | `data/sources/sbdb.py` | NASA JPL Small Body Database API client |
| `NEODySSource` | `data/sources/neodys.py` | ESA NEO Dynamics Site API client |
| `MPCSource` | `data/sources/mpc.py` | Minor Planet Center API client |
| `HorizonsSource` | `data/sources/horizons.py` | JPL Horizons System API client |

### Value Objects

| Value Object | Code | Description |
|-------------|------|-------------|
| `TimeRange` | `ChunkConfig` fields | Immutable (start_date, end_date) query window |
| `ChunkConfig` | `polling/historical_chunked_poller.py` | chunk_size_years=5, overlap_days=7, max_objects_per_chunk=50K |
| `APIEndpoint` | `config/settings.py:APIConfig` | URL + timeout + retry parameters per source |
| `CacheKey` | `data/cache.py:_safe_key()` | Filesystem-safe query fingerprint |
| `CacheStats` | `data/cache.py:CacheStats` | hit_count, miss_count, eviction_count |
| `DataQualityScore` | `data/fetcher.py` | Completeness metric for source merger decisions |

### Domain Services

| Service | Purpose |
|---------|---------|
| `DataFetcher` | Multi-source orchestrator; parallel ThreadPoolExecutor; priority-based merge |
| `CacheManager` | RLock thread-safe LRU + TTL; JSON/pickle disk persistence |
| `CircuitBreaker` | Three-state (CLOSED/OPEN/HALF_OPEN) failure isolation per API source |
| `HistoricalChunkedPoller` | 200-year window decomposition with boundary overlap |

### Domain Events

| Event | Trigger |
|-------|---------|
| `DataFetched(source, object_count, timestamp)` | Successful API response |
| `ChunkCached(chunk_id, time_range, object_count)` | Chunk written to disk cache |
| `APIHealthChanged(source_name, is_healthy, timestamp)` | Circuit breaker state change |
| `FallbackToSimulation(reason)` | **Currently silent** — should be an explicit observable event (ADR-032 risk; ADR-050 partial remediation in menu path) |
| `CloseApproachDataFetched(designation, n_approaches, source)` | CAD API fetch complete — unconditional after primary orbital fetch (ADR-001 Update) |

### Anti-Corruption Layer
Each `DataSourceBase` subclass translates the source-specific JSON format into
internal `NEOData` / `OrbitalElements` types, preventing external schema changes
from propagating into the domain.

### Context Boundary Issues
- No `physical.py` indicators fed from physical data (see ADR-007): the data
  acquisition layer delivers physical properties but they are not consumed by
  the indicator pipeline
- `DataFetcher` performs inline deserialization — the ACL is embedded in the
  fetcher rather than being a separate translation layer

---

## Bounded Context 2: NEO Core Domain (Core Domain)

**Purpose**: Define the canonical representation of a Near Earth Object and all
observable properties. This is the central domain; all other contexts operate
on projections of it.

**Code**: `aneos_core/data/models.py`, `aneos_core/config/`

---

### Aggregate Root: `NEOObject` (implemented as `NEOData`)
The primary entity. Identity is the `designation` string (MPC/JPL provisional
or permanent identifier). All analysis contexts receive a `NEOData` instance.

### Entities

| Entity | Code | Key Fields |
|--------|------|-----------|
| `NEOData` | `data/models.py:NEOData` | designation, orbital_elements, physical_properties, close_approaches, analysis_results |
| `OrbitalElements` | `data/models.py:OrbitalElements` | eccentricity, inclination, semi_major_axis, ra_of_ascending_node, arg_of_periapsis, mean_anomaly, epoch |
| `PhysicalProperties` | `data/models.py:PhysicalProperties` | diameter, albedo, rot_per, spectral_type, absolute_magnitude |
| `CloseApproach` | `data/models.py:CloseApproach` | date, nominal_distance_AU, velocity_km_s, uncertainty_3sigma |
| `AnalysisResult` | `data/models.py:AnalysisResult` | designation, overall_score, classification, indicator_scores, created_at |

### Value Objects

| Value Object | Validation Rule |
|-------------|----------------|
| `Designation` | Non-empty string; MPC format preferred |
| `Eccentricity` | Must be in [0, 1) — enforced in `OrbitalElements._validate()` |
| `Inclination` | Must be in [0°, 180°] |
| `SemiMajorAxis` | Must be positive (AU) |
| `Albedo` | Must be in [0, 1] |
| `AbsoluteMagnitude` | H value (Johnson V band) |

### Domain Events

| Event | Trigger |
|-------|---------|
| `NEOIngested(designation, source, timestamp)` | First time object enters system |
| `NEOUpdated(designation, fields_changed)` | Source data refresh improves fields |
| `NEOFlaggedForAnalysis(designation, trigger_reason)` | Meets criteria for analysis run |

### Configuration Entities

| Entity | Code | Purpose |
|--------|------|---------|
| `ThresholdConfig` | `config/settings.py` | Per-indicator detection thresholds |
| `WeightConfig` | `config/settings.py` | Per-category scoring weights |
| `ANEOSConfig` | `config/settings.py` | Root configuration combining all sub-configs |
| `ConfigManager` | `config/settings.py` | File + env var loading; YAML/JSON support |

### Domain Issue
`OrbitalElements` carries physical properties (diameter, albedo, rotation period,
spectral type) that logically belong in `PhysicalProperties`. This violates
single responsibility and makes it unclear which model to query for physical data.

### Phase 10 Updates (2026-03-08)
- `NEOData.to_dict()` and `from_dict()` now serialise `physical_properties`
  (5 fields) and `fetched_at` — cache round-trips no longer silently drop
  physical evidence.
- `NEOData.close_approaches` is now populated from SBDB CAD API on every live
  fetch (future approaches within 0.2 AU).

### Implemented Extension (ADR-040, Phase 11)
`NonGravitationalParameters` value object added to `aneos_core/data/models.py`:
| Field | Description |
|---|---|
| `a1` | Radial Marsden component (AU/day²) |
| `a2` | Transverse component (Yarkovsky dominant) |
| `a3` | Normal component |
| `model` | Parameterization type |
| `epoch` | Reference epoch for measurement |
`NEOData.nongrav: Optional[NonGravitationalParameters]` — populated from SBDB `nongrav=1`
optional fetch; None for ~97% of objects. A2 bounds validated in `__post_init__` (±1e-9 AU/day²).

---

## Bounded Context 3: Anomaly Scoring

**Purpose**: Transform raw NEO observables into a weighted composite anomaly
score across indicator categories. Produces the first-stage filter verdict.

**Code**: `aneos_core/analysis/`

---

### Aggregate Root: `AnomalyAssessment`
Ties NEO identity to all per-indicator and per-category scores into a single
scorable unit for pipeline routing.

### Entities

| Entity | Code | Description |
|--------|------|-------------|
| `ScoreCalculator` | `analysis/scoring.py` | Weighted composite from IndicatorResults; 4-tier classification |
| `AdvancedScoreCalculator` | `analysis/advanced_scoring.py` | ATLAS 6-clue continuous scoring |
| `AnomalyScore` | `analysis/scoring.py:AnomalyScore` | designation, overall_score, confidence, classification, indicator_scores, risk_factors |
| `AdvancedAnomalyScore` | `analysis/advanced_scoring.py` | overall_score, clue_contributions, debris_penalty_applied |
| `StatisticalAnalyzer` | `analysis/scoring.py` | Population-level significance calculator |
| `AnalysisPipeline` | `analysis/pipeline.py` | ThreadPoolExecutor(10) parallel indicator runner |
| `EnhancedAnalysisPipeline` | `analysis/enhanced_pipeline.py` | Wrapper adding 5-stage validation |

### Indicator Entities (Pluggable via `AnomalyIndicator` ABC)

**Orbital Indicators** (`indicators/orbital.py`):
| Indicator | Anomaly Detected |
|-----------|-----------------|
| `EccentricityIndicator` | e > 0.8 (normal), > 0.95 (extreme) |
| `InclinationIndicator` | High inclination (polar or retrograde orbits) |
| `SemiMajorAxisIndicator` | a outside 0.3–3.5 AU natural range |
| `OrbitalResonanceIndicator` | Unusual mean motion resonance with planets |
| `OrbitalStabilityIndicator` | Lyapunov instability indicators |

**Velocity Indicators** (`indicators/velocity.py`):
| Indicator | Anomaly Detected |
|-----------|-----------------|
| `VelocityShiftIndicator` | Delta-v between successive close approaches |
| `AccelerationIndicator` | Non-gravitational acceleration A1/A2 beyond Yarkovsky |
| `VelocityConsistencyIndicator` | V_inf consistency across observations |
| `InfinityVelocityIndicator` | Hyperbolic excess velocity anomalies |

**Temporal Indicators** (`indicators/temporal.py`):
| Indicator | Anomaly Detected |
|-----------|-----------------|
| `CloseApproachRegularityIndicator` | Non-random temporal spacing of approaches |
| `ObservationGapIndicator` | Suspicious gaps in observation history |
| `PeriodicityIndicator` | Regular period suggesting station-keeping |
| `TemporalInertiaIndicator` | Approach pattern persistence over decades |

**Geographic Indicators** (`indicators/geographic.py`):
| Indicator | Anomaly Detected |
|-----------|-----------------|
| `SubpointClusteringIndicator` | Ground track clustering over specific regions |
| `GeographicBiasIndicator` | Non-uniform subpoint distribution |

**Missing Physical Indicators** (`indicators/physical.py` — DOES NOT EXIST):
| Intended Indicator | Status | Impact |
|--------------------|--------|--------|
| `DiameterAnomalyIndicator` | Missing | Physical category contributes 0 to composite score |
| `AlbedoAnomalyIndicator` | Missing | Physical category contributes 0 to composite score |
| `SpectralAnomalyIndicator` | Missing | Physical category contributes 0 to composite score |

This is a **concept-to-code misalignment** (ADR-053): `scoring.py` maps the `physical`
category but no file implements its indicators. Implementation deferred until SBDB physical
data coverage is sufficient (~30% diameter, ~25% albedo, ~15% spectral type). ATLAS weight
of 0.20 for Physical Traits is effectively wasted at present.

### Value Objects

| Value Object | Description |
|-------------|-------------|
| `IndicatorResult` | raw_score, weighted_score, confidence, metadata, contributing_factors |
| `IndicatorConfig` | weight, enabled, confidence_threshold, parameters |
| `ClueContribution` | clue_id, raw_score, sigma_deviation, weight (ATLAS) |
| `DebrisPenalty` | Applied when human_origin_confidence > 0.8; magnitude = 0.4 |

### ATLAS Clue Category Weights (advanced_scoring.py)

| Category | Default Weight | Scientific Basis |
|----------|---------------|-----------------|
| Encounter Geometry | 0.15 | Close approach distance + relative velocity |
| Orbit Behavior | 0.25 | Repeat passes, non-gravitational acceleration |
| Physical Traits | 0.20 | Area-to-mass ratio, radar cross-section, thermal |
| Spectral Identity | 0.20 | Reflectance spectrum vs taxonomic class |
| Dynamical Sanity | 0.15 | Yarkovsky drift coherence |
| Human Origin | 0.05 | Debris/spacecraft cross-match (veto) |

### Standard Category Weights (scoring.py / WeightConfig)

| Category | Default Weight |
|----------|---------------|
| velocity_shifts | 2.0 |
| close_approach_regularity | 2.0 |
| purpose_driven | 2.0 |
| acceleration_anomalies | 2.0 |
| orbital_mechanics | 1.5 |
| spectral_anomalies | 1.5 |
| physical_anomalies | 1.0 |
| temporal_anomalies | 1.0 |
| geographic_clustering | 1.0 |

### Classification Thresholds

| Score Range | Classification | Source |
|-------------|---------------|--------|
| 0.0 – 0.30 | natural | scoring.py |
| 0.30 – 0.59 | suspicious | scoring.py |
| 0.60 – 0.79 | highly_suspicious | scoring.py |
| ≥ 0.80 | artificial | scoring.py |

### Domain Events

| Event | Trigger |
|-------|---------|
| `CandidateFlagged(designation, composite_score, clue_breakdown)` | Score > first_stage threshold |
| `CandidateRejected(designation, score, stage=1)` | Score ≤ first_stage threshold |
| `DebrisPenaltyApplied(designation, penalty_magnitude, human_origin_confidence)` | THETA veto triggered |

---

## Bounded Context 4: Multi-Modal Validation

**Purpose**: Subject stage-1 candidates to independent physical-domain
validation. Each SWARM is an anti-corruption layer translating domain-specific
measurements into sigma contributions.

**Code**: `aneos_core/validation/`

---

### Aggregate Root: `ValidationReport`
Collects all SWARM verdicts and the 5-stage pipeline result for a single
candidate, determines final composite sigma, and may block output via
`ConsistencyResult.blocked_report`.

### Entities — Five-Stage Pipeline

| Stage | Entity | File | FP Reduction Target |
|-------|--------|------|---------------------|
| 1 | Data Quality Filter | `multi_stage_validator.py` | 60% |
| 2 | Known Object Cross-Match | `false_positive_prevention.py` | 80% |
| 3 | Physical Plausibility + Delta-BIC | `physical_sanity.py`, `delta_bic_analysis.py` | 90% |
| 4 | Statistical Significance | `statistical_testing.py` | 95% |
| 5 | Expert Review Threshold | `multi_stage_validator.py` | >98% |

`ValidationStageResult(stage_number, stage_name, passed, score, confidence,
false_positive_reduction, details, processing_time_ms)` per stage.

`EnhancedAnalysisResult` wraps original result + all stage results via
`__getattr__` proxy.

### SWARM Entities

| SWARM | Entity | File | Physical Domain |
|-------|--------|------|----------------|
| KAPPA | `RadarPolarizationAnalyzer` | `radar_polarization_analysis.py` | SC/OC polarization ratio vs natural rock |
| LAMBDA | `ThermalIRAnalyzer` | `thermal_ir_analysis.py` | NEATM beaming parameter deviation |
| MU | `GaiaAstrometricCalibrator` | `gaia_astrometric_calibration.py` | Non-gravitational acceleration vs Yarkovsky |
| CLAUDETTE | `StatisticalTesting` | `statistical_testing.py` | Multi-test correction; hypothesis testing |
| CLAUDETTE | `FalsePositivePrevention` | `false_positive_prevention.py` | Debris cross-match; confusion matrix |
| THETA | `HumanHardwareAnalyzer` | `human_hardware_analysis.py` | TLE database; known spacecraft catalog |
| — | `SpectralOutlierAnalyzer` | `spectral_outlier_analysis.py` | Taxonomic class outlier scoring |
| — | `DeltaBICAnalyzer` | `delta_bic_analysis.py` | Natural vs artificial model BIC comparison |
| — | `LargeScaleMonteCarlo` | `large_scale_monte_carlo.py` | Synthetic FPR calibration |
| — | `UncertaintyAnalysis` | `uncertainty_analysis.py` | Bootstrap + MC confidence propagation |
| — | `PhysicalSanityValidator` | `physical_sanity.py` | Calibration Plan v1.2 physical checks |
| — | `ConsistencyValidator` | `consistency_validator.py` | Cross-field contradiction blocking |

### Value Objects

| Value Object | Description |
|-------------|-------------|
| `PolarizationRatio` | SC/OC radar circular polarization ratio |
| `ThermalSignature` | NEATM beaming parameter η and emissivity |
| `AstrometricDeviation` | A1/A2 non-gravitational parameters vs Yarkovsky model |
| `DeltaBICResult` | delta_bic, preferred_model, model_confidence |
| `SpaceDebrisMatch` | Matched TLE entry + match confidence |
| `HumanHardwareMatch` | Matched spacecraft + orbit correlation confidence |
| `SpectralOutlierResult` | Taxonomic distance, closest class, outlier sigma |
| `PhysicalValidationResult` | PASS/WARNING/FAIL + issues + corrected_values |
| `ConsistencyResult` | is_valid, violations[], errors[], blocked_report |
| `UncertaintyResult` | confidence_interval_95, confidence_interval_99, indicator_uncertainties |
| `MonteCarloFPR` | Empirical false-positive rate from N synthetic runs |
| `MultipleTestingResult` | Bonferroni/BH corrected p-values per indicator |

### Consistency Violation Types

| Type | Triggered When |
|------|---------------|
| `RISK_PROBABILITY_CONTRADICTION` | risk_label = "high" but P_impact = 0 |
| `ARTIFICIAL_CLASSIFICATION_MISMATCH` | is_artificial=True but classification="natural" |
| `PHYSICS_VIOLATION` | Crater < 1 km for a 500m+ impactor |
| `SEVERE_INCONSISTENCY` | Multiple contradictions co-present |

### Domain Events

| Event | Trigger |
|-------|---------|
| `ValidationPassed(designation, composite_sigma, swarm_verdicts)` | All 5 stages pass |
| `ValidationFailed(designation, failing_stages, composite_sigma)` | One or more stages fail |
| `FalsePositiveDetected(designation, reason, triggering_swarm)` | Stage 2 or CLAUDETTE rejection |
| `PhysicalInconsistencyBlocked(designation, violations)` | ConsistencyValidator blocks report |
| `DebrisMatchFound(designation, matched_tle, confidence)` | THETA SWARM match |

---

## Bounded Context 5: Detection & Classification

**Purpose**: Maintain the lifecycle of detection candidates from first sigma-5
scoring through expert review queue. Owns the final artificial/natural verdict.

**Code**: `aneos_core/detection/`, `aneos_core/interfaces/`

---

### Aggregate Root: `DetectionCandidate`
Combines NEO identity + multi-modal evidence package + composite sigma score
+ Bayesian calibrated probability + queue status.

### Entities

**Canonical (production path):**

| Entity | Code | Description |
|--------|------|-------------|
| `DetectionManager` | `detection/detection_manager.py` | Priority-based detector registry; unified entry point (ADR-011) |
| `ValidatedSigma5ArtificialNEODetector` | `detection/validated_sigma5_artificial_neo_detector.py` | **Canonical — priority 0**; Bayesian corrected; scientifically validated (ADR-008 Update) |
| `MultiModalSigma5ArtificialNEODetector` | `detection/multimodal_sigma5_artificial_neo_detector.py` | Multi-modal evidence fusion; priority 1 fallback (ADR-013) |

**Archived variants (fallback only; ADR-013):**

| Entity | Code | Description |
|--------|------|-------------|
| `ProductionArtificialNEODetector` | `detection/production_artificial_neo_detector.py` | Hard-coded thresholds; priority 2 |
| `CorrectedSigma5ArtificialNEODetector` | `detection/corrected_sigma5_artificial_neo_detector.py` | First calibration pass; priority 3 |
| `Sigma5ArtificialNEODetector` | `detection/sigma5_artificial_neo_detector.py` | Original basic; priority 4 |
| `Sigma5CorrectedStatisticalFramework` | `detection/sigma5_corrected_statistical_framework.py` | Statistical framework shared by archived variants |

**Supporting:**

| Entity | Code | Description |
|--------|------|-------------|
| `GroundTruthDatasetBuilder` | `datasets/ground_truth_dataset_preparation.py` | Operational (Phase 3/4); sensitivity=1.00, specificity=1.00 on small corpus (ADR-038) |
| `ArtificialNEOTestSuite` | `detection/artificial_neo_test_suite.py` | Unit test harness; should reside in `tests/` |

### `ProductionArtificialNEODetector` Hard-Coded Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `semi_major_axis_threshold` | 1.5 AU | Conservative (addresses false positives) |
| `eccentricity_threshold` | 0.6 | Higher than natural mode |
| `inclination_low_threshold` | 50° | More restrictive |
| `inclination_high_threshold` | 80° | Captures polar orbits |
| `confidence_threshold` | 0.60 | Balanced |
| `combined_indicator_requirement` | 2 | Minimum strong indicators needed |

### Value Objects

| Value Object | Description |
|-------------|-------------|
| `DetectionResult` | is_artificial, confidence, sigma_level, artificial_probability, classification, analysis, risk_factors |
| `CompositeScore` | Multi-modal weighted sigma (final) |
| `CalibratedProbability` | Bayesian posterior P(artificial | evidence) |
| `ConfidenceInterval` | (lower_95, upper_95) Bayesian bounds |
| `FalsePositiveRisk` | Empirical FPR estimate for this detection |
| `DetectorType` | Enum: BASIC, CORRECTED, MULTIMODAL, PRODUCTION, VALIDATED, AUTO |

### Domain Services

| Service | Purpose |
|---------|---------|
| `DetectionManager` | Registry + AUTO selection + per-detector output adapters |
| `OrbitalElementsNormalizer` | Translates dual-naming (a/semi_major_axis, e/eccentricity) |
| `_display_neo_data()` | Presents orbital elements, physical properties, observation arc, completeness, and nearest approach before the detection result in Options 1 and 2 (ADR-051) |

### Interfaces (ABCs)

| Interface | Purpose |
|-----------|---------|
| `ArtificialNEODetector` | Base contract: `detect(orbital_elements) -> DetectionResult` |
| `EnhancedDetector` | Adds physical data to detection |
| `MultiModalDetector` | Full multi-modal evidence fusion |

### `aNEOSAnalysisResult` Fields (unified output)

| Field | Type | Description |
|-------|------|-------------|
| `designation` | str | Object identifier |
| `is_artificial` | bool | Binary classification |
| `artificial_probability` | float | Calibrated Bayesian probability |
| `confidence_level` | float | Statistical confidence |
| `classification` | str | natural/suspicious/highly_suspicious/artificial |
| `risk_assessment` | str | low/moderate/high/critical |
| `sigma_statistical_level` | float? | Composite sigma score |
| `orbital_analysis` | Dict? | Per-indicator orbital results |
| `physical_analysis` | Dict? | Per-indicator physical results |
| `temporal_analysis` | Dict? | Per-indicator temporal results |

### `AnalysisCapability` Enum

```
ARTIFICIAL_DETECTION, ORBITAL_ANALYSIS, PHYSICAL_ANALYSIS,
TEMPORAL_ANALYSIS, MULTI_SOURCE_ENRICHMENT, STATISTICAL_VALIDATION,
REAL_TIME_MONITORING, HISTORICAL_ANALYSIS
```

### Domain Events

| Event | Trigger |
|-------|---------|
| `CandidateDetected(designation, composite_sigma, evidence_package)` | sigma >= 5 |
| `CandidateRejected(designation, sigma, stage)` | Below threshold at any stage |
| `ExpertReviewQueued(designation, priority, sigma)` | Passes all stages; score >= 0.80 |
| `GroundTruthValidated(designation, true_label, detector_label)` | **Future** — when dataset exists |

### Pipeline Inconsistency (RESOLVED — Phase 19)

`automatic_review_pipeline.py` previously hardcoded `MultiModalSigma5ArtificialNEODetector`,
bypassing `DetectionManager` and the `VALIDATED` detector (priority 0).
**Fixed**: pipeline now uses `DetectionManager(preferred_detector=DetectorType.AUTO)` at
line 175, correctly honouring the priority-0 VALIDATED detector.

---

## Bounded Context 6: Impact Assessment (Planetary Defense)

**Purpose**: Calculate Earth and Moon impact probabilities for candidates and
all Earth-crossing NEOs. This is the secondary mission defined in the concept
document.

**Code**: `aneos_core/analysis/impact_probability.py`,
`aneos_core/analysis/impact_enhanced_pipeline.py`

---

### Aggregate Root: `ImpactAssessment`
Complete impact risk profile for a single NEO covering both Earth and Moon
scenarios.

### Entities

| Entity | Code | Description |
|--------|------|-------------|
| `ImpactProbabilityCalculator` | `analysis/impact_probability.py` | Core physics engine |
| `ImpactEnhancedPipeline` | `analysis/impact_enhanced_pipeline.py` | Wires impact data into standard pipeline output |

### `ImpactProbability` Value Object Fields

| Field | Description |
|-------|-------------|
| `collision_probability` | Overall Earth impact probability |
| `collision_probability_per_year` | Annual expected rate |
| `time_to_impact_years` | Most probable impact time |
| `probability_uncertainty` | 95% CI (lower, upper) |
| `calculation_confidence` | 0–1 quality score |
| `data_arc_years` | Observation arc length |
| `earth_collision_cross_section_km2` | Geometric + gravitational focusing |
| `moon_collision_probability` | Lunar impact probability |
| `moon_earth_probability_ratio` | Comparative risk factor |
| `impact_energy_mt_tnt` | Energy equivalent (megatons TNT) |
| `crater_diameter_km` | Estimated crater via pi-scaling |
| `damage_radius_km` | Overpressure damage radius |
| `risk_level` | negligible/low/moderate/high/extreme |
| `gravitational_keyholes` | List of keyhole orbital regions |
| `regional_probability_distribution` | Geographic impact probability map |
| `temporal_evolution` | Time-series of probability evolution |
| `artificial_object_considerations` | Enhanced uncertainty for propulsive objects |

### Calculation Methods

| Method | Physics |
|--------|---------|
| Collision cross-section | Geometric + gravitational focusing: σ = πR²(1 + v_esc²/v_inf²) |
| Moon impact | Separate lunar σ with Moon's v_esc and R_moon |
| Keyhole analysis | Resonant return orbital element mapping |
| Monte Carlo propagation | `scipy.integrate.solve_ivp` over orbital uncertainty ellipsoid |
| Impact energy | E = ½mv² with bulk density assumptions |
| Crater scaling | Pi-scaling laws (regime-dependent: simple, complex, basin) |

### Domain Events

| Event | Trigger |
|-------|---------|
| `ImpactRiskAssessed(designation, earth_prob, moon_prob, energy_mt)` | Assessment complete |
| `HighRiskFlagged(designation, collision_probability, timeline)` | P_impact > threshold |
| `KeyholeIdentified(designation, keyhole_date, orbital_region)` | Resonant return path found |
| `CraterSizeEstimated(designation, crater_diameter_km, uncertainty_pct)` | Crater calculation complete |
| `EarthLunarRatioCalculated(designation, earth_prob, moon_prob, ratio)` | Comparative impact output produced |

---

## Bounded Context 7: Reporting & Visualization

**Purpose**: Transform detection and validation results into publication-ready
reports, dashboards, and exportable data packages. Includes the interactive
terminal menu as the primary presentation layer for human users.

**Code**: `aneos_core/reporting/`, `aneos_menu_v2.py` (primary UI), `aneos_menu.py` (legacy UI)

---

### Aggregate Root: `AnalysisReport`
A complete, exportable record of one analysis run including all candidates,
scores, methodology references, and uncertainty bounds.

### Entities

| Entity | Code | Output Formats |
|--------|------|---------------|
| `ANEOSMenuV2` | `aneos_menu_v2.py` | Rich terminal, 15-option menu (primary) |
| `DetectionAnalytics` (Option 13) | `aneos_menu_v2.py` | σ-tier + ATLAS-tier session statistics, JSON export via Exporter |
| `ANEOSMenu` | `aneos_menu.py` | Rich terminal, 121-option full menu (legacy) |
| `ReportGenerator` | `reporting/generators.py` | Rich terminal, structured JSON |
| `Exporter` | `reporting/exporters.py` | CSV, JSON, FITS |
| `Visualizer` | `reporting/visualizers.py` | matplotlib, plotly charts |
| `AIValidationAnnotator` | `reporting/ai_validation.py` | AI-generated academic-style notes |
| `ProgressTracker` | `reporting/progress.py` | Real-time progress bars (Rich) |
| `AnalyticsCalculator` | `reporting/analytics.py` | Aggregate statistics |
| `ProfessionalSuite` | `reporting/professional_suite.py` | Bundled multi-format package |

### Value Objects

| Value Object | Description |
|-------------|-------------|
| `ReportFormat` | Enum: SUMMARY, DETAILED, PRIORITY, ANOMALY |
| `ExportMetadata` | run_id, timestamp, object_count, methodology_version |

### Domain Events

| Event | Trigger |
|-------|---------|
| `ReportGenerated(run_id, format, path, timestamp)` | Report written to disk |
| `ExportCompleted(run_id, format, path, object_count)` | Export file created |
| `ProgressMilestone(stage, pct_complete, eta_seconds)` | Long-run status update |

### Risk
`AIValidationAnnotator` generates human-readable validation language. AI-generated
text may overstate confidence and contribute to the documentation drift risk
identified in `maturity_assessment.md`.

---

## Bounded Context 8: API & External Integration

**Purpose**: Expose aNEOS capabilities over HTTP to external research consumers
and provide a web dashboard.

**Code**: `aneos_api/`

---

### Aggregate Root: `APIRequest`
Validated, authenticated HTTP request with rate-limiting and CORS applied.

### Entities

| Entity | Code | Description |
|--------|------|-------------|
| `FastAPIApp` | `aneos_api/app.py` | Core application; CORS + GZip middleware; 52 endpoints |
| `AuthToken` | `aneos_api/auth.py` | JWT token with expiry and scope |
| `Dashboard` | `aneos_api/dashboard.py` | Real-time web UI |
| `DatabaseSession` | `aneos_api/database.py` | SQLAlchemy 2.0 session management |
| `EnhancedModel` | `aneos_api/enhanced_models.py` | Extended Pydantic response models |

### Endpoint Groups

| Group | File | Responsibility |
|-------|------|---------------|
| Analysis | `endpoints/analysis.py` | Trigger and retrieve analysis runs |
| Enhanced Analysis | `endpoints/enhanced_analysis.py` | Advanced multi-modal analysis |
| Monitoring | `endpoints/monitoring.py` | System health, component status |
| Prediction | `endpoints/prediction.py` | ML inference (deferred — ADR-033) |
| Streaming | `endpoints/streaming.py` | Real-time event streaming (Redis pub/sub) |
| Admin | `endpoints/admin.py` | User management, configuration |

### Value Objects

| Value Object | Description |
|-------------|-------------|
| `APIEndpointSpec` | Path, method, auth requirement, rate limit |
| `AuthScope` | Permission set encoded in JWT |

### Context Boundary Issues
- No DTO/schema separation: endpoint handlers use domain types directly,
  coupling API versioning to domain model evolution
- No OpenAPI spec maintained separately; auto-generated by FastAPI only when
  installed, creating potential for drift with `docs/api/rest-api.md`
- All imports wrapped in `HAS_FASTAPI` guard — API silently disabled if
  FastAPI not installed (ADR-032)

---

## Bounded Context 9: Infrastructure & Monitoring

**Purpose**: Maintain operational health, collect performance and analysis
quality metrics, and surface alerts.

**Code**: `aneos_core/monitoring/`, `docker-compose.yml`, `k8s/`,
`prometheus.yml`

---

### Aggregate Root: `SystemHealthReport`
Snapshot of all monitored components at a point in time.

### Entities

| Entity | Code | Description |
|--------|------|-------------|
| `MetricsCollector` | `monitoring/metrics.py` | psutil-based system metrics |
| `AlertManager` | `monitoring/alerts.py` | Rule-based alerts; email notification |
| `MonitoringDashboard` | `monitoring/dashboard.py` | Component health aggregator |
| `PrometheusExporter` | `prometheus.yml` + API `/metrics` | Time-series metrics |
| `GrafanaDashboard` | `docker-compose.yml:grafana` | Visualization |

### `SystemMetrics` Value Object Fields

| Field | Description |
|-------|-------------|
| `cpu_percent` | Process CPU utilization |
| `memory_percent` | System memory usage |
| `memory_used_mb` / `memory_available_mb` | Absolute memory |
| `disk_usage_percent` / `disk_free_gb` | Storage health |
| `network_bytes_sent/recv` | Network throughput |
| `process_count` | Active process count |
| `load_average` | System load (1/5/15 min) |

### Alert Types

| Alert Type | Trigger |
|------------|---------|
| `ANOMALOUS_NEO` | High-sigma candidate detected |
| `SYSTEM_PERFORMANCE` | CPU/memory threshold breach |
| `MODEL_DRIFT` | ML prediction distribution shift |
| `DATA_QUALITY` | Incomplete or anomalous API data |
| `SYSTEM_ERROR` | Unhandled exception in pipeline |

### Alert Levels: `LOW | MEDIUM | HIGH | CRITICAL`

### Notification Channels
`AlertManager` supports SMTP email notifications via `smtplib`/`ssl`.
No Slack, PagerDuty, or webhook integration exists yet.

### Critical Dependency Issue
`monitoring/alerts.py` imports `Alert` from `ml.prediction`:
```python
from ..ml.prediction import Alert as MLAlert, PredictionResult
```
This creates a hard dependency on the ML module from the monitoring layer.
If ML is unavailable, alerting breaks. See ADR-033 and ADR-056 for design rationale
and accepted risk. `Alert` is used as a carrier class only — no model inference at alert time.

### Infrastructure Components

| Component | Image | Port |
|-----------|-------|------|
| aneos-api | Custom Dockerfile | 8000 |
| postgres | postgres:15-alpine | 5432 |
| redis | redis:7-alpine | 6379 |
| nginx | nginx:alpine | 80, 443 |
| prometheus | prom/prometheus | 9090 |
| grafana | grafana/grafana | 3000 |

### Kubernetes Manifests
`k8s/deployment.yml`, `k8s/postgres.yml`, `k8s/redis.yml` — production-grade
configuration. No service mesh or ingress controller configured.

---

## Bounded Context 10: Ground Truth (Operational — Phase 3/4 Complete)

**Purpose**: Provide a verified, labelled dataset of confirmed artificial and
natural objects to enable empirical accuracy, recall, and FPR measurement.

**Status**: Operational. `GroundTruthDatasetBuilder` and `GroundTruthValidator`
are active. External validation has been performed on 3 confirmed artificials
and 20+ natural NEOs. Blind-test set not yet built.

**Code**: `aneos_core/datasets/`

---

### Aggregate Root: `GroundTruthDataset`
A versioned, immutable collection of labelled objects split into training,
validation, and blind-test sets.

### Entities

| Entity | Status | Description |
|--------|--------|-------------|
| `GroundTruthDatasetBuilder` | Operational | Compiles 9 confirmed artificials (3 SBDB + 6 Horizons) + up to 250 natural NEOs from SBDB |
| `GroundTruthObject` | Operational | object_id, is_artificial, orbital_elements, source, verification_notes |
| `GroundTruthValidator` | Operational | Runs canonical detector on corpus; computes accuracy metrics |
| `BlindTestSuite` | Not started | Randomized test set with withheld labels |
| `AccuracyReport` | Operational | precision, recall, F1, FPR, ROC-AUC at calibrated threshold |

### Validation Results (Phase 3 external run, 2026-03-07)

| Metric | Value | Corpus |
|---|---|---|
| Sensitivity (recall) | 1.00 | 3 confirmed artificials |
| Specificity | 1.00 | 20+ JPL natural NEOs |
| F1 | 1.00 | Combined |
| ROC-AUC | 1.00 | Full corpus |
| Calibrated threshold | 0.037 | Bayesian posterior |

### Value Objects

| Value Object | Description |
|-------------|-------------|
| `TrueLabel` | `ARTIFICIAL | NATURAL | UNKNOWN` |
| `PredictedLabel` | Detector output |
| `ConfusionMatrix` | TP/TN/FP/FN counts |
| `DetectionMetric` | precision, recall, F1, FPR at sigma-5 threshold |

### Verification Sources (Planned)

| Source | Object Types |
|--------|-------------|
| Space-Track.org TLE | Rocket bodies, dead satellites in heliocentric orbit |
| JPL Horizons | Confirmed spacecraft trajectories (Voyager, Pioneer, etc.) |
| MPC unusual objects | Oumuamua-class interlopers (natural anomalies) |
| JPL SBDB | Natural NEO population sample |

### Domain Events

| Event | Trigger | Status |
|-------|---------|--------|
| `GroundTruthValidated(designation, true_label, predicted_label)` | Single object labelled against corpus | Active (Phase 3) |
| `DatasetVersionReleased(version, n_artificial, n_natural)` | New ground truth corpus published | Planned — next corpus expansion |
| `BlindTestCompleted(detector_type, precision, recall, fpr)` | Held-out blind-test run completes | Planned — requires ≥50 confirmed artificials |

---

---

## Bounded Context 11: Population Pattern Analysis (Stage 1 Implemented — Phase 19, Stage 2 Deferred)

**Purpose**: Identify statistically anomalous structures in a collection of NEO
objects by comparing their collective orbital, temporal, and dynamical properties
against the natural NEO population model (Granvik et al. 2018 debiased reference).

**Code**: `aneos_core/pattern_analysis/`

**Status**: Stage 1 implemented (Phase 11–19). PA-1 (clustering), PA-3 (harmonics),
PA-5 (correlation), network sigma combiner, and PA-6 Stage 1 rendezvous scanner are
all live. BC11 integrates modularly without touching any v1.0 component.
Stage 2 (PA-6 REBOUND propagation) deferred pending dependency adoption.

---

### Aggregate Root: `NetworkAnalysisSession`

**Design ADRs**: ADR-042 (BC architecture), ADR-043 (clustering PA-1), ADR-044 (harmonics PA-3),
ADR-045 (rendezvous PA-6), ADR-046 (correlation PA-5), ADR-047 (network sigma), ADR-048 (API endpoint)

Owns the lifecycle of one population-level analysis run: from the input batch
of designations through sub-module execution to `NetworkReport` production.

| Field | Type | Description |
|---|---|---|
| `session_id` | str | Unique run identifier |
| `designations` | List[str] | Objects in scope |
| `config` | `PatternAnalysisConfig` | Which sub-modules are enabled |
| `status` | str | pending / running / complete / failed |
| `report` | Optional[NetworkReport] | Populated when complete |

### Entities

| Entity | Sub-module | Prerequisite | Status |
|---|---|---|---|
| `OrbitalElementClusterer` | `clustering.py` | None | Implemented (Phase 11, PA-1) |
| `SynodicHarmonicAnalyzer` | `harmonics.py` | ADR-041 ✅ done | Implemented (Phase 11, PA-3) — Note: ~60% of objects skipped (insufficient historical epochs) |
| `NonGravCorrelator` | `correlation.py` | ADR-040 ✅ done | Implemented (Phase 11, PA-5) — Note: ~97% of clusters produce no output (A2 data available for ≈3% of NEOs) |
| `RendezvousDetector Stage 1` | `rendezvous.py` | aiohttp | Implemented (Phase 19, PA-6 Stage 1) — Option 15 |
| `RendezvousDetector Stage 2` | `rendezvous.py` | `rebound` compiled dep | Deferred — REBOUND N-body propagation pending dependency adoption |
| `NetworkSigmaCombiner` | `network_sigma.py` | At least one sub-module | Implemented (Phase 11) |

### Value Objects

| Value Object | Description |
|---|---|
| `OrbitalCluster` | members, centroid (a,e,i,ω,Ω), density_sigma, known_family, p_value |
| `HarmonicSignal` | designation, dominant_period_days, power_excess_sigma, p_value |
| `RendezvousPair` | designation_a, designation_b, n_encounters, min_distance_au, encounter_epochs, p_value |
| `CorrelationMatrix` | cluster_id, designations, matrix, flagged_pairs, bonferroni_threshold |
| `NetworkSigma` | combined_p_value, sigma, tier, sub_module_contributions |
| `PatternAnalysisConfig` | clustering: bool, harmonics: bool, correlation: bool, rendezvous: bool, max_objects: int |

### Domain Services

| Service | Purpose |
|---|---|
| `NetworkAnalysisSession` | Orchestrates sub-module execution; respects `PatternAnalysisConfig` |
| `GranvikReferenceModel` | Provides expected density/distribution from Granvik 2018 Table 1 constants |
| `MoidPreFilter` | Computes MOID from orbital elements; eliminates non-candidate rendezvous pairs |

### External Dependencies

| Dependency | Guard | Purpose |
|---|---|---|
| `numpy`, `scipy` | None (already required) | Clustering, distance metrics, Lomb-Scargle |
| `astropy.timeseries` | None (already required) | LombScargle periodogram |
| `hdbscan` | `HAS_HDBSCAN` | Primary clustering; fallback: `sklearn.cluster.DBSCAN` |
| `rebound` | `HAS_REBOUND` | Rendezvous propagation (PA-6 only) |

### Domain Events

| Event | Trigger |
|---|---|
| `OrbitalFamilyDetected(cluster_id, members, density_sigma)` | New cluster exceeds 3σ background density |
| `ResonantGroupDetected(designation, period_days, power_sigma)` | Lomb-Scargle excess power > 3σ |
| `DynamicalPairFound(designation_a, designation_b, n_encounters)` | Repeated mutual close approach |
| `NonGravCorrelationFound(cluster_id, pair, r_value)` | Correlated A2 across cluster members |
| `NetworkAnomalyFlagged(session_id, network_sigma, tier)` | Network sigma ≥ 3.0 |
| `NetworkExceptionalFlagged(session_id, network_sigma)` | Network sigma ≥ 5.0 |
| `HarmonicAnalysisSkipped(designation, reason, epoch_count)` | Object has < 5 historical close-approach epochs — PA-3 cannot run |
| `CorrelationAnalysisSkipped(cluster_id, reason)` | Cluster has zero members with A2 data — PA-5 produces no output |

### Context Boundary Rules

1. BC11 reads `NEOData` from BC1 (Data Acquisition). It never modifies `NEOData` records.
2. BC11 reads `DetectionResult` from BC5 as a pre-filter: only objects with σ ≥ 1.0
   (NOTABLE or above) are included in the population analysis.
3. Historical approach data is fetched by BC1's `DataFetcher.fetch_historical_approaches()`
   on behalf of BC11 — BC11 does not call external APIs directly.
4. `NetworkReport` (BC11 output) is consumed by BC7 (Reporting) and BC8 (API).
5. BC11 never contributes to individual `DetectionResult` or `DetectionResponse` objects.
   Network findings are reported separately.

---

## Domain Evolution History & Roadmap (Updated 2026-03-10)

### v1.0 → v1.1 Completed (Phases 11–19)
- **PA-1** ✅: `OrbitalElementClusterer` + `NetworkSigmaCombiner` — BC11 Stage 1 milestone
- **ADR-040** ✅: `NonGravitationalParameters` added to `NEOData` (Phase 11)
- **ADR-041** ✅: `fetch_historical_approaches()` in `DataFetcher` (Phase 17)
- **PA-3** ✅: `SynodicHarmonicAnalyzer` using Lomb-Scargle on historical approach epochs
- **PA-5** ✅: `NonGravCorrelator` Pearson A2 correlation within clusters
- **ADR-048** ✅: `POST /analyze/network` + `GET /analyze/network/{job_id}/status` (Phase 11)
- **PA-6 Stage 1** ✅: `RendezvousDetector` MOID pre-filter; Option 15 (Phase 19)

### v1.2 Deferred Roadmap
- **Blind-test set** for BC10 — withheld-label validation; requires ≥50 confirmed artificials
- **PA-6 Stage 2**: REBOUND N-body propagation for rendezvous encounter epoch confirmation
- **ML Classification Context**: activate `aneos_core/ml/` training with expanded ground truth
- **Physical Indicators** (ADR-053): `indicators/physical.py` when SBDB data coverage sufficient
- **Real-Time Streaming Context**: activate `endpoints/streaming.py` with Redis pub/sub

### Long-Term
- **Publication Pipeline Context** — evidence package → peer-review document generator
- **Collaboration Context** — multi-institution data sharing and result federation
- **TLE Registry Context** — live Space-Track feed for THETA SWARM enrichment

---

## Concept Document Alignment Matrix (v1.1.0)

| Scientific Doc Requirement | Implementation | Status |
|---------------------------|----------------|--------|
| Primary mission: artificial detection | All detection modules | Implemented |
| Secondary mission: planetary defense | `impact_probability.py`; 16-field API | Implemented |
| Multi-modal Sigma-5 detection | `ValidatedSigma5ArtificialNEODetector` | Implemented + ground-truth validated (Phase 3) |
| 5 indicator categories | 4 categories; physical indicators in `indicators/physical.py` | Physical category implemented but not yet wired to main pipeline |
| Bayesian base rate correction | `validated_sigma5` detector | Implemented |
| Multiple testing correction | `statistical_testing.py` | Implemented |
| Uncertainty quantification | `uncertainty_analysis.py`; `ImpactResponse.probability_uncertainty` | Implemented |
| Reproducible methodology | Version-controlled code + config + OpenAPI spec | Implemented |
| Peer-review ready output | Report generator | Implemented |
| Ground truth validation | `ground_truth_dataset_preparation.py` + `GroundTruthValidator` | Operational; sensitivity=1.00, specificity=1.00 (small corpus) |
| Real NASA/JPL data | `data/sources/` (SBDB, Horizons, CAD, NEODyS, MPC) | Implemented; silent fallback risk remains (ADR-032) |
| Moon impact assessment | `impact_probability.py`; `ImpactResponse.moon_*` fields | Implemented |
| FPR < 5.7×10⁻⁷ | Monte Carlo validator | Theoretically modeled; not empirically verified at scale |
| Publication-standard sigma-5 | Detectors + `statistical_utils.py` | Methodology correct; blind-test validation still needed |
| Population-level pattern analysis | `aneos_core/pattern_analysis/` | Stage 1 Implemented (PA-1, PA-3, PA-5, PA-6 MOID — Phases 11–19); Stage 2 (PA-6 REBOUND) Deferred |

---

_End of Domain-Driven Design Document_
