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
| **Artificial NEO** | An object whose technological origin is independently supported; a screening rank alone cannot establish this |
| **Natural NEO** | An NEO whose orbital and physical properties are consistent with solar system formation and gravitational evolution |
| **Screening score** | Uncalibrated legacy ranking heuristic; not a Gaussian-equivalent significance |
| **Gaussian-equivalent significance** | A calibrated null-tail probability expressed on a specified Gaussian convention; unavailable in the installed screening model |
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

**Physical Indicators** (Phase 21A — ADR-053 closed):
| Indicator | Status | Notes |
|-----------|--------|-------|
| `DiameterAnomalyIndicator` | **Implemented (Phase 21A)** | Wired in `detection_manager.py` `ValidatedWrapper.analyze_neo()` when SBDB supplies `diameter_km`; μ flag in ATLAS output |
| `AlbedoAnomalyIndicator` | **Implemented (Phase 21A)** | Same path; α flag in ATLAS output |
| `SpectralAnomalyIndicator` | Deferred | Spectral type coverage ~15% — insufficient for reliable scoring |

ADR-053 is now closed for the single-object detection path. Physical Traits ATLAS weight
(0.20) is partially active when SBDB supplies diameter or albedo data (~30% and ~25%
coverage respectively). The **CAD-API pipeline** (Option 8 / `automatic_review_pipeline.py`)
remains 0.0 for all physical scores (ADR-059): batch objects lack individual physical
observations.

**Phase 20 Pipeline Proxy Fix (ADR-059)**: The CAD-API pipeline previously synthesised
`radar_score`, `thermal_score`, and `spectral_score` from orbital elements (eccentricity,
inclination). These fabricated values fed ATLAS and produced misleading explanations ("Radar
polarization indicates artificial surface properties"). From Phase 20 these are zeroed out
in `automatic_review_pipeline.py`; real values only appear if a caller provides measured
data. Similarly, ΔBIC confidence is lowered to 0.35 (proxy) and the ATLAS Δ flag threshold
raised from 0.4 → 0.65 so no proxy score can trigger it.

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
| `NetworkSigma` | combined_p_value, sigma, tier, sub_module_contributions, method |
| `PatternAnalysisConfig` | clustering: bool, harmonics: bool, correlation: bool, rendezvous: bool, max_objects: int, combination_method: str (default 'fisher') |

### Domain Services

| Service | Purpose |
|---|---|
| `NetworkAnalysisSession` | Orchestrates sub-module execution; respects `PatternAnalysisConfig` |
| `NetworkSigmaCombiner` | Combines sub-module p-values; supports `method='fisher'` (default) and `method='stouffer'` with per-key weights (Phase 21B — SCI-003 mitigated) |
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
| 5 indicator categories | 4 full categories + physical (partial); `DiameterAnomalyIndicator`/`AlbedoAnomalyIndicator` in `detection_manager.py` | Physical category active for single-object path when SBDB data available (Phase 21A) |
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

## Non-gravitational evidence implementation — 2026-11-12

BC1 acquisition and BC2 core data own the new `OrbitModelParameter` and
`OrbitalCovariance` value objects, composed by the existing
`NonGravitationalParameters`. They preserve native values, uncertainties,
estimation kind, model constants, coordinate labels, provenance and distinct
TDB epochs. Their purpose is faithful evidence transport, not origin inference.
BC7/BC8 expose this evidence through the existing Rich/REST surfaces. BC4 retains
responsibility for future physical-model fitting and epoch-aware covariance
propagation. No new bounded context is needed for these changes (ADR-064).

Implementation and offline path coverage do not mean that the legacy detector's
sigma/probability output or a KY26 origin hypothesis has been validated. Those
requirements remain open in the programme issues.


## Screening contract correction — 2026-11-12 (ADR-065)

Use the existing analysis/detection boundary, not a new bounded context.
`ScreeningAnalysisService` owns deterministic ranking and follow-up status.
Acquisition owns source evidence; scheduling and persistence remain separate.

A `ScreeningAssessment` identifies its ranking-input snapshot and policy by
content hashes. Its contributions are labeled heuristic proxies and dependency
groups, not independent origin measurements. Typed orbit-fit coefficients retain
the ADR-064 evidence contract and cannot become propulsion proof through adapter
conversion. Unavailable inference is null, never zero or a natural-origin verdict.

**Implemented/integrated:** canonical manager/facade, menu single/batch, REST
fetched/raw/batch, and the automatic screening helper. **Regression-tested:**
bounded rank controls, actual adapter parity, frozen KY26 comparison, replay,
missing inputs and unavailable-probability rejection. **Not validated:**
scientific discovery significance, origin probability, independent performance,
full automatic review gates, network/legacy interfaces or restart durability.
Earlier capability descriptions must be read with these limits; the correction
does not certify their implementation. Detailed remaining scope and deprecated
contracts: `docs/engineering/screening-result-contract.md`.

## Required review-gate correction — 2026-11-12 (ADR-066)

The existing validation boundary owns required review prerequisites. A screening
rank is neither a prerequisite outcome nor a validation probability. A review
report identifies its evidence, ranking policy, validation configuration and
required-stage contract. The automatic orchestrator and expert queue enforce
that contract; a missing, failed, inconclusive, stale or incomplete report cannot
advance as validated. Capacity deferral is not scientific rejection.

The canonical installed validator can inventory actual inputs but cannot yet
validate observational physical hypotheses or calibrated discovery significance.
Its aggregate is inconclusive. Five stage records are prerequisites, not five
independent evidence streams. A gate-positive synthetic fixture is engineering
control evidence only. Legacy helper/direct-call science remains unverified and
requires #23. Normal JSON report history protection is implemented; transactional
jobs/restart recovery remain designed work under #15.

## CAD intake correction — 2026-11-12 (ADR-067)

A **computed encounter** is a catalogue prediction with object/body/time identity,
not a raw observation, fitted orbit, discovery date or origin indication. The
existing acquisition boundary owns CAD schema/units/native TDB/provenance. Orbit
acquisition remains a separate whole-solution responsibility. The existing
analysis service owns ranking; absence of orbit means abstention, not natural.

Intake counts distinguish records, overlap-deduplicated encounters and unique
designations. All records are retained across the historical→review boundary;
only their review eligibility is filtered. Source coverage and execution status
are independent of scientific validation status. New operational caches are
versioned/checksummed; they are not authenticated scientific archives. Physical
measurement source ingestion (#25), provider-wide scheduling (#16), restart-safe
execution (#15) and calibrated observation-based inference remain unfinished.

## Legacy scientific-contract correction — 2026-11-12 (ADR-068)

Validation owns prerequisite status, not an origin verdict inferred from a rank.
A field-presence fraction is descriptive completeness, not source reliability,
quality or independent-evidence count. Actual observation span and orbital
coordinates may be reported; unavailable inference cannot be filled with constants.

Enhanced results retain legacy input under `original_result` and expose canonical
claim-bearing fields. Standalone research diagnostics remain distinct from
installed validated model evidence. Monitoring consumes authoritative outcomes
rather than independently orchestrating science. Its pass fractions and transport
counts are operational metrics, not measured FDR or classifier performance.

Automatic origin alerts require a real origin model; a scalar threshold does not
supply one. Current capability is unavailable, explicitly not a confirmed natural
finding. Core legacy AnalysisPipeline/API/CLI migration and broader interface
claim review remain unfinished under #6/#12/#17. Durable assessment identity and
case lifecycle remain #15/#13; current collector deduplication is in-process only.

## Core analysis adapter correction — 2026-11-12 (ADR-069)

The existing analysis boundary now orchestrates typed acquisition and shared
screening, instead of independently interpreting legacy indicators. The core
result carries actual NEO evidence and canonical assessment together. Operational
status is separate from candidate status and from unavailable origin inference.
Field-presence completeness is not source reliability. No new bounded context.

**Implemented/integrated:** core single/batch, `/analyze`, existing batch request
flags, core CLI displays, JSON/CSV export, required-validator/fallback transport.
ML integration rejects incompatible rank/probability units and requires explicit
independent label inputs. **Regression evidence:** network-denied contract tests,
actual HTTP serialization, candidate positive/negative controls, source failure,
timeout, replay and nullable display. **Not scientifically validated:** physical
origin, probability/significance, reference-population calibration or independent
performance. The enhanced REST endpoint still supplies mock acquisition/pipeline
and is explicitly outside this increment; #6/#12 remain open. Old research model
modules do not establish a deployed validated capability.

## Enhanced REST correction — 2026-11-12 (ADR-070)

The existing analysis/reporting boundaries now connect the previously declared
enhanced routes to the current application pipeline. The core owns acquisition,
evidence and screening; the review wrapper owns explicit required-prerequisite
outcomes. An initialized component, disabled review, request preference or
available database is not scientific validation. No new bounded context.

**Implemented/integrated:** enhanced single/batch/summary registration and typed
HTTP transport, current-service identity, refresh/options, nullable uncertainty,
all ordered outcomes and descriptive persisted-screening summaries.
**Regression-tested:** actual main-app HTTP routing, source/missing/invalid cases,
candidate controls, synthetic review-transport positive control, concurrent
request options, timeout, native A3/covariance, read-only SQLite summary and
OpenAPI parity. **Not independently validated:** physical models, calibrated null
or origin probabilities, detection performance, provider-wide limits or durability.
Summary units are persisted assessment rows, not independent objects. Enhanced
reports currently log but are not durable case history; #15/#13 remain open.
This corrects the enhanced REST migration gap noted in ADR-069, not the remaining
legacy menu/pattern/standalone research-model claim audit (#6/#12).

## Primary legacy menu/launcher correction — 2026-11-12 (ADR-071)

The existing presentation boundary consumes shared analysis rather than creating
another scientific decision. Primary legacy single/batch/interactive, suite and
predefined workflow routes now carry canonical assessment and actual normalized
source data. Computed history is separate from observed acceleration/residuals.
Menu startup is not an authorization to clean research files or change settings.
No new bounded context.

**Implemented/integrated:** primary menu/launcher adapters, current-policy replay,
JSON/Markdown reports, native acceleration evidence, exclusive normal report
creation and scoped software/input/report fingerprints. **Regression-tested:**
Rich/plain parity, candidate positives, source failures, aliases without demo data,
ordered duplicates, history errors, native A3 sigma/covariance, malicious-designation
nonexecution, CLI flags/exits and tamper/binding checks. **Not established:** full
legacy/standalone/impact/report coverage, calibrated inference, independent
performance, complete environment reproduction or restart-safe case history.
Retained names for prototype inference raise explicit controlled deprecation
rather than silently produce numbers; core scientific libraries remain available.

## Standalone and advanced facade correction — 2026-11-12 (ADR-072)

Simple/advanced names are adapters to the existing analysis boundary, not authority
to manufacture observations or select an unvalidated fallback model. Their
structured reports carry actual normalized NEO data, canonical screening and
nullable scientific inference. Raw caller inputs are explicit, not silently mixed
with a catalogue fetch. Unsupported filtering/conversions fail explicitly.

An impact-prerequisite inventory belongs to the existing impact/risk boundary; it
is not an impact-probability engine. Catalogue covariance presence is not propagated
impact uncertainty. Dated arc length is not a quality score. Screening cannot
supply impact, strategic-threat or maneuver probabilities. Core impact research/
scenario modules remain available; public heuristic claims require #32 remediation.

**Implemented/integrated:** simple/advanced library aliases, async paths, standalone
CLI command delegation, native evidence/report transport and implementation-only
capability inventory. **Regression-tested:** synthetic candidate controls, actual
network-denied CLI processes, missing/failed sources/core, no name/hash fallback,
raw unit/epoch validation, input preservation, A3/covariance and explicit unavailable
impact/quality. **Not validated:** origin/impact models, provider health, independent
performance, complete environment reproduction, durable execution or case history.
No new bounded context is introduced. This supersedes the standalone migration gap
in ADR-071, not all remaining scientific/legacy-interface findings.

## Provider transport correction — 2026-11-12 (ADR-073)

Data Acquisition owns provider-family scheduling and response transport, not just
individual-source rate limiters. Its shared JPL adapter coordinates buffered GETs,
application identification, persistent backoff and safe failure. Source adapters
still own identity/schema/version checks, units, provenance and evidence parsing.
Scheduling metadata is neither observational data nor a provider-health verdict.
An acquired lease cannot upgrade a response into scientific evidence.

**Implemented/integrated:** inventoried SBDB/Horizons/CAD/PHA-query transport and
legacy tools, explicit health diagnostics, per-object probe removal and returned-
failure circuit accounting. **Regression-tested:** synthetic positive responses,
threads, real child processes, process kill, cancellation, Retry-After and unsafe
state. **Not established:** cross-host serialization, nonparticipating clients,
full API-version/cache/pagination compliance, live provider performance, durable
jobs/cases or scientific accuracy. No new bounded context is needed. The impact
boundary still requires #32's authoritative-source and scientific-contract work;
a transport adapter does not validate impact probability.

## Authoritative impact evidence — 2026-11-12 (ADR-074)

Data Acquisition's Sentry anti-corruption adapter owns source version, object
identity, native method/event parsing and response provenance. The existing Impact
Assessment boundary owns the interpretation: a scoped external Earth-impact model
output is not origin inference, an observed collision, a local covariance fit,
lunar risk or a damage prediction. A source catalogue absence/removal is not a
negative scientific measurement. Shared reports carry that distinction to REST.

**Implemented/integrated:** strict source service, per-process TTL/single-flight
reuse, existing REST route/model, JSON and replay-checked Markdown helper; one
current KY26 not-listed capture. **Regression-tested:** all supported methods,
positive events, absent/removed/failing/mismatched/tampered records, refresh/cohort
failure, actual application routing and frozen capture replay. **Not established:**
local impact probability/calibration, legacy menu/core enhanced-impact correctness,
full alias resolution, distributed cache, durable cases/jobs, or readiness for
planetary-defense decisions. No new bounded context is introduced; #32 stays open.

## Impact entry points and local evidence snapshots — 2026-11-12 (ADR-075)

Impact Assessment is not authority to infer risk from origin scores, orbit geometry
or data-quality duration proxies. Menu/enhanced entry points now use Data Acquisition's
scoped Sentry evidence contract and preserve independent screening state. Source,
validation and persistence statuses have separate meanings. Legacy numerical/scenario
research is not a fitted impact solver; absent domain values remain unavailable.

Reporting owns immutable local source-report snapshots and explicit save confirmation,
not probability values written into origin-score database fields. File atomicity and
local restart read-back are bounded engineering evidence, not durable job scheduling,
full case lifecycle, backups, distributed storage or independently validated science.
No new bounded context is required. Restored/get-by-ID evidence must replay against its
source input; a complete case history/workflow and API-wide persistence remain open.

## Durable execution storage foundation — 2026-11-12 (ADR-076)

Execution persistence belongs to existing API/database infrastructure, separate from
scientific analysis and source acquisition. A stored `completed` state means an
execution result was committed; it is not scientific validation, origin confirmation
or five-sigma significance. A handler must validate its own versioned, evidence-bound
result before commit. Missing results remain null; no scientific default is generated.

**Implemented/regression-tested:** same-Base SQL job/admission/checkpoint/event
storage, owner scope filters, idempotency, lease fencing/recovery, immutable snapshots,
atomic transitions, rollback and process restart controls; side-effect-free storage
imports and explicit full-API schema startup. **Not integrated:** current network/
automatic/polling workers, REST persistence, authenticated principal mapping, domain
report validation or case lifecycle. **Not validated:** PostgreSQL execution, cross-host
clock behavior, backups, hardware/power-loss recovery or scientific accuracy. No new
queue service or bounded context is introduced. Current API jobs remain #6/#15 work.

## Population evidence contract — 2026-11-13 (ADR-077)

BC11 orchestrates immutable input snapshots and BC5 canonical per-object screening;
it does not become an independent detection engine. Ordered screening requests and
unique grouping identities have different semantics. Conflicting duplicate snapshots
are invalid; a missing orbital angle is not zero. Exploratory neighbourhoods and
computed encounter gaps are descriptors, not independent observations, calibrated
significance, physical families or origin evidence. A scalar fitted A1/A2/A3 inventory
preserves native units, estimation kind, uncertainty and within-orbit covariance;
paired temporal correlation remains unavailable without the required observations.

Reports carry version/configuration/policy/input/result IDs and replayable complete
snapshots. Report verification and domain interpretation are separate from execution
storage: future handlers must verify report and request/checkpoint identity before
terminal commits. Current network REST/menu consume the report; legacy API job
scheduling still uses process memory with no principal ownership. Durable activation,
case histories, independent population calibration and observational model fits
remain separate open work. No new bounded context is introduced.

## Registered population execution — 2026-11-13 (ADR-078)

Execution infrastructure now composes acquisition, BC11 reporting and SQL storage
through a versioned `network-evidence-job-v1` handler. Requests bind ordered identities,
scientific configuration/reference policy, acquisition fingerprint and fixed calendar
query bounds. Immutable checkpoints bind each unique source snapshot/history and
final report to the request. Replaying a report against its own snapshots does not
alone prove execution provenance: terminal commit also compares acquisitions to the
actual stored checkpoints. A committed execution is not a calibrated scientific finding.

Transient provider outage leaves missing acquisition work for bounded retry; valid
committed evidence is reused, never refreshed. Missing physical properties may yield
canonical scientific inconclusive results, whereas source outage, invalid identity,
stop/deadline, storage uncertainty and internal errors have separate execution codes.
Heartbeat/fencing/recovery handle old workers without granting them write authority.
This library is not yet hosted by the API lifecycle or authenticated REST ownership;
no additional bounded context, queue product or detection engine is introduced.

## Persistent network API and ownership — 2026-11-13 (ADR-079)

Existing network submission/polling now crosses verified-principal -> scoped SQL
execution -> registered acquisition/report handler -> verified persisted presentation.
Execution status, evidence completeness and scientific validity remain distinct.
Stable service/SQL-user principals own jobs; role grants write capability, not access
to another principal's history. Scoped credentials are revalidated against active
configuration/database state. No client-supplied owner identity or development mock
authorizes this boundary.

API lifecycle owns worker admission, start/stop/join and observed readiness. Storage
and handler contracts remain reusable, not independent detection engines. Current
other API job families and legacy authentication are not upgraded by this integration.
Object-wide assertion/enrichment/conflict history, explanations, graphical/menu QA,
scientific suitability and documentation/version consistency are tracked #40–#46;
review existing boundaries before adding any material new context.

Credential-provisioning/admin authority is inside this verified boundary as well: a
legacy development administrator must not mint SQL users to bypass persistent job
ownership. Provisioning/listing use the same runtime database and verified admin
role. This does not certify all legacy user-management implementations; #47 tracks
remaining lifecycle/default-grant defects.

## Application identity versus domain versions — 2026-11-13

Dependency-free application identity (`aneos_version.py`) belongs to shared build/
presentation metadata. It is not a data-source schema, orbit solution, screening
policy, model, API contract or readiness certificate. Preserve those independent
identifiers and recorded historical versions. Menu navigation/labels share one
presentation catalogue; dispatch/renders do not certify feature prerequisites or
scientific results. No new bounded context is needed for this distinction.

## Replay-bound object review — 2026-11-13 / ADR-080

The existing analysis context owns the screening calculation trace and deterministic
object-evidence review. Presentation consumes the verified review; it does not infer
origin from rank, coefficient sign or missing evidence. The review freezes one typed
object snapshot and retains source alternatives without reconciling incompatible
solutions. It is not the normalized multi-source evidence ledger (#40), an append-only
research-case lifecycle, a physical fit, or independent calibration. Coverage profiles
have frozen named denominators and describe presence only. Full acquisition coverage
and operation-specific scientific readiness remain separate, currently unavailable
outputs. Existing source/domain and research boundaries remain unchanged.

## Atomic physical assertions — 2026-11-13 / ADR-081

Acquisition owns native catalogue assertions and documented normalization. The common
reconciler owns priority selection of value/uncertainty pairs and preservation of all
source snapshots; it does not decide physical origin or average incompatible solutions.
NEO metadata carries the versioned contract through cache/export/report reconstruction.
Analysis validates projection consistency before consuming scalar compatibility fields.
Presentation explains the selected/ambiguous/unavailable status. Physical fitting must
separately establish uncertainty convention, observation epoch, covariance/dependencies
and source comparability. A global reference cannot describe a mixed-source object;
per-selection references remain authoritative only as catalogue assertions. No new
bounded context, independent detector or database migration is required for this fix.
Full multi-source object-ledger/history/pollability (#40), alternative spin coordinate
models, observation likelihoods and calibrated uncertainty use remain unimplemented.

## Native orbital time evidence — 2026-11-13 / ADR-082

Acquisition preserves native astronomical representations/time-scale labels and binds
computed element rows to their headers. OrbitalElements owns an optional NativeEpoch
alongside, not silently replacing, the legacy datetime. Source reconciliation keeps
an orbit solution and epoch atomic. Analysis checks source/projection consistency;
it does not propagate covariance or convert scales. Presentation distinguishes orbital,
covariance, computed-encounter and observation dates. Numeric epoch equality is not
physical fit validity or frame compatibility. Existing acquisition, evidence and
presentation boundaries suffice; no new bounded context is created. Full observational
likelihoods, epoch/frame-aware propagation, source identity and operation-specific
scientific readiness remain separate requirements.

## HTTP evidence adapters — 2026-11-13 / ADR-083

The existing API boundary preserves domain evidence instead of narrowing or coercing
it into a different scientific contract. A native history DTO replays the epoch and
binds its computed header/source/role; presentation metadata describes returned samples,
not observations or calibrated significance. Acquisition owns bounded query outcomes
and source attempt metadata. HTTP maps those outcomes without inventing empty evidence
on outage. Canonical analysis remains the only screening engine, with matching finite
numeric/conic semantics. Request-error serialization owns redacted client diagnostics,
not authentication policy. No new bounded context is needed. All-source local persistence,
identity, enrichment history, independent scientific validation and complete menu/API
runtime prerequisites remain separate required workstreams.

## Local correlated evidence persistence — 2026-11-13 / ADR-084

Acquisition produces source-specific typed or explicitly legacy assertions. Persistence
owns ingestion time, idempotent receipt identity, append-only history and atomic replayable
projections in the existing database boundary. The shared acquisition reconciler selects
whole orbital solutions and physical value/uncertainty pairs, retaining conflicts rather
than treating them as independent evidence. Latest receipt per source is an explicit
projection policy, not an astronomical epoch comparison or alias-resolution service.

Analysis still owns scientific admissibility and canonical screening. Persistence does
not fit observations, calibrate probabilities, establish measurement independence or
supply a missing source-coverage denominator. Presentation reads verified projections and
exposes source history, missingness and limitations. Legacy compatibility columns are a
rebuildable view; unsupported origin/quality values are not current scientific outputs.
The legacy polling enrichment path now uses this contract. Ordinary DataFetcher query
persistence, all-source eligibility, identity ambiguity and complete research-case lifecycle
remain pending in #40/#13. No additional bounded context or detection engine is introduced.

## Policy inspection and configuration safety — 2026-11-13 / ADR-085

The canonical screening service owns its policy/reference description and fingerprint;
inspection reuses that identity rather than inventing UI thresholds or running another
engine. Presentation describes the default service, diagnostics and unsupported actions.
A read-only archive is not an active configuration, source authentication, a physical fit
or a validator receipt. Snapshot verification compares records without modifying runtime
policy or required review. Custom service/job policy IDs remain authoritative for their
own results; this menu does not audit every live process.

The existing configuration boundary owns explicit file/environment selection and
candidate validation before manager-instance publication. This preserves prior state on
rejected updates/reloads but does not persist settings, synchronize workers or validate
all consumer behavior. Direct accessors still expose mutable objects. Acquisition source
eligibility and polling, observational fitting/calibration and whole-menu readiness remain
separate requirements. Existing boundaries suffice; no new context or global policy store.

## Ordinary query acquisition — 2026-11-13 / ADR-086

Acquisition owns the frozen installed source plan and typed source outcomes;
normalization owns explicit adapter mappings/native identity checks; reconciliation
owns atomic evidence selection, not a physical model. The existing repository
owns one transaction for query, source assertions, projection and compatibility
view. Presentation owns counts, timestamps and stale-source bindings, not inferred
scientific certainty. Repeated bodies are not additional independent evidence.

An acquisition receipt is local committed provenance, not source authentication.
The exact requested identity remains the warehouse key; cross-query canonical
alias resolution is still required. A cached query retains its original receipt
and evidence dates; partial-cache backfill, all observation/history roles and a
full research-case lifecycle remain open. No new bounded context is necessary for
this increment. File-SQLite is validated; other backend transactions are not.

## Source-native NEODyS normalization — 2026-11-13 / ADR-087

The acquisition adapter owns safe file identifiers, optional bounded alias resolution
and explicit HTTP/source outcomes. The pure normalizer owns native OEF identity,
frame/state admission, coordinate conversion and data-record accounting. The existing
physical-evidence boundary owns a source-specific replayable MAG/H assertion, not
an imitation SBDB packet. G and matrix/model records remain native with explicit
limits; no independent measurement or transformed covariance is inferred.

The existing repository commits the resulting source assertion and query history;
presentation displays the source/uncertainty/alternatives. A normalization revision
changes new cache contexts, not historical records. These responsibilities fit
existing boundaries. A global canonical alias registry and scientific fit/covariance
models remain separate pending requirements, not newly implemented contexts.

## Date-role contracts — 2026-11-13 / ADR-088

Acquisition owns the SBDB discovery request/native assertion and adapter receipts.
The scientific evidence boundary owns literal discovery/observation labels and
native orbital/covariance scales, not invented UTC instants. The repository owns
local append receipts and sequence order; it does not infer discovery from intake.
A pure chronology read model references these inputs and keeps legacy date semantics
unverified. Presentation owns compact role/scale labels and safe literal output,
not DST selection, propagation or source authentication.

No second lifecycle database or bounded context is introduced. Compatibility columns
remain caches, not authoritative scientific dates. Verified-format ledger timestamps
are distinct from original source freshness. ADR-089 below implements receipt review
and explicit refresh (#66); automatic selective backfill remains pending (#40). Frozen source histories and review-v1 assessments remain
available; new rendered context is not a rewritten historical scientific result.

## Source-receipt review and explicit refresh — 2026-11-13 / ADR-089

Acquisition owns the canonical installed-source plan, typed outcome and original
adapter-read timestamp. Persistence owns ordered immutable queries/assertions and
replay verification. A pure receipt-age policy consumes that history; it does not
interpret orbital epochs, retag legacy dates or authenticate providers. Its current
30-day review/one-hour failure cooldown are operational policy, not scientific age
limits. A repository read adapter includes failed-query-only objects, keeps unknown
legacy values and reports bounded coverage. No read-path schema mutation is allowed.

The explicit refresh coordinator rechecks stored eligibility and uses the existing
DataFetcher/repository/transport path. Whole-plan refresh is distinct from selective
backfill; commit receipt replay precedes a recorded result. Presentation owns clear
recent/stale/unknown/deferred/partial labels and safe rendering. Existing menu entry
points delegate here instead of the legacy SBDB-only re-poll. No additional bounded
context or database is needed. All-source backfill, eligibility/health discovery and
a durable scheduling lifecycle remain pending; this is not scientific completeness.

## Polling dashboard and browser — 2026-11-13 / ADR-090

Acquisition owns source identity, units and typed snapshots; a renderer must not
invent those from a legacy score or object name. The canonical analysis service owns
the ranking decision and nullable scientific inference. The existing object-review
read model owns the replayed calculation trace, fixed field-presence profile, source
chronology and evidence-grounded explanations. A polling-screening record binds its
original finite JSON input to that review and records a local screening receipt.
Legacy or invalid input stays distinct from typed/replayed evidence.

Presentation owns dashboard counts, navigation, safe labels and the graph of record
outcomes. Counts are not origin probabilities; repeated designation labels are not
independent observations. Sorting by local screening receipt is not sorting discovery.
Field presence is not calibrated quality or scientific completeness. The existing
Sentry boundary alone supplies source-listed impact evidence and identity checks.

Versioned dashboard exports retain per-record frozen policy descriptors and verify
replay on import. JSON/Markdown are new derived artifacts, not edits to source files
or historical assessments. Source-file hashes are reproducibility metadata, not
provider authentication. Unsupported older probability/heuristic helpers have explicit
controlled deprecations. No additional bounded context or persistence engine is needed.
Root candidate/source polling remains #71/#73, and independent model fitting,
calibration, search-wide error control and all-feature QA remain separate obligations.

## Polling acquisition handoff — 2026-11-13 / ADR-091

The legacy workflow orchestrates collection but must not collapse source assertions
into a first-payload/source-name union. Acquisition hands all returned payloads to
the existing transactional enrichment boundary. That boundary owns repeat
idempotency, immutable source/session history, conflict retention and materialized
object projections. Presentation may count distinct exact labels but does not own
measurement fusion or scientific identity resolution. A dashboard is published
only after an acknowledged commit; it displays original records rather than
manufacturing a cross-source measurement. Discovery failures without an object key
and chunk-level per-source receipts still need the source-restoration work (#73).

Root-poller presentation follow-up (#81): the root module is a caller of the
existing screening views, not a domain model. A small adapter supplies the shared
browser callbacks, replay-bound source-listed impact filter and explicit export.
Request context belongs to the export receipt, not measurement provenance or a
claim of date/source coverage. Acquisition restoration remains #73; these read-only
views do not fetch or repair missing observations.

## Bounded candidate acquisition — 2026-11-13 / ADR-092

Acquisition has two related aggregates: an object query with a known designation,
and a candidate collection query whose object set is unknown before the response.
The latter owns selection, transport outcome and native entity bytes, including
empty/failed outcomes without invented object identities. Grouped object packets
flow through the existing assertion/projection transaction; the collection receipt
binds those storage references. Exact native row repeats and alternative predictive
solutions remain traceable, not independent observations or averaged measurements.

The CAD service performs one explicitly bounded source request and no analysis,
automatic paging or additional-source enrichment. The SQL adapter owns atomic
commit/readback, immutability guards and source-body archive storage. Presentation
and scientific assessment remain existing consumers, not acquisition side effects.
Raw catalogue physical values are retained pending their atomic physical adapter.
Root/menu restoration #73 must consume committed receipts and explicitly budget
further enrichment rather than multiplying provider calls or re-ingesting a source
query as a fresh observation. This foundation does not establish full source coverage,
alias resolution, scientific completeness or calibrated detection performance.

### Root CAD collection integration — 2026-11-13 / ADR-093 / #84

Data Acquisition owns bounded collection and explicit named-object enrichment;
the existing evidence ledger owns receipts and original assertions. The physical
anti-corruption layer admits only same-row native CAD assertions under an optional
versioned receipt policy. Core shared screening owns all ranking decisions; the root
controller and Rich/dashboard/export paths consume its result, not legacy origin
scores. Receipt replay is read-only. No new bounded context is introduced.
