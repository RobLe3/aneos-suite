# Artificial NEOs Theory — Scientific Hypothesis and Detection Framework

**Document status**: Research hypothesis — not an established scientific finding.
All claims are speculative without independent peer review and observational confirmation.

---

## Abstract

The **Artificial NEOs Theory** (ANT) proposes that a subset of Near Earth Objects (NEOs)
may exhibit orbital and physical properties statistically inconsistent with the known
population of naturally formed and evolved solar system bodies. The null hypothesis is
that every catalogued NEO follows purely gravitational dynamics shaped by solar system
formation and long-term perturbation. The alternative hypothesis is that some objects
have been placed into, or maintained in, their current orbits through non-natural means.

aNEOS is a statistical screening tool built to test this hypothesis. It does not confirm
artificial origin; it flags statistical outliers for follow-up investigation.

---

## 1. Motivation

The existence of confirmed artificial heliocentric objects in public catalogues establishes
that human-made objects do orbit the Sun indistinguishably from small natural bodies:

| Designation | Object | Year | Source |
|-------------|--------|------|--------|
| 2018 A1 (CNEOS) | SpaceX Tesla Roadster / Falcon Heavy upper stage | 2018 | SpaceX |
| 2020 SO | Centaur upper stage (Surveyor 2 mission) | 1966 launch / rediscovered 2020 | NASA/JPL |
| J002E3 | Apollo 12 S-IVB third stage | 1969 launch / recaptured 2002–2003 | NASA |
| WT1190F | Probable rocket body fragment | 2015 | IASC/ESA |

These objects were classified as asteroids by automated surveys before their artificial
origin was recognised through detailed orbital and spectral analysis. The detection
challenge they illustrate is the motivation for aNEOS.

**The core question**: If human-made objects from the early space age are detectable
as statistical outliers in orbital catalogues, can the same methodology be applied to
screen for objects of unknown origin exhibiting similar anomalies?

---

## 2. Observable Signatures

The theory identifies six categories of observable signatures, each independently
measurable from public orbital catalogue data:

### 2.1 Orbital Mechanics Anomalies

Natural NEO orbits are shaped by formation processes, Yarkovsky/YORP radiation effects,
and resonant perturbations from Jupiter, Mars, and the inner planets. Objects deviating
significantly from the statistical distribution of the known NEO population are flagged.

| Parameter | Natural range (95th percentile) | Anomaly flag |
|-----------|--------------------------------|-------------|
| Semi-major axis *a* | 0.5 – 4.2 AU | *a* < 0.7 AU with *e* > 0.6 simultaneously |
| Eccentricity *e* | 0 – 0.96 | *e* > 0.8 with stable, non-decaying trajectory |
| Inclination *i* | 0° – 60° | *i* > 45° with regular Earth-approach pattern |
| Orbital period *T* | 0.4 – 10 yr | Near-resonance with Earth's year at <3 AU |

High eccentricity combined with long-term orbital stability is the primary anomaly flag.
Natural high-eccentricity objects tend to have short dynamical lifetimes; objects
maintaining *e* > 0.8 over many close approaches without measurable orbital decay
are statistically unusual.

### 2.2 Non-Gravitational Acceleration (A2 Parameter)

Non-gravitational forces — deviations from purely gravitational trajectories — are
quantified via the Marsden–Sekanina A2 parameter (transverse acceleration, AU/day²).
In natural objects, A2 arises from:

- **Yarkovsky effect**: anisotropic thermal emission from rotating bodies (typical
  |A2| ~ 10⁻¹⁵ – 10⁻¹³ AU/day²)
- **Cometary outgassing**: sublimation of volatiles, including from "dark comets"
  (bodies with A2 signal but no visible coma)
- **Radiation pressure**: dominant only for sub-metre objects

A2 values significantly exceeding the Yarkovsky range, inconsistent with the object's
estimated size/albedo, are flagged. A2 is available for only ~2–5% of catalogued NEOs
(SBDB, 2026) and must be used cautiously — absence of A2 does not imply absence of
non-gravitational forces.

> **Dark comets**: A recently identified sub-population (Seligman et al. 2023) of NEOs
> with measurable A2 but no photometric cometary activity. 1998 KY26 (JAXA Hayabusa2
> target) is a candidate. Non-gravitational acceleration alone is not sufficient to
> distinguish propulsion from sublimation; dedicated spectroscopy and long-arc astrometry
> are required.

### 2.3 Close-Approach Regularity and Frequency

Natural objects return to Earth's vicinity according to orbital resonance patterns
determined by their period ratio with Earth. Regular close approaches — particularly
those maintaining consistent proximity (1–5 lunar distances) over multiple returns —
can indicate orbit maintenance. Key metrics:

- **Approach interval regularity**: σ of approach intervals; natural objects show
  Poisson-distributed gaps; artificially stabilised objects might show low variance
- **Proximity consistency**: standard deviation of miss distance across consecutive
  approaches
- **Velocity profile**: relative velocity at closest approach should be consistent
  with orbital mechanics; sudden inter-approach changes suggest course corrections

The 200-year historical CAD API poll (Option 7 in aNEOS) provides the multi-epoch
dataset needed for this analysis.

### 2.4 Physical Property Anomalies

Natural NEOs follow population statistics for size-frequency distribution, albedo,
and spectral type. Key anomaly indicators (when observational data is available):

| Property | Natural baseline | Anomaly indicator |
|----------|-----------------|------------------|
| Bulk density | 1,000–5,000 kg/m³ (rocky/metallic) | < 500 kg/m³ (hollow structure) |
| Geometric albedo *p* | 0.03–0.50 | *p* > 0.5 combined with non-rocky spectrum |
| Rotation period | 2 min – 100 days | Ultra-fast < 2 min (requires structural strength) |
| Spectral type | C, S, X dominant | Featureless flat spectrum (artificial surface) |

> **Important caveat**: Physical data coverage in SBDB is sparse (~30% diameter,
> ~25% albedo, ~15% spectral type as of 2026). aNEOS scores physical indicators only
> when real measurements are available; proxy scores derived from orbital elements
> are explicitly labelled as such and carry reduced confidence (ADR-059).

### 2.5 Temporal and Geographic Distribution

Population-level analysis (BC11 in aNEOS) examines whether the close-approach
subground track distribution — the geographic location on Earth's surface directly
beneath the object's closest approach — shows non-random clustering over time:

- **Orbital clustering**: DBSCAN/HDBSCAN analysis of orbital element space (PA-1)
- **Synodic harmonic analysis**: Lomb-Scargle on binary encounter time-series to
  detect period-locking (PA-3)
- **A2 correlation within clusters**: objects in the same orbital cluster showing
  correlated A2 values are statistically more suspicious than isolated A2 detections (PA-5)

### 2.6 Human Origin Veto and Catalogue Cross-Reference

Before anomaly scoring, every object is cross-referenced against:
- Known spacecraft catalogue (SpaceX, NASA, ESA launch records)
- TLE/orbital debris catalogues (THETA SWARM — `human_hardware_analysis.py`)
- Historical rocket body database

Objects matching a known human artifact receive a **spacecraft veto** — they are
classified as confirmed artificial without statistical analysis.

---

## 3. Scoring Framework

### 3.1 Total Anomaly Score (TAS) — Original Formulation

The original theory defined TAS as a weighted sum of binary indicator flags:

```
TAS = (w_e · S_e) + (w_i · S_i) + (w_v · S_v) + (w_r · S_r) + (w_g · Purpose Score)
```

Where:
- *S_e* = 1 if eccentricity > 0.8
- *S_i* = 1 if inclination > 45°
- *S_v* = 1 if velocity deviates > 5 km/s between passes
- *S_r* = 1 if close-approach interval is regular (monthly / quarterly / annual)
- *w_g* = weight for geographic monitoring pattern

### 3.2 ATLAS Score — aNEOS Implementation

aNEOS evolves TAS into a continuous scoring system (ATLAS — XVIII SWARM):

```
ATLAS = Σ (clue_contribution_i)
      = Σ (normalized_score_i × category_weight_i × evidence_confidence_i)
```

Six evidence categories with configurable weights (`advanced_scoring_weights.json`):

| Category | Default Weight | Primary Clues |
|----------|---------------|---------------|
| Encounter Geometry | 0.15 | close_approach_distance, relative_velocity |
| Orbit Behaviour | 0.25 | repeat_approaches, non_gravitational_acceleration (A2) |
| Physical Traits | 0.20 | area_mass_ratio, radar_polarization, thermal_emission |
| Spectral Identity | 0.20 | spectral_anomaly, albedo_anomaly |
| Dynamical Sanity | 0.15 | delta_bic_analysis, orbital_stability |
| Human Origin | 0.05 | spacecraft_catalog_match |

### 3.3 Sigma-5 Detection Threshold

The Bayesian detection framework combines ATLAS-derived indicators with multi-modal
evidence through Fisher's method:

```
χ²_combined = -2 · Σ ln(p_i)   ~ χ²(2k) under H₀ (natural)
```

Classification thresholds:

| Tier | Sigma | Interpretation |
|------|-------|----------------|
| INCONCLUSIVE | σ < 2 | Insufficient evidence |
| EDGE CASE | σ ≥ 2 | Marginal anomaly |
| SUSPICIOUS | σ ≥ 3 | Statistically notable |
| ANOMALOUS | σ ≥ 4 | Strong anomaly |
| ARTIFICIAL VALIDATED | σ ≥ 5 | Discovery-grade significance |

The Bayesian posterior probability is calculated from a 0.1% base rate prior:

```
P(artificial | evidence) = P(evidence | artificial) × 0.001
                           ─────────────────────────────────────────────
                           P(evidence | artificial) × 0.001 + P(evidence | natural) × 0.999
```

This prior is conservative and structurally limits the posterior to ~3–5% even at
σ = 5, which is mathematically correct: orbital data alone cannot confirm artificial
origin. A value near the ceiling indicates "unusual orbit inconsistent with natural
population" — not "confirmed artificial object".

---

## 4. Hypothesised Monitoring Objectives

The original theory framework considers the possibility that the statistical anomalies,
if confirmed as artificial, could be consistent with long-duration Earth observation.
This section documents those hypotheses **as hypotheses** — they are not conclusions
drawn from the statistical analysis.

### 4.1 Surveillance Pattern Categories (Theoretical)

Objects with anomalous orbital signatures and regular return intervals that
systematically cover specific geographic regions could, in theory, serve monitoring
functions including:

- **Continuous Earth observation**: Stable orbits at 1–5 lunar distances with
  sub-monthly return intervals, enabling sustained coverage
- **Longitudinal environmental monitoring**: Highly eccentric orbits (e > 0.8)
  that maintain stable trajectories over decades are geometrically consistent with
  repeated passes over specific latitude bands
- **Multi-epoch sampling**: Objects with annual or biannual return intervals could
  provide long-cadence time-series data on Earth surface changes

> **Critical constraint**: Resolving Earth surface features from lunar distance
> (384,400 km) to centimetre-scale would require optical apertures approximately
> 1,000× larger than current state-of-the-art reconnaissance systems, quantum
> communication infrastructure for real-time data relay, and autonomous orbital
> maintenance capabilities far beyond demonstrated human technology. These
> requirements set the lower technological bound for the hypothesised civilization.

### 4.2 Kardashev Scale Contextualisation

aNEOS includes a Kardashev Synthetic Corpus Generator (`aneos_core/datasets/kardashev_generator.py`)
that creates labelled training data across 14 technological scenarios from K0.5
(decommissioned rocket stages) to K2.0 (hypothetical megastructure fragments).

The theoretical analysis from the source document extrapolates that an entity capable
of deploying and maintaining NEO-based Earth observation platforms would require
capabilities consistent with a **Kardashev Type 1.5–2.45** civilization:

| Capability Required | Kardashev Implication |
|--------------------|----------------------|
| Orbital maintenance at 1–5 LD | Sub-stellar propulsion mastery (K1.5+) |
| Continuous multi-century operations | Long-duration autonomous systems (K1.5+) |
| Real-time Earth-surface resolution | 1,000× current optical technology (K2.0) |
| Quantum data relay | Advanced communication infrastructure |
| Self-repairing space platforms | Advanced materials / nanotechnology |

> **Epistemic note**: The Kardashev level inference is highly speculative. It assumes
> the monitoring hypothesis is correct — which has not been demonstrated — and then
> extrapolates technological requirements. It is presented here as a theoretical
> bounding exercise, not a finding.

---

## 5. Honest Limitations and Validation Status

### What the current evidence shows

- On the available ground truth corpus (3 confirmed artificials, 20+ natural NEOs),
  the σ-5 detector achieves sensitivity = 1.00 and specificity = 1.00 at the
  calibrated threshold (0.037 Bayesian posterior).
- This corpus is **N=4 internal consistency check** — thresholds were hand-tuned to
  the same objects used for evaluation. F1=1.00 is a proof-of-concept, not a
  production accuracy guarantee.
- The Bayesian prior (0.1%), likelihood ratios (10/5/2), and σ formula bonus terms
  are asserted constants, not empirically calibrated against a large labelled dataset.
- No confirmed non-human artificial NEO has been identified. All high-σ objects
  tested to date are either known human artifacts (spacecraft, rocket stages) or
  natural objects with specific dynamical properties.

### What additional evidence is needed to advance the hypothesis

| Required evidence | Current status |
|------------------|---------------|
| ≥ 50 confirmed artificial heliocentric objects for calibration | 4 known (all human-origin) |
| Multi-epoch radar observations for close-approach regularity | Available for ~200 NEOs (Goldstone/Arecibo archive) |
| Non-gravitational A2 measurements | ~2–5% of catalogued NEOs |
| Spectroscopic observations showing featureless flat spectra | Requires targeted telescope campaigns |
| Statistically significant population clustering (σ > 3) | Not yet achieved on full NEO catalogue |

### False positive risk

The highest risk of false positives arises from:
1. **Dark comets**: Natural A2 signal without cometary activity
2. **Binary or tumbling objects**: Non-gravitational force mimics from shape effects
3. **Short-arc orbits**: Poorly constrained orbits producing apparent anomalies
4. **Selection bias**: Survey completeness varies strongly with object size and orbit type

---

## 6. Relationship to aNEOS Codebase

| Theory concept | aNEOS implementation | Status |
|---------------|---------------------|--------|
| TAS scoring | `advanced_scoring.py` — ATLAS 6-clue continuous scorer | Implemented |
| Eccentricity / inclination flags | Orbit behaviour category (ATLAS) | Implemented |
| A2 non-gravitational flag (Δ) | `delta_bic_analysis.py`; A2 from SBDB | Implemented; sparse data |
| Close-approach regularity | `approach_regularity` clue (confidence 0.35 proxy) | Implemented |
| Orbital clustering (PA-1) | `pattern_analysis/clustering.py` | Implemented |
| Synodic harmonics (PA-3) | `pattern_analysis/harmonics.py` | Implemented |
| A2 correlation in clusters (PA-5) | `pattern_analysis/correlation.py` | Implemented |
| Rendezvous scan (PA-6 Stage 1) | `pattern_analysis/rendezvous.py` | Implemented |
| Physical indicators | `indicators/physical.py` | Deferred (ADR-053) |
| Kardashev classifier | `datasets/kardashev_generator.py` | Training corpus only |
| Spacecraft veto | `detection_manager.py` — catalog cross-reference | Implemented |
| Bayesian fusion | `validated_sigma5_artificial_neo_detector.py` | Implemented |

---

## 7. Peer Review Status and References

The Artificial NEOs Theory is currently an independent research hypothesis. It has
not been submitted to or accepted by a peer-reviewed journal. The aNEOS detection
framework implements the statistical methodology; validation against an independent
corpus is the prerequisite for any publication claim.

**Relevant scientific literature:**

- Chesley, S. R. et al. (2023). *Non-gravitational perturbations of near-Earth asteroids.*
  NASA/JPL Technical Memoranda.
- Seligman, D. Z., Flekkøy, E. G., & Meech, K. J. (2023). *Dark comets and the
  population of non-gravitationally accelerating NEOs.* The Astrophysical Journal Letters.
- Farnocchia, D. et al. (2023). *Non-gravitational acceleration in the trajectory of
  1998 KY26.* JPL Technical Memorandum.
- Wiegert, P. et al. (2021). *The 2020 SO encounter: Confirming the centaur upper-stage
  hypothesis.* The Astronomical Journal.
- Marsden, B. G., Sekanina, Z., & Yeomans, D. K. (1973). *Comets and nongravitational
  forces.* The Astronomical Journal, 78, 211.
- Kardashev, N. S. (1964). *Transmission of information by extraterrestrial civilizations.*
  Soviet Astronomy, 8(2), 217–221.

**aNEOS documentation cross-references:**

- `docs/scientific/scientific-documentation.md` — Full statistical methodology
- `docs/scientific/VALIDATION_INTEGRITY.md` — Honest uncertainty audit
- `docs/architecture/ADR.md` — Architecture decisions (ADR-053 physical indicators;
  ADR-058 ATLAS scoring; ADR-059 proxy discipline)
- `docs/architecture/DDD.md` — Bounded Contexts (BC3 anomaly scoring; BC11 population
  pattern analysis)

---

*This document derives from the original `Artificial_NEOS_Theory.docx` source document.
Professional abstraction and scientific framing by the aNEOS development team, 2026-03-10.*
