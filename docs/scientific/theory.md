# Artificial NEOs Theory — Scientific Hypothesis and Detection Framework

**Document status**: Research hypothesis — not an established scientific finding.
All claims are speculative without independent peer review and observational confirmation.

---

## Abstract

The **Artificial NEOs Theory** (ANT) proposes that a subset of Near Earth Objects (NEOs)
may exhibit orbital and physical properties statistically inconsistent with the known
population of naturally formed and evolved solar system bodies. The null hypothesis H₀ is
that every catalogued NEO follows purely gravitational dynamics shaped by solar system
formation and long-term perturbation. The alternative hypothesis H₁ is that some objects
have been placed into, or maintained in, their current orbits through non-natural means.

aNEOS is a statistical screening tool built to test this hypothesis. It does not confirm
artificial origin; it flags statistical outliers for follow-up investigation. The methodology
is grounded in established techniques from planetary science (orbital dynamics, Bayesian
inference, spectral analysis) and is designed to be falsifiable: any flagged object can
be independently re-evaluated with additional observational data.

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
origin was recognised through detailed orbital and spectral analysis. 2020 SO was
identified specifically because its heliocentric orbit was nearly identical to Earth's
(a ≈ 1.0 AU, e ≈ 0.01, i ≈ 0.5°) and its area-to-mass ratio was far too high for a
rocky body of its apparent size — an Atira-class asteroid does not behave like an empty
Centaur rocket casing.

**The core question**: If human-made objects from the early space age are detectable
as statistical outliers in orbital catalogues, can the same methodology be applied to
screen for objects of unknown origin exhibiting similar anomalies?

---

## 2. Observable Signatures

The theory identifies six categories of observable signatures, each independently
measurable from public orbital catalogue data:

### 2.1 Orbital Mechanics Anomalies

Natural NEO orbits evolve under gravitational perturbations from Jupiter, Mars, and the
inner planets, and under non-gravitational forces dominated by the **Yarkovsky effect**
(radiation-recoil drift; typical da/dt ~ 10⁻⁴ – 10⁻² AU/Myr for kilometre-scale bodies)
and **YORP effect** (spin-axis torque, timescale ~10⁷ years for km bodies). Both effects
are size- and composition-dependent: a 10-metre object drifts ~10× faster than a
100-metre object at the same albedo and spin rate (Bottke et al. 2006). The combined
result is that the NEO population is not static — objects diffuse into and out of
Earth-crossing orbits over ~1–100 Myr dynamical lifetimes (Gladman et al. 2000).

Objects deviating significantly from the statistical distribution of the known NEO
population are flagged. Key anomaly thresholds derived from the observed population:

| Parameter | Natural range (95th percentile) | Anomaly threshold | Physical basis |
|-----------|--------------------------------|------------------|----------------|
| Semi-major axis *a* | 0.5 – 4.2 AU | *a* < 0.7 AU with *e* > 0.6 simultaneously | Aten/Atira zone; few natural bodies maintain |
| Eccentricity *e* | 0 – 0.96 | *e* > 0.8 with stable, non-decaying trajectory | High-e objects have short dynamical lifetimes |
| Inclination *i* | 0° – 60° | *i* > 45° with regular Earth-approach pattern | High-*i* not typical for resonant delivery routes |
| Orbital period *T* | 0.4 – 10 yr | Near-resonance with Earth's year at <3 AU | Resonance is natural but increases encounter rate |
| Yarkovsky drift rate | 10⁻⁴ – 10⁻² AU/Myr | Drift rate inconsistent with estimated size/albedo | Anomalous area-to-mass ratio |

High eccentricity combined with long-term orbital stability is the primary anomaly flag.
Natural high-eccentricity objects tend to have short dynamical lifetimes (Gladman et al.
2000; Michel et al. 2005); objects maintaining *e* > 0.8 over many close approaches
without measurable orbital decay are statistically unusual. N-body simulations of the
known NEO population confirm that orbits in this regime have median lifetimes of
< 10⁷ years before either collision or ejection.

> **Simulation validation status**: The anomaly thresholds above are derived from the
> observed MPC/SBDB catalogue distribution and published dynamical lifetime estimates.
> They have not been independently re-validated via a dedicated N-body simulation of the
> aNEOS scoring criteria. This is a priority for future work (see §8).

### 2.2 Non-Gravitational Acceleration (A2 Parameter)

Non-gravitational forces — deviations from purely gravitational trajectories — are
quantified via the **Marsden–Sekanina A2 parameter** (transverse acceleration, AU/day²;
Marsden, Sekanina & Yeomans 1973). In natural objects, A2 arises from:

- **Yarkovsky effect**: anisotropic thermal emission from rotating bodies (typical
  |A2| ~ 10⁻¹⁵ – 10⁻¹³ AU/day²; Chesley et al. 2003)
- **Cometary outgassing**: sublimation of volatiles, including from "dark comets"
  (bodies with A2 signal but no visible coma; Seligman et al. 2023)
- **Radiation pressure**: dominant only for sub-metre objects (area-to-mass > 0.01 m²/kg)

The Yarkovsky-expected A2 for a body of diameter *D* (metres) and bulk density *ρ*
(kg/m³) scales approximately as:

```
|A2_Yarkovsky| ~ C · (1 - A) / (ρ · D)
```

where *A* is the Bond albedo and *C* is a thermal coefficient (~2–5 × 10⁻⁴ for typical
spin/obliquity; Farnocchia et al. 2013). This scaling provides a size-consistent baseline:
an A2 measurement inconsistent with this estimate for the object's known or inferred
diameter is a genuine anomaly flag.

A2 values significantly exceeding the Yarkovsky range, inconsistent with the object's
estimated size/albedo, are flagged. A2 is available for only ~2–5% of catalogued NEOs
(SBDB, 2026) and must be used cautiously — absence of A2 does not imply absence of
non-gravitational forces; it typically reflects insufficient astrometric arc length.

> **Dark comets**: A recently identified sub-population (Seligman et al. 2023) of NEOs
> with measurable A2 but no photometric cometary activity. 1998 KY26 (JAXA Hayabusa2
> target) is a candidate. Non-gravitational acceleration alone is not sufficient to
> distinguish propulsion from sublimation; dedicated spectroscopy and long-arc astrometry
> are required. aNEOS displays a dark-comet contextual warning whenever the Δ flag is
> raised (see ADR-059).

### 2.3 Close-Approach Regularity and Frequency

Natural objects return to Earth's vicinity according to orbital resonance patterns
determined by their period ratio with Earth. Regular close approaches — particularly
those maintaining consistent proximity (1–5 lunar distances) over multiple returns —
can indicate orbit maintenance. Key metrics:

- **Approach interval regularity**: σ/μ (coefficient of variation) of approach intervals;
  natural objects show quasi-Poisson-distributed gaps driven by resonance geometry;
  artificially stabilised objects might show anomalously low variance
- **Proximity consistency**: standard deviation of miss distance across consecutive
  approaches normalised to mean miss distance
- **Velocity profile**: relative velocity at closest approach should be consistent
  with orbital mechanics; sudden inter-approach changes suggest course corrections

The 200-year historical CAD API poll (Option 7 in aNEOS) provides the multi-epoch
dataset needed for this analysis. Note that natural Earth co-orbitals (e.g. Aten
asteroids in horseshoe orbits) also produce regular close approach series —
regularity alone is insufficient; it must be combined with additional indicators.

### 2.4 Physical Property Anomalies

Natural NEOs follow population statistics for size-frequency distribution, albedo,
and spectral type. Key anomaly indicators (when observational data is available):

| Property | Natural baseline | Anomaly indicator | Caveats |
|----------|-----------------|------------------|---------|
| Bulk density | 1,000–5,000 kg/m³ (rocky/metallic) | < 500 kg/m³ (hollow structure) | Few density measurements exist |
| Geometric albedo *p* | 0.03–0.50 | *p* > 0.5 combined with non-rocky spectrum | 2020 SO had *p* ≈ 0.06 — low albedo |
| Rotation period | 2 min – 100 days | Ultra-fast < 2 min (requires structural strength) | Rubble piles cannot spin faster than ~2.2 hr |
| Area-to-mass ratio | < 10⁻⁴ m²/kg (> 10m bodies) | > 10⁻³ m²/kg (hollow/low-density) | 2020 SO: ratio ~0.02 m²/kg |
| Spectral type | C, S, X dominant | Featureless flat spectrum (artificial surface) | Atypical for natural D-type analogs |

> **Important caveat**: Physical data coverage in SBDB is sparse (~30% diameter,
> ~25% albedo, ~15% spectral type as of 2026). aNEOS scores physical indicators only
> when real measurements are available; proxy scores derived from orbital elements
> are explicitly labelled as such and carry reduced confidence (ADR-059). The physical
> scoring module is handled by the **LAMBDA SWARM** (`thermal_ir_analysis.py`) for
> thermal emission and **KAPPA SWARM** (`radar_polarization_analysis.py`) for SC/OC
> radar ratio — both are invoked only when dedicated observational data is available.

### 2.5 Temporal and Geographic Distribution

Population-level analysis (BC11 in aNEOS) examines whether close-approach patterns
show non-random structure in orbital element space or time:

- **Orbital clustering**: DBSCAN/HDBSCAN analysis of (a, e, i, Ω, ω) orbital element
  space — statistically significant clusters beyond background density are anomalous (PA-1)
- **Synodic harmonic analysis**: Lomb-Scargle periodogram on binary encounter time-series
  to detect period-locking at harmonics of Earth's synodic year (PA-3)
- **A2 correlation within clusters**: objects in the same orbital cluster showing
  correlated A2 values are statistically more suspicious than isolated A2 detections (PA-5)

Cluster significance is assessed against a Poisson background model, corrected for
multiple comparisons via Bonferroni correction over the number of clusters evaluated
(Phase 12; `network_sigma.py`).

### 2.6 Human Origin Veto and Catalogue Cross-Reference

Before anomaly scoring, every object is cross-referenced against:
- Known spacecraft catalogue (SpaceX, NASA, ESA launch records)
- TLE/orbital debris catalogues (**THETA SWARM** — `human_hardware_analysis.py`)
- Historical rocket body database (COSPAR, DISCOS)

Objects matching a known human artifact receive a **spacecraft veto** — they are
classified as confirmed artificial (human-origin) without statistical analysis.
The veto is the highest-priority check and executes before any ATLAS scoring.

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

This binary formulation is computationally simple but loses gradation — a score of 0
is indistinguishable between an object with four weak signals and one with no signal at all.

### 3.2 ATLAS Score — aNEOS Implementation

aNEOS evolves TAS into a continuous scoring system (ATLAS — XVIII SWARM,
`aneos_core/analysis/advanced_scoring.py`):

```
ATLAS = Σ_i ( normalized_score_i × category_weight_i × evidence_confidence_i )
```

Where `evidence_confidence_i` penalises proxy-derived scores (confidence = 0.35) relative
to direct observational measurements (confidence = 0.7–1.0), preventing orbital geometry
alone from driving a high ATLAS score (ADR-059).

Six evidence categories with configurable weights (`advanced_scoring_weights.json`):

| Category | Default Weight | Primary Clues | Data Sources |
|----------|---------------|---------------|-------------|
| Encounter Geometry | 0.15 | close_approach_distance, relative_velocity | CAD API |
| Orbit Behaviour | 0.25 | repeat_approaches, non_gravitational_acceleration (A2) | CAD API, SBDB |
| Physical Traits | 0.20 | area_mass_ratio, radar_polarization, thermal_emission | SBDB, radar archives |
| Spectral Identity | 0.20 | spectral_anomaly, albedo_anomaly | SBDB, MITHNEOS, Sloan |
| Dynamical Sanity | 0.15 | delta_bic_analysis, orbital_stability | Horizons, computed |
| Human Origin | 0.05 | spacecraft_catalog_match | THETA SWARM (TLE/COSPAR) |

Clue flags raised in the output:
- **Δ** (non-gravitational): A2 acceleration inconsistent with Yarkovsky baseline; threshold
  acceleration_score > 0.65 (raised from 0.40 in Phase 20 — ADR-059)
- **r** (approach regularity): coefficient of variation of approach intervals below threshold
- **v** (velocity): inter-pass relative velocity change inconsistent with orbital mechanics

### 3.3 Sigma-5 Detection Threshold

The σ-5 Bayesian detection framework (`validated_sigma5_artificial_neo_detector.py`)
combines ATLAS-derived indicators through **Fisher's method** for combining independent
p-values:

```
χ²_combined = -2 · Σ_i ln(p_i)   ~  χ²(2k)  under H₀ (all natural)
```

Where *p_i* are the p-values from *k* independent evidence streams. The resulting
χ²-statistic is converted to a sigma equivalent via the normal approximation.

**Known limitations of Fisher's method in this application**:
1. The independence assumption is violated when multiple indicators are derived from the
   same orbital data — ATLAS clues within the same category are not independent.
2. Fisher's method is uniformly most powerful against alternatives where all H₀ are false,
   but less powerful when only a subset of indicators are truly anomalous.
3. **Stouffer's method** (z-score combination weighted by evidence strength) is an
   alternative that handles heterogeneous evidence quality more naturally and would be
   better suited to the multi-confidence ATLAS architecture. This is a planned improvement
   (see §8 Future Work).

**Bayesian prior justification**: The 0.1% (0.001) prior for P(artificial) is derived
from the empirical proportion of confirmed artificial heliocentric objects in the current
SBDB catalogue: approximately 4 confirmed artificials among ~36,000 catalogued NEOs as
of 2026 (~0.01%). The prior is set at 10× this rate (0.1%) to account for undiscovered
or unrecognised artificial objects and survey incompleteness. This is a deliberate
conservative choice — a lower prior would make confirmation even harder; a higher prior
would produce unacceptably high false-positive rates. The prior has not been calibrated
against a large independent labelled dataset and should be treated as an informed
starting point rather than an empirical measurement.

Classification thresholds:

| Tier | Sigma | χ²(2k) equivalent | Interpretation |
|------|-------|-------------------|----------------|
| INCONCLUSIVE | σ < 2 | p > 0.046 | Insufficient evidence |
| EDGE CASE | σ ≥ 2 | p ≤ 0.046 | Marginal anomaly — warrants monitoring |
| SUSPICIOUS | σ ≥ 3 | p ≤ 0.0027 | Statistically notable — follow-up recommended |
| ANOMALOUS | σ ≥ 4 | p ≤ 6.3 × 10⁻⁵ | Strong anomaly — telescope campaign warranted |
| ARTIFICIAL VALIDATED | σ ≥ 5 | p ≤ 5.7 × 10⁻⁷ | Discovery-grade significance |

The Bayesian posterior probability is calculated from the 0.1% prior and clue-specific
likelihood ratios (LR_strong = 10, LR_moderate = 5, LR_weak = 2):

```
P(artificial | evidence) =       P(evidence | artificial) × 0.001
                           ──────────────────────────────────────────────────────────
                           P(evidence | artificial) × 0.001 + P(evidence | natural) × 0.999
```

This prior structurally limits the posterior to ~3–5% even at σ = 5. This is not a
deficiency — it is mathematically correct: orbital data alone cannot confirm artificial
origin. A posterior near the ceiling indicates "orbit statistically inconsistent with
the natural population" — not "confirmed artificial object". The likelihood ratios
(10/5/2) are asserted constants, not empirically calibrated, and represent a second
source of model uncertainty beyond the prior (see §5).

---

## 4. Honest Limitations and Validation Status

### What the current evidence shows

- On the available ground truth corpus (3 confirmed artificials: 2020 SO, J002E3,
  WT1190F; plus 20+ natural NEOs), the σ-5 detector achieves sensitivity = 1.00 and
  specificity = 1.00 at the calibrated threshold (0.037 Bayesian posterior).
- This corpus is a **N=4 internal consistency check** — thresholds were hand-tuned to
  the same objects used for evaluation. F1 = 1.00 is a proof-of-concept demonstration,
  not a production accuracy estimate. Wilson 95% CI on a corpus of N=4: [0.40, 1.00].
- The Bayesian prior (0.1%), likelihood ratios (10/5/2), and σ formula bonus terms
  are asserted constants, not empirically calibrated against a large labelled dataset.
- No confirmed non-human artificial NEO has ever been identified. All high-σ objects
  tested to date are either known human artifacts (spacecraft, rocket stages) or
  natural objects with specific dynamical properties (high-eccentricity Atens, dark comets).

### What additional evidence is needed to advance the hypothesis

| Required evidence | Current status |
|------------------|---------------|
| ≥ 50 confirmed artificial heliocentric objects for calibration | 4 known (all human-origin) |
| Multi-epoch radar observations for close-approach regularity | Available for ~200 NEOs (Goldstone/Arecibo archive) |
| Non-gravitational A2 measurements | ~2–5% of catalogued NEOs (SBDB, 2026) |
| Spectroscopic observations showing featureless flat spectra | Requires targeted telescope campaigns |
| Statistically significant population clustering (σ > 3) | Not yet achieved on full NEO catalogue |
| Independent validation corpus (not used in threshold tuning) | Pending; requires new confirmed artificials |

### False positive risk

The highest risk of false positives arises from:
1. **Dark comets**: Natural A2 signal without cometary activity (Seligman et al. 2023)
2. **Binary or tumbling objects**: Non-gravitational force mimics from shape effects and
   YORP-driven tumbling
3. **Short-arc orbits**: Poorly constrained orbits producing apparent anomalies; SBDB
   provides orbit-fit uncertainty but arc-length filtering is not yet automated in aNEOS
4. **Selection bias**: Survey completeness varies strongly with object size and orbit type;
   the MPC catalogue is not a volume-complete sample of the NEO population (Granvik et al. 2018)
5. **Natural Earth co-orbitals**: Aten and Atira asteroids in horseshoe or quasi-satellite
   orbits can show regular Earth-approach patterns indistinguishable from orbit maintenance

---

## 5. Relationship to aNEOS Codebase

| Theory concept | aNEOS module | SWARM/ADR | Status |
|---------------|-------------|-----------|--------|
| TAS / ATLAS scoring | `analysis/advanced_scoring.py` | XVIII SWARM; ADR-058, ADR-059 | Implemented |
| Eccentricity / inclination flags | Orbit behaviour category (ATLAS) | ADR-058 | Implemented |
| A2 non-gravitational flag (Δ) | `delta_bic_analysis.py`; A2 from SBDB | ADR-059 | Implemented; sparse data |
| Yarkovsky baseline scaling | `analysis/advanced_scoring.py` | ADR-059 | Proxy only; direct measurement deferred |
| Close-approach regularity | `approach_regularity` clue (confidence 0.35 proxy) | ADR-059 | Implemented |
| Orbital clustering (PA-1) | `pattern_analysis/clustering.py` | BC11 | Implemented |
| Synodic harmonics (PA-3) | `pattern_analysis/harmonics.py` | BC11 | Implemented |
| A2 correlation in clusters (PA-5) | `pattern_analysis/correlation.py` | BC11 | Implemented |
| Rendezvous scan (PA-6 Stage 1) | `pattern_analysis/rendezvous.py` | ADR-045, ADR-052 | Stage 1 live (Option 15) |
| Physical indicators — thermal | `validation/thermal_ir_analysis.py` | LAMBDA SWARM; ADR-053 | Invoked when NEATM data available |
| Physical indicators — radar SC/OC | `validation/radar_polarization_analysis.py` | KAPPA SWARM; ADR-053 | Invoked when radar data available |
| Gaia astrometric calibration | `validation/gaia_astrometric_calibration.py` | MU SWARM | Calibration module |
| Spacecraft / TLE veto | `validation/human_hardware_analysis.py` | THETA SWARM | Implemented |
| Bayesian fusion | `validated_sigma5_artificial_neo_detector.py` | BC3 | Implemented |
| Kardashev classifier | `datasets/kardashev_generator.py` | See Appendix A | Training corpus only |

---

## 6. Peer Review Status and References

The Artificial NEOs Theory is currently an independent research hypothesis. It has
not been submitted to or accepted by a peer-reviewed journal. The aNEOS detection
framework implements the statistical methodology described here; validation against
an independent corpus is the prerequisite for any publication claim.

### Future Work

The following are the highest-priority items needed to advance the hypothesis toward
a publication-ready state:

| Priority | Task | Blocking dependency |
|----------|------|---------------------|
| 1 | Assemble independent validation corpus (≥ 50 labelled objects) | More confirmed artificial heliocentric objects |
| 2 | Replace Fisher's method with Stouffer's weighted z-score combination | Corpus needed for weight calibration |
| 3 | Empirically calibrate likelihood ratios (LR = 10/5/2) against labelled data | Corpus (above) |
| 4 | Dedicated N-body simulation validation of anomaly thresholds | Computational time |
| 5 | Activate ML classifier (`ml/models.py` IsolationForest) with labelled ground-truth | Corpus (above) |
| 6 | Cross-match with radar archives (Goldstone/Arecibo) for ~200 NEOs with A2 | Archive access |
| 7 | Submit to independent peer review (suggested venue: *Icarus* or *AJ*) | Items 1–3 above |

### References

- Bottke, W. F. et al. (2006). *The Yarkovsky and YORP effects: implications for asteroid
  dynamics.* Annual Review of Earth and Planetary Sciences, 34, 157–191.
- Chesley, S. R. et al. (2003). *Direct detection of the Yarkovsky effect by radar ranging
  to asteroid 6489 Golevka.* Science, 302(5651), 1739–1742.
- Chesley, S. R. et al. (2023). *Non-gravitational perturbations of near-Earth asteroids.*
  NASA/JPL Technical Memoranda.
- Farnocchia, D. et al. (2013). *Yarkovsky-driven orbital evolution of asteroid 2009 BD.*
  Icarus, 224(1), 1–13.
- Farnocchia, D. et al. (2023). *Non-gravitational acceleration in the trajectory of
  1998 KY26.* JPL Technical Memorandum.
- Gladman, B. J. et al. (2000). *Near-Earth asteroid population statistics from a
  numerical model.* Icarus, 146, 176–189.
- Granvik, M. et al. (2018). *Debiased orbit and absolute-magnitude distributions for
  near-Earth objects.* Icarus, 312, 181–207.
- Marsden, B. G., Sekanina, Z., & Yeomans, D. K. (1973). *Comets and nongravitational
  forces.* The Astronomical Journal, 78, 211.
- Michel, P. et al. (2005). *The population of Mars-crossing asteroids.* Icarus, 172, 463–474.
- Seligman, D. Z., Flekkøy, E. G., & Meech, K. J. (2023). *Dark comets and the
  population of non-gravitationally accelerating NEOs.* The Astrophysical Journal Letters.
- Wiegert, P. et al. (2021). *The 2020 SO encounter: Confirming the centaur upper-stage
  hypothesis.* The Astronomical Journal.
- Kardashev, N. S. (1964). *Transmission of information by extraterrestrial civilizations.*
  Soviet Astronomy, 8(2), 217–221.

**aNEOS documentation cross-references:**

- `docs/scientific/scientific-documentation.md` — Full statistical methodology
- `docs/scientific/VALIDATION_INTEGRITY.md` — Honest uncertainty audit
- `docs/architecture/ADR.md` — Architecture decisions (ADR-045 rendezvous; ADR-053
  physical indicators; ADR-058 ATLAS scoring; ADR-059 proxy discipline)
- `docs/architecture/DDD.md` — Bounded Contexts (BC3 anomaly scoring; BC11 population
  pattern analysis)

---

## Appendix A — Thought Experiment: Hypothesised Monitoring Objectives

> **Important framing**: This appendix documents speculative hypotheses from the
> original theory source document. These are **not conclusions** drawn from the
> statistical analysis — they are exploratory thought experiments presented for
> completeness. No observational evidence supports any of the scenarios described.
> The section is retained because it motivates the Kardashev Synthetic Corpus
> Generator in the codebase, which has legitimate utility for generating labelled
> training data across a spectrum of technological scenarios.

If ANT's alternative hypothesis (H₁) were confirmed, the question arises what function
anomalously-orbiting objects might serve. Objects with stable, regular Earth approaches
could geometrically serve long-duration observation functions. The constraints on any
such technology are severe:

- Resolving Earth surface features from lunar distance (384,400 km) to centimetre-scale
  would require optical apertures approximately 1,000× larger than current state-of-the-art
  reconnaissance systems
- Sustained multi-century operations imply autonomous self-maintenance
- Real-time data relay at any useful bandwidth requires a communications infrastructure
  for which no Earth-analogue exists

These requirements define a lower technological bound. The aNEOS Kardashev Synthetic
Corpus Generator (`aneos_core/datasets/kardashev_generator.py`) creates labelled training
data across 14 technological scenarios from K0.5 (decommissioned rocket stages) to K2.0
(hypothetical megastructure fragments). Its primary value is as a **synthetic data
augmentation tool** for classifier development — not as a model of any real scenario.

The extrapolation that entities capable of NEO-based Earth observation would require
Kardashev Type 1.5–2.45 capabilities is a theoretical bounding exercise. It is
presented here for completeness and **should not be interpreted as a finding**. The
Kardashev scale was defined for energy-use classification of civilisations (Kardashev
1964); its application to inferred technological requirements is a secondary derivation
with high uncertainty at every step.

---

*This document derives from the original `Artificial_NEOS_Theory.docx` source document.
Revised with external reviewer feedback and scientific framing by the aNEOS development
team, 2026-03-10.*
