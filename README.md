<a name="top"></a>

# aNEOS: Advanced Near Earth Object Suite

**v2.2.0.dev0 | Development review | Unreleased reliability work | 2026-11-13**

aNEOS is a Python research platform for:

1. **NEO candidate screening:** reproducible, uncalibrated anomaly ranking. It does
   not establish natural, human-made or extraterrestrial origin.
2. **Impact research support:** scoped external JPL Sentry evidence and explicitly
   conditional physical scenarios, not a validated local Earth/Moon risk solver.

> **Current contract:** this review branch does not provide independently calibrated
> origin probabilities, FDR estimates or five-sigma discovery significance. Earlier
> numerical examples and aspirational capability descriptions below are not current
> scientific validation. Follow the scoped engineering reports and ADR-065–079;
> remaining work is tracked in [programme #2](https://github.com/RobLe3/aneos-suite/issues/2).
> This is not an operational warning or planetary-defense decision service.

---

## Local evidence storage — scoped development capability

Explicit file-SQLite enrichment now preserves source-specific history, atomic physical
value/uncertainty selections and replayable object projections. The existing database
detail view exposes provenance and missingness. See the [storage contract and runnable
example](docs/engineering/durable-enrichment-ledger.md) before enabling writes to a
research database. This is not yet automatic persistence of every source query, complete
source coverage or scientific validation.

## Analysis policy and configuration

The legacy Analysis Policy view shows the actual default ranking policy and explicitly
synthetic diagnostics. It exports read-only snapshots and verifies archives without
applying them; alpha/BIC/threshold mutation is not supported by that menu. Explicit
configuration-file errors now fail rather than silently activate defaults, and rejected
manager updates/reloads preserve the prior object. See the [policy guide and executed
example](docs/engineering/truthful-policy-menu.md) and [configuration reference](docs/reference/configuration-reference.md)
for scope and remaining limits. Neither feature establishes scientific calibration.

## Long-Term Goal: Addressing the Fermi Paradox Empirically

The **Fermi Paradox** asks a deceptively simple question: given the age and scale of the galaxy, where is everybody? If intelligent civilisations arise at even a modest rate across billions of stars over billions of years, our solar system should statistically have been visited, observed, or contacted by now. The absence of any confirmed evidence is the paradox.

Most proposed resolutions fall into two camps: either civilisations are rare or short-lived (the "rare Earth" and "great filter" families), or the universe is populated but we are not looking in the right places or with the right instruments (the "dark forest", "zoo", and "artefact" families).

This is a long-term research motivation, not an established capability of aNEOS.
An anomalous catalogue entry may reflect natural physics, uncertain measurements,
a fitting artifact or human technology. Screening alone cannot distinguish these
explanations, much less establish extraterrestrial origin.

Even a properly calibrated five-sigma result would concern a specified statistical
null model; it would not identify an object's origin. Hypotheses need comparable
observations, physical fits, uncertainty and independent checks. A credible natural
explanation or an unresolved result is a useful outcome. Results from a bounded
screening population cannot settle the Fermi Paradox.

The current reference investigation is hypothesis-neutral 1998 KY26 research. Its
engineering/scientific requirements are tracked in programme #2 and the versioned
case-study register; most observational fitting and independent evaluation remain
unfinished. See `docs/scientific/VALIDATION_INTEGRITY.md`.

---

## Table of Contents

**Part 1: Introduction**
- [What is aNEOS (Plain English)](#part-1-what-is-aneos)
- [Long-Term Goal: Addressing the Fermi Paradox](#long-term-goal-addressing-the-fermi-paradox-empirically)
- [What it is not](#what-it-is-not)

**Part 2: Getting Started**
- [Quick Start](#part-2-getting-started)
- [Start the REST API](#start-the-rest-api)

**Part 3: Features**
- [Detection (Mission 1)](#detection-mission-1)
- [Impact Assessment (Mission 2)](#impact-assessment-mission-2)
- [Population and Polling Analysis](#population-and-polling-bc11)
- [Terminal UI](#terminal-ui)

**Part 4: Examples**
- [Screen a single NEO](#1-screen-a-single-neo-in-the-terminal)
- [Classify a known artificial object](#2-classify-a-known-artificial-object)
- [Multi-evidence verbose breakdown](#3-run-multi-evidence-analysis-with-verbose-breakdown)
- [Batch-screen a list](#4-batch-screen-a-list-of-neos-with-progress-bar)
- [Scoped impact evidence](#5-inspect-scoped-impact-evidence)
- [10-year orbital history](#6-view-10-year-orbital-history)
- [200-year historical pipeline](#7-run-the-200-year-historical-pipeline)
- [Population clustering and harmonics](#8-screen-a-population-for-clustering-and-harmonics)
- [REST API quick tour](#9-rest-api-programmatic-access)
- [Python library usage](#10-python-library-usage)

**Part 5: Use Cases by Profession**
- [Planetary Defense Scientists](#planetary-defense-scientists-nasa-pdco--esa-neocc)
- [Astronomers](#astronomers--observational-astronomers)
- [Astrophysicists](#astrophysicists-orbital-mechanics--statistical-methods)
- [Astrodynamicists and Mission Planners](#astrodynamicists--mission-planners)
- [SETI and Technosignature Researchers](#seti--technosignature-researchers)
- [Software Developers and API Integrators](#software-developers--api-integrators)

**Part 6: Accuracy and Limitations**
- [Detection Quality: Verified Claims](#part-6-accuracy-and-limitations)
- [Capabilities and Limitations](#capabilities-and-limitations)
- [Responsible Use Thresholds](#responsible-use-thresholds)

**Part 7: Technical Reference**
- [System Architecture](#part-7-technical-reference)
- [REST API Reference](#rest-api-reference)

**Part 8: Scientific Foundation**

**Part 9: Contributing and License**

---

## Part 1: What is aNEOS?

**Near Earth Objects (NEOs)** include asteroids and comets with orbits near Earth's.
Catalogue investigations can also encounter human-made objects. Origin needs
object-specific evidence rather than a name match or anomaly threshold.

aNEOS asks two questions about any object in those catalogues:

**Question 1: Does this object behave like a natural rock?**

The software acquires typed public-catalogue evidence and compares available
features to produce an uncalibrated screening rank.
A high rank can motivate follow-up; it is not Gaussian significance or an origin
probability. Natural processes, source/measurement uncertainty and technological
hypotheses require separate compatible observational models. Independent controls,
calibration and held-out evaluation remain unfinished.

**Question 2: What impact evidence is available?**

The existing REST impact route, V2 menu and impact-enhanced wrapper can retrieve
scoped JPL Sentry object evidence: source method, listed virtual impactors, dates,
probabilities and uncertainty limitations. Unlisted/removed/source-failed outcomes
are distinct and do not mean zero risk. No lunar probability, confidence interval,
local Torino class, damage or risk-period model is invented from catalogue elements.

A separate explicit conditional energy function computes classical kinetic energy
from caller-supplied mass and speed. That scenario is not a predicted collision,
measured input or validated damage model. Legacy geometric/arc formula replay is
retained only as labelled unvalidated diagnostics; its former scientific-probability
entry point raises a controlled-deprecation error.

**Evidence and limits**

Typed provider acquisition, shared screening/report transport and scoped Sentry
source integration have regression controls. Those tests do not establish real-world
origin-detection accuracy or a local impact model. Synthetic training/evaluation is
not independent scientific validation. Current counts, commands and remaining gaps
are recorded in the engineering reports rather than fixed historical test totals.

### What it is not

aNEOS is independent research software, not an agency service or an operational
space-surveillance/impact-warning system. Catalogue/model outputs carry their source
scope and limitations. It does not replace authoritative monitoring, observational
follow-up or independent scientific review.

[Back to top](#top)

---

## Part 2: Getting Started

```bash
git clone https://github.com/RobLe3/aneos-suite.git
cd aneos-suite
pip install -r requirements.txt   # install dependencies
python aneos.py                   # launch 15-option interactive menu
```

### Start the REST API

```bash
python aneos.py api --dev
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Run the legacy full menu (121 options)

```bash
python aneos.py --legacy-menu
```

[Back to top](#top)

---

## Part 3: Features

All items below are implemented, tested (360 unit + integration tests pass / 0 fail), and verified against real data.

### Detection (Mission 1)

| Capability | Status | Notes |
|---|---|---|
| Fetch orbital elements from JPL SBDB | Working | Apophis, Bennu, Ceres confirmed live |
| Fetch orbital elements from JPL Horizons | Working | Element tables parsed via regex |
| Fetch orbital elements from NEODyS, MPC | Working (fallback) | Graceful; fails without blocking pipeline |
| Fetch close-approach data from SBDB CAD API | Working | `date-min=now`, `dist-max=0.2 AU` |
| Fetch orbital history time-series from Horizons | Working | 10-year Keplerian element series |
| Cache with full round-trip fidelity | Working | `physical_properties` + `fetched_at` preserved across cache hits |
| Sigma-5 multi-modal detector | Working | 6 evidence types: orbital, physical, trajectory, temporal, statistical, behavioral |
| Bayesian probability calibration | Working | 0.1% base rate prior; posterior 1-4% from orbital+physical |
| Known-spacecraft catalog veto | Working | Tesla Roadster, 2020 SO, J002E3 instantly classified without statistical analysis |
| Batch detection (concurrent) | Working | `POST /analyze/batch` with `ThreadPoolExecutor` |
| Orbital history course-correction analysis | Working | `_analyze_course_corrections()` fired from API |
| REST `GET /detect` endpoint | Working | Returns `DetectionResponse` with sigma, tier, evidence, p-values |
| REST `POST /detect` endpoint | Working | Accepts user-supplied orbital elements + optional history |
| REST `GET /history` endpoint | Working | Returns Horizons 10-year Keplerian time-series |
| Physical indicators: diameter + albedo anomaly | Working (Phase 21) | `DiameterAnomalyIndicator` + `AlbedoAnomalyIndicator` active in detection path |
| OrbitalInput Pydantic bounds | Working (Phase 24) | `a` 0.1-1000 AU; `e` 0-2; `i` 0-180 deg -- invalid inputs rejected with HTTP 422 |
| NEODyS provisional designation fix | Working (Phase 24) | `"1998 KY26"` and `"2004 MN4"` no longer misparsed as year integers |

### Impact Assessment (Mission 2)

| Capability | Status | Notes |
|---|---|---|
| Scoped Earth-event model output | Integrated, source-dependent | JPL Sentry method/events/dates and material uncertainty; not a local fit |
| Moon collision probability | Not implemented as a validated model | Null, not zero risk |
| Conditional kinetic energy | Implemented, explicit supplied scenario | Mass kg, speed km/s; no probability/damage inference |
| Legacy crater/damage/arc formulas | Unvalidated diagnostics only | No current scientific-risk claim |
| Local keyhole/peak-risk/interval/confidence model | Not validated | Do not use prototype outputs as fitted probability/CI |
| Observation arc inventory | Actual dated inputs only | No quality-score-to-days or 30-day missing-value default |
| REST/V2 menu/enhanced impact wrapper | Shared scoped source | JSON/Markdown evidence; origin screening remains separate |
| Menu/enhanced report archive | Local append-only snapshots | Verified writes, restart read-back; not a durable job queue/case workflow |

### Population and Polling (BC11)

| Capability | Status | Notes |
|---|---|---|
| 200-year historical close-approach poll | Working | 40 x 5-year chunks via SBDB CAD API |
| Orbital catalogue grouping | Descriptive, not calibrated | Deterministic circular neighbourhoods; no missing-angle imputation/family claim |
| Computed encounter gaps | Descriptive, native TDB dates | No synthetic daily observations or periodicity significance |
| Non-gravitational correlation | Unavailable without paired time series | A1/A2/A3, uncertainty and covariance inventory retained |
| Network significance | Not calibrated | Nullable sigma/p; legacy fusion restricted to explicit historical diagnostics |
| Existing network REST/menu | Evidence-bound report transport | Process-memory jobs remain nondurable and not owner-isolated (#15) |

### Terminal UI

| Capability | Status | Notes |
|---|---|---|
| 15-option Rich terminal menu | Working | `python aneos.py`; 4 groups: Detection, Impact, Polling, System |
| Rich progress bars on all batch operations | Working | `ANEOSMenuBase.track_progress()` generator |
| File browser for designation files | Working | `ANEOSMenuBase.browse_files()` numbered table picker |
| Interactive results browser | Working | Pick `#` to see full verbose detection detail |
| API server launch from menu (Option 12) | Working | uvicorn subprocess, PID shown |
| Session detection analytics (Option 13) | Working | sigma-tier breakdown + JSON export |
| Scientific help viewer (Option 14) | Working | Reads `docs/scientific/` in-terminal |
| JWT bearer token endpoint | Working (Phase 21) | `POST /api/v1/auth/token` exchanges API key for signed JWT |
| Gaia stdout suppression | Working (Phase 24) | Gaia import warnings redirected to debug log, not terminal |

[Back to top](#top)

---

## Part 4: Examples

### 1. Screen a single NEO in the terminal

```
python aneos.py
-> Choice: 1 (Detect NEO)
-> Designation: 99942          # Apophis

Output:
  sigma = 1.82   INCONCLUSIVE (sigma<2)
  P(artificial) = 0.1%
  Combined p-value = 0.18
  Evidence: orbital OK  physical OK  trajectory OK  temporal OK  statistical OK  behavioral OK
```

### 2. Classify a known artificial object

```
python aneos.py
-> Choice: 1
-> Designation: 2020 SO        # Centaur upper stage

Output:
  SPACECRAFT VETO -- Known catalog match
  Classification: ARTIFICIAL VALIDATED (sigma>=5)
  sigma = 6.97   P(artificial) = 3.7%
  Reason: Object matches SpaceX/NASA spacecraft catalog
```

### 3. Run multi-evidence analysis with verbose breakdown

```
python aneos.py
-> Choice: 2 (Multi-Evidence Analysis)
-> Designation: J002E3         # Apollo 12 S-IVB stage

Output:
  sigma = 5.76   ARTIFICIAL VALIDATED
  Evidence breakdown:
    orbital_anomaly     p=0.00094  effect=1.23  quality=0.88  analyzed
    physical_properties p=0.00011  effect=2.41  quality=0.91  analyzed
    trajectory_analysis p=0.00430  effect=0.87  quality=0.85  analyzed
    temporal_patterns   p=0.01200  effect=0.61  quality=0.79  analyzed
    statistical_anomaly p=0.00320  effect=1.05  quality=0.82  analyzed
    behavioral_pattern  p=0.02100  effect=0.54  quality=0.75  analyzed
```

### 4. Batch-screen a list of NEOs (with progress bar)

Create `targets.txt`:
```
99942
433
3200
25143
2020 SO
J002E3
```

```
python aneos.py
-> Choice: 3 (Batch Detection)
-> Browse files -> select targets.txt

Output (Rich progress bar):
  Detecting 6 NEOs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%  6/6
  ┌─────────────────┬──────┬──────────────┬────────────────────────────┐
  │ Designation     │  σ   │ P(artificial)│ Classification             │
  ├─────────────────┼──────┼──────────────┼────────────────────────────┤
  │ 99942           │ 1.82 │ 0.0010       │ ❔ INCONCLUSIVE (σ<2)      │
  │ 433             │ 0.94 │ 0.0010       │ ❔ INCONCLUSIVE (σ<2)      │
  │ 3200            │ 2.31 │ 0.0012       │ ⚠️  EDGE CASE (σ≥2)        │
  │ 25143           │ 1.56 │ 0.0010       │ ❔ INCONCLUSIVE (σ<2)      │
  │ 2020 SO         │ 6.97 │ 0.0367       │ 🤖 ARTIFICIAL VALIDATED    │
  │ J002E3          │ 5.76 │ 0.0367       │ 🤖 ARTIFICIAL VALIDATED    │
  └─────────────────┴──────┴──────────────┴────────────────────────────┘
```

### 5. Inspect scoped impact evidence

```sh
python aneos.py
# Choice: 5 (Impact Evidence)
# Primary number/provisional designation: 1998 KY26
```

The result identifies a source outcome and dated evidence, not a default probability.
The frozen 2026-11-12 KY26 lookup was **not listed**, with null probability and
not-assessed risk. A future query may differ. Inspect source method, event span and
limits for an available result; not listed/removed is not zero risk. The V2 menu
confirms a local archive only after writing and replay verification succeed.
See [the dated record](docs/case-studies/1998-KY26/sentry/20261112T211912Z/README.md).

### 6. View 10-year orbital history

```
python aneos.py
-> Choice: 4 (Orbital History Analysis)
-> Designation: 433            # Eros

Output (table of Keplerian elements by epoch):
  ┌──────────────┬────────┬────────┬────────┬────────┬──────────────┐
  │ Epoch        │  a(AU) │   e    │  i(°)  │  Ω(°)  │ Da/epoch     │
  ├──────────────┼────────┼────────┼────────┼────────┼──────────────┤
  │ 2016-03-09   │ 1.4580 │ 0.2229 │ 10.83  │ 304.3  │ n/a          │
  │ 2017-03-09   │ 1.4581 │ 0.2229 │ 10.83  │ 304.2  │ +0.0001      │
  │ ...          │ ...    │ ...    │ ...    │ ...    │ ...          │
  └──────────────┴────────┴────────┴────────┴────────┴──────────────┘
  No anomalous course corrections detected.
```

### 7. Run the 200-year historical pipeline

```
python aneos.py
-> Choice: 7 (Live Pipeline Dashboard)

Output:
  Checking API sources ... 4/4 online
  Historical Data Polling  100%  27,632 objects retrieved
  ATLAS First-Stage Review 100%  candidates flagged
  Multi-Stage Validation   100%  validated
  Expert Review Queue      100%  final candidates

  Objects Processed : 27,632
  Processing Time   : 124.7 s
```

### 8. Screen a population for clustering and harmonics

```
python aneos.py
-> Choice: 8 (Population Pattern Analysis)
-> Browse files -> select a multi-object list

Output (illustrative counts, not an observation):
  Objects analysed: 3
  Descriptive groups: 1
  Statistical significance: not calibrated. Origin: unresolved.
  clustering: descriptive_only
  harmonics: descriptive_only_no_calibrated_periodicity
```

### 9. REST API: programmatic access

```bash
# Start server
python aneos.py api --dev &

# Detect
curl "http://localhost:8000/api/v1/analysis/detect?designation=2020%20SO"

# Impact
curl "http://localhost:8000/api/v1/analysis/impact?designation=99942"

# Batch
curl -X POST "http://localhost:8000/api/v1/analysis/analyze/batch" \
     -H "Content-Type: application/json" \
     -d '{"designations": ["99942","433","2020 SO"], "include_evidence": true}'

# 10-year orbital history
curl "http://localhost:8000/api/v1/analysis/history?designation=433"

# Health
curl "http://localhost:8000/api/v1/health"
```

### 10. Python library usage

```python
import asyncio
from aneos_core.analysis.pipeline import AnalysisPipeline
from aneos_core.analysis.impact_evidence import get_sentry_impact_service, render_sentry_markdown

async def inspect(designation):
    result = await AnalysisPipeline().analyze_neo(designation)
    print(result.to_dict())  # Structured screening or an explicit unavailable/error state.
    report = await asyncio.to_thread(get_sentry_impact_service().assess, designation)
    print(render_sentry_markdown(report))  # Source-scoped impact evidence, not origin inference.

asyncio.run(inspect("1998 KY26"))
```

Both acquisition paths can fail; callers must handle provider/input errors. Missing
probability/significance is not zero. This example does not claim durable job
execution, automatic API-wide report persistence or independent scientific validation.


### Planetary Defense Scientists (NASA PDCO / ESA NEOCC)

**What aNEOS provides:**

- A secondary screening layer for newly discovered NEOs. Feed a designation into `GET /detect` and receive a structured `DetectionResponse` with six independent evidence types (orbital, physical, trajectory, temporal, statistical, behavioral), sigma confidence, and a Fisher-combined p-value.
- `GET /impact` returns a 16-field impact assessment including probability uncertainty bounds, keyhole passage analysis, peak risk decade, and a list of human-readable primary risk factors, supplementing (not replacing) your Monte Carlo orbit determination pipelines.
- Batch screening of observation lists via `POST /analyze/batch`.

**What you bring:**

- Authoritative orbit solutions. aNEOS uses JPL SBDB/Horizons orbital elements; for newly-discovered objects with short arcs, your own reduced elements will be more accurate.
- Telescope resources for follow-up. aNEOS flags anomalies; confirmation requires additional observations.

**Example workflow:**

```bash
# Screen a new discovery
curl "http://localhost:8000/api/v1/analysis/detect?designation=2024%20YR4"

# Get full impact profile
curl "http://localhost:8000/api/v1/analysis/impact?designation=2024%20YR4"

# Force fresh fetch (bypass cache)
curl "http://localhost:8000/api/v1/analysis/detect?designation=Apophis&force_refresh=true"
```

**Current quality**: On the 3 confirmed artificial heliocentric objects in our ground truth set (Tesla Roadster, 2020 SO, J002E3), the detector achieves sigma >= 5.76 for all three. On 20+ real JPL natural NEOs, specificity = 1.00 (zero false positives at the calibrated threshold of 0.037). This is a small validation set; treat it as a proof of concept, not a production accuracy guarantee.

---

### Astronomers / Observational Astronomers

**What aNEOS provides:**

- An API endpoint (`GET /history`) that retrieves 10-year Keplerian element time-series from JPL Horizons for any named body. Useful for visualising long-term orbital evolution without writing Horizons query scripts yourself.
- Automated anomaly scoring across 6 indicator categories. If you observe an object that seems unusual, submit its orbital elements to `POST /detect` and receive a structured assessment in seconds.
- Close-approach data (within 0.2 AU, from present onward) fetched automatically from SBDB CAD API for every object analysed.

**Example: check whether a newly reported object's orbit is consistent with natural dynamics**

```python
import requests

# Submit your reduced orbital elements directly
resp = requests.post("http://localhost:8000/api/v1/analysis/detect", json={
    "a": 1.91,       # AU
    "e": 0.33,
    "i": 11.4,       # degrees
    "diameter_km": 0.006,
    "designation": "my_target",
})
result = resp.json()
print(f"sigma={result['sigma_confidence']:.2f}, tier={result['sigma_tier']}")
print(f"P(artificial)={result['artificial_probability']*100:.3f}%")
print(f"Evidence sources: {len(result['evidence_sources'])}")
```

**What aNEOS cannot do**: It cannot process raw photometry, spectra, or astrometric residuals. It works only from orbital elements and physical properties (diameter, albedo), the same inputs available in public catalogues.

---

### Astrophysicists (Orbital Mechanics / Statistical Methods)

**What aNEOS provides:**

- A transparent Bayesian detection framework you can read, audit, and extend. The canonical detector (`aneos_core/detection/validated_sigma5_artificial_neo_detector.py`) implements Fisher's method for combining p-values across independent evidence streams, followed by Bayesian updating with a 0.1% base-rate prior. All statistical methods are documented in `docs/scientific/scientific-documentation.md`.
- A ground truth dataset builder (`aneos_core/datasets/ground_truth_dataset_preparation.py`) that constructs a labelled corpus from JPL SBDB (confirmed naturals) and known heliocentric spacecraft (confirmed artificials). The corpus currently covers 9 artificials and up to 250 naturals.
- Honest uncertainty quantification: `ImpactResponse` returns `probability_uncertainty` (lower/upper bounds), `calculation_confidence`, and `impact_probability_by_decade`, making temporal evolution visible.

**Key statistical property to understand:**

| Observation | Sigma | P(artificial) |
|---|---|---|
| Tesla Roadster orbital+physical | 5.76 | ~3.7% |
| Apophis orbital only | < 2.0 | ~0.1% |
| Any natural NEO (observed threshold) | < 3.0 | ~0.1-0.2% |

The posterior is bounded by the base prior. To exceed 10% P(artificial), propulsion signatures or observed course corrections are required; no such automated data source currently exists. This is mathematically correct, not a software limitation.

---

### Astrodynamicists / Mission Planners

**What aNEOS provides:**

- Impact keyhole analysis: `ImpactResponse.keyhole_passages` lists resonant return opportunities with associated probability amplification.
- Temporal risk evolution: `impact_probability_by_decade` and `peak_risk_period` ([start_year, end_year]) allow mission timeline planning against the probability curve.
- Earth vs Moon impact ratio: `moon_earth_ratio` quantifies which body is at greater risk, relevant for protecting future lunar infrastructure.
- Artificial object uncertainty flag: if `is_artificial=True` or `artificial_probability > 0.037`, the impact assessment is annotated with `artificial_object_considerations`, indicating that trajectory uncertainty from potential propulsion capability is not modelled.

**Example: screen a candidate for a kinetic deflection mission**

```python
import requests

r = requests.get("http://localhost:8000/api/v1/analysis/impact",
                 params={"designation": "99942"})  # Apophis
data = r.json()
print(f"P(Earth) = {data['collision_probability']:.2e}")
print(f"P(Moon)  = {data['moon_collision_probability']:.2e}")
print(f"Peak risk: {data['peak_risk_period']}")
print(f"Keyholes: {len(data['keyhole_passages'])}")
print(f"Energy: {data['impact_energy_mt']:.0f} MT TNT")
print(f"Damage radius: {data['damage_radius_km']:.1f} km")
for factor in data['primary_risk_factors']:
    print(f"  * {factor}")
```

---

### SETI / Technosignature Researchers

**What aNEOS provides:**

- The world's first open-source statistical framework specifically designed to test whether a heliocentric object's orbital and physical properties are inconsistent with the natural NEO population, the foundation of the Artificial NEOs Theory.
- Six independent anomaly indicators (orbital dynamics, physical properties, trajectory, temporal patterns, statistical anomaly, behavioral patterns), each contributing an independent p-value combined via Fisher's method.
- A calibrated interpretation tier system: ROUTINE / NOTABLE / INTERESTING / SIGNIFICANT (sigma>=3) / ANOMALOUS (sigma>=4) / EXCEPTIONAL (sigma>=5).
- `analysis_metadata` in every `DetectionResponse`: detector version, population reference statistics, and method parameters for reproducibility.

**What aNEOS cannot prove**: It cannot confirm artificial intelligence control, propulsion, or intentionality. High sigma (unusual orbit) + high P(artificial) is a flag for follow-up, not a discovery claim. The gap between "statistically unusual" and "artificial" requires propulsion/course-correction evidence that no automated catalogue provides today.

---

### Software Developers / API Integrators

**What aNEOS provides:**

- A FastAPI application with 52+ REST endpoints, auto-generated OpenAPI schema (`docs/api/openapi.json`, regenerated by `make spec`), and typed Pydantic models for every request and response.
- `GET /detect?designation=...&force_refresh=true` for real-time detection with cache control.
- `POST /detect` to supply your own orbital elements; optionally pass `orbital_history` (from `GET /history`) to enable course-correction analysis.
- `GET /history` for JPL Horizons 10-year Keplerian time-series.
- `GET /impact` for a 16-field impact assessment.
- `POST /analyze/batch` for concurrent batch detection with evidence detail in each result.
- `GET /health` for a typed health check with per-component status.

**Quick API test:**

```bash
# Install and start
python install.py --core && python aneos.py api --dev &

# Health check
curl http://localhost:8000/api/v1/health

# Detect 2020 SO (Centaur upper stage -- should return spacecraft_veto=true)
curl "http://localhost:8000/api/v1/analysis/detect?designation=2020%20SO"

# Get Apophis impact profile
curl "http://localhost:8000/api/v1/analysis/impact?designation=99942"

# Browse interactive docs
open http://localhost:8000/docs
```

[Back to top](#top)

---

## Part 6: Accuracy and Limitations

### Detection Quality: Verified Claims

The following results are reproducible by running `python -m pytest tests/ -m "not network" -q` and the ground truth validation suite.

| Metric | Value | Source |
|---|---|---|
| Unit / integration tests | 360 pass, 0 fail | `pytest tests/ aneos_core/tests/ -m "not network"` |
| Ground truth artificial objects | 3 confirmed | Tesla Roadster (SpaceX 2018-017A), 2020 SO (Centaur/Surveyor-2), J002E3 (Apollo 12 S-IVB) |
| Ground truth natural NEOs | 20+ | JPL SBDB query, real orbital data |
| Sensitivity (recall) | 1.00 | All 3 artificials correctly classified at calibrated threshold 0.037 |
| Specificity | 1.00 | Zero natural NEOs falsely flagged at calibrated threshold |
| F1 score | 1.00 | At calibrated threshold 0.037 |
| ROC-AUC | 1.00 | Validated externally |
| Tesla Roadster sigma | 5.76 | `Sigma5DetectionResult` |
| 2020 SO sigma | 6.97 | `Sigma5DetectionResult` |
| J002E3 sigma | 5.76 | `Sigma5DetectionResult` |
| Max P(artificial) from orbital+physical | 3-4% | Bayesian posterior, 0.1% prior |

**Important context**: The ground truth set is small (3 artificials, 20+ naturals). Perfect discrimination on this set demonstrates the approach works; it does not guarantee the same performance on a large, diverse unseen corpus. The Bayesian posterior ceiling (3-4%) is mathematically correct given the 0.1% base rate and available evidence types; it is not a software bug.

---

### Capabilities and Limitations

**What aNEOS supports today:**

- Statistical screening of NEOs by orbital and physical anomaly indicators
- Multi-source data acquisition: JPL SBDB, JPL Horizons, NEODyS, MPC (with graceful fallback)
- Close-approach history (upcoming, within 0.2 AU) via SBDB CAD API
- 200-year historical close-approach polling (40 x 5-year chunks, SBDB CAD API)
- Time-series orbital element history via JPL Horizons
- Earth and Moon impact probability with uncertainty bounds, keyholes, risk periods
- Population-level orbital clustering, synodic harmonic analysis, non-grav correlation (BC11)
- REST API with OpenAPI specification, Pydantic models, and batch processing
- Clean 15-option Rich terminal menu (`python aneos.py`)
- JSON/CSV export of analysis results
- shelve-based caching for CAD data (24-hour TTL)

**What aNEOS does NOT support:**

| Gap | Reason |
|---|---|
| Propulsion / manoeuvre signature detection | No automated data source exists; requires dedicated tracking campaigns |
| Radar polarimetry (SC/OC ratio) | SWARM KAPPA implemented but no live data feed |
| Thermal infrared modelling (NEATM) | SWARM LAMBDA implemented but requires WISE/NEOWISE photometry input |
| Gaia astrometric anomaly detection | SWARM MU implemented but requires Gaia epoch astrometry input |
| Real-time NEO discovery alerts | No MPC/JPL Scout webhook integration |
| Observation scheduling or telescope control | Out of scope for this platform |
| Production authentication (JWT) | Mock tokens in dev mode; not suitable for public deployment |
| ML classifier activation | Deferred (G-015); scikit-learn pipeline exists but is not wired into the default detection path |
| Processing raw photometry or spectra | Only processed orbital elements and physical properties are accepted |
| IAU Torino / Palermo scale ratings | aNEOS uses its own risk classification; Torino/Palermo require authoritative orbit solutions |
| ESA/NASA operational endorsement | aNEOS is an independent research tool |

---

### Responsible Use Thresholds

- **Report a result as "anomalous"** only if `sigma_confidence >= 3.0` (SIGNIFICANT tier).
- **Report a result as "potential artificial"** only if `sigma_confidence >= 5.0` (EXCEPTIONAL tier) **and** independent follow-up confirms the anomaly.
- **Do not cite `artificial_probability`** as proof of artificiality. The maximum posterior from orbital+physical evidence alone is 3-4%. Values in this range indicate "unusual orbit", not "confirmed artificial object".
- **Cross-check impact probabilities** against JPL Scout or ESA NEOCC for any object with `collision_probability > 1e-6`.

[Back to top](#top)

---

## Part 7: Technical Reference

### System Architecture

```
aneos-suite/
+-- aneos_core/           # Core science and data packages
|   +-- data/             # DataFetcher, CacheManager, SBDB/Horizons/NEODyS/MPC sources
|   +-- detection/        # ValidatedSigma5ArtificialNEODetector (canonical), DetectionManager
|   +-- analysis/         # ImpactProbabilityCalculator, scoring, pipeline
|   +-- validation/       # 6 SWARMs (KAPPA/LAMBDA/MU/CLAUDETTE/THETA/ATLAS), stats
|   +-- datasets/         # Ground truth dataset builder and validator
|   +-- ml/               # ML classifier (deferred, behind HAS_TORCH guard)
|   +-- monitoring/       # Prometheus/Grafana, psutil metrics, SMTP alerts
|   +-- config/           # APIConfig, settings
+-- aneos_api/            # FastAPI application
|   +-- endpoints/        # analysis, dashboard, monitoring, admin, data
|   +-- schemas/          # Pydantic models: DetectionResponse, ImpactResponse, OrbitalInput, ...
|   +-- app.py            # Application factory
+-- aneos_dashboard/      # Web dashboard (Flask)
+-- aneos_menu_v2.py      # 15-option Rich terminal menu (primary)
+-- aneos_menu.py         # Legacy 121-option menu (--legacy-menu flag)
+-- aneos_menu_base.py    # Shared UI helpers: progress bars, file browser, display
+-- aneos.py              # CLI entry point
+-- tests/                # 360 unit and integration tests
+-- docs/                 # Architecture (ADR, DDD), API spec, user guide, scientific docs
```

**Detection pipeline:**

```
GET /detect?designation=Apophis
  |
  +- Known spacecraft catalog veto -> instant ARTIFICIAL (no analysis)
  |
  +- DataFetcher.fetch_neo_data()
  |   +- JPL SBDB          -> orbital elements + physical properties
  |   +- JPL Horizons      -> element table (fallback)
  |   +- NEODyS / MPC      -> element table (fallback)
  |   +- SBDB CAD API      -> close-approach history (supplemental, never blocks)
  |
  +- HorizonsSource.fetch_orbital_history()  -> 10-year Keplerian time-series
  |
  +- ValidatedSigma5ArtificialNEODetector.analyze_neo_validated()
      +- 6 evidence modules -> individual p-values
      +- Fisher's method    -> combined_p_value
      +- Bayesian update    -> bayesian_probability (0.1% prior)
      +- DetectionResponse  -> sigma_confidence, sigma_tier, evidence_sources, ...
```

---

### REST API Reference

Full OpenAPI specification: `docs/api/openapi.json` (regenerate with `make spec`).

Interactive documentation at `http://localhost:8000/docs` when the API server is running.

**Key endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/analysis/detect` | Run Sigma-5 detector on a named NEO |
| POST | `/api/v1/analysis/detect` | Run detector on caller-supplied orbital elements |
| GET | `/api/v1/analysis/impact` | Compute Earth/Moon impact probability |
| GET | `/api/v1/analysis/history` | Fetch 10-year Keplerian history from Horizons |
| POST | `/api/v1/analysis/analyze/batch` | Batch detection for multiple designations |
| GET | `/api/v1/analysis/batch/{id}/status` | Poll batch job with full evidence detail |
| GET | `/api/v1/health` | Per-component health check |
| GET | `/api/v1/data/neo/{designation}` | Raw NEO data fetch |

**`DetectionResponse` fields:**

| Field | Description |
|---|---|
| `sigma_confidence` | Sigma above natural NEO null hypothesis |
| `sigma_tier` | ROUTINE / NOTABLE / INTERESTING / SIGNIFICANT / ANOMALOUS / EXCEPTIONAL |
| `artificial_probability` | Bayesian posterior (0.1% prior) |
| `combined_p_value` | Fisher's combined p-value across evidence types |
| `false_discovery_rate` | Expected FDR at current sigma threshold |
| `evidence_sources` | List of `EvidenceSummary` with `anomaly_score`, `p_value`, `effect_size` |
| `analysis_metadata` | Detector version, method, population statistics |
| `spacecraft_veto` / `veto_reason` | Instant classification for known spacecraft |

**`ImpactResponse` fields (16 total):**

| Field | Description |
|---|---|
| `collision_probability` | Earth impact probability |
| `probability_uncertainty` | [lower, upper] confidence interval |
| `moon_collision_probability` | Moon impact probability |
| `moon_earth_ratio` | Relative risk between Moon and Earth |
| `impact_energy_mt` | Energy released in megatons TNT |
| `crater_diameter_km` | Estimated crater diameter |
| `damage_radius_km` | Estimated damage radius |
| `keyhole_passages` | Resonant return windows |
| `peak_risk_period` | Decade of highest risk |
| `impact_probability_by_decade` | Time-resolved probability evolution |
| `primary_risk_factors` | Human-readable rationale list |
| `comparative_risk` | Risk context relative to historical events |

[Back to top](#top)

---

## Part 8: Scientific Foundation

aNEOS builds on the **Artificial NEOs Theory**, the hypothesis that some Near Earth Objects may exhibit orbital or physical properties statistically inconsistent with natural formation and evolution, potentially indicating artificial origin. This is a scientific hypothesis, not an established finding.

**The statistical framework is grounded in:**

- **Bayesian inference**: Posterior probability updated from a 0.1% base rate (estimated fraction of heliocentric objects that could be artificial). This prior is conservative; the true rate is unknown.
- **Fisher's method**: Independent p-values from 6 evidence types are combined into a single test statistic. Under the natural null hypothesis, this statistic follows a chi-squared distribution with 2k degrees of freedom.
- **Sigma-5 threshold**: Classification as artificial requires combined significance >= 5-sigma (p < 5.7e-7), matching standard astronomical discovery criteria.
- **Calibrated threshold**: At p(Bayesian) >= 0.037, the current ground truth validation achieves sensitivity = 1.00 and specificity = 1.00 on the available corpus.

**Further reading:**

- Full methodology: `docs/scientific/scientific-documentation.md`
- Theory: `docs/scientific/theory.md`
- Honest caveats: `docs/scientific/VALIDATION_INTEGRITY.md`
- Architecture decisions: `docs/architecture/ADR.md` (60 ADRs)
- Domain model: `docs/architecture/DDD.md` (11 bounded contexts)
- Current state summary: `docs/current-state-summary.md`

[Back to top](#top)

---

## Part 9: Contributing and License

See `CONTRIBUTING.md` for development guidelines. The project follows the C&C + Implementation + Q&A agent structure defined in `DEVELOPMENT_FRAMEWORK.md`.

Run the test suite before opening a pull request:

```bash
python -m pytest tests/ aneos_core/tests/ -m "not network" -q   # 360 tests, 0 fail
make spec                                      # regenerate OpenAPI spec
git diff --stat docs/api/openapi.json          # confirm spec is current
```

**License**: Scientific research and educational use. See `LICENSE` for complete terms.

[Back to top](#top)

---

*aNEOS v1.2.2 | Phase 24 complete | Bug fixes: NEODyS provisional designation misparse, Gaia stdout suppression, DetectionResult falsy-zero guard, monitoring alert kwarg | Input validation: Pydantic bounds on OrbitalInput (a/e/i), designation length on AnalysisRequest | Endpoint wiring: get_validation_summary uses real DB, SSE streaming uses real alert_manager | Demo NEO list updated with real designations | REST API: 52+ endpoints | Test suite: 360 pass / 0 fail*

### Review-branch scientific screening contract (2026-11-12)

The primary launcher and legacy menu analysis routes now use the same typed
acquisition/shared screening path as the REST core. A rank is not Gaussian sigma,
a calibrated origin probability or a natural-origin finding. Missing evidence and
provider errors remain inconclusive; scientific fitting/calibration is unfinished.

```sh
python aneos.py analyze "1998 KY26"                  # human-readable Markdown
python aneos.py analyze "1998 KY26" --json           # versioned JSON report
python aneos.py analyze "1998 KY26" --force-refresh  # bypass source cache
python aneos.py analyze "1998 KY26" --json --output result.json
```

`analysis`, `a`, `simple` and `detect` are aliases for this same service, not a
separate classifier. `--output` creates a new file and refuses an existing path.
This prevents normal overwrite, not interrupted/partial writes or later manual
editing/deletion. New reports do not overwrite the frozen KY26 baseline. Entering
the legacy menu no longer silently deletes cache files or changes environment.

Reports retain normalized source data, native A1/A2/A3 units/uncertainty, covariance,
evidence/policy/review IDs and a report checksum. Source-file hashes and selected
runtime versions are not an environment lock or proof of loaded code; preserve
any dirty source separately. See [the exact scope and limitations](docs/engineering/legacy-menu-contract.md).
The standalone `simple_neo_analyzer.py` and remaining prototype paths are still
under review in [issue #30](https://github.com/RobLe3/aneos-suite/issues/30); the
launcher migration does not validate those library APIs or claim five-sigma readiness.

### Standalone facade correction (ADR-072 review branch)

This supersedes the standalone-analyzer warning in the preceding review-branch
section. `SimpleNEOAnalyzer` and `AdvancedaNEOSAnalyzer` now use actual typed
acquisition/shared screening, not name/hash-based inputs or a detector fallback.
The existing standalone commands delegate to the shared CLI:

```sh
python simple_neo_analyzer.py single "1998 KY26" --json
python simple_neo_analyzer.py analyze "1998 KY26" --force-refresh
python simple_neo_analyzer.py status
```

A designation is required for analysis; there is no implicit demo named `test`.
Scalar `calculate_artificial_probability` and fabricated/private calibration/input
contracts raise explicit controlled-deprecation errors; structured reports retain
probability/confidence/significance as null. Use `await analyzer.analyze_neo_async(...)`
inside an active event loop. Importable components and capability counts are not
scientific readiness or provider-health checks.

Impact bridges report actual input/prerequisite availability, not a probability or
safe verdict. The separate public impact REST/menu/core heuristic contract remains
[issue #32](https://github.com/RobLe3/aneos-suite/issues/32); core scenario/research
functions remain available but are not validated by this facade correction.
[Exact scope, tests and reproduction](docs/engineering/standalone-analysis-contract.md).

### Durable execution foundation (ADR-076, review branch)

`aneos_api.job_store.JobStore` adds transactional execution storage on the existing
SQLAlchemy database: scoped reads, idempotent admission, fenced leases, immutable
checkpoints, atomic terminal commits and retained history. Storage import does not
assemble web/ML services or create a database. Schema setup is explicit; starting
the full API confirms setup before services and cleans up on lifecycle failure.

**Not yet integrated:** current network/automatic/polling API workers. They are not
made durable merely by adding this component. ADR-077 corrects normal network
reporting; registered recoverable worker and authenticated API integration remain #15. Execution
completion does not mean scientific validation. See
[the storage controls, setup and integration gates](docs/engineering/durable-job-store.md).


### Population evidence reports (ADR-077, review branch)

The normal BC11 session now freezes actual NEOData and uses canonical per-object
screening. Descriptive groups, native-TDB computed encounter gaps, explicit module
availability and full A1/A2/A3 inventory replace unsupported network significance.
`verify_network_report` fully recomputes the report; `network_report_to_markdown`
includes verified JSON for replay. No network acquisition occurs inside this session.
See [the exact scope and remaining acceptance gates](docs/engineering/network-evidence-report.md).

### Registered resumable worker (ADR-078, review branch)

`aneos_api.network_jobs.NetworkWorker` executes the registered population job against
the SQL store: fixed request/query bounds, successful immutable source checkpoints,
resume without refreshing committed evidence, lease heartbeats/fencing and full
request/checkpoint/report verification before completion. It is an explicit Python
worker API; no automatic daemon or new CLI/REST option is introduced.

**Not yet integrated:** current authenticated API ownership, worker lifespan/health
and persistent REST jobs. See [the exact worker controls and remaining gates](docs/engineering/resumable-network-worker.md).
Completion is execution evidence, not calibrated scientific validation.

### Durable owned population API (ADR-079, review branch)

The existing population API now commits SQL jobs, runs a managed registered worker,
serves owner-scoped verified reports and supports cancellation/idempotent retries.
Configured/SQL API keys or explicitly scoped signed tokens are required; development
mocks do not authorize these jobs. Restart/partial-acquisition, positive-owner and
cross-owner controls are documented in [the scoped integration report](docs/engineering/durable-network-api.md).
Other job families and general authentication still need review. Successful execution
is not complete object knowledge, calibrated significance or origin confirmation.

### Additional user-quality programme

Requested source completeness/local enrichment, every-menu render QA, verified object
explanations/graphics, scientific suitability, current user documentation and version
consistency are tracked separately in [the requirements register](docs/engineering/user-quality-programme.md).
They remain acceptance work, not claims already achieved by engineering test counts.


Application version is defined in `aneos_version.py`. Run `python aneos.py --version`
without starting scientific or API services. The REST API contract remains2.0.0;
`application_version` and `X-Application-Version` identify this development build.
Model/schema/provider versions remain independent. No release tag or scientific
readiness is implied; historical result versions are preserved.

[Menu-entry before/after captures and scope](docs/engineering/version-and-menu-identity.md)
include actual text/HTML/SVG at80/120/160 columns, current Scientific Help and a
machine-readable inventory. Fifteen feature dispatch routes are checked; detailed
prerequisite/behavior/result QA remains explicitly pending in #41.

For the canonical V2 object-detail scope and its limits, see [evidence-bound object details](docs/engineering/evidence-bound-object-detail.md).

Physical source normalization, paired uncertainty and compatibility limits: [atomic physical evidence](docs/engineering/atomic-physical-evidence.md).

Native orbital/covariance epoch handling and computed-history limits: [native orbital epochs](docs/engineering/native-orbital-epochs.md).

HTTP history/native-epoch and raw-input contracts: [HTTP evidence parity](docs/engineering/http-evidence-parity.md).

### Acquisition history and honest completeness (2.2.0.dev0)

Ordinary queries now retain a fixed-profile source plan, source outcomes and local
correlated evidence before returning a result. Object detail and JSON export keep
acquisition dates and provenance visible. A complete response profile is not proof
of complete scientific evidence. See the [acquisition guide and reproducible
80/120/160-column QA](docs/engineering/object-acquisition-ledger.md) for usage,
cache/persistence behavior and remaining limits. This is unreleased draft work;
independent five-sigma/origin calibration is not claimed.

NEODyS-specific source identity, H/G photometry, native matrix retention and
reproducible object-review examples are described in the
[NEODyS evidence guide](docs/engineering/neodys-evidence.md). Typed H enrichment
does not imply a fitted phase law, transformed covariance or complete scientific data.

### Dates and chronology (2.2.0.dev0)

Object detail/export now separates source discovery days, observations, dynamical
epochs and local receipts. The ambiguous `first_discovered` / `last_updated` export
aliases are deprecated; use sourced `discovery_date` or explicit ledger receipt
fields. Old values remain preserved and unverified, not silently retagged UTC.
See the [chronology guide](docs/engineering/object-chronology.md) for the provider's
numbered-object discovery limitation, Pacific-local fit dates, partial observations,
compatibility changes and reproducible display examples.

### Source receipt review and refresh (2.2.0.dev0)

The legacy enrichment workflow now reviews replay-checked receipts for all five
installed object adapters, including Horizons and failed-query-only objects.
Unknown legacy timestamps are not assigned UTC. Explicit refresh uses the existing
acquisition/SQL-history path, respects stored cooldown eligibility and distinguishes
partial, empty, deferred and failed outcomes. A committed adapter query is not proof
of new observations or scientific completeness. See the [receipt review guide](docs/engineering/source-freshness.md)
for menu behavior, limits and reproducible examples. Automatic selective backfill
and exhaustive source/menu QA remain open.
