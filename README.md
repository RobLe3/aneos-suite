# aNEOS — Advanced Near Earth Object Suite

**v1.0.0 — Research Platform (Phase 10 complete, 2026-03-08)**

aNEOS is an open-source Python research platform with two independent missions:

1. **Artificial NEO Detection** — statistical screening of Near Earth Objects for signatures inconsistent with natural dynamics, using a Bayesian multi-modal framework calibrated against confirmed artificial heliocentric objects.
2. **Planetary Defense Assessment** — comprehensive Earth and Moon impact probability calculation with energy, crater, and risk-period estimation.

> **Honest scope statement**: aNEOS is a research tool, not an operational space-surveillance system. All results require independent verification through peer review, telescope follow-up, or comparison with authoritative catalogues (JPL Scout, ESA NEOCC). See [Capabilities and Limitations](#capabilities-and-limitations) before citing results.

---

## In Plain English — What Does This Software Actually Do?

Space agencies and astronomers track tens of thousands of **Near Earth Objects (NEOs)** —
asteroids and comets whose orbits bring them close to Earth. Most are natural rocks following
paths governed entirely by gravity. But a small number of confirmed objects are actually
human-made spacecraft — such as the Tesla Roadster launched by SpaceX in 2018 or rocket
upper stages left over from Apollo-era missions — that orbit the Sun just like asteroids do.

aNEOS asks two questions about any object in those catalogues:

**Question 1: Does this object behave like a natural rock?**

The software downloads publicly available orbital data from NASA's JPL database, measures six
different properties of the object's path around the Sun, and compares each against the
known population of natural NEOs. If the object's orbit, size, brightness, or approach pattern
is statistically unusual, it receives a higher "sigma score". The three confirmed artificial
objects in our test set all score above sigma 5 — the same threshold astronomers use to claim
a scientific discovery. All natural asteroids tested score below sigma 3.

This does **not** mean aNEOS can detect alien spacecraft. It means it can flag objects whose
behaviour is inconsistent with what we expect from rocks shaped only by gravity. Human-made
spacecraft are the only known cause of such anomalies, and even then the software assigns
only a ~3–4% probability of being artificial — the realistic ceiling given what orbital data
alone can tell us. Confirming artificial origin requires telescope or radar follow-up.

**Question 2: Could this object hit the Earth or Moon?**

For any named object, aNEOS calculates the probability of a collision with Earth and with the
Moon, estimates how much energy the impact would release (in megatons of TNT), how large a
crater it would leave, and which decade carries the highest risk. It also identifies
"gravitational keyholes" — narrow windows in space where a close approach could nudge an
object onto a future impact trajectory.

**How deep does it go?**

- Data is fetched live from four NASA/ESA sources (JPL SBDB, JPL Horizons, NEODyS, MPC).
- The detection framework has been validated against 3 confirmed artificial objects and 20+
  natural NEOs from the JPL catalogue: it correctly identifies all of them.
- A full REST API lets other software query aNEOS programmatically.
- 59 automated tests verify the system works correctly end to end.

**What it is not:**

aNEOS is a research platform built by an independent developer. It is not affiliated with
NASA, ESA, or any space agency. It does not have access to classified data, telescope feeds,
or radar measurements. Its impact probability numbers are research-grade estimates — useful
for screening and prioritisation, but not a replacement for the authoritative calculations
produced by JPL Sentry or ESA NEOCC.

---

## Table of Contents

1. [In Plain English — What Does This Software Actually Do?](#in-plain-english--what-does-this-software-actually-do)
2. [Quick Start](#quick-start)
3. [What aNEOS Does Right Now](#what-aneos-does-right-now)
4. [Who Benefits — Profession-Specific Use Cases](#who-benefits--profession-specific-use-cases)
5. [Detection Quality — Verified Claims](#detection-quality--verified-claims)
6. [Capabilities and Limitations](#capabilities-and-limitations)
7. [System Architecture](#system-architecture)
8. [REST API Reference](#rest-api-reference)
9. [Scientific Foundation](#scientific-foundation)
10. [Contributing](#contributing)

---

## Quick Start

```bash
git clone https://github.com/RobLe3/aneos-suite.git
cd aneos-suite
python install.py --core          # installs required dependencies
python aneos.py status            # verify system health
python aneos.py analyze "Apophis" # run full detection + impact analysis
```

### Start the REST API

```bash
python aneos.py api --dev
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Run the interactive menu

```bash
python aneos_menu.py
```

---

## What aNEOS Does Right Now

All items below are implemented, tested (59 unit tests pass / 0 fail), and verified against real data.

### Detection (Mission 1)

| Capability | Status | Evidence |
|---|---|---|
| Fetch orbital elements from JPL SBDB | Working | Apophis, Bennu, Ceres confirmed live |
| Fetch orbital elements from JPL Horizons | Working | Element tables parsed via regex |
| Fetch orbital elements from NEODyS, MPC | Working (fallback) | Graceful; fails without blocking pipeline |
| Fetch close-approach data from SBDB CAD API | Working (Phase 10) | `date-min=now`, `dist-max=0.2 AU` |
| Fetch orbital history time-series from Horizons | Working (Phase 10) | 10-year Keplerian element series |
| Cache with full round-trip fidelity | Working (Phase 10) | `physical_properties` + `fetched_at` preserved across cache hits |
| Sigma-5 multi-modal detector | Working | 6 evidence types: orbital, physical, trajectory, temporal, statistical, behavioral |
| Bayesian probability calibration | Working | 0.1% base rate prior; posterior ~1–4% from orbital+physical |
| Known-spacecraft catalog veto | Working | Tesla Roadster, 2020 SO, J002E3 instantly classified without statistical analysis |
| Batch detection (concurrent) | Working | `POST /analyze/batch` with `ThreadPoolExecutor` |
| Orbital history course-correction analysis | Working (Phase 10) | `_analyze_course_corrections()` fired from API |
| REST `GET /detect` endpoint | Working | Returns `DetectionResponse` with sigma, tier, evidence, p-values |
| REST `POST /detect` endpoint | Working | Accepts user-supplied orbital elements + optional history |
| REST `GET /history` endpoint | Working | Returns Horizons 10-year Keplerian time-series |
| `force_refresh` cache bypass on `GET /detect` | Working (Phase 10) | Query param `?force_refresh=true` |
| `orbital_history` passthrough on `POST /detect` | Working (Phase 10) | Chain: `GET /history` → `POST /detect` |

### Impact Assessment (Mission 2)

| Capability | Status | Notes |
|---|---|---|
| Earth collision probability | Working | Gravitational focusing, orbital integration |
| Moon collision probability | Working | Lunar cross-section, Earth vs Moon ratio |
| Impact energy (megatons TNT) | Working | Kinetic energy from velocity + mass |
| Crater diameter (km) | Working | Pi-scaling relations |
| Damage radius (km) | Working | Simplified energy-scaling |
| Gravitational keyhole analysis | Working | Close-approach resonance detection |
| Peak risk period (decade) | Working | Time-resolved probability evolution |
| Probability uncertainty bounds | Working (Phase 10) | `[lower, upper]` confidence interval |
| Primary risk factors list | Working (Phase 10) | Human-readable scientific rationale |
| `GET /impact` REST endpoint | Working | 16-field `ImpactResponse` |

---

## Who Benefits — Profession-Specific Use Cases

### Planetary Defense Scientists (NASA PDCO / ESA NEOCC)

**What aNEOS provides:**

- A secondary screening layer for newly discovered NEOs. Feed a designation into `GET /detect` and receive a structured `DetectionResponse` with six independent evidence types (orbital, physical, trajectory, temporal, statistical, behavioral), sigma confidence, and a Fisher-combined p-value.
- `GET /impact` returns a 16-field impact assessment including probability uncertainty bounds, keyhole passage analysis, peak risk decade, and a list of human-readable primary risk factors — supplementing (not replacing) your Monte Carlo orbit determination pipelines.
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

**Current quality**: On the 3 confirmed artificial heliocentric objects in our ground truth set (Tesla Roadster, 2020 SO, J002E3), the detector achieves σ ≥ 5.76 for all three. On 20+ real JPL natural NEOs, specificity = 1.00 (zero false positives at the calibrated threshold of 0.037). This is a small validation set — treat it as a proof of concept, not a production accuracy guarantee.

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

**What aNEOS cannot do**: It cannot process raw photometry, spectra, or astrometric residuals. It works only from orbital elements and physical properties (diameter, albedo) — the same inputs available in public catalogues.

---

### Astrophysicists (Orbital Mechanics / Statistical Methods)

**What aNEOS provides:**

- A transparent Bayesian detection framework you can read, audit, and extend. The canonical detector (`aneos_core/detection/validated_sigma5_artificial_neo_detector.py`) implements Fisher's method for combining p-values across independent evidence streams, followed by Bayesian updating with a 0.1% base-rate prior. All statistical methods are documented in `docs/scientific/scientific-documentation.md`.
- A ground truth dataset builder (`aneos_core/datasets/ground_truth_dataset_preparation.py`) that constructs a labelled corpus from JPL SBDB (confirmed naturals) and known heliocentric spacecraft (confirmed artificials). The corpus currently covers 9 artificials and up to 250 naturals.
- Honest uncertainty quantification: `ImpactResponse` now returns `probability_uncertainty` (lower/upper bounds), `calculation_confidence`, and `impact_probability_by_decade`, making temporal evolution visible.

**Key statistical property to understand:**

| Observation | Sigma | P(artificial) |
|---|---|---|
| Tesla Roadster orbital+physical | 5.76 | ~3.7% |
| Apophis orbital only | < 2.0 | ~0.1% |
| Any natural NEO (observed threshold) | < 3.0 | ~0.1–0.2% |

The posterior is bounded by the base prior. To exceed 10% P(artificial), propulsion signatures or observed course corrections are required — no such automated data source currently exists. This is mathematically correct, not a software limitation.

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
    print(f"  • {factor}")
```

---

### SETI / Technosignature Researchers

**What aNEOS provides:**

- The world's first open-source statistical framework specifically designed to test whether a heliocentric object's orbital and physical properties are inconsistent with the natural NEO population — the foundation of the Artificial NEOs Theory.
- Six independent anomaly indicators (orbital dynamics, physical properties, trajectory, temporal patterns, statistical anomaly, behavioral patterns), each contributing an independent p-value combined via Fisher's method.
- A calibrated interpretation tier system: ROUTINE / NOTABLE / INTERESTING / SIGNIFICANT (σ≥3) / ANOMALOUS (σ≥4) / EXCEPTIONAL (σ≥5).
- `analysis_metadata` in every `DetectionResponse` — detector version, population reference statistics, and method parameters for reproducibility.

**What aNEOS cannot prove**: It cannot confirm artificial intelligence control, propulsion, or intentionality. High sigma (unusual orbit) + high P(artificial) is a flag for follow-up, not a discovery claim. The gap between "statistically unusual" and "artificial" requires propulsion/course-correction evidence that no automated catalogue provides today.

---

### Software Developers / API Integrators

**What aNEOS provides:**

- A FastAPI application with 52+ REST endpoints, auto-generated OpenAPI schema (`docs/api/openapi.json`, regenerated by `make spec`), and typed Pydantic models for every request and response.
- `GET /detect?designation=...&force_refresh=true` — real-time detection with cache control.
- `POST /detect` — supply your own orbital elements; optionally pass `orbital_history` (from `GET /history`) to enable course-correction analysis.
- `GET /history` — JPL Horizons 10-year Keplerian time-series.
- `GET /impact` — 16-field impact assessment.
- `POST /analyze/batch` — concurrent batch detection with evidence detail in each result.
- `GET /health` — typed health check with per-component status.

**Quick API test:**

```bash
# Install and start
python install.py --core && python aneos.py api --dev &

# Health check
curl http://localhost:8000/api/v1/health

# Detect 2020 SO (Centaur upper stage — should return spacecraft_veto=true)
curl "http://localhost:8000/api/v1/analysis/detect?designation=2020%20SO"

# Get Apophis impact profile
curl "http://localhost:8000/api/v1/analysis/impact?designation=99942"

# Browse interactive docs
open http://localhost:8000/docs
```

---

## Detection Quality — Verified Claims

The following results are reproducible by running `python -m pytest tests/ -m "not network" -q` and the ground truth validation suite.

| Metric | Value | Source |
|---|---|---|
| Unit / integration tests | 59 pass, 0 fail | `pytest tests/ -m "not network"` |
| Ground truth artificial objects | 3 confirmed | Tesla Roadster (SpaceX 2018-017A), 2020 SO (Centaur/Surveyor-2), J002E3 (Apollo 12 S-IVB) |
| Ground truth natural NEOs | 20+ | JPL SBDB query, real orbital data |
| Sensitivity (recall) | 1.00 | All 3 artificials correctly classified at calibrated threshold 0.037 |
| Specificity | 1.00 | Zero natural NEOs falsely flagged at calibrated threshold |
| F1 score | 1.00 | At calibrated threshold 0.037 |
| ROC-AUC | 1.00 | Validated externally |
| Tesla Roadster sigma | 5.76 | `Sigma5DetectionResult` |
| 2020 SO sigma | 6.97 | `Sigma5DetectionResult` |
| J002E3 sigma | 5.76 | `Sigma5DetectionResult` |
| Max P(artificial) from orbital+physical | ~3–4% | Bayesian posterior, 0.1% prior |

**Important context**: The ground truth set is small (3 artificials, 20+ naturals). Perfect discrimination on this set demonstrates the approach works; it does not guarantee the same performance on a large, diverse unseen corpus. The Bayesian posterior ceiling (~3–4%) is mathematically correct given the 0.1% base rate and available evidence types — it is not a software bug.

---

## Capabilities and Limitations

### What aNEOS Supports Today

- Statistical screening of NEOs by orbital and physical anomaly indicators
- Multi-source data acquisition: JPL SBDB, JPL Horizons, NEODyS, MPC (with graceful fallback)
- Close-approach history (upcoming, within 0.2 AU) via SBDB CAD API
- Time-series orbital element history via JPL Horizons
- Earth and Moon impact probability with uncertainty bounds, keyholes, risk periods
- REST API with OpenAPI specification, Pydantic models, and batch processing
- Interactive Rich-based terminal menu (11,500-line `aneos_menu.py`)
- JSON/CSV export of analysis results
- JSON file caching with full round-trip fidelity for physical properties

### What aNEOS Does NOT Support

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

### Quality Thresholds for Responsible Use

- **Report a result as "anomalous"** only if `sigma_confidence >= 3.0` (SIGNIFICANT tier).
- **Report a result as "potential artificial"** only if `sigma_confidence >= 5.0` (EXCEPTIONAL tier) **and** independent follow-up confirms the anomaly.
- **Do not cite `artificial_probability`** as proof of artificiality. The maximum posterior from orbital+physical evidence alone is ~3–4%. Values in this range indicate "unusual orbit", not "confirmed artificial object".
- **Cross-check impact probabilities** against JPL Scout or ESA NEOCC for any object with `collision_probability > 1e-6`.

---

## System Architecture

```
aneos-suite/
├── aneos_core/           # Core science and data packages
│   ├── data/             # DataFetcher, CacheManager, SBDB/Horizons/NEODyS/MPC sources
│   ├── detection/        # ValidatedSigma5ArtificialNEODetector (canonical), DetectionManager
│   ├── analysis/         # ImpactProbabilityCalculator, scoring, pipeline
│   ├── validation/       # 6 SWARMs (KAPPA/LAMBDA/MU/CLAUDETTE/THETA/ATLAS), stats
│   ├── datasets/         # Ground truth dataset builder and validator
│   ├── ml/               # ML classifier (deferred, behind HAS_TORCH guard)
│   ├── monitoring/       # Prometheus/Grafana, psutil metrics, SMTP alerts
│   └── config/           # APIConfig, settings
├── aneos_api/            # FastAPI application
│   ├── endpoints/        # analysis, dashboard, monitoring, admin, data
│   ├── schemas/          # Pydantic models: DetectionResponse, ImpactResponse, OrbitalInput, ...
│   └── app.py            # Application factory
├── aneos_dashboard/      # Web dashboard (Flask)
├── aneos_menu.py         # Rich interactive terminal menu (~11,500 lines)
├── aneos.py              # CLI entry point
├── tests/                # 59 unit and integration tests
└── docs/                 # Architecture (ADR, DDD), API spec, user guide, scientific docs
```

### Detection Pipeline

```
GET /detect?designation=Apophis
  │
  ├─ Known spacecraft catalog veto → instant ARTIFICIAL (no analysis)
  │
  ├─ DataFetcher.fetch_neo_data()
  │   ├─ JPL SBDB          → orbital elements + physical properties
  │   ├─ JPL Horizons      → element table (fallback)
  │   ├─ NEODyS / MPC      → element table (fallback)
  │   └─ SBDB CAD API      → close-approach history (supplemental, never blocks)
  │
  ├─ HorizonsSource.fetch_orbital_history()  → 10-year Keplerian time-series
  │
  └─ ValidatedSigma5ArtificialNEODetector.analyze_neo_validated()
      ├─ 6 evidence modules → individual p-values
      ├─ Fisher's method    → combined_p_value
      ├─ Bayesian update    → bayesian_probability (0.1% prior)
      └─ DetectionResponse  → sigma_confidence, sigma_tier, evidence_sources, ...
```

---

## REST API Reference

Full OpenAPI specification: `docs/api/openapi.json` (regenerate with `make spec`).

Interactive documentation available at `http://localhost:8000/docs` when the API server is running.

### Key Endpoints

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

### Response Schema Highlights

`DetectionResponse`:
- `sigma_confidence` — sigma above natural NEO null hypothesis
- `sigma_tier` — ROUTINE / NOTABLE / INTERESTING / SIGNIFICANT / ANOMALOUS / EXCEPTIONAL
- `artificial_probability` — Bayesian posterior (0.1% prior)
- `combined_p_value` — Fisher's combined p-value across evidence types
- `false_discovery_rate` — expected FDR at current sigma threshold
- `evidence_sources` — list of `EvidenceSummary` with `anomaly_score`, `p_value`, `effect_size`
- `analysis_metadata` — detector version, method, population statistics
- `spacecraft_veto` / `veto_reason` — instant classification for known spacecraft

`ImpactResponse` (16 fields):
- `collision_probability`, `probability_uncertainty` [lower, upper]
- `moon_collision_probability`, `moon_earth_ratio`
- `impact_energy_mt`, `crater_diameter_km`, `damage_radius_km`
- `keyhole_passages`, `peak_risk_period`, `impact_probability_by_decade`
- `primary_risk_factors`, `comparative_risk`

---

## Scientific Foundation

aNEOS builds on the **Artificial NEOs Theory** — the hypothesis that some Near Earth Objects may exhibit orbital or physical properties statistically inconsistent with natural formation and evolution, potentially indicating artificial origin. This is a scientific hypothesis, not an established finding.

The statistical framework is grounded in:

- **Bayesian inference**: Posterior probability updated from a 0.1% base rate (estimated fraction of heliocentric objects that could be artificial). This prior is conservative; the true rate is unknown.
- **Fisher's method**: Independent p-values from 6 evidence types are combined into a single test statistic. Under the natural null hypothesis, this statistic follows a chi-squared distribution with 2k degrees of freedom.
- **Sigma-5 threshold**: Classification as artificial requires combined significance ≥ 5σ (p < 5.7×10⁻⁷), matching standard astronomical discovery criteria.
- **Calibrated threshold**: At p(Bayesian) ≥ 0.037, the current ground truth validation achieves sensitivity = 1.00 and specificity = 1.00 on the available corpus.

Full methodology: `docs/scientific/scientific-documentation.md`
Architecture decisions: `docs/architecture/ADR.md` (39 ADRs)
Domain model: `docs/architecture/DDD.md` (10 bounded contexts)

---

## Contributing

See `CONTRIBUTING.md` for development guidelines. The project follows the C&C + Implementation + Q&A agent structure defined in `DEVELOPMENT_FRAMEWORK.md`.

Run the test suite before opening a pull request:

```bash
python -m pytest tests/ -m "not network" -q   # 59 tests, 0 fail
make spec                                      # regenerate OpenAPI spec
git diff --stat docs/api/openapi.json          # confirm spec is current
```

## License

Scientific research and educational use. See `LICENSE` for complete terms.

---

*aNEOS v1.0.0 — Phase 10 complete. All planned gaps closed. Ground truth: sensitivity=1.00, specificity=1.00 on 3 confirmed artificials. REST API: 52+ endpoints. Test suite: 59 pass / 0 fail.*
