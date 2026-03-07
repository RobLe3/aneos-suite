# aNEOS Profession Guide

_Version: 1.0.0 | Last updated: 2026-03-08_

This guide explains what aNEOS provides, and what it does not provide, for each professional
context. Every capability listed here is implemented and verified in the test suite
(59 tests pass / 0 fail as of Phase 10).

---

## Planetary Defense Scientists (NASA PDCO / ESA NEOCC)

### What aNEOS gives you

**Secondary anomaly screening** for objects in existing catalogues. You submit a designation
and aNEOS fetches orbital elements from JPL SBDB + Horizons, augments them with close-approach
data from SBDB CAD API, retrieves a 10-year Keplerian history from Horizons, and runs a
six-evidence-stream Sigma-5 detector.

The `DetectionResponse` you receive includes:

- `sigma_confidence` — how many sigma above the natural NEO population null hypothesis
- `sigma_tier` — ROUTINE / NOTABLE / INTERESTING / SIGNIFICANT / ANOMALOUS / EXCEPTIONAL
- `artificial_probability` — Bayesian posterior (0.1% base rate prior)
- `combined_p_value` — Fisher's combined p-value across all evidence types
- `false_discovery_rate` — expected FDR at the current threshold
- `evidence_sources` — per-evidence anomaly scores, p-values, effect sizes, confidence intervals
- `analysis_metadata` — detector version, population reference statistics

**Impact assessment** via `GET /impact` returns:

- `collision_probability` with `probability_uncertainty` [lower, upper]
- `moon_collision_probability` and `moon_earth_ratio`
- `keyhole_passages` — resonant return windows
- `peak_risk_period` — decade of highest probability
- `impact_energy_mt`, `crater_diameter_km`, `damage_radius_km`
- `primary_risk_factors` — human-readable scientific rationale

### What aNEOS does NOT give you

- Authoritative orbit determination. aNEOS uses whatever elements are in JPL SBDB. For new
  discoveries with short arcs, use your own orbit solution.
- Torino or Palermo scale ratings. These require authoritative Monte Carlo orbit propagation
  over decades; aNEOS uses a simplified analytical model.
- Radar, spectral, or photometric data processing.
- Real-time alert ingestion from MPC discovery feeds.

### Verified performance

On the ground truth set (3 confirmed heliocentric spacecraft + 20+ natural NEOs from JPL SBDB):
sensitivity = 1.00, specificity = 1.00, F1 = 1.00 at calibrated threshold 0.037.
This is a small corpus — treat it as a proof of concept.

### Example workflow

```bash
# Screen a potentially unusual object
curl "http://localhost:8000/api/v1/analysis/detect?designation=2024%20YR4"

# Force fresh data (bypass cache)
curl "http://localhost:8000/api/v1/analysis/detect?designation=2024%20YR4&force_refresh=true"

# Full impact profile
curl "http://localhost:8000/api/v1/analysis/impact?designation=2024%20YR4"

# Batch screen a list of new discoveries
curl -X POST http://localhost:8000/api/v1/analysis/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"designations": ["2024 YR4", "2024 AB1", "Apophis"]}'
```

---

## Astronomers / Observational Astronomers

### What aNEOS gives you

- **Orbital anomaly screening** from your own reduced elements. Use `POST /detect` to submit
  an object's semi-major axis, eccentricity, and inclination without going through a full
  catalogue lookup. Optionally include `diameter_km`, `albedo`, and `orbital_history` from
  `GET /history`.

- **10-year orbital history** from JPL Horizons via `GET /history?designation=...`. Returns
  one Keplerian element set per year: `epoch`, `a`, `e`, `i`, `node`, `peri`, `M`.

- **Close-approach table** for any named object: fetched automatically when you call
  `GET /detect`; also available by parsing the `CloseApproach` records returned in the raw
  data endpoint `GET /api/v1/data/neo/{designation}`.

### What aNEOS does NOT give you

- Astrometric reduction, photometry pipeline, or spectral classification.
- Access to observation archives (IRSA, ESO archive, etc.).
- Survey scheduling or target-of-opportunity triggering.

### Example: check an object you observed that seems unusual

```python
import requests

# You reduced the orbit from your own observations
resp = requests.post("http://localhost:8000/api/v1/analysis/detect", json={
    "a": 2.31,
    "e": 0.41,
    "i": 6.2,
    "diameter_km": 0.008,
    "designation": "my_object_2026_A1",
})
r = resp.json()
print(f"Sigma: {r['sigma_confidence']:.2f}  Tier: {r['sigma_tier']}")
print(f"P(artificial): {r['artificial_probability']*100:.3f}%")
for e in r['evidence_sources']:
    print(f"  {e['type']:30s} score={e['anomaly_score']:.3f}  p={e['p_value']:.4f}")
```

Interpretation: if `sigma_tier` is SIGNIFICANT or above (σ ≥ 3.0), the orbit is statistically
unusual relative to the known NEO population and may warrant additional observations.
This does not imply artificial origin.

### Example: get a 10-year history and pass it to the detector

```python
import requests

# Step 1: get history
hist = requests.get("http://localhost:8000/api/v1/analysis/history",
                    params={"designation": "Apophis", "years": 10}).json()

# Step 2: detect with history (enables course-correction analysis)
resp = requests.post("http://localhost:8000/api/v1/analysis/detect", json={
    "a": 0.9223,
    "e": 0.1912,
    "i": 3.337,
    "designation": "Apophis",
    "orbital_history": hist["points"],
})
print(resp.json()["sigma_confidence"])
```

---

## Astrophysicists (Statistical / Orbital Mechanics Research)

### What aNEOS gives you

- **Transparent Bayesian framework**: Every detection step is auditable Python code. The
  canonical detector is `aneos_core/detection/validated_sigma5_artificial_neo_detector.py`.
  Fisher's method combines p-values across 6 evidence streams. Bayesian updating uses a
  0.001 (0.1%) base prior.

- **Reproducible ground truth dataset**: `aneos_core/datasets/ground_truth_dataset_preparation.py`
  builds a labelled corpus from SBDB and known heliocentric spacecraft. The dataset and
  validation results are reproducible from a fresh clone.

- **Uncertainty quantification** in impact assessment: `probability_uncertainty` [lower, upper],
  `calculation_confidence`, `impact_probability_by_decade`.

- **`analysis_metadata`** in every `DetectionResponse`: detector version, population statistics
  used for z-score normalisation, Fisher combination parameters.

### Statistical properties to understand before using

| Observation | Sigma | P(artificial) |
|---|---|---|
| Tesla Roadster (orbital+physical) | 5.76 | ~3.7% |
| 2020 SO (orbital+physical) | 6.97 | ~3.7% |
| Typical Aten/Apollo/Amor NEO | 1.0–2.5 | ~0.1–0.2% |

The Bayesian posterior is bounded by the base prior. From orbital + physical evidence alone,
the maximum posterior is ~3–4%. This is not a bug; it reflects the evidence available.
To exceed 10%, propulsion signatures or direct observation of course corrections are needed.
No automated source for these exists.

The σ value measures rarity under the natural NEO null hypothesis. σ = 5 means the observation
has probability < 5.7×10⁻⁷ if the object is natural. It does not mean P(artificial) = 99.99994%.

### Extending the framework

The detection pipeline is modular. To add a new evidence type:

1. Implement a new method in `ValidatedSigma5ArtificialNEODetector` that returns an
   `EvidenceSource` dataclass with `evidence_type`, `anomaly_score`, `p_value`,
   `quality_score`, `effect_size`, `confidence_interval`, `sample_size`.
2. Add the result to the `evidence_sources` list before the Fisher combination step.
3. Add a test in `tests/test_validated_sigma5_detector.py`.

---

## Astrodynamicists / Mission Planners

### What aNEOS gives you

**Impact risk timeline** for planning deflection or redirection missions:

- `peak_risk_period` — [start_year, end_year] of highest impact probability
- `impact_probability_by_decade` — time-resolved probability evolution
- `keyhole_passages` — list of close-approach windows that could amplify impact probability
  via gravitational resonance
- `time_to_impact_years` — most probable impact time from present epoch

**Earth vs Moon risk comparison**:

- `moon_earth_ratio` — if < 1.0, Moon impact is more likely than Earth impact
- `moon_collision_probability` — lunar cross-section collision probability
- Critical for protecting future lunar infrastructure (Gateway, Artemis landing sites)

**Artificial object uncertainty flag**: when `is_artificial = True` or
`artificial_probability > 0.037`, the response includes `artificial_object_considerations`
indicating that trajectory uncertainty from potential propulsion is not modelled.

### What aNEOS does NOT give you

- High-fidelity trajectory propagation (JPL Horizons or GMAT for that).
- Deflection efficiency calculations.
- Launch window analysis.

### Example: screen a candidate for kinetic impactor mission

```python
import requests

r = requests.get("http://localhost:8000/api/v1/analysis/impact",
                 params={"designation": "99942"})
d = r.json()

print(f"P(Earth impact) = {d['collision_probability']:.3e}")
print(f"P(Moon impact)  = {d['moon_collision_probability']:.3e}")
print(f"Moon/Earth ratio = {d['moon_earth_ratio']:.2f}")
print(f"Peak risk decade: {d['peak_risk_period']}")
print(f"Keyhole passages: {len(d['keyhole_passages'])}")
print(f"Impact energy: {d['impact_energy_mt']:.0f} MT TNT")
print(f"Crater: {d['crater_diameter_km']:.2f} km diameter")
print(f"Damage radius: {d['damage_radius_km']:.1f} km")
print("\nPrimary risk factors:")
for f in d['primary_risk_factors']:
    print(f"  • {f}")
print("\nProbability by decade:")
for decade, p in d['impact_probability_by_decade'].items():
    print(f"  {decade}: {p:.3e}")
```

---

## SETI / Technosignature Researchers

### What aNEOS gives you

- The first open-source implementation of a multi-modal statistical test specifically designed
  to detect orbital and physical properties inconsistent with the natural NEO population.

- A calibrated interpretation framework that separates "statistically unusual" from "possibly
  artificial" — avoiding the common error of equating high sigma with high probability of
  artificial origin.

- Six independent anomaly dimensions: orbital dynamics, physical properties, trajectory
  (close-approach pattern), temporal pattern, statistical anomaly, behavioral pattern.
  Each contributes an independent p-value; Fisher's method combines them.

- `analysis_metadata` provides complete traceability: detector version, population reference
  statistics, Fisher combination parameters — sufficient for a methods section.

### The hypothesis and its current status

**Hypothesis**: Some Near Earth Objects exhibit orbital or physical characteristics
statistically inconsistent with natural formation and evolution processes.

**Current evidence**: The detector successfully identifies all 3 confirmed heliocentric
spacecraft in the ground truth set at σ ≥ 5.76 with P(artificial) ≈ 3.7%. All 20+ natural
NEOs score < 3.0σ with P(artificial) ≈ 0.1–0.2%. Score separation is clear and statistically
significant on this corpus.

**What this does NOT prove**: No natural NEO in any public catalogue has yet produced a
σ ≥ 5.0 detection through aNEOS. The hypothesis is untested against the full NEO population.
The 3.7% Bayesian posterior for the known spacecraft reflects the prior (0.1%) updated by
orbital+physical evidence — it is not "99.99994% confident artificial".

**What would change the picture**: A sustained observation campaign providing propulsion burn
vectors, non-gravitational accelerations, or radar cross-section anomalies inconsistent with
the object's brightness. These could push the posterior above 50%.

### Sigma tier interpretation

| Tier | Sigma | Interpretation |
|---|---|---|
| ROUTINE | < 1.0 | Indistinguishable from typical NEO population |
| NOTABLE | 1.0–2.0 | Slightly unusual; note but no action |
| INTERESTING | 2.0–3.0 | Worth logging; continue monitoring |
| SIGNIFICANT | 3.0–4.0 | Anomalous; consider follow-up observations |
| ANOMALOUS | 4.0–5.0 | Highly unusual; follow-up recommended |
| EXCEPTIONAL | ≥ 5.0 | Extreme rarity; cross-check all data; independent verification required |

---

## Software Developers / API Integrators

### What aNEOS gives you

A fully typed FastAPI application with:

- **52+ REST endpoints** across 6 groups (analysis, dashboard, monitoring, admin, data, auth)
- **OpenAPI 3.0 specification** at `docs/api/openapi.json` (regenerate with `make spec`)
- **Pydantic v2 models** for all requests and responses — IDE autocompletion works out of the box
- **Interactive docs** at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`
- **Batch processing** with concurrent `ThreadPoolExecutor` under the hood
- **Cache with force-refresh**: `GET /detect?force_refresh=true` bypasses the JSON file cache

### Quick integration test

```bash
# Start server
python aneos.py api --dev &

# Health check
curl -s http://localhost:8000/api/v1/health | python -m json.tool

# Known spacecraft (instant veto, no analysis)
curl -s "http://localhost:8000/api/v1/analysis/detect?designation=2020%20SO" | python -m json.tool

# Supply your own elements
curl -s -X POST http://localhost:8000/api/v1/analysis/detect \
  -H "Content-Type: application/json" \
  -d '{"a": 1.91, "e": 0.33, "i": 11.4, "designation": "test_obj"}' \
  | python -m json.tool

# 10-year history
curl -s "http://localhost:8000/api/v1/analysis/history?designation=Apophis&years=10" \
  | python -m json.tool

# Batch detection
curl -s -X POST http://localhost:8000/api/v1/analysis/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"designations": ["Apophis", "Bennu", "Ryugu"]}' \
  | python -m json.tool
```

### What aNEOS does NOT provide (for production deployment)

- Production JWT. The current auth is mock tokens. Wire a real OIDC provider before exposing
  the API publicly.
- Rate limiting or DDoS protection. Add an API gateway (nginx, Traefik) in front.
- Database persistence. Analysis results are held in memory (`_analysis_cache`); they are lost
  on restart. A SQLAlchemy 2.0 ORM is present but not wired to the analysis cache.
- Horizontal scaling. The in-memory `_batch_store` and `_analysis_cache` are process-local.
  Use Redis-backed caching for multi-process deployments.
