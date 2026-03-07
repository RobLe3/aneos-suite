# aNEOS — User Expectation, Logic & Outcome (UELO) Analysis

**Version**: 1.0
**Date**: 2026-03-07
**Purpose**: Map user personas, journeys, and expected outcomes to the current system's
actual delivery. This document is the precursor to the Phase 8 gap analysis.

---

## 1. User Personas

### P1 — The Anomaly Researcher
Academic or independent researcher studying the Artificial NEOs Theory.

**Profile**
- Has astronomical background; understands σ-levels and Bayesian inference
- Wants to investigate specific objects by designation (e.g., "2020 SO", "J002E3")
- Needs clear separation between *statistical rarity* and *artificial probability*
- Compares results across multiple objects over time
- May publish or present results — needs reproducibility guarantees

**What they bring**: A NEO designation or a list of designations
**What they need**: Evidence breakdown, probability with base-rate context, classification rationale

---

### P2 — The Planetary Defense Analyst
Scientist or policy analyst assessing Earth/Moon impact risk.

**Profile**
- Focused on the secondary mission (impact probability)
- Needs quantitative outputs: P(Earth impact), P(Moon impact), energy in MT TNT, crater diameter
- Requires temporal risk evolution — "when is the peak risk window?"
- May need to justify risk levels to non-technical stakeholders

**What they bring**: NEO designation + analysis parameters
**What they need**: Earth/Moon collision probability, damage estimates, confidence intervals, risk narrative

---

### P3 — The System Integrator / Developer
Technical user building on top of aNEOS via the REST API.

**Profile**
- Consumes `GET /detect` and `POST /analyze` programmatically
- Expects typed, consistent JSON responses (no null surprises, no field name drift)
- Needs OpenAPI spec to be accurate and complete
- May run batches of hundreds of objects

**What they bring**: API calls with designations
**What they need**: Stable contract (response_model), predictable error shapes, working batch endpoint

---

### P4 — The Casual Explorer / Enthusiast
Non-expert user of the interactive CLI menu.

**Profile**
- Not fluent in orbital mechanics; wants human-readable outputs
- Uses `python aneos.py` and the Rich menu interface
- Needs the system to explain its own results ("what does σ=2.1 mean?")
- Doesn't know what "equinoctial elements" are

**What they bring**: Curiosity + a NEO name they read about
**What they need**: Plain-English interpretation, clear classification, "so what" statement

---

## 2. Core User Journeys

### J1 — Single Object Artificial NEO Analysis (P1, P4)

```
User input:  "Analyze 2020 SO for artificial signatures"

Step 1 — DATA:     Fetch orbital elements from SBDB / NEODyS / MPC / Horizons
Step 2 — SCORE:    Run 18 indicators across 6 categories
Step 3 — DETECT:   ValidatedSigma5 detector → bayesian_probability + sigma_confidence
Step 4 — EXPLAIN:  Map evidence_sources to plain-English rationale
Step 5 — DISPLAY:  Present σ-level, P(artificial), classification, evidence breakdown

Expected output:
  Object: 2020 SO
  Classification: ARTIFICIAL (confirmed spacecraft)
  σ-confidence: 3.2 (rarity under natural NEO null hypothesis)
  Artificial probability: 3.7%  ← 0.1% base prior × evidence
  Interpretation: "Statistically rare orbit but low absolute probability of
                  artificial origin given base rate. Propulsion/maneuver
                  evidence would increase this substantially."
  Evidence: [lunar transfer orbit, high albedo, non-gravitational accel detected]
```

**Current delivery**: Menu paths `[2] Detection Analysis`, `[1] Single Object`
**Gap**: Interpretation is shown but may not clearly distinguish σ from P(artificial);
         `evidence_sources` breakdown not always surfaced to user.

---

### J2 — Impact Probability Assessment (P2)

```
User input:  "What is the impact risk for 2024 YR4?"

Step 1 — DATA:     Fetch orbital elements + close approaches (CAD API)
Step 2 — CALC:     ImpactProbabilityCalculator → Earth/Moon probabilities
Step 3 — SCALE:    Compute energy (MT TNT), crater diameter, damage radius
Step 4 — TEMPORAL: Derive peak risk period from approach timeline
Step 5 — NARRATIVE: Assign risk tier (NEGLIGIBLE / LOW / MEDIUM / HIGH / CRITICAL)

Expected output:
  Earth P(impact): 8.52×10⁻¹⁵  Risk tier: NEGLIGIBLE
  Moon  P(impact): 6.65×10⁻⁹   (3.4× more likely)
  Peak risk window: [date range]
  Impact energy: 8,863 MT TNT  Crater: ~12 km
  Narrative: "Negligible Earth impact risk. Moon impact unlikely but
              not zero — relevant for lunar infrastructure planning."
```

**Current delivery**: Menu path `[3] Impact Assessment`; ImpactProbabilityCalculator implemented
**Gap**: Temporal risk evolution not always connected to close-approach data; risk tier narrative
         not always shown; Moon output conditional on orbit type.

---

### J3 — REST API Programmatic Detection (P3)

```
Request:  GET /detect?designation=2020+SO
Response: {
  "designation": "2020 SO",
  "is_artificial": true,
  "artificial_probability": 0.037,
  "sigma_confidence": 3.2,
  "classification": "ARTIFICIAL",
  "confidence": 0.64,
  "evidence_count": 5,
  "interpretation": "sigma_confidence = rarity under natural NEO null hypothesis..."
}
```

**Current delivery**: `GET /detect` endpoint added Phase 7; `DetectionResponse` in OpenAPI spec
**Gap**: `confidence = sigma_confidence / 5.0` is an approximation not documented in spec;
         no authentication bypass for research use; no pagination or batch endpoint returning typed results.

---

### J4 — Batch Multi-Object Analysis (P1, P3)

```
Request:  POST /analyze/batch  { "designations": ["2020 SO", "J002E3", "Apophis"] }
Response: { "batch_id": "...", "status": "processing" }

Follow-up: GET /batch/{id}/status → results list
```

**Current delivery**: Endpoint exists; always returns mock status "completed" with 0 results
**Gap**: Batch processing is entirely non-functional end-to-end; results never populated.

---

### J5 — System Health & Data Freshness (P3, P4)

```
Request:  GET /health
Response: { "status": "healthy", "components": {...}, "data_freshness": {...} }
```

**Current delivery**: `/health` endpoint exists; Redis check, SBDB ping, NEODyS probe
**Gap**: `HealthResponse` schema not wired as `response_model`; data freshness timestamp
         not exposed; no indication of when cache was last populated.

---

## 3. Expectation → Logic → Outcome Mapping

| # | User Expectation | Internal Logic | Delivered Outcome | Delta |
|---|-----------------|----------------|-------------------|-------|
| E-01 | "Classify this object" | 18 indicators → Bayesian → DetectionResult | sigma_confidence + artificial_probability | ✅ works; display clarity improvable |
| E-02 | "Why is it classified this way?" | evidence_sources list in DetectionResult | evidence_count integer only in API response | ⚠ evidence_sources not exposed in API |
| E-03 | "What is the impact probability?" | ImpactProbabilityCalculator | P(Earth), P(Moon), energy, crater | ✅ menu; ❌ no REST endpoint |
| E-04 | "Is this object natural or spacecraft?" | Spacecraft veto (THETA), TLE cross-reference | Veto flag + rationale | ⚠ veto result not surfaced in API |
| E-05 | "Show me the orbital history" | Horizons ELEMENTS API → JPL Horizons | Keplerian elements table | ✅ menu only |
| E-06 | "Run analysis on 100 objects" | POST /analyze/batch | batch_id only; results never returned | ❌ broken end-to-end |
| E-07 | "What data source was used?" | DataFetcher priority chain | _source metadata in raw data | ⚠ not surfaced in DetectionResponse |
| E-08 | "How fresh is the data?" | cache TTL + fetched_at timestamp | Not exposed | ❌ missing |
| E-09 | "Export results to CSV/JSON" | ExportRequest → ExportResponse | Placeholder data only | ❌ non-functional |
| E-10 | "Is the API secure for research use?" | JWT auth mock (startup guard in dev) | Mock tokens in dev; blocks in prod | ⚠ blocks legitimate research API use |
| E-11 | "Tell me in plain English what this means" | Interpretation string in DetectionResponse | Static string; not object-specific | ⚠ generic, not contextualized |
| E-12 | "What's σ=2.1 vs σ=5.0 mean to me?" | sigma_confidence field | No human-readable risk tier for detection | ❌ missing tier/label |

---

## 4. Outcome Quality Dimensions

For each core output, what makes it *good enough* from the user's perspective:

### Detection Result Quality

| Dimension | Threshold | Current |
|-----------|-----------|---------|
| Correct classification | sens=1.00, spec=1.00 on ground truth | ✅ validated |
| Uncertainty is quantified | Bayesian CI or confidence range shown | ⚠ probability shown; CI not shown |
| Evidence is traceable | User can see which indicators fired | ⚠ count only in API |
| Interpretation is plain English | Non-expert can understand output | ⚠ generic static string |
| Result is reproducible | Same input → same output | ✅ deterministic detector |

### Impact Assessment Quality

| Dimension | Threshold | Current |
|-----------|-----------|---------|
| Earth impact probability | Scientific calculation with uncertainty | ✅ calculated |
| Moon impact included | Yes (integrated framework) | ✅ calculated |
| Risk tier label | NEGLIGIBLE / LOW / MEDIUM / HIGH | ⚠ in menu; not in API |
| Temporal peak | When is highest risk? | ⚠ partial via CAD data |
| Energy & crater | MT TNT + km | ✅ calculated |

### API Contract Quality

| Dimension | Threshold | Current |
|-----------|-----------|---------|
| Schema in OpenAPI spec | All endpoints have response_model | ⚠ partial (DetectionResponse wired; HealthResponse not) |
| No field name drift | Schema fields match internal model | ✅ sigma_confidence fixed in Phase 7 |
| Errors are typed | 404/500 have consistent shape | ⚠ raw HTTPException only |
| Batch works end-to-end | Results populated in follow-up GET | ❌ broken |

---

## 5. User-Facing Gap Seed for Phase 8

These are not engineering gaps (covered by GAP_ANALYSIS.md) but user-expectation gaps:

| UE-ID | Gap | Persona | Priority |
|-------|-----|---------|----------|
| UE-001 | Evidence sources (which indicators fired) not exposed in API DetectionResponse | P1, P3 | P1 |
| UE-002 | No impact probability REST endpoint (ImpactProbabilityCalculator not wired to API) | P2, P3 | P1 |
| UE-003 | Batch analysis entirely non-functional end-to-end | P1, P3 | P1 |
| UE-004 | No human-readable σ tier label ("ANOMALOUS", "NOTABLE", "ROUTINE") in detection output | P1, P4 | P2 |
| UE-005 | Data source and freshness not surfaced in API responses | P3 | P2 |
| UE-006 | HealthResponse not wired as response_model for /health | P3 | P2 |
| UE-007 | Interpretation string in DetectionResponse is generic (not object-specific) | P4 | P2 |
| UE-008 | No spacecraft veto result in API (THETA SWARM result hidden) | P1, P3 | P2 |
| UE-009 | Export (CSV/JSON) non-functional | P1, P3 | P3 |
| UE-010 | Auth blocks research API use in non-dev mode (no API-key-only path) | P3 | P3 |

---

## 6. Interplay Model

```
                    ┌─────────────────────────────────────────────┐
                    │                USER LAYER                   │
                    │  P1: Researcher  P2: Analyst  P4: Explorer  │
                    └─────────────┬───────────────────────────────┘
                                  │ expects
                    ┌─────────────▼───────────────────────────────┐
                    │            EXPERIENCE LAYER                  │
                    │  CLI Menu (P1,P4)   REST API (P3)           │
                    │  Plain-English interpretation required        │
                    └─────────────┬───────────────────────────────┘
                                  │ invokes
                    ┌─────────────▼───────────────────────────────┐
                    │             LOGIC LAYER                      │
                    │  DataFetcher → 18 Indicators → Detector     │
                    │  ImpactCalc → Earth/Moon probabilities       │
                    │  Bayesian calibration (0.1% base prior)      │
                    └─────────────┬───────────────────────────────┘
                                  │ produces
                    ┌─────────────▼───────────────────────────────┐
                    │            OUTCOME LAYER                     │
                    │  DetectionResult (sigma, probability, why)   │
                    │  ImpactResult (P(Earth), P(Moon), energy)    │
                    │  HealthStatus (components, freshness)        │
                    └─────────────────────────────────────────────┘
```

**Critical path for P1**: `DataFetcher → Indicators → Detector → DetectionResponse (with evidence)`
**Critical path for P2**: `DataFetcher → CAD API → ImpactCalc → ImpactResponse`
**Critical path for P3**: `REST /detect → DetectionResponse | POST /impact → ImpactResponse`

The biggest current gap is that the **Outcome Layer** is partially disconnected from the
**Experience Layer** — specifically: evidence_sources, impact probability, and batch results
never reach the user through the API surface.

---

## 7. Proposed Phase 8 Focus

Based on this analysis, Phase 8 should prioritize closing the Experience↔Outcome gap:

```
Phase 8A — Evidence Exposure (UE-001, UE-004, UE-007)
  Add evidence_sources list to DetectionResponse
  Add sigma_tier label (ROUTINE/NOTABLE/ANOMALOUS/SIGNIFICANT/EXCEPTIONAL)
  Contextualize interpretation string with object name + top indicators

Phase 8B — Impact REST API (UE-002)
  POST /impact endpoint → ImpactResponse
  Wire ImpactProbabilityCalculator through DataFetcher → response_model
  ImpactResponse: {P_earth, P_moon, energy_mt, crater_km, risk_tier}

Phase 8C — Batch Fix (UE-003)
  POST /analyze/batch → stores results in SQLite analysis table
  GET /batch/{id}/status → returns paginated DetectionResponse list
  Use ThreadPoolExecutor for concurrent fetch + detect

Phase 8D — API Surface Completeness (UE-005, UE-006, UE-008)
  Wire HealthResponse as response_model for /health
  Add data_source + fetched_at to DetectionResponse
  Add spacecraft_veto bool + veto_reason to DetectionResponse
```

---

*This document complements ADR.md (decision record) and DDD.md (bounded contexts) by
providing the user-facing perspective. The next step is a Phase 8 gap analysis that
maps UE-IDs to specific code changes, test criteria, and acceptance conditions.*
