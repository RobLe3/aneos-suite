# aNEOS — Current State Summary

**Version**: 1.2.2 (Phase 24)
**Date**: 2026-03-11
**Test suite**: 360 pass / 0 fail

---

## What aNEOS Is

aNEOS is an open-source Python research platform with two missions:

1. **Artificial NEO Detection** — statistical screening of Near Earth Objects for orbital and physical signatures inconsistent with the natural asteroid population.
2. **Planetary Defense Assessment** — Earth and Moon impact probability with energy, crater, damage radius, and risk-period estimates.

It is a **research tool**, not an operational system. Results require independent verification.

### Long-Term Goal

The long-term scientific goal of aNEOS is to address the **Fermi Paradox** empirically. The paradox rests partly on the assumption that no engineered objects exist in our solar system -- but that assumption has never been systematically tested against the NEO catalogue with a rigorous statistical framework. aNEOS is built to perform that test.

If the complete NEO population screens as consistent with natural dynamics, that is a quantified null result. If even one object survives the full validation pipeline and all natural explanations are ruled out by follow-up observation, it constitutes direct physical evidence that changes the terms of the paradox entirely.

See `docs/scientific/theory.md` Section 7 for the full scientific argument.

---

## What Works Right Now (v1.2.2)

### Detection
- Fetches live orbital data from JPL SBDB, JPL Horizons, NEODyS, and MPC
- Runs 6-evidence Sigma-5 multi-modal detector (orbital, physical, trajectory, temporal, statistical, behavioral)
- Bayesian probability with 0.1% base-rate prior → posterior typically 1–4%
- Instant spacecraft veto for known heliocentric spacecraft (Tesla Roadster, 2020 SO, J002E3)
- Single-object, multi-evidence, batch (concurrent), and orbital history analysis
- Physical indicators: diameter and albedo anomaly detection active in detection path

### Impact Assessment
- Earth and Moon collision probability with uncertainty bounds
- Impact energy (megatons TNT), crater diameter, damage radius
- Gravitational keyhole analysis, peak risk decade, probability-by-decade time-series
- Uses real observation arc length from SBDB for improved probability estimates

### Population Analysis (BC11)
- 200-year historical close-approach polling via SBDB CAD API (40 × 5-year chunks)
- DBSCAN/HDBSCAN orbital clustering with density σ-gate
- Synodic harmonic analysis (Lomb-Scargle)
- Non-gravitational acceleration correlation (A2 Pearson r)
- Fisher's method + Bonferroni correction for network-level sigma; Stouffer weighted z-score optional

### REST API (52+ endpoints)
- `GET /detect`, `POST /detect` — Sigma-5 detection by name or by supplied orbital elements
- `GET /impact` — 16-field impact assessment
- `GET /history` — 10-year Keplerian element time-series from Horizons
- `POST /analyze/batch` — concurrent batch with evidence detail
- `GET /health` — typed per-component health check
- `POST /api/v1/auth/token` — JWT bearer token endpoint
- `GET /api/v1/analysis/enhanced/summary` — DB-backed validation statistics
- Server-Sent Events streaming for real-time alerts and system status
- Full OpenAPI/Swagger UI at `/docs`

### Terminal UI
- 15-option Rich interactive menu (`python aneos.py`)
- Rich progress bars on all batch and polling operations
- File browser for designation list input
- Interactive results browser (pick by number for full detail)
- Options 12–15: API server launch, session analytics, scientific help viewer, system health

### Infrastructure
- Pydantic v2 input validation: `OrbitalInput` bounds, designation length limits
- 360 automated tests (unit + integration, property-based via Hypothesis)
- CI gate at 40% coverage
- shelve-based CAD cache (24-hour TTL); 7-day TTL for orbital history

---

## Known Limitations

| Area | Status |
|------|--------|
| Ground truth corpus | N=4 confirmed artificials — F1=1.00 on this set is proof-of-concept only |
| Bayesian posterior ceiling | ~3–4% max from orbital+physical alone; propulsion data needed to exceed 10% |
| ML classifier | Exists but not wired into default detection path (G-015 deferred) |
| Production JWT | Dev-mode mock tokens only; real OIDC needed for public deployment |
| Radar, thermal, Gaia SWARMs | Implemented but require live instrument data feeds not currently available |
| Propulsion/manoeuvre detection | No automated data source exists anywhere |
| PostgreSQL | SQLite-only for research; schema tested only with SQLite |

---

## How to Run

```bash
# Interactive menu
python aneos.py

# REST API (dev mode)
python aneos.py api --dev
# → http://localhost:8000/docs

# Tests
python -m pytest tests/ aneos_core/tests/ -m "not network" -q
# Expected: 360 passed, 0 failed
```

### Menu Options at a Glance

| # | Option | What it does |
|---|--------|-------------|
| 1 | Detect NEO (single) | Sigma-5 detection for one object by designation |
| 2 | Multi-Evidence Analysis | Full 6-indicator breakdown with p-values and effect sizes |
| 3 | Batch Detection | Screen a file of designations with a progress bar |
| 4 | Orbital History | 10-year Keplerian time-series from Horizons |
| 5 | Impact Probability | Earth/Moon collision probability + energy + crater |
| 6 | Close Approach History | Upcoming passes within 0.2 AU |
| 7 | Live Pipeline Dashboard | 200-year historical poll → ATLAS → validation funnel |
| 8 | Population Pattern Analysis | Clustering, harmonics, non-grav correlation on a set |
| 9 | Browse Results | Session and database results browser |
| 10 | Export Results | JSON or CSV export of current session |
| 11 | System Health | 8-component health check + API connectivity |
| 12 | Launch API Server | Start FastAPI server in subprocess |
| 13 | Session Analytics | σ-tier breakdown + JSON export of session |
| 14 | Scientific Help | In-terminal viewer for `docs/scientific/` |
| 15 | Exit | — |

---

## Architecture in One Page

```
aneos.py (CLI)
    └─ aneos_menu_v2.py (15-option Rich menu)
           │
           ├─ aneos_core/
           │   ├─ data/          DataFetcher → SBDB + Horizons + NEODyS + MPC
           │   ├─ detection/     ValidatedSigma5ArtificialNEODetector (canonical)
           │   │                 DetectionManager (registry, priority-ordered)
           │   ├─ analysis/      ImpactProbabilityCalculator, ATLAS scoring
           │   ├─ validation/    6 SWARMs: KAPPA/LAMBDA/MU/CLAUDETTE/THETA/ATLAS
           │   ├─ pattern_analysis/  BC11: clustering, harmonics, correlation
           │   ├─ pipeline/      AutomaticReviewPipeline (200-year funnel)
           │   └─ ml/            RandomForest classifier (deferred)
           │
           └─ aneos_api/ (FastAPI)
               ├─ endpoints/   analysis, impact, monitoring, admin, streaming, auth
               ├─ schemas/     Pydantic models (DetectionResponse, ImpactResponse, …)
               └─ app.py       Application factory + lifespan management
```

Detection flow:
```
designation  →  spacecraft veto?  yes → ARTIFICIAL VALIDATED (instant)
              ↓ no
              DataFetcher  →  orbital elements + physical properties
              ↓
              ValidatedSigma5Detector
              ├─ 6 indicators → individual p-values
              ├─ Fisher's method → combined_p
              └─ Bayesian update (0.1% prior) → P(artificial)
```

---

## Key File Locations

| Purpose | File |
|---------|------|
| CLI entry | `aneos.py` |
| Primary menu | `aneos_menu_v2.py` |
| Shared UI helpers | `aneos_menu_base.py` |
| Canonical detector | `aneos_core/detection/validated_sigma5_artificial_neo_detector.py` |
| Detection manager | `aneos_core/detection/detection_manager.py` |
| Data fetcher | `aneos_core/data/fetcher.py` |
| Impact calculator | `aneos_core/analysis/impact_probability.py` |
| Main pipeline | `aneos_core/pipeline/automatic_review_pipeline.py` |
| FastAPI app factory | `aneos_api/app.py` |
| API schemas | `aneos_api/schemas/` |
| Tests | `tests/` + `aneos_core/tests/` |
| Architecture decisions | `docs/architecture/ADR.md` (60 ADRs) |
| Domain model | `docs/architecture/DDD.md` (11 bounded contexts) |
| Scientific methodology | `docs/scientific/scientific-documentation.md` |
| Honest caveats | `docs/scientific/VALIDATION_INTEGRITY.md` |

---

## Validation Claims (Honest)

| Metric | Value | Caveats |
|--------|-------|---------|
| Test suite | 360 pass / 0 fail | |
| Ground truth artificials | 4 (Tesla Roadster, 2020 SO, J002E3, WT1190F) | Small set |
| Sensitivity on GT set | 1.00 | N=4 only |
| Specificity on GT set | 1.00 | N≈20 naturals |
| F1 on GT set | 1.00 | **Not a production guarantee** — threshold hand-tuned to same objects |
| Max P(artificial) from orbital+physical | ~3–4% | Posterior bounded by 0.1% prior; propulsion data needed for more |
| σ threshold for discovery claim | ≥ 5.0 | Matches standard astronomical convention |

---

*Summary generated 2026-03-11. For full methodology, see `docs/scientific/scientific-documentation.md`.*
