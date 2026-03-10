# Contributing to aNEOS

Thank you for your interest in aNEOS — an open-source Python research platform for
statistical screening of Near Earth Objects. This document covers everything you need
to contribute code, tests, documentation, or scientific methodology improvements.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Running the Test Suite](#running-the-test-suite)
4. [Making Changes](#making-changes)
5. [Architecture Decisions (ADR Process)](#architecture-decisions-adr-process)
6. [API Changes](#api-changes)
7. [Scientific Methodology Changes](#scientific-methodology-changes)
8. [Reporting Issues](#reporting-issues)

---

## Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/aneos-suite.git
cd aneos-suite

# 2. Install core dependencies (includes testing tools)
pip install -r requirements-core.txt

# 3. Verify the suite runs
python aneos.py            # opens 15-option interactive menu
python aneos.py api --dev  # starts REST API at http://localhost:8000

# 4. Run tests to confirm your environment is clean
python -m pytest tests/ aneos_core/tests/ -m "not network" -q
# Expected: 308 passed, 0 failed (277 in tests/ + 31 in aneos_core/tests/)
```

---

## Project Structure

```
aneos-suite/
├── aneos_core/           # Core science and data packages
│   ├── data/             # DataFetcher, CacheManager, SBDB/Horizons/NEODyS/MPC
│   ├── detection/        # ValidatedSigma5ArtificialNEODetector (canonical)
│   │                     # DetectionManager (registry pattern — ADR-011)
│   ├── analysis/         # ImpactProbabilityCalculator, ATLAS scoring, pipeline
│   │   └── advanced_scoring.py   # XVIII SWARM — 6-clue continuous scoring
│   ├── validation/       # KAPPA (radar), LAMBDA (thermal), MU (Gaia),
│   │                     # CLAUDETTE (stats), THETA (hardware), multi_stage_validator
│   ├── pipeline/         # AutomaticReviewPipeline — 200-year historical funnel
│   ├── polling/          # HistoricalChunkedPoller — 40×5-year CAD chunks
│   ├── pattern_analysis/ # BC11: clustering (PA-1), harmonics (PA-3),
│   │                     # correlation (PA-5), rendezvous (PA-6 Stage 1)
│   ├── datasets/         # Ground truth dataset builder and validator
│   ├── ml/               # ML classifier (deferred, behind HAS_TORCH guard)
│   ├── monitoring/       # Prometheus/psutil metrics
│   ├── config/           # APIConfig, settings
│   └── tests/            # Internal unit/integration tests
│       ├── unit/         # test_cache.py, test_config.py
│       └── integration/  # (skeleton — contributions welcome)
├── aneos_api/            # FastAPI application — 52+ REST endpoints
│   ├── endpoints/        # analysis, dashboard, monitoring, admin, data
│   └── schemas/          # Pydantic models: DetectionResponse, ImpactResponse, …
├── aneos_menu_v2.py      # 15-option Rich terminal menu (primary UI)
├── aneos_menu.py         # Legacy 121-option menu (--legacy-menu flag)
├── aneos_menu_base.py    # Shared UI helpers: progress bars, file browser
├── aneos.py              # CLI entry point
├── tests/                # pytest test suite (277 tests)
└── docs/
    ├── architecture/
    │   ├── ADR.md        # 59 Architecture Decision Records
    │   └── DDD.md        # 11 Bounded Contexts (Domain-Driven Design map)
    ├── scientific/
    │   ├── scientific-documentation.md
    │   └── VALIDATION_INTEGRITY.md   # Honest uncertainty audit
    └── api/
        └── openapi.json  # Auto-generated OpenAPI spec (regenerate with `make spec`)
```

---

## Running the Test Suite

```bash
# All non-network tests (CI baseline — must pass before any PR)
python -m pytest tests/ aneos_core/tests/ -m "not network" -q

# With coverage report
python -m pytest tests/ aneos_core/tests/ -m "not network" --cov=aneos_core --cov-report=term-missing

# Network tests (require live JPL/NASA APIs — run locally, not in CI)
python -m pytest -m "network" -v

# Integration tests only
python -m pytest -m "integration" -v

# Single file
python -m pytest tests/test_pipeline_scoring.py -v
```

### Test markers

| Marker | Meaning | CI? |
|--------|---------|-----|
| *(none)* | Fast, fully mocked | Yes |
| `network` | Requires live NASA/JPL APIs | No |
| `integration` | Slower, multi-component flows | Yes |

### Where to add new tests

| What you're testing | Where to add |
|--------------------|-------------|
| Single function / class | `tests/test_<module>.py` |
| Config / settings | `aneos_core/tests/unit/test_config.py` |
| Cache behaviour | `aneos_core/tests/unit/test_cache.py` |
| Multi-component flow | `aneos_core/tests/integration/` |
| Full CLI / menu flow | `tests/test_menu_v2_e2e.py` |
| Pipeline scoring logic | `tests/test_pipeline_scoring.py` |

### Property-based tests (Hypothesis)

`hypothesis` is installed via `requirements-core.txt`. Use it for orbital mechanics
invariants:

```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.0, max_value=0.99, allow_nan=False))
def test_natural_eccentricity_stays_inconclusive(e):
    """Any natural-eccentricity orbit without anomalies should score < SIGNIFICANT."""
    ...
```

---

## Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following existing code style (PEP 8).

3. **Run the full test suite** — it must pass with zero failures:
   ```bash
   python -m pytest tests/ aneos_core/tests/ -m "not network" -q
   ```

4. **If you changed an API endpoint**, regenerate the OpenAPI spec:
   ```bash
   make spec
   git add docs/api/openapi.json
   ```
   CI will fail with a clear message if the spec is stale.

5. **If your change affects architecture**, add or update an ADR (see below).

6. **Commit with a descriptive message** and open a pull request against `main`.

---

## Architecture Decisions (ADR Process)

aNEOS uses Architecture Decision Records to track every significant design choice.
Before adding a new component, data source, or detection method:

1. Read `docs/architecture/ADR.md` (currently 59 ADRs) — find the closest existing
   decision. If your change is within its scope, add an update note.
2. If it's a genuinely new decision, append a new ADR at the bottom:
   ```markdown
   ### ADR-060: Your Decision Title
   **Status**: Proposed / Accepted / Deferred
   **Context**: Why this decision was needed.
   **Decision**: What was decided.
   **Consequences**: (+) benefits; (-) tradeoffs.
   **Files**: affected source files
   ```
3. Also update `docs/architecture/DDD.md` if the change affects a Bounded Context.

Key ADRs to read before contributing:
- **ADR-008**: ATLAS scoring is canonical for full-data scenarios
- **ADR-011**: DetectionManager registry — how to add a new detector
- **ADR-050**: No silent fallback — errors must be explicit
- **ADR-053**: Physical indicators are not available from CAD API (proxy discipline)
- **ADR-059**: Pipeline proxy score discipline — do not synthesise missing data

---

## API Changes

When adding or modifying a FastAPI endpoint:

1. Add typed Pydantic request/response models in `aneos_api/schemas/`.
2. Register the endpoint in the appropriate router in `aneos_api/endpoints/`.
3. Add tests in `tests/test_phase<N>.py` or the relevant module test file.
4. Regenerate the OpenAPI spec: `make spec`
5. Commit both the code change and the updated `docs/api/openapi.json`.

CI enforces that the spec is current — it will fail if you forget step 4.

---

## Scientific Methodology Changes

aNEOS's detection framework makes explicit scientific claims. Changes to scoring
weights, thresholds, or statistical methods must:

1. **Be documented in `docs/scientific/scientific-documentation.md`** — add or
   update the relevant section.
2. **Have a test** verifying the old behaviour does not regress (or explaining why
   the behaviour is being changed).
3. **Update `docs/scientific/VALIDATION_INTEGRITY.md`** if the change affects
   sensitivity, specificity, or the Bayesian posterior ceiling.
4. **Add an ADR** if the change alters a threshold or method that other components
   depend on.

The current validation ground truth is small (3 confirmed artificials, 20+ naturals).
F1=1.00 at calibrated threshold is a proof-of-concept on this set, not a production
accuracy guarantee. Any PR that claims improved accuracy must specify the evaluation
corpus used.

---

## Reporting Issues

Open an issue in the [GitHub issue tracker](https://github.com/RobLe3/aneos-suite/issues).

For detection-related issues, please include:
- The NEO designation and the output you observed (paste the full panel)
- Whether you ran with `--legacy-menu` or the default v2 menu
- The data source that was active (check with Option 11 — System Health)

For scientific methodology questions, the forum thread at
https://community.openastronomy.org/t/open-source-python-tool-for-checking-neo-anomalies-aneos/1374
is also a good place to discuss.
