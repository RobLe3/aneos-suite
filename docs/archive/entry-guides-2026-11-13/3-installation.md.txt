# aNEOS Installation Guide

Complete installation and setup guide for aNEOS v1.2.0.

---

## Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/RobLe3/aneos-suite.git
cd aneos-suite

# 2. Install core dependencies
pip install -r requirements-core.txt

# 3. Verify the environment
python aneos.py status

# 4. Launch the interactive menu
python aneos.py
```

That's it. No database initialisation step is required for the core research workflow —
aNEOS uses SQLite and creates `aneos.db` automatically on first run.

---

## System Requirements

### Minimum

| Requirement | Value |
|-------------|-------|
| Python | 3.10 or later |
| RAM | 4 GB |
| Storage | 2 GB free (for NEO data cache) |
| Network | Internet — JPL, NASA, ESA APIs queried live |
| OS | Linux, macOS, or Windows |

### Recommended

| Requirement | Value |
|-------------|-------|
| Python | 3.11 (used in CI) |
| RAM | 8 GB (200-year pipeline poll processes ~30,000 objects) |
| Storage | 10 GB (full cache + DB after extended use) |
| CPU | 4+ cores (batch detection uses ThreadPoolExecutor) |

---

## Dependency Profiles

aNEOS ships three requirements files:

| File | Purpose | Install command |
|------|---------|----------------|
| `requirements-core.txt` | Science + testing — covers all menu options | `pip install -r requirements-core.txt` |
| `requirements.txt` | Full stack: API + ML + Docker + monitoring | `pip install -r requirements.txt` |
| `requirements-minimal.txt` | Absolute minimum (no API, no ML) | `pip install -r requirements-minimal.txt` |

**For research use** (the interactive menu, all 15 options): `requirements-core.txt` is sufficient.

**For API deployment** (REST endpoints, Prometheus monitoring): use `requirements.txt`.

### Key packages in `requirements-core.txt`

| Package | Purpose |
|---------|---------|
| `astropy`, `astroquery` | Astronomical calculations, JPL Horizons queries |
| `numpy`, `scipy`, `pandas` | Numerical analysis |
| `scikit-learn` | DBSCAN clustering, anomaly detection |
| `requests`, `aiohttp` | HTTP calls to JPL/NASA/ESA APIs |
| `rich` | Terminal UI (progress bars, panels, tables) |
| `skyfield` | Orbital mechanics |
| `pytest`, `pytest-cov`, `pytest-mock` | Test suite |
| `hypothesis` | Property-based testing |

---

## Virtual Environment (recommended)

```bash
python -m venv aneos-env
source aneos-env/bin/activate   # Linux / macOS
# aneos-env\Scripts\activate    # Windows

pip install -r requirements-core.txt
python aneos.py status
```

---

## Full Installation (includes REST API + monitoring)

```bash
pip install -r requirements.txt

# Optional: initialise PostgreSQL (SQLite is used by default)
# export ANEOS_DATABASE_URL=postgresql://user:pass@localhost/aneos

python aneos.py api --dev
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## Configuration

aNEOS works out of the box without any configuration file. Optional overrides:

```bash
# Copy the example environment file
cp .env.example .env

# Edit as needed — all fields are optional
# ANEOS_DATABASE_URL=sqlite:///./aneos.db   (default)
# ANEOS_ENV=development                      (default)
# ANEOS_HOST=0.0.0.0                        (API only)
# ANEOS_PORT=8000                            (API only)
# ANEOS_SECRET_KEY=<random-string>          (API auth, dev mode only)
```

Admin-level config overrides (Option 12 → API Server → Admin panel) are stored in
`aneos_config_override.json` and take precedence over `.env`.

---

## Verification

```bash
# Preflight check (8 components: pipeline, detection, API modules, DB, cache…)
python aneos.py status

# Run the full test suite
python -m pytest tests/ aneos_core/tests/ -m "not network" -q
# Expected: 308 passed, 0 failed

# Smoke test — import core modules
python -c "import aneos_core; print('Core OK')"
python -c "from aneos_menu_v2 import ANEOSMenuV2; print('Menu OK')"

# If you installed the full stack, verify the API
python aneos.py api --dev &
curl http://localhost:8000/api/v1/health
```

---

## Docker (optional)

```bash
# Build and start all services
make bootstrap   # creates certs and init.sql
docker-compose up -d

# API:       http://localhost:8000
# Dashboard: http://localhost:3000 (Grafana — admin/aneos)
```

Docker includes: aNEOS API, PostgreSQL, Redis, Nginx, Prometheus, Grafana.

---

## Troubleshooting Installation

### Python version

```bash
python --version   # must be 3.10 or later
python3 --version
```

If you have multiple Python versions, use `python3.11 -m pip install ...` and
`python3.11 aneos.py`.

### Dependency conflicts

```bash
# Start clean in a fresh virtual environment
python -m venv fresh-env
source fresh-env/bin/activate
pip install -r requirements-core.txt
```

### `astropy` or `scikit-learn` install fails

```bash
# Install with binary wheels (avoids compilation)
pip install --prefer-binary astropy scikit-learn numpy scipy
```

### API server won't start (missing `torch`, `redis`, etc.)

The REST API requires the full `requirements.txt` stack. If you only installed
`requirements-core.txt`, the interactive menu (all 15 options) works fully but
the API server needs:

```bash
pip install fastapi uvicorn sqlalchemy pydantic httpx
```

### Database issues

```bash
# Reset the SQLite database (deletes all stored results)
rm aneos.db
python -c "from aneos_api.database import init_database; init_database()"
```

### Network / firewall

aNEOS queries these external APIs — ensure they are reachable:

| API | URL | Used for |
|-----|-----|---------|
| JPL SBDB | `ssd-api.jpl.nasa.gov` | Orbital elements, physical properties |
| JPL CAD | `ssd-api.jpl.nasa.gov` | Close-approach data (200-year poll) |
| JPL Horizons | `ssd.jpl.nasa.gov` | Orbital history time-series |
| NEODyS | `newton.spacedys.com` | Orbital elements (fallback) |
| MPC | `minorplanetcenter.net` | Orbital elements (fallback) |

All external calls have circuit-breaker protection and graceful fallback — a single
unavailable API will not break the workflow.

---

## Next Steps

After a successful install:

1. `python aneos.py` — opens the 15-option interactive menu
2. Choose **Option 1** (Detect NEO — single) → enter `99942` (Apophis)
3. Choose **Option 7** (Live Pipeline Dashboard) → run the 200-year historical poll
4. Choose **Option 11** (System Health) → verify all components are online
5. Read `docs/scientific/theory.md` — the theoretical foundation
6. Read `docs/scientific/scientific-documentation.md` — the full statistical methodology
7. See `CONTRIBUTING.md` for how to run tests and extend the system

---

## Related Documentation

- [Quick Start Guide](quick-start.md)
- [Menu System Guide](menu-system.md)
- [Artificial NEOs Theory](../scientific/theory.md)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [Troubleshooting Guide](../troubleshooting/troubleshooting-guide.md)
