# aNEOS Suite

## Project overview

aNEOS (Advanced Near Earth Object detection Suite) is an open research
platform for investigating whether engineered or otherwise anomalous
Near-Earth Objects can be detected in public astronomy catalogues. The
repository combines data acquisition, anomaly-indicator scoring, CLI and
API surfaces, and dashboard tooling so experiments can be run end to end.

The codebase grew out of the earlier `neo-analyzer` research scripts and
now focuses on building a reproducible pipeline that can ingest real
orbital data, evaluate multiple detection methods, and surface results in
a consistent way. While the long-term ambition is a pipeline-driven virtual
observatory, the current release is still a research sandbox rather than a
production system.

## Core capabilities today

### Mission control launcher
- `python aneos.py` opens the mission-control menu that drives analysis,
  polling, diagnostics, and system management flows. With `rich` installed
  the interface renders the cinematic dashboard seen in demos; in plain
  terminal mode it falls back to the text menus exercised below, with
  several visualisations (enhanced validation, cross-reference browser,
  etc.) disabled until a rich console is available.
- The launcher also exposes shortcuts (`python aneos.py status`,
  `python aneos.py analyze <designation>`, `python aneos.py api --dev`, …)
  so you can script specific tasks outside of the interactive menu.

### Validated anomaly scoring prototypes
- `aneos_core.analysis.pipeline.AnalysisPipeline` and the validated
  `DetectionManager` are wired into the “Quick Scan” workflow. Running it
  today initialises the SBDB, NEODyS, and MPC connector classes, but the
  real-time fetch path is incomplete (`DataSourceManager.get_neo_data`
  is still missing) so analyses fall back to simulated orbital elements and
  report a neutral “natural” classification.
- The scoring stack still produces the weighted anomaly output and exposes
  smoking-gun toggles, giving contributors a concrete place to add proper
  data acquisition and richer result handling.

### Polling and automation experiments
- Historical polling helpers (`HistoricalChunkedPoller`,
  `enhanced_neo_poller.py`, `neo_poller.py`) remain available for scripted
  experiments, but the mission-control “Continuous Monitoring” status check
  currently reports several blockers (for example, the
  `aneos_core.validation.delta_bic_analysis` module has not been packaged,
  which keeps the automatic pipeline flags in the “not ready” state).
- Until those validation modules are restored, large-scale backfills and the
  multi-stage automation funnel should be treated as design sketches rather
  than production jobs.

### API, dashboard, and documentation support
- The FastAPI application in `aneos_api/app.py` still wraps the pipeline and
  monitoring utilities for REST access, and the `aneos_dashboard` package
  provides templates/websocket hooks for a future control room UI once the
  backend stabilises.
- Rich documentation lives under `docs/`, covering installation, menu
  workflows, API usage, architecture, and troubleshooting, and `install.py`
  continues to offer guided dependency installs (`--minimal`, `--core`,
  `--full`).

## Current project status

aNEOS 0.7.0 is in a stabilization phase. The automated unit and integration
suite runs, but several risks remain:

- Quick Scan runs through the validated detector but logs
  `'DataSourceManager' object has no attribute 'get_neo_data'`, so every
  analysis still relies on simulated orbital elements until the real fetch
  path is implemented.
- The Continuous Monitoring diagnostics report that several validation
  modules (for example `aneos_core.validation.delta_bic_analysis`) are
  missing, leaving the automated historical polling funnel marked as “not
  ready”.
- `python aneos.py status` confirms the API and database layers start, yet
  it flags the default absence of `logs/` and `cache/`; create those
  directories or update the installer so health checks pass cleanly.
- External data-source integrations are not yet exercised in CI; when API
  credentials or network access are missing the software simulates results
  instead of failing fast.
- No recent, end-to-end runs against live catalogues are documented, so
  published anomaly scores should be treated as research prototypes that
  still require human review.
- Some documentation inherited from earlier phases overstated production
  readiness; this README and the maturity notes aim to reset expectations
  until the remaining verification work lands.

## Getting started

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/RobLe3/aneos-suite.git
   cd aneos-suite
   python install.py --core
   ```
   Use `--minimal` or `--full` for alternate dependency sets.
2. **Verify the environment**
   ```bash
   python aneos.py status
   ```
   This checks the analysis pipeline, API components, database layer, and
   expected project directories. On a fresh checkout it is normal to see
   warnings about missing `logs/` and `cache/`; create them manually or run
   the installer with a flag that bootstraps runtime directories.
3. **Explore the menu or run targeted commands**
   ```bash
   python aneos.py              # interactive mission control menu (rich UI when available)
   python aneos.py analyze "2024 AB"  # on-demand pipeline run
   python aneos.py api --dev    # launch FastAPI server in dev mode
   ```
4. **Review outputs** – cached data and analysis results are stored under
   `neo_data/`, while API and dashboard components log activity to `logs/`
   when enabled.

## Repository guide

- `aneos_core/` – analysis pipeline, data connectors, polling utilities,
  scoring, ML helpers, and validation modules.
- `aneos_api/` – FastAPI application, authentication, endpoints, and
  service wiring.
- `aneos_dashboard/` – dashboard web app scaffold and websocket handlers.
- `docs/` – user guides, API references, engineering notes, and release
  documentation.
- `scripts & tools` – top-level helpers (`aneos.py`, `aneos_menu.py`,
  `enhanced_neo_poller.py`, `simple_neo_analyzer.py`, etc.).

## Mission-control check-in

Running the menu without the rich renderer (so the plain-text fallback is
active) yielded the following observations:

- **Quick Scan** initialises the validated detector stack, emits
  `'DataSourceManager' object has no attribute 'get_neo_data'`, and returns a
  neutral “natural” score using simulated orbital elements while waiting for
  live data retrieval to be implemented.
- **Continuous Monitoring → Pipeline Component Status** reports every
  automation flag as unavailable because several validation modules are not
  packaged yet. This is the clearest indicator of the work left to harden the
  historical poller.
- **Mission Status & Intelligence → System Health** and **Advanced Mission
  Control → Emergency Diagnostics** confirm that the API, database, and core
  analysis layers initialise, albeit with the runtime-directory warnings noted
  above.

## Help us build the pipeline-based virtual observatory

The long-term goal is to assemble these components into a pipeline-based
virtual observatory that can compare multiple detection methods, monitor
incoming discoveries, and present curated investigation queues. We welcome
help with:

- **Data acquisition fixes** – finish the `DataSourceManager.get_neo_data`
  implementation (or equivalent) so Quick Scan and the API can operate on
  live SBDB/NEODyS/MPC data instead of the current simulated placeholders.
- **Automation pipeline packaging** – restore the validation modules the
  Continuous Monitoring check expects (e.g. `aneos_core.validation` pieces)
  and script an end-to-end historical poll so the readiness indicators flip
  to green.
- **Runtime environment hygiene** – teach `install.py` to create `logs/`,
  `cache/`, and other runtime directories and surface clearer errors when
  optional components (such as the rich UI dependencies) are missing.
- **Multi-method fusion** – expand the indicator set, integrate ML models
  through the API, and document how different approaches contribute to the
  final anomaly score.
- **Observability and dashboards** – wire the monitoring hooks, metrics, and
  dashboard front-end to give mission control visibility into live runs and
  queued investigations.
- **Documentation & testing** – keep the docs in sync with verified
  behaviour and extend the automated test coverage so future releases stay
  grounded in reproducible evidence.

If you experiment with the suite or add new capabilities, please update the
relevant documentation and share your findings so the community can build
on verifiable results.
