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

### Modular anomaly analysis pipeline
- `aneos_core.analysis.pipeline.AnalysisPipeline` coordinates the anomaly
  indicators, scoring, caching, and data-source access needed to analyze a
  single NEO or batches of designations asynchronously.
- The pipeline loads connector classes for NASA SBDB, ESA NEODyS, and MPC
  data. When those services are unreachable it falls back to cached or mock
  data rather than terminating the run, which keeps experiments moving but
  means real detections still need independent validation.
- Indicator results feed into a weighted scoring framework that combines
  orbital, velocity, temporal, geographic, physical, and behavioural signal
  groups to produce an anomaly classification and confidence estimate.

### Historical polling and automation primitives
- The `HistoricalChunkedPoller` breaks long spans of time into manageable
  chunks, handles caching, and can be wired to the enhanced poller and
  SWARM scoring components for large-scale backfills or reprocessing
  experiments.
- Utility scripts such as `enhanced_neo_poller.py`, `neo_poller.py`, and
  `simple_neo_analyzer.py` demonstrate how to access the polling and
  analysis layers without going through the full menu system.

### Interfaces for running experiments
- `python aneos.py` launches an interactive menu that exposes analysis,
  polling, diagnostics, and data-management workflows. The script also
  provides shortcuts for launching the API server, simple analyzer, and
  other entry points directly.
- The FastAPI application in `aneos_api/app.py` bundles the analysis
  pipeline, optional machine learning helpers, and monitoring utilities
  behind REST endpoints and a dashboard-friendly data layer.
- The `aneos_dashboard` package offers a web UI scaffold (templates,
  static assets, and websocket hooks) that can consume the API for
  visualization once the backend services are configured.

### Documentation and configuration support
- Rich documentation lives under `docs/`, covering installation, menu
  workflows, API usage, architecture, and troubleshooting.
- `install.py` provides a guided installer with minimal/core/full
  dependency sets and environment checks so contributors can reproduce the
  toolkit on their systems.

## Current project status

aNEOS 0.7.0 is in a stabilization phase. The automated unit and integration
suite runs, but several risks remain:

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
   expected project directories.
3. **Explore the menu or run targeted commands**
   ```bash
   python aneos.py              # interactive mission control menu
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

## Help us build the pipeline-based virtual observatory

The long-term goal is to assemble these components into a pipeline-based
virtual observatory that can compare multiple detection methods, monitor
incoming discoveries, and present curated investigation queues. We welcome
help with:

- **Hardening data ingestion** – add health checks, credentials support,
  and failure reporting for the SBDB/NEODyS/MPC connectors so live polling
  is reliable.
- **Reproducible pipeline runs** – script historical backfills with the
  chunked poller, capture real data snapshots, and publish validation
  notebooks so results can be replayed by other researchers.
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
