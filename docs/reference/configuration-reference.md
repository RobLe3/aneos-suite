# aNEOS Configuration Reference

This reference documents the configuration surface that is currently honoured by
`aneos_core.config.settings.ANEOSConfig`.  It focuses on the environment
variables and file options that the loader actually parses so that operators can
confidently tune the running system without relying on aspirational or
deprecated knobs.

The configuration loader follows this precedence order (highest first):

1. Programmatic overrides (e.g. `ConfigManager.update_config`)
2. Environment variables
3. Configuration files (`.json`, `.yaml`/`.yml`)
4. Built-in defaults defined in the dataclasses

> **Tip:** Boolean variables accept `1`, `true`, `yes`, or `on` (case
> insensitive) for truthy values.  Anything else is treated as `False`.

---

## Environment Variables

Only the variables listed below affect the shipping loader.  Any other `ANEOS_*`
variables that may appear in legacy documentation are ignored by the current
code and are safe to omit.

### API endpoints & timeouts

| Variable | Default | Description |
| --- | --- | --- |
| `ANEOS_NEODYS_URL` | `https://newton.spacedys.com/neodys/api/` | Override the NEODyS API endpoint. |
| `ANEOS_MPC_URL` | `https://www.minorplanetcenter.net/` | Override the Minor Planet Center endpoint. |
| `ANEOS_HORIZONS_URL` | `https://ssd.jpl.nasa.gov/api/horizons.api` | Override the JPL Horizons endpoint. |
| `ANEOS_SBDB_URL` | `https://ssd-api.jpl.nasa.gov/sbdb.api` | Override the Small Body Database endpoint. |
| `ANEOS_REQUEST_TIMEOUT` | `10` | HTTP request timeout (seconds) for all data sources. |
| `ANEOS_MAX_RETRIES` | `3` | Maximum retry attempts for HTTP requests. |
| `ANEOS_DATA_SOURCES_TIMEOUT` | (unset) | Alternative name for the HTTP timeout; takes precedence over `ANEOS_REQUEST_TIMEOUT` when provided. |
| `ANEOS_DATA_SOURCES_RETRY_ATTEMPTS` | (unset) | Alternative name for the retry count; takes precedence over `ANEOS_MAX_RETRIES` when provided. |

### Data-source priority

| Variable | Default | Description |
| --- | --- | --- |
| `ANEOS_DATA_SOURCES_PRIMARY` | — | When set, becomes the first entry in the data-source priority list (e.g. `SBDB`). |
| `ANEOS_DATA_SOURCES_FALLBACK` | — | Comma-separated fallback priority list appended after the primary source (e.g. `NEODyS,MPC`). |

If neither variable is supplied the loader keeps the default order of
`["SBDB", "NEODyS", "MPC", "Horizons"]`.

### Paths

| Variable | Default | Description |
| --- | --- | --- |
| `ANEOS_DATA_DIR` | `dataneos` | Base directory for downloaded/processed data (`PathConfig.data_neos_dir`).  The loader creates the directory on startup. |
| `ANEOS_LOG_FILE` | `dataneos/neos_analyzer.log` | Location of the primary analyser log file. |

> Other path fields (`data_dir`, `orbital_dir`, `output_dir`, `cache_dir`) are
> derived from the defaults inside `PathConfig` and currently do not have
dedicated environment overrides.

### Analysis execution

| Variable | Default | Description |
| --- | --- | --- |
| `ANEOS_ANALYSIS_MAX_WORKERS` | `10` | Number of worker processes/threads for the pipeline. |
| `ANEOS_MAX_WORKERS` | `10` | Legacy alias consulted only when `ANEOS_ANALYSIS_MAX_WORKERS` is absent. |
| `ANEOS_ANALYSIS_PARALLEL` | `true` | Toggle for running the analysis pipeline in parallel. |
| `ANEOS_ANALYSIS_QUEUE_SIZE` | `1000` | Maximum queued objects awaiting analysis. |
| `ANEOS_ANALYSIS_TIMEOUT` | `300` | Per-object timeout (seconds). |
| `ANEOS_BATCH_PROCESSING_ENABLED` | `true` | Enable/disable batch mode. |
| `ANEOS_BATCH_SIZE` | `100` | Preferred batch size. |
| `ANEOS_BATCH_MAX_SIZE` | `1000` | Hard ceiling on batch size. |
| `ANEOS_BATCH_TIMEOUT` | `3600` | Batch processing timeout (seconds). |
| `ANEOS_ANALYSIS_MEMORY_LIMIT` | `4096` | Memory limit per analysis run (MB). |
| `ANEOS_ANALYSIS_TEMP_DIR` | `/tmp/aneos` | Temporary directory for intermediate files. |
| `ANEOS_ANALYSIS_CLEANUP_TEMP_FILES` | `true` | Whether temporary files are removed after processing. |
| `ANEOS_CACHE_TTL` | `3600` | Cache time-to-live for cached orbital data (seconds). |

---

## Configuration Files

`ANEOSConfig.from_file` accepts either JSON or YAML documents.  Unknown keys are
ignored, and missing sections fall back to the dataclass defaults.  The
structure mirrors the dataclasses in `aneos_core.config.settings`:

```yaml
api:
  neodys_url: https://newton.spacedys.com/neodys/api/
  mpc_url: https://www.minorplanetcenter.net/
  horizons_url: https://ssd.jpl.nasa.gov/api/horizons.api
  sbdb_url: https://ssd-api.jpl.nasa.gov/sbdb.api
  request_timeout: 10
  max_retries: 3
  data_sources_priority:
    - SBDB
    - NEODyS
    - MPC
    - Horizons
paths:
  data_neos_dir: dataneos
  data_dir: dataneos/data
  orbital_dir: dataneos/orbital_elements
  output_dir: dataneos/daily_outputs
  cache_dir: dataneos/cache
  log_file: dataneos/neos_analyzer.log
  cache_file: dataneos/orbital_elements_cache
thresholds:
  eccentricity: 0.8
  inclination: 45.0
  # ... remaining threshold values ...
weights:
  orbital_mechanics: 1.5
  velocity_shifts: 2.0
  # ... remaining weight values ...
max_workers: 10
max_subpoint_workers: 20
analysis_parallel: true
analysis_queue_size: 1000
analysis_timeout: 300
batch_processing_enabled: true
batch_size: 100
batch_max_size: 1000
batch_timeout: 3600
analysis_memory_limit: 4096
analysis_temp_dir: /tmp/aneos
analysis_cleanup_temp_files: true
cache_ttl: 3600
```

> **Validation:** After loading, the configuration manager verifies numeric
> ranges (e.g. positive worker counts, timeout values within range) and raises a
> `ValueError` when constraints are violated.

---

## Legacy compatibility

`ConfigManager.get_legacy_config()` exposes a dictionary compatible with the old
`CONFIG` global.  It includes the derived paths, API URLs, and the raw `weights`
and `thresholds` dictionaries for older modules that still expect the legacy
format.

---

## Unsupported / future variables

Earlier revisions of this document listed many more `ANEOS_*` variables covering
areas such as authentication, database tuning, Kubernetes deployment, and ML
training.  Those keys are currently **not** consumed by the runtime and have
been removed to avoid confusion.  When new configuration surfaces are
implemented they should be added to both the loader and this reference.
