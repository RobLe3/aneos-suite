# aNEOS Configuration Reference

This reference describes what `aneos_core.config.settings.ANEOSConfig` parses.
Parsing a field does not prove that every provider, worker or entry point uses it.
Other modules also read environment variables; this is not a complete application-wide
configuration inventory. Provider polling/coverage is still under review (#16/#40).

`ConfigManager` selects one loading path:

1. An explicitly supplied, existing file is loaded; environment variables are **not**
   merged over that file. Missing file sections use dataclass defaults.
2. Otherwise `ANEOSConfig.from_env()` overlays supported environment variables on defaults.
3. `update_config()` changes that manager's in-memory object and validates afterward;
   it does not persist a file or update every running service. Rejected updates/reloads
   leave the previous manager object intact; valid changes publish a validated copy.
   Unknown update fields are rejected, not silently ignored.

**Explicit file errors fail closed:** missing files raise `FileNotFoundError`;
malformed/unsupported construction raises `ValueError`. Defaults are not silently
activated in their place. This corrects earlier fallback behavior. Direct
`ANEOSConfig.from_file()` parses dataclasses; use `ConfigManager` for its bounded
range/type validation. No file write or global hot-reload is implied.

## Scientific policy menu

The legacy **Analysis Policy** menu is read-only. It shows the default canonical
ranking policy, offers explicitly synthetic controls, and exports/verifies policy
snapshots. It does not change alpha, toggle a validated physical BIC model, apply
thresholds or import runtime configuration. Snapshot JSON is not an `ANEOSConfig`
file. See [policy inspection and examples](../engineering/truthful-policy-menu.md).
A candidate threshold is in raw ranking units, not sigma or origin confidence.

> **Tip:** Boolean variables accept `1`, `true`, `yes`, or `on` (case
> insensitive) for truthy values.  Anything else is treated as `False`.

---

## Environment Variables

The following variables are parsed by this loader. Other application modules may
read additional variables, including database and authentication settings.

### API endpoints & timeouts

| Variable | Default | Description |
| --- | --- | --- |
| `ANEOS_NEODYS_URL` | `https://newton.spacedys.com/~neodys2/epoch/` | Override the NEODyS API endpoint. |
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

| Variable | Default | Loader behavior |
| --- | --- | --- |
| `ANEOS_BASE_DIR` | `dataneos` | Sets `DirectoryConfig.base_dir` and recomputes derived paths. |
| `ANEOS_DATA_DIR` | `dataneos/data` | Overrides `directories.data_dir`, not the base directory. |
| `ANEOS_ORBITAL_DIR` | `dataneos/orbital_elements` | Overrides orbital-data path. |
| `ANEOS_OUTPUT_DIR` | `dataneos/daily_outputs` | Overrides output path. |
| `ANEOS_CACHE_FILE` | `dataneos/orbital_elements_cache` | Overrides cache-file path. |
| `ANEOS_LOG_FILE` | `dataneos/neos_analyzer.log` | Overrides log path. |

Path derivation itself does not create directories. Consumers may create their own
paths. `DirectoryConfig` is the current type; `paths` is a compatibility property.
There is no dedicated cache-directory environment override in this loader.

### Legacy threshold override

`ANEOS_ECCENTRICITY_THRESHOLD` overrides the loader's legacy eccentricity threshold
(default0.8). It is not the canonical follow-up ranking policy or a significance level.

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

`ANEOSConfig.from_file` accepts JSON or YAML documents (YAML requires PyYAML).
Unknown top-level and nested dataclass keys are rejected rather than silently
ignored or replaced with defaults. The legacy `paths` alias remains supported. Missing sections use dataclass defaults.  The
structure mirrors the dataclasses in `aneos_core.config.settings`:

```yaml
api:
  neodys_url: https://newton.spacedys.com/~neodys2/epoch/
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

> **Validation scope:** The manager checks eccentricity/inclination bounds, positive integer
> request timeout/worker count and nonnegative integer retry count. Booleans, strings
> and nonfinite numbers are rejected in these checked fields. It does not validate every
> parsed execution field or certify that a provider honors it. These are configuration
> checks, not scientific calibration or physical-model validity.

---

## Legacy compatibility

`ConfigManager.get_legacy_config()` exposes a dictionary compatible with the old
`CONFIG` global.  It includes the derived paths, API URLs, and the raw `weights`
and `thresholds` dictionaries for older modules that still expect the legacy
format.

---

## Other modules and unresolved work

`ANEOS_DATABASE_URL`, authentication settings and provider pacing/state controls are
read outside `ANEOSConfig`; absence from this table does not make them unsupported.
Consult their scoped implementation and engineering reports. The loader's legacy
`thresholds`/`weights` are not the canonical `ScreeningPolicy` or a validated sigma model.
An example field appearing here is not evidence of active policy mutation.

`ANEOSConfig.save()` creates parent directories and **overwrites** its target file.
That existing general configuration API is separate from the menu's exclusive-create
policy snapshot export. Back up an existing configuration before calling it. Whole-app
configuration consistency, all remaining field validation and consumer application still
need verification/remediation under #16/#17; this reference does not claim those are complete.
