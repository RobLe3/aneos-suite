# Changelog

All notable changes to this project will be documented in this file.

## [v1.2.2] - 2026-03-11 (Phase 24)
### Fixed
- `DetectionResult.__init__`: falsy-zero bug — `artificial_probability=0.0` was replaced
  by fallback due to `or` antipattern; fixed to `is not None` guard (`interfaces/detection.py`)
- `monitoring.py` dashboard endpoint: `get_recent_alerts(limit=5)` → `get_recent_alerts(hours=1)`;
  `limit` kwarg does not exist in `AlertManager`
- NEODyS `_resolve_number()`: provisional designations like `"1998 KY26"` no longer misparse
  the year prefix as a catalogue number; provisional-year regex guard added (`neodys.py`)
- Gaia import stdout suppression: astroquery Gaia TAP notice redirected to debug log via
  `contextlib.redirect_stdout`; terminal no longer polluted on startup (`gaia_astrometric_calibration.py`)
- `demo_neos.txt`: replaced placeholder entries with 5 real NEO designations
  (Apophis, Bennu, Ryugu, Didymos, 2004 MN4); pre-flight format guard in menu (`aneos_menu_v2.py`)
### Added
- Pydantic input validation on `OrbitalInput`: `a` 0.1–1000 AU, `e` 0.0–2.0, `i` 0.0–180°;
  invalid inputs return HTTP 422 (`aneos_api/schemas/detection.py`)
- Designation length validation on `AnalysisRequest` (1–50 chars) and per-element
  validator on `BatchAnalysisRequest` (`aneos_api/models.py`)
- `get_validation_summary` endpoint wired to real `AnalysisResult` ORM query with
  UTC-aware date filtering (`enhanced_analysis.py`)
- SSE streaming `_get_next_sse_event` wired to real `alert_manager.get_recent_alerts()`
  and `get_health_status()` — no longer uses `random` mock (`streaming.py`)
### Infrastructure
- Test baseline: 360 pass / 0 fail (+24 from Phase 24 tests)
- CI coverage gate raised from 30% to 40%

## [v1.2.1] - 2026-03-10 (Phase 23)
### Fixed
- SA 2.0 scoring alignment fixes in `advanced_scoring.py`
- Auth DB wiring: `_load_users_from_db()` called at startup; API key map populated from DB
- ADR-032 closure: `rendezvous.py` raises `ImportError` with install hint instead of silent return
- Warning cleanup across 15+ files; deprecated usage of `datetime.utcnow` fully purged
### Infrastructure
- Version bump 1.1.0 → 1.2.1
- Test baseline: 336 pass / 0 fail

## [v1.2.0] - 2026-03-10 (Phase 21)
### Added
- Physical indicators wiring: `DiameterAnomalyIndicator` + `AlbedoAnomalyIndicator` active
  in single-object detection path via `detection_manager.py` (ADR-053 closed)
- Stouffer's weighted z-score method in `NetworkSigmaCombiner`; `PatternAnalysisConfig`
  gains `combination_method` field (default `'fisher'`) (SCI-003 mitigated)
- JWT bearer token endpoint: `POST /api/v1/auth/token` exchanges API key for signed JWT
  (G-031 partial)
- Property-based hypothesis tests: orbital invariants, ATLAS bounds, sigma/p round-trips
  (TST-001/TST-003 closed)
### Fixed
- `PHAMoidScanner.fetch_phas()` raises `ImportError` with install instructions when
  `aiohttp` is absent, replacing silent `return []` (ADR-032 partial)
### Infrastructure
- ADR-060 added (property-based testing); ADR-032/047/053 updated
- Test baseline: 336 pass / 0 fail (+28 from Phase 21)

## [v1.1.0] - 2026-03-10 (Phase 20)
### Added
- Pipeline proxy score discipline: radar/thermal/spectral zeroed in CAD-API pipeline;
  approach/ΔBIC confidence 0.35; Δ flag threshold raised to 0.65 (ADR-059)
- Dark comet context in pipeline detail panel; data-source disclaimer
- UI: category score bar chart scaled to max; `display_panel` expand=False
- `docs/scientific/theory.md` — professional abstract of Artificial NEOs theory document
### Fixed
- Results browser no longer dumps all DB entries after detail view
- ADR gap closure: ADR-045/051/052/053-058 alignment; ADR-059 added
### Infrastructure
- Test baseline: 308 pass / 0 fail; CI coverage gate 30%; pytest-cov/mock/hypothesis added
- CONTRIBUTING.md rewritten; README/installation.md stale links fixed

## [Unreleased]
- Work in progress on refining analysis methods for NEO anomaly scoring.
- Updated reporting module to include enhanced visualization.

## [v0.7.0] - 2025-09-26
- Tagged the previous stable snapshot on the `version/0.5` branch to preserve the 0.5 series for reference.
- Updated all published package metadata to report version 0.7.0 for the active development line.
- Added release documentation to keep downstream consumers aligned with the new versioning scheme.

## [v3.0] - 2025-02-20
- Initial release of reporting_neos_ng_v3.0.py.
- Integration with neos_o3high_v6.16.py for unified analysis and reporting.
- Added detailed logging and improved error handling.

*(Future versions will be added here as development progresses.)*

