# Changelog

All notable changes to this project will be documented in this file.

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

