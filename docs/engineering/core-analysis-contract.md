# Core analysis contract correction

Date: 2026-11-12. Review branch: `codex/core-analysis-contract`.
Base: `e9b6c22ff0e3ce994007b348fd0518dca4efe27b` (draft PR #27).
Programme: #2; implementation/interface contracts #6/#12; independent labels #11;
remaining provider/durability/verification #16/#15/#17. ADR-069 records the decision.

## Confirmed defects and corrections

- Core AnalysisPipeline was a second classifier: legacy indicator weighting,
  natural/zero error fallbacks and an uncalibrated statistical population tracker.
  It now acquires typed NEOData through DataFetcher and delegates to the shared
  ScreeningAnalysisService. Legacy calculators/indicator methods remain research
  utilities, not deployed classification evidence. The separate mock-capable
  source-manager factory is controlled-deprecated with NotImplementedError.
- Core results lacked neo_data and a canonical assessment. The result now has both,
  explicit execution status, policy/evidence IDs and a nullable compatibility view.
  `overall_score` is raw uncalibrated rank, not [0,1] probability or Gaussian sigma.
  Source failure, invalid input, internal error and incomplete evidence do not
  become natural classifications. Full provided/cached NEO evidence is retained.
- Field quality invented reliability from field presence and orbit epoch age,
  used the wrong physical attribute, and treated zeros as missing in some checks.
  The new inventory counts 13 named catalogue fields with an explicit denominator;
  reliability is unavailable and retrieval time is distinct from orbit epoch.
  Presence is not uncertainty, measurement accuracy or fitness for physical fitting.
- Batch scheduling used nested event loops/thread futures, lost input order and
  could abort on an outer timeout. Bounded async scheduling retains per-object
  identity/order/failure, including duplicate requests and progress-callback errors.
  Timeout cancels the await, not a running provider thread. This is not a durable
  scheduler or global rate limiter.
- `/analyze` accessed a dict's __dict__, expected nonexistent neo_data and read a
  designation/flag-only cache. It now uses typed serialization, forwards refresh,
  returns 503/422 for source/measurement problems, and preserves assessment fields.
  Cached snapshots are retrieval/export conveniences, never read as fresh analysis.
  Latest in-process detection retrieval is refreshed by a new core assessment.
- Compatibility diagnostics are labeled proxy contributions with nullable
  confidence/weighted scores; `include_indicators` does not remove canonical
  evidence needed for replay. Typed source sidecars and native CAD metadata survive
  API serialization. Batch refresh/raw-data request flags are actually forwarded.
- Existing JSON export used stringification of model objects and CSV lost scores.
  JSON now preserves structured assessment/replay metadata, CSV labels raw ranking
  units and IDs, search accepts finite ranking bounds above one, and null ranks
  are excluded from numerical filters. Unsupported export formats fail explicitly.
  Filtering/date/sort completeness and durable export jobs remain separate work.
- Core-result Rich/plain/subprocess CLI displays no longer format null inference
  as a float. The broader old menu/detector/advanced-display claim audit is unfinished.
- Legacy ML assumed bounded scoring units, could return old cached predictions,
  and generated supervised labels from classifications. Prediction single/batch
  and their API adapters now reject the new core contract before cache/dispatch;
  API returns 503. Direct indicator fallback rejects canonical ranks as probability.
  Training requires independently supplied binary labels and provenance and aligns
  them to returned object order; classification-only labels fail. This does not
  verify supplied labels or install a compatible feature/calibration pipeline.
- Existing enhanced wrapper/validator consumes a core assessment and acquired NEO
  data rather than losing a valid candidate when validation is unavailable. The
  separately mocked enhanced REST acquisition/pipeline is still **not repaired**.

## Verification

`tests/test_core_analysis_contract.py` contains 34 network-denied controls using
synthetic, explicitly non-observational markers. Fifteen initial core/API cases
exposed legacy failures; subsequent controls extend that coverage, not independent
scientific validation. The final full offline run passed 608 tests, with four
network tests deselected and four warnings, in 26.39 seconds. Coverage was 40.24%
against the unchanged 30% minimum. Controls exercise:

- Candidate positive and unflagged controls through the real initialized core and
  shared service; legacy indicator/scoring execution forbidden.
- No-data/outage/processing error, invalid NaN, mismatched identity, missing orbit,
  zero-valued elements, changed evidence, lazy typed fetcher and force refresh.
- Ordered mixed batches, duplicate identity, timeout and callback failure.
- Actual FastAPI HTTP route serialization and typed 503; raw/diagnostic flags,
  canonical payload parity, native A3/CAD transport, JSON replay and CSV semantics.
- Required validation/fallback retains candidacy but never grants scientific pass.
- Old ML cache cannot bypass contract rejection; direct fallback cannot reinterpret
  rank as probability. Classification-only labels fail, independently supplied
  synthetic label-contract controls succeed with object-order alignment.
- Rich/plain null display, finite rank search and latest in-process retrieval.
- Generated documented OpenAPI response matches the actual response model.

No prior tests were changed or disabled. The full offline suite includes existing
frozen KY26 baseline replay, source/covariance, numerical tail and automatic review
regressions. Run with the existing unchanged 30% coverage requirement:

```sh
PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-core-review.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core \
  --cov-report=term-missing --cov-fail-under=30
```

The PR records the final commit and CI results. Work was isolated in a separate
worktree; the dirty main checkout and its database were not overwritten. No live
JPL request, new source capture, model fitting, threshold tuning, changed frozen
historical measurement or independent scientific evaluation was performed here.

## Scientific assessment and remaining work

The preserved user-reported KY26 rank (4.142), posterior and historical p-value
remain historical outputs, not calibrated scientific claims. This change supplies
no updated significance, artificial-origin probability or valid physical-model
ranking. A3 transport is tested, not a force interpretation. Five-sigma permission
remains false. No finding for or against Phobos 1, extraterrestrial technology,
outgassing or another natural explanation follows from these engineering tests.

Next priorities: remove mock enhanced REST data/scoring and audit old menu paths
(#6/#12); complete dated primary-source KY26 measurements and raw astrometry
(#7/#25/#8); implement compatible natural/technological/artifact fits (#9); address
conditional populations/dependencies and independent calibration (#10/#11).
Provider-wide backoff/configuration/pagination and recoverable jobs/results remain
#16/#15. The current DataFetcher still uses its own global source configuration;
this increment does not establish per-pipeline provider isolation. All current
result/job caches are in-process, not append-only scientific case history (#13).
Full covariance epoch propagation, observation likelihoods and 2031 frozen
predictions remain unimplemented. Existing issue acceptance gates stay open.
