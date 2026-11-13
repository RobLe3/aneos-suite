# Legacy validation integrity correction

Programme #2; legacy scientific contracts #23; shared-pipeline work #6.
Stacked on CAD integrity PR #26. This increment does not install a calibrated
null, observational physical model, independent evaluation or origin classifier.

## Reproduced defects and correction

Direct MultiStageValidator callers could bypass canonical containment and use
fixed source reliability .85, observation span45days, uncertainty .05, a fabricated
orbit, three anomalous categories and a true high-impact flag. Exceptions could
pass stages. The aggregate converted assumed FP reductions into a probability;
its uncertainty helper returned an invented .1. These are withdrawn contracts,
not useful measurements to preserve as current scientific outputs.

All existing public function names remain. Direct helpers now read actual orbit
values/observation dates or return unavailable. Zero eccentricity/inclination is
present data. Reversed/incompatible arcs and invalid numbers fail explicitly.
Field coverage reports whole counts and a clearly defined six-field presence
fraction; it is neither scientific quality nor independent evidence count. Mixed
orbital covariance has no invented unitless scalar uncertainty. Non-gravitational
fit evidence remains descriptive and traceable, not propulsion proof.

The constructor no longer registers legacy research analyzers as validated
models or installs fixed null distributions. Compatibility configuration flags
remain, but cannot enable a calibration artifact by assertion. Standalone
research analyzer classes/functions remain in their own modules; their output is
not promoted into required validation by this orchestrator.

Canonical and legacy inputs use the same screening service. The enhanced result
keeps its original input explicitly in `original_result`; its screening assessment
and claim-bearing aliases expose recomputed canonical results. Caller-edited
probability/classification/metadata cannot restore significance. Invalid/missing
NEO data gives explicit unavailable validation, not a half-probability.

Five-stage aggregation requires complete, explicit, consistent outcomes with no
execution-error markers. Empty/missing/duplicate/truthy/legacy reports cannot
pass. Even a complete gate-positive fixture provides only eligibility for expert
review; no aggregate confidence, false-positive probability, synthetic percentile
or artificial-origin likelihood is invented.

## Callers and triage

A real CanonicalAnalysisAdapter replaces the integration's MockOriginalPipeline
constant score. Enhanced fallback retains a valid canonical screening positive
when scientific validation is unavailable, without granting validation. The old
advanced wrapper cannot turn unavailable stage results into measured clues;
it returns its unavailable status and the separate canonical rank. Its function
and dictionary interface remain; `advanced_score` is nullable. Standalone
AdvancedScoreCalculator remains available for explicitly sourced research use.

Automatic ATLAS-named wrappers no longer synthesize radar/thermal/spectral,
proxy-BIC or zero-orbit evidence. Measured encounter geometry can still produce
clearly labelled, uncalibrated review triage. Real orbital candidacy comes only
from the shared service. A missing proxy score is not zero evidence, and does not
erase a valid ranking-positive control. No KY26 threshold/weight is retuned.

## Dashboard propagation—not a new UI

Existing dashboard monitoring invokes the authoritative validator once and
publishes its actual outcomes, rather than running another stage orchestrator.
Outcome publication timestamps are not represented as stage execution timing.

Metrics retain unavailable values and count their availability separately.
Repeated transport of one report does not double-count that report in the same
in-memory collector. Workflow pass fractions are not detection accuracy, and
workflow rejection counts are not measured false-positive prevention. REST,
WebSocket, summary and JavaScript paths do not coerce null into zero confidence,
perfect `1-FPR` score or calibrated artificial probability. Chart coordinates use
explicit ranking units and skip unavailable coordinates; valid numeric zero is
retained. Automatic threshold-only origin alerts are withdrawn while no origin
model is installed. Historical inputs remain accessible, not rewritten.

This is bounded in-process deduplication and nullable transport, not durable
exactly-once execution, trusted-source authentication or independent certification.
Those concerns remain #15/#16 and scientific evaluation #11.

## Verification and unfinished scope

23 new regressions failed initially; dashboard and JavaScript controls separately
reproduced nullable-output defects. `tests/test_legacy_validation_integrity.py`
contains32 controls, including real source-value extraction, measured31-day arc,
invalid/missing cases, explicit gate-positive fixture, claim tampering, actual
wrapper/dashboard/transport paths, and executable JavaScript null/positive/zero
controls. Node.js is required for the existing dashboard's frontend regression.
No pre-existing tests are edited, disabled, skipped or weakened in this increment.

```sh
node --version
python -m pytest tests/test_legacy_validation_integrity.py -q
python -m pytest tests/ aneos_core/tests/ -m 'not network' -q \
  --cov=aneos_core --cov-report=term-missing --cov-fail-under=30
```

The controls prove bounded integrity behavior, not real-world sensitivity,
probability calibration, physical fit correctness or five-sigma readiness.

Still open: core AnalysisPipeline and legacy `/analyze`/CLI score-result migration
(#6), remaining raw research/legacy interface claims (#12/#17), physical source
uncertainty ingestion (#25), actual fits/BIC/hypotheses (#7–#9), conditional models
and independent evaluation (#10/#11), provider-wide scheduling (#16), durable
jobs/results (#15) and append-only case/prediction lifecycle (#13/#14).
In particular, the core legacy AnalysisPipeline's natural-on-error result and
its API adapter require separate regression/correction; this increment does not
claim every application entry point is already canonical.
