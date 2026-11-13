# Required automatic-review gate

Programme #2; gate integrity #21; depends on screening-contract PR #20.
This increment repairs advancement control, not scientific detection accuracy.

## Defects reproduced

The automatic multi-stage stage ignored its injected validator. It accepted a
repeated detector output or a fraction of the first-stage rank, called one path
`natural_object_standard`, and assigned `fp_probability = 1 - validation_score`.
A high rank could therefore bypass required validation. The enhanced wrapper
also marked unavailable validation as passed in its minimal fallback.

Eighteen new regressions failed before the implementation. No pre-existing test
was disabled, skipped, weakened or rewritten in this increment.

## Implemented contract

Every eligible candidate receives a fresh canonical screening assessment and an
invocation of `validate_analysis_result`. Cached analysis, sigma claims and
first-stage flags cannot substitute for that invocation. Capacity-deferred
candidates are explicitly recorded rather than counted as rejected detections.

A report can advance only when:

1. Its contract is `required-review-v1` and its evidence/ranking-policy IDs match
   the assessment supplied to the validator.
2. It contains exactly one outcome for each required stage 1–5.
3. Every stage has an explicit boolean pass and matching `passed` status, with no
   execution-error marker or invalid/non-finite serialized diagnostic data.
4. The aggregate explicitly agrees with every stage. No truthy strings or empty
   lists can grant a pass.
5. The validator configuration has not changed during execution.

The gate records separate validation-policy and full validation-input fingerprints, statuses, reasons and
stage details. Full review inputs include observation dates, source labels and
physical uncertainties that are not part of the ranking-only snapshot. They
cannot change during execution or reuse a prior pass. The frozen input snapshot
survives report export. Expert-queue preparation rechecks the gate, current evidence and
validation policy. It recomputes the screening rank for ordering; it does not
reuse a caller-edited rank. Queue capacity deferrals are visible.

Advancement means **eligible for expert review**, never confirmed origin. Gate
records keep confidence, false-positive probability and validation score null.
The old multi-stage score threshold remains a compatibility configuration field,
but cannot replace required stages; its ignored value is reported in metrics.
No scientific accuracy or calibration acceptance requirement is waived.

Missing validator, exception, timeout, incomplete report, stale identity/policy or
invalid diagnostic data yields unavailable validation. Failed and inconclusive
reports remain distinct. Retrying replaces stale positive state. A failure on
one object remains visible without discarding another object's verified positive
control in the complete pipeline.

## Installed scientific capability

For canonical screening, `MultiStageValidator` now records five explicit
prerequisite outcomes without invoking its legacy placeholder physics, orbit,
source-quality or null-distribution helpers. It preserves actual observation
span when known and descriptive fitted A1/A2/A3/covariance evidence. Unknown source
reliability and uncertainty are not filled with constants.

There is no installed calibrated null or compatible observation-fit comparison.
The real canonical validator therefore reports **inconclusive**, not passed or
natural. Its p-value, FP probability, synthetic percentile and confidence are
unavailable. Component initialization is not scientific validation.

The passing fixture is an explicitly synthetic validator-contract positive
control. It verifies that the gate and expert queue are usable, including mixed
outage cases. It is **not** a scientifically validated NEO, independent evaluator,
calibration artifact or real-world sensitivity estimate. A fully initialized real
validator is also tested with network connections denied; it reports inconclusive
without provider access.

Legacy/noncanonical validation and direct helper entry points still contain
placeholder science and pass-on-error behavior. They are not credited by this
gate and remain a separate P0 issue #23. Historical CAD integrity remains #22.

## Reporting, history and verification

Existing pipeline JSON exports now retain all gate outcomes/reasons, evidence and
policy IDs, and stage details, including non-advancing/capacity cases. Result
filenames are unique and created exclusively rather than overwriting an earlier
same-second report. This is bounded history protection, **not** restart-safe jobs,
transactional storage or crash-safe atomic publication; those remain #15.

The Rich pipeline detail/result paths and enhanced summaries accept unavailable
values without formatting them as zero probability. Enhanced fallback preserves
the original result alongside explicit unavailable, non-passed validation.

Regression module: `tests/test_required_validation_gate.py` (37 controls).

```sh
python -m pytest tests/test_required_validation_gate.py -q
python -m pytest tests/ aneos_core/tests/ -m 'not network' -q \
  --cov=aneos_core --cov-report=term-missing --cov-fail-under=30
```

Existing planetary-defense and historical baseline tests remain in the full run.
Local tests are implementation evidence, not independent scientific certification.
