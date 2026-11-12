# Screening result contract: uncalibrated ranking, unavailable inference

Programme [#2](https://github.com/RobLe3/aneos-suite/issues/2);
statistical reporting and shared service [#6](https://github.com/RobLe3/aneos-suite/issues/6).
This increment depends on baseline PR #18 and typed evidence PR #19. It is not
scientific readiness or an independent performance evaluation.

## Confirmed defects

The old facade called a weighted root-sum-of-squares rank “sigma”, derived an
origin posterior from assumed likelihood ratios, and reported zero FDR from
constructor examples. The API added name-based spacecraft decisions, while menu,
raw API, fetched API and batch used different adapters. Predictions and fitted
acceleration coefficients could activate propulsion/manoeuvre heuristics without
a comparative physical fit. A validator converted scoring failures into zero
probability, allowing unavailable outputs to count as correct natural decisions.

Those are contract defects, not evidence for or against KY26's origin.

## Implemented boundary

`ScreeningAnalysisService` is a synchronous, acquisition-free ranking service.
Menu single/batch, REST fetched/raw/batch, the manager's canonical wrapper, the
legacy-named detector facade, and the automatic pipeline's screening helper use
it. They expose:

- `screening_score`: legacy uncalibrated rank, or null for incomplete/out-of-domain orbits;
- `candidate`: follow-up triage, not confirmed origin;
- `classification`: `FOLLOW_UP_CANDIDATE`, `NOT_FLAGGED`, or `INCONCLUSIVE`;
- null sigma, p-value, FDR, artificial-origin probability, binary origin and confidence;
- explicit unavailable significance and inconclusive origin status;
- SHA-256 evidence and policy IDs, versioned ranking-input snapshots, limitations,
  and descriptive fitted A1/A2/A3 evidence including covariance and provenance.

The policy ID includes the threshold, engine version and fixed reference values.
The evidence ID includes input/provenance content. These hashes identify content;
they do not authenticate sources or prove scientific validity. Replay uses the
installed policy; matching evidence IDs alone does not establish matching policy.

The default follow-up threshold remains 5 **ranking units**, not five Gaussian
standard deviations. KY26 is not used to tune the policy. Its rounded frozen
inputs retain rank 4.141288008229254; the user-recorded historical display 4.142
and old posterior/p-value remain in the separate historical baseline. Neither is
silently replaced by the new output.

The ranking still includes fixed global population assumptions, dependent
size/photometry features and hand-set inclination bonuses. It is unsuitable as
a calibrated discovery or origin statistic. Proxy contributions carry dependency
groups; their count is not a count of independent observations. Conditional
populations, quantified dependencies and independent calibration remain #10/#11.

## Controlled deprecations and tests

Existing public facade/function names remain importable. The misleading name
`ValidatedSigma5ArtificialNEODetector` confers no validation. Its private
`_bayesian_evidence_fusion` and `_perform_cross_validation` methods now raise
`NotImplementedError` explaining the withdrawal. Constructor examples remain
available as legacy data but cannot enable a validation claim by assignment.
Archived wrappers are not registered as automatic fallbacks; failure is explicit.
Public component adapters no longer expose heuristic p-values as significance.
The old API `_sigma_tier`, `_build_interpretation` and `_spacecraft_veto` helpers
also remain importable but explicitly reject their withdrawn inference contracts.

Old tests that required Tesla to reach sigma5, forced natural-origin verdicts,
or required toy F1>0.8 tested the withdrawn contract. They are replaced with
executable unavailable-inference, frozen-rank, candidate positive/negative,
replay, entry-point parity and evaluator fail-closed regressions. Conditional
assertions that passed without exercising validation are replaced with direct
assertions. Source-string checks for sigma-tier UI text become rendered-output
checks. No calibration acceptance threshold or coverage gate is lowered.

The former live “accuracy” test selected a threshold on its evaluation objects
and converted failures to zero. It was not an independent evaluation. Its
replacement verifies that this deployed uncalibrated service cannot obtain a
probability-accuracy report from unavailable outputs. This test **does not pass
the scientific accuracy requirement**; independent evaluation remains open #11.
The existing optional provider integration tests remain network-marked.

Tests: `tests/test_screening_service.py`, `tests/test_screening_numerics.py`,
`tests/test_validated_sigma5_detector.py`, `tests/test_ky26_baseline.py`,
and the affected menu/API/evidence contract tests.

```sh
python -m pytest tests/ aneos_core/tests/ -m "not network" -q \
  --cov=aneos_core --cov-report=term-missing --cov-fail-under=30
```

Arithmetic regressions cover normal-tail cancellation, Fisher log tails and
underflow bounds. Correct arithmetic is not validation of an astronomical null.

## Deliberately unfinished

- Automatic multi-stage review still contains legacy validation logic; sharing
  its screening helper is not proof that required validation cannot be bypassed (#21).
- Historical CAD polling still has placeholder orbit/score paths; no validated
  evidence credit is given to those paths. Repair and outage tests remain open (#22).
- Network-pattern, ATLAS proxy, legacy front ends and scientific-help documents
  still need an end-to-end claim audit. This increment is not “all interfaces
  scientifically corrected.”
- API batch/export state remains process-local and menu persistence uses daemon
  threads. Restart-safe jobs/results remain #15.
- Provider-wide scheduling, production settings, observation fitting, valid BIC,
  hypothesis comparisons, independent evaluation, complete Markdown reports and
  case lifecycle remain the linked programme issues.

No origin model is evaluated here. “Not flagged” does not mean natural.
No new endpoint, bounded context, external inference service or production
deployment is introduced.
