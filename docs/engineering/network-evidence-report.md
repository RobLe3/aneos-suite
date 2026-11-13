# Population evidence report correction

Date: 2026-11-13. Scope: BC11 normal session, existing REST/menu report transport,
controlled deprecation of unvalidated population prototypes. Issues #6/#10/#12/#15/#17.
This is not the whole-app or complete 1998 KY26 scientific review.

## A. Engineering summary

Confirmed defects in the preceding session:
- Mutable legacy `dynamic_anomaly_score` selected objects under a sigma label.
- Minimum selected cluster/period p-values were fused as independent calibrated
  streams, with inadequate search-wide accounting and routine defaults for absence.
- Missing angular elements became zero, orbital bounds clipped extremes and optional
  HDBSCAN installation silently changed the grouping algorithm.
- Computed close-approach dates became zero-filled daily pseudo-observations; power
  standardized over selected frequencies became a claimed normal-null p-value.
- One A2 coefficient per object was passed to pairwise temporal Pearson correlation;
  undefined evidence could become zero/one or an exception.
- REST queued zero sigma/p=1, ignored requested rendezvous, returned the wrong status
  URL and could omit acquisition failures or expose exception details.

Changes: `NetworkAnalysisSession` owns full frozen JSON NEO snapshots, shared
ScreeningAnalysisService/policy results, ordered duplicate assessments, explicit cap
exclusions, config/input/result IDs and whole-report deterministic replay. Caller
objects are not mutated. No hidden provider work or stored-score selection occurs.
Conflicting duplicate snapshots fail rather than silently selecting one solution.

Exploratory grouping uses explicit scaled seven-component (a,e,i and sine/cosine of
node/periapsis) circular distance and connected components. Scales: 0.1 AU, 0.05 e,
5° i and 10° angular chord scale; radius 1 and minimum 3 unique members by default.
These are versioned **engineering exploration choices**, not fitted population or
measurement-uncertainty scales. No clipping, imputation, density null, family label
or probability. Chaining, angular degeneracies, incompatible catalogue epochs/frames,
survey selection and unpropagated covariance limit physical interpretation. Members
are sorted deterministically; optional installed clustering libraries cannot alter
normal output. A positive synthetic group tests computation, not real-world accuracy.

Computed encounter inventory uses only matching-identity Earth events with finite
native TDB JD, deduplicates equal epochs and reports gaps/exclusions/unestablished
coverage. Calendar dates are not silently interpreted as UTC/TDB, other bodies are
not mixed in, and no artificial zeros or spectral claims are generated. Complete
raw encounter records remain in snapshots. The requested historical horizon is
recorded, explicitly not acquired/applied by this pure boundary. Future acquisition
must freeze its query/window/coverage evidence before analysis.

Correlation preserves all A1/A2/A3 estimation/units/uncertainty/covariance evidence
but requires paired temporal observations; within-orbit coefficient covariance is
not cross-object temporal covariance. Requested rendezvous remains unavailable, not
silently successful. Disabled/descriptive/unavailable states are distinct.

Legacy public classes/functions remain available behind `legacy_diagnostic=True`;
otherwise their normal `run`/`combine` methods raise a controlled deprecation.
Their historical formulas and numerical tests remain for comparison, not calibration.
The old sigma-filter/fusion configuration fields reject operational values explicitly.
An additional model defect found by a failing regression test is fixed:
`OrbitalElements.from_dict` must use an explicit None check so angular zero does not
fall back to a contradictory alias. Both values survive serialization for diagnosis.

Existing REST responses preserve the complete scientific report and ordered
acquisition outcomes, reject wrong-object responses and redact public exceptions.
No available inputs is an error, partial acquisition remains visible. Queued and
finished sigma/p remain null; the registered status URL is returned. Menu uses actual
typed/matching identities, preserves requested duplicates, lists unavailable inputs,
verifies report replay and displays ranking candidates/descriptive groups, not sigma.
Markdown rendering verifies before emitting summary plus complete JSON.

## B. Scientific assessment

No new provider observations, KY26 measurements, fitted force models, probabilities,
calibration populations or physical-origin results were generated. The historical
KY26 baseline is unchanged. The earlier heuristic p-values do not become empirical
significance merely through Fisher/Stouffer or multiplying by object count. These
procedures are withdrawn from normal population inference, not tuned to an object.
Five-sigma claims remain disabled; an unflagged ranking is not a natural verdict.
Natural/Phobos/artifact models, valid observational BIC, conditional population
calibration, independent labels/holdout and scientific uncertainty remain #7–#11/#25.
No disabled-only gate is credited: actual shared-service candidate and actual
synthetic-neighbourhood positives are tested alongside missing and invalid inputs.

## C. Architecture impact

ADR-077/DDD clarify BC11's existing aggregate and its dependency on canonical
screening, acquisition and execution storage. No additional engine, UI, external
service or bounded context. Files changed: session/prototype modules, the angular
model deserializer, existing network API/schema, V2 menu, regression tests,
README/REST/OpenAPI, ADR/DDD and this report. New tests are force-added because the
repository ignores `test_*.py`; no scientific tests are disabled or thresholds lowered.

## D. Remaining work and next gates

1. Register an actual resumable handler using ADR-076: checkpoint frozen successfully
   acquired inputs and historical query coverage, reuse on retry, validate request/
   policy/report identity before finish. Distinguish acquisition failure, invalid
   evidence, scientific abstention and execution failure; bound provider retries.
2. Managed worker startup/shutdown/heartbeat/cancellation and durable API submission/
   polling/results. Derive owner scope from verified principals, not a caller string;
   actual authorized positive and cross-owner denial controls are required.
3. Actual full source-injected worker/API success, outages, cancellation, process
   termination/restart, stored-result replay and request ordering. Current real-session
   synchronous worker tests do not prove durable execution or authenticated isolation.
4. Independent population nulls/selection accounting, held-out sensitivity/false
   positives/abstentions and scientifically fitted observational comparisons before
   any significance/origin assertion. Real physical grouping and periodicity remain
   unvalidated; descriptor outputs are not their replacement.

Current network API **still uses daemon threads/process-memory storage** and is
not owner-isolated. No generic JobStore activation is claimed. Existing automatic/
polling jobs, case lifecycle, full source provenance audit, full application startup,
security, backups and production readiness remain open. All PRs are drafts/unmerged.

## E. Reproducibility and verification

Tests use frozen synthetic inputs, disposable databases and the existing offline
network-denial gate; no provider data is silently substituted or fetched. Initial
new-contract regression run: 22 failures against the preceding session. Subsequent
fixtures corrected mistaken assumptions (out-of-domain orbit was rightly inconclusive;
duplicate test inputs must share the same complete snapshot). This did not alter
scientific scoring. The alias-zero regression exposed an actual deserializer defect.
The first full run had 936 passing tests plus the expected stale OpenAPI snapshot
failure; the snapshot is regenerated from the actual app, not its test relaxed.

Final commands/results are recorded below and in the draft PR after completion.

```sh
PYTHONDONTWRITEBYTECODE=1 ANEOS_DATABASE_URL=sqlite:////tmp/aneos-network-final1.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core --cov-report=term-missing --cov-fail-under=30

COVERAGE_FILE=/tmp/aneos-network-component2.coverage PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-network-component2.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/test_network_evidence_report.py \
  tests/test_pattern_api.py -q --tb=short --cov=aneos_core \
  --cov-config=tests/coverage-network-report.ini --cov-report=term-missing --cov-fail-under=90
```

Final full suite: **949 passed, five network deselected, five warnings in 39.00s;
45.39% core coverage against the unchanged 30% gate**. This branch adds 47 controls:
40 report/menu/deprecation cases, six actual-worker cases and one native fitted
A3/covariance case, alongside strengthened existing transport tests. Targeted report/
API suite: **51 passed, 95.86% session coverage against the new 90% gate**, 3.88s.
The existing 85% job-store gate remains unchanged. No live provider calls were needed.

An initial deep dotted-source coverage invocation failed during eager parent imports
with a SciPy native-module import error and NumPy-reload warning. The component gate
now uses top-level source discovery plus an explicit report include for the session;
the command above measures its 169 statements without that early dotted-source
import path. No dependency reinstall, warning suppression, coverage threshold
reduction or scientific-test exclusion was used. Full-suite and targeted receipts
are distinct and both complete; earlier exploratory runs are superseded.

Remote CI/commit and draft PR references are recorded separately after publication.
The root experimental working tree/database and dated KY26 artifacts are unchanged.
