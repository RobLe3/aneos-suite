# Enhanced REST contract correction

Date: 2026-11-12. Branch: `codex/enhanced-rest-contract`.
Base: `7ba0ee1734a2169a72c7a854132248304232d28a` (draft #28).
Programme #2; analysis/reporting #6/#12; review integrity #21/#23;
provider/durability/full verification #16/#15/#17. Decision: ADR-070.

## A. Engineering summary

### Confirmed defects

The declared enhanced REST adapter returned MockNEOData and a MockPipeline with
fixed score .75, confidence .85 and potentially-artificial classification. Its
models failed import in the installed Pydantic runtime (`regex` removed), required
unavailable inference as floats and supplied fabricated uncertainty examples.
Main-app routing did not register the declared routes. Single analysis ignored
request flags; batch analysis lost flags, item outcomes and timing. Summary code
converted mixed legacy ranks into FP probabilities and accepted natural/inconclusive
labels without review evidence. Logging formatted unknown FP probability as a float.

### Changes

- Register the existing single/batch/summary routes in the real app under the
  existing prefix/auth dependency. This makes previously disconnected routes
  reachable; it does not introduce another engine or interface.
- Use the same actual application AnalysisPipeline as `/analyze`, once per object.
  Rebind the review wrapper when the application pipeline changes; an unavailable
  current service cannot silently use a stale cached wrapper or mock result.
- Carry the canonical assessment, IDs, raw typed data on request, proxy diagnostics,
  required-review outcomes and explicit execution status. Actual elapsed time is
  measured; uninstrumented stage/validation timing defaults become null in REST.
- Forward refresh to acquisition, preserve raw/indicator flags, and record method/
  sampling preferences as requested-but-unperformed where no model is installed.
  `fast`/`statistical_only` cannot bypass prerequisites. Disabling optional review
  is `not_requested`, not a pass; concurrent requests do not mutate shared flags.
- Preserve ordered batch outcomes, duplicates, source/invalid/processing errors,
  abstentions and timeouts. Enforce existing list-shaped request with 1–100 nonblank
  designations before acquisition. Batch counts describe execution, not accuracy.
- Check purported workflow passes for five unique explicit boolean/stage outcomes,
  no stage errors, required contract/status, matching ranking IDs and matching full
  review-input hash. Incomplete, truthy, errored or stale passes become unavailable
  without erasing a valid core candidate. This transport guard is not a physical
  validator. No calibrated numerical artifact is installed: legacy probabilities,
  reduction/confidence/score scalars and synthetic percentiles are not promoted
  through review transport. Original objects and stored historical rows are not
  rewritten; native measured inputs remain in their provenance-bearing sidecars.
- Summary reads existing stored screening payloads, replays supported current-policy
  inputs and checks inference fields. Legacy, modified and unsupported-policy rows
  are counted as excluded, not rewritten. Legacy scalar columns are not used to
  manufacture probabilities. Summary units are persisted rows, not independent
  objects; the date filter is assessment date, not source observation date.
- Report unavailable FP estimates as null/`not_reliably_estimated`. Without valid
  stored review evidence recommendations remain unknown, never automatic acceptance.
  Empty rank statistics are null. Database availability is not scientific validity.
- Update nullable models, request validation, OpenAPI and actual route documentation.
  Existing function names and route shapes remain; unsupported scientific promises
  are withdrawn, not replaced by invented numbers.

### Verification

`tests/test_enhanced_rest_contract.py`: **37 new controls**, denying external socket
connections. Initial ten controls failed at model import; after model correction,
nine still failed on real adapter/routing/response behavior. Additional malformed
pass and unsupported probability regressions failed before their guards. No prior
suite tests were edited or disabled.

Controls include real initialized core/review components and shared service;
candidate positive/negative and missing/invalid/source cases; actual main-app HTTP
routing/serialization; concurrent flags; list-body/query flag compatibility;
ordered mixed/timeout outcomes; current-service identity; full native A3 sigma,
units, correlations and raw provenance; actual read-only SQLite date filtering;
legacy/tampered/unsupported summary exclusion; OpenAPI parity; strict stale review
bindings and a complete synthetic review-transport positive control. That control
is workflow evidence, explicitly not a real scientific model or accuracy result.

Final suite selected by `-m 'not network'`: **645 passed, four deselected, five
warnings, 27.29 seconds; 40.25% coverage**, unchanged 30% minimum. `git diff --check`
is clean. No new live provider query or external scientific observation occurred.

## B. Scientific assessment

This increment adds no updated KY26 significance, calibrated artificial-origin
probability, physical fit, likelihood/BIC comparison or assessment of Phobos 1
versus natural explanations. Historical KY26 fixtures/results are unchanged and
included in the full regression suite. A3/covariance transport is demonstrated,
not a force interpretation or covariance epoch propagation. Screening status is
not origin identification. Five-sigma permission remains false. A review workflow
pass is not independent scientific confirmation.

## C. Architecture impact

Changed production files: `aneos_api/app.py`,
`aneos_api/endpoints/enhanced_analysis.py`, `aneos_api/enhanced_models.py`,
`aneos_core/analysis/enhanced_pipeline.py`. Added test file above. Updated ADR-070,
DDD, scientific integrity notice and API/OpenAPI documentation. No new bounded
context, research engine or UI. GitHub tracking remains open pending complete
coverage/review; the PR supplies commit/CI provenance, not a claim of merged readiness.

## D. Remaining work

Legacy menu/direct-detector/advanced views and pattern/network/standalone model
claims still require audit under #6/#12. New enhanced reports currently log; they
are not persisted by these routes and are not a durable case history. Summary is
restricted to existing supported screening rows, not complete enhanced review
history. Provider-wide scheduling/configuration/backoff, worker cancellation,
restart recovery and append-only cases remain #16/#15/#13. Cancellation stops an
await, not a blocking provider thread. This is not a resource/durability assurance.

Real source-property uncertainty, full dated KY26 primary-source investigation,
astrometric observations, fitted natural/technological/artifact comparisons,
conditional populations, dependent evidence and independent evaluation/calibration
remain #25/#7–#11. No confidence interval or origin finding is supplied to fill
those gaps. No independent reviewer certification was performed by the coding agent.

## E. Reproduction

Review branch/base above; final commit and CI are recorded on the draft PR.
No new dataset, measurement capture or scientific fixture/hash was created or
changed. Synthetic markers exercise contracts only. Current frozen KY26 input
references/hashes remain in the existing case fixture/baseline documentation.

```sh
PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-enhanced-review.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core \
  --cov-report=term-missing --cov-fail-under=30
```

Work was performed in an isolated Git worktree. The original dirty checkout and
its database were not overwritten. The SQLite summary control creates its own
temporary database; it neither inspects nor writes unrelated user records.
