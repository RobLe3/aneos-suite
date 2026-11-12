# Scoped Sentry impact evidence and REST correction

Date: 2026-11-12. Branch `codex/authoritative-impact-evidence`.
Base `e9b3c54f655e115941cce5a30e075308e682e0e8` (draft #34).
Programme #2; impact #32; source #16; reporting/verification #12/#17; case #7.
ADR-074. This increment fixes the existing REST impact contract, not all legacy
impact callers, a local orbital solver or planetary-defense readiness.

## A. Engineering summary

Confirmed: `/analysis/impact` called the legacy geometric/arc-scaling calculator
without a residual fit or propagated covariance, implicitly used a 30-day arc, and
required a probability. Numerical intervals/confidence/damage outputs implied
unsupported certainty. Shared JPL transport did not remedy that scientific defect.

The route now delegates to a strict Sentry Mode O evidence adapter via the shared
provider guard. It validates source/version, primary designation, known method,
finite values, event counts and native dates. Exact UTF-8 bytes/hash, source method,
observation/computation/retrieval dates and complete events are retained. A report
replays from its raw snapshot and is bound to the requested object. The existing
nullable response model exposes unsupported legacy quantities as null, not empty
or fabricated intervals. Errors remain explicit 422/502/503/500 outcomes with
redacted public details, not successful zero-risk or natural-origin assessments.

The source service reuses valid dated outcomes in a bounded one-hour/128-object
per-process cache and returns copies. Expiry/refresh obtains a new snapshot without
mutating previously returned reports. Single-flight callers share the actual
attempt's result **or failure**; failed refreshes cannot substitute old success
for waiting callers. Shared JPL transport still governs cross-process wire access.
Cache content is not durable history or a provider-health claim.

JSON and replay-checked Markdown helpers share the same evidence. No new route,
CLI option or UI is invented. The actual registered application route is exercised.
The published OpenAPI snapshot was stale for earlier core/screening changes as well
as this route; it is regenerated from the current app, without service startup or
source acquisition, and checked against registered schemas/routes.

Review exposed defects in the first new implementation: a valid report for
another object could pass transport replay; refresh waiters could get an old cached
success after the active refresh failed. Failing regressions were added and both
were corrected before publication. A second guarded live source-format control
for `101955` returned 157 MC virtual impactors but exposed unsupported native
fractional-day observation dates (including an unpadded day). The same captured
bytes now pass the corrected parser and full guarded-service/HTTP replay offline;
no duplicate live request was needed. Native UTC date strings remain unchanged.
The diagnostic capture did not persist its exact retrieval time; the fixture
metadata records that limit rather than guessing a time from source fields.

## B. Scientific assessment

Sentry output is an **external model prediction**, not aNEOS observational fitting,
independent validation, observed impact, origin evidence or Gaussian discovery
significance. Source probability applies to tabulated virtual impactors, not all
future trajectories. Native method-specific orbit-fit metrics retain their role;
no conversion into discovery sigma is made. Source diameter/mass/energy estimates
are not automatically treated as independent measured physical properties.

JPL documents IOBS, LOV and MC methods, substantial probability uncertainty,
method-dependent fields and separate removed/not-found outcomes. These constraints
were checked against the [primary Sentry contract](https://ssd-api.jpl.nasa.gov/doc/sentry.html)
on 2026-11-12. The report carries the material uncertainty caveat rather than an
invented probability interval. No lunar/annual/decadal probability, local confidence,
crater/damage estimate or local Torino class is inferred.

One current guarded live KY26 transaction returned **not listed** at
`2026-11-12T21:19:13.177037Z`. This is not zero risk and says nothing about natural,
human-technological or extraterrestrial origin. Its exact short public response,
report and execution metadata are frozen in
`docs/case-studies/1998-KY26/sentry/20261112T211912Z/`.
Raw SHA-256:
`722bec8de705159e59e00f6be8592e1233b0301370c65a786e2a0072cfeba2b9`.

Historical screening score/results and the separate SBDB measurement capture are
unchanged. No acceleration, hypothesis fit, independent predictive comparison,
Bayesian calibration, FDR estimate or five-sigma claim is produced here. A real
negative catalogue lookup is bounded source evidence, not solver validation.

## C. Architecture and verification

No new bounded context: Data Acquisition owns the Sentry anti-corruption adapter;
Impact Assessment owns scoped interpretation; the existing API/report boundary
transports evidence. Source absence/removal, source failure and unsupported local
inference are distinct contracts. ADR-074, DDD, REST/OpenAPI and the KY26 register
are updated. The new adapter does not consume a possibly incompatible local orbit
or fabricate missing covariance/observations.

`tests/test_sentry_impact_contract.py` provides external-socket-denied
controls for all methods, valid positive event reports, absence/removal, schema/
identity/method/count/numeric/date errors, duplicate/nonstandard JSON, tampering,
real guarded service transport, cache copies/expiry/refresh/concurrency/failure,
actual application routing, errors/redaction, Markdown, the frozen live capture
and published OpenAPI. Synthetic payloads are explicitly not observed impact data. The real MC source
fixture is separately identified and hashed; it is not synthetic calibration data.
`tests/test_sentry_network.py` adds one optional KY26 live integration test; offline
CI does not run it. The single live transaction described above was executed
separately and preserved, not reported as an executed network pytest.

Root test scheduling isolation excludes explicit network-marked tests so optional
live calls share the normal deployment coordination directory instead of silently
bypassing provider serialization. No prior test is disabled and no gate is reduced. Two existing OpenAPI tests
now compare the actual registered application schemas rather than raw Pydantic
model output: FastAPI omits `default: null` while retaining nullable branches.
Their nullable scientific-inference checks remain explicit, and the full published
spec is compared against the real application. This corrects a documentation-test
seam; it does not change scientific outputs or weaken their nullability contract. Final full suite: **824 passed, five network tests deselected, five warnings in
31.23 seconds; 43.25% core coverage against the unchanged 30% gate**. There are
**56 new offline controls**, plus one optional network test. The final commit and
remote CI state are recorded with the PR. A concurrent-first-request singleton
control also exposed multiple initial source-service construction; explicit locked
initialization now preserves one per-process cache/single-flight service.

## D. Remaining work

Issue #32 remains **open/P0** for V2/legacy menu, the research calculator's claimed
probability/confidence outputs and ImpactEnhancedPipeline's proxy inputs/interpretation.
This REST correction must not certify those paths. Keep actual physical/scenario
computations available with explicit assumptions and controlled deprecation where
scientific contracts are defective. Alias/name resolution is not implemented;
unsupported selectors fail explicitly rather than risk cross-object attribution.

Source-wide configuration/version/cache/coverage/backpressure work remains #16.
Durable jobs/results and append-only research cases remain #15/#13. The Sentry
cache is process-local, ephemeral and not an incident-warning service. Real KY26
literature extraction, physical uncertainty ingestion, compatible observational
fits, origin hypotheses, independent controls/calibration and frozen predictions
remain required. No issue is closed by this partial PR.

## E. Reproducibility

```sh
PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-sentry-final5.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core \
  --cov-report=term-missing --cov-fail-under=30
```

Optional, **live network** (one fresh query, not required for offline tests):

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/roble/.venvs/aneos-suite/bin/python \
  -m pytest tests/test_sentry_network.py -m network -q
```

Dated source artifacts include exact bytes and acquisition-code file hashes.
Their metadata identifies the base revision and then-uncommitted source hashes;
it is not a complete environment/dependency lock. Replay hashes detect content
changes, not malicious authorship or cryptographic source authenticity. New source
queries must create new dated records, not overwrite this or the original baseline.
The root experimental worktree/database are not modified by this branch.

### CI environment correction

The first remote run had **823 passing tests and one OpenAPI snapshot failure**.
Inspection of the completed job's exact schema diff showed only ten missing
optional dashboard/streaming paths, not changed scientific/impact fields. The CI
`requirements-core.txt` installed FastAPI but omitted Jinja2/SSE, while the published
full API profile and local environment included them. Both packages were already
in `requirements.txt`. They are now explicit in the core API/CI dependency set;
the full snapshot assertion remains unchanged. No failure is skipped and no
scientific or schema requirement is reduced. Minimal custom installations can
still omit optional UI packages, but are not the documented full API test profile.
