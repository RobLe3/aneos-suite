# Resumable population worker

2026-11-13. Scope: actual registered handler/worker on the existing SQL JobStore,
fixed-window CAD acquisition and request/checkpoint/report validation. #6/#15/#16/#17.
This is the dependency between storage/report contracts and managed authenticated API
execution; it does not by itself migrate the current endpoints.

## A. Engineering summary

`aneos_api.network_jobs` registers `network-evidence-job-v1` and exposes an explicit
NetworkJobHandler/NetworkWorker, request builder and result verifier. There is no
import-time execution, new REST endpoint, CLI flag, service or bounded context.

A request contains ordered bounded designations, exact immutable pattern configuration,
screening policy and current reference fingerprint, report engine identifier, source
configuration fingerprint and historical query bounds. It rejects unknown options,
invalid units/structure, incompatible policy versions and batches above their explicit
cap rather than ignoring or silently upgrading them. The default anchor is the UTC
calendar date used as a query label; it is **not** a UTC-to-TDB observation conversion.
The historical horizon is explicitly years ×365 days, not a claim of calendar-year
or observation-arc precision. A caller can supply an explicit ISO date.

The configured source fingerprint hashes current API adapter settings without storing
raw credentials. The default factory honors configured source priority. These checks
are neither source authentication nor a complete source/environment lock; actual
native source, solution and query metadata remain in frozen records. Drift blocks new
acquisition; fully checkpointed recovery need not initialize a changed/unavailable
provider. Default source/cache behavior is retained; acquisition does not promise a
fresh orbit solution merely because execution is new.

Successful typed, matching NEO snapshots are validated and committed before the next
provider request. Successful bounded historical responses get separate immutable
checkpoints. Repeated requested identities share one acquisition while retaining
ordered per-request assessments. Source failure/missing return is retryable, not a
frozen zero or scientific negative; malformed/wrong-identity evidence is refused.
Prior successful snapshots are reused after outage/restart. Conflicting or unknown
checkpoints cannot be silently selected. Public errors are bounded codes, not provider
URLs/credentials/exception details. Internal execution failure is not scientific
abstention. Failed/cancelled/uncertain writes do not become completed reports.

`DataFetcher.fetch_approaches_between` uses explicit date-min/date-max labels with
the existing native-TDB parser/provider coordination. The old relative-horizon method
remains available. JPL documents date filters, native TDB epochs and the meaning of
limit/total in its [CAD contract](https://ssd-api.jpl.nasa.gov/doc/cad.html), checked
2026-11-13. This adapter intentionally retains a bounded ten-row query and explicit
partial/complete count evidence; it is **not** a complete population/time-series
acquisition or pagination implementation. An empty valid response is not an outage.
Alias identity not established by the typed source must be resolved upstream, not
silently rewritten by the worker. Fixed query bounds prevent retry-time window drift;
new orbit solutions can still change not-yet-frozen measurements.

For historical descriptors, the derived report input uses the frozen historical
events rather than mixing supplemental current/future encounters into that series.
Original NEO encounters and historical rows/query evidence remain separately preserved
in the acquisition register. No computed event becomes an observed acceleration.
The pure report continues to state that it does not itself acquire/filter a window;
the execution envelope records which acquisition supplied its inputs.

Before terminal commit, the worker recomputes the report and compares its acquisition
register against the actual immutable SQL checkpoints. Hash-only or self-consistent
report replay is insufficient: a rehashed alternative input must not replace committed
evidence. A final report checkpoint also supports recovery without provider calls.
Storage failure remains unconfirmed/raises rather than acknowledged success; lease
expiry permits later recovery, subject to the persisted attempt budget.

Provider calls and calculations occur outside SQL transactions. A supervised heartbeat
renews leases while work remains live; stop/attempt deadline ends renewal even if a
provider is blocked. Another worker can reclaim after expiry; the old worker cannot
checkpoint, fail or finish that reclaimed execution. Retry deadlines/attempt counts
persist. Exposed provider cooldown deadlines are honored; existing JPL shared pacing
still governs actual requests when intermediate adapters wrap exception detail.

## B. Scientific assessment

The worker executes the same versioned, uncalibrated canonical screening and population
descriptor reports as ADR-077. Real synthetic candidate positives remain candidates;
missing physical measurements can remain inconclusive. Completion is not validation,
a natural verdict, artificial probability or five-sigma significance. No observations,
KY26 score, force models, fit likelihoods, physical-origin ranking or independent
calibration were generated or altered in this change. #7–#11/#25 remain open.

## C. Architecture and operating boundary

ADR-078/DDD describe composition across existing acquisition, BC11 and execution
infrastructure. JobStore schema is unchanged; only the new kind is claimed. Unknown
kinds remain available to other registered workers. Production setup remains explicit:
initialize the existing file-backed SQL store, build the request, submit with a trusted
owner scope/idempotency key, and explicitly call NetworkWorker.run_once or host its
blocking serve loop. These are Python APIs, not invented command-line options.

Current API ownership/authentication is not adequate merely because a backend accepts
an owner string. **No endpoint or application-lifespan worker is activated here.**
Existing network API still runs a daemon/process-memory job. Do not represent these
library tests as whole-app durable execution. Tests inject synthetic sources into the
actual handler; they are not independent astronomical validation.

## D. Remaining work and limits

- Integrate verified principals and authorization, durable submission/status/results/
  cancellation into existing routes, including idempotency and safe error semantics.
- Host an explicitly configured managed worker in API lifespan or an approved process;
  expose actual readiness/worker failure, stop admission/shutdown predictably and test
  actual application/API restart/outage/cross-owner positives and denials.
- Retain request/checkpoint/result history; migrate/deprecate existing in-memory stores
  without deleting research. Scientific version changes require explicit compatibility
  policy; incompatible queued jobs are not silently upgraded.
- Provider schema/canonical alias resolution, source-wide config/cache/version policy,
  complete historical pagination and backpressure remain #16. Ten-row partial results
  cannot establish absence of a pattern or completeness of an observation history.
- Deadlines are cooperative calculation boundaries and renewal limits, not hard wire/
  process cancellation. Blocked calls/driver waits cannot be forcibly killed inside
  Python. Process supervision and deployment database/provider timeouts are required;
  serve does not swallow storage failure and advertise success. Heartbeat-thread joins
  are subject to the configured database driver returning. PostgreSQL, cross-host
  clock coherence, hardware/power-loss recovery and backup/restore are unverified.
- Provider GETs can repeat if interruption precedes checkpoint commit. No exactly-once
  external side effects, source authentication or cryptographic tamper proof is claimed.
- Full scientific case lifecycle, calibrated independent populations, observational
  natural/technological/artifact fits and whole-app readiness remain separate work.

Rollback means stop the explicitly hosted worker and review queued/current jobs;
never reset/delete existing tables, checkpoints, reports, research or source files.
An old worker lacking this registered kind must not consume these requests.

## E. Reproducibility

New-contract red run: eleven failures before the worker module existed. Targeted
SQL tests subsequently passed, including a real child-process kill after a committed
input while a second acquisition blocked; recovery reused the committed first input,
completed the second, and another process read and replayed the terminal report.
Separate controls cover unconfirmed final storage and recovery from the final report
checkpoint with a forbidden provider, actual slow-provider heartbeats, cancellation,
stop/deadline recovery while an old provider remains blocked, request/source-policy
drift, native query coverage/count binding, forged rehashed result versus committed
inputs, bounded exhaustion, unknown-kind isolation and the cooperative serving loop.
No live network observations are used, and the existing offline denial guard remains.

Final commands/counts/commit and remote CI are recorded below and in the draft PR.

```sh
PYTHONDONTWRITEBYTECODE=1 ANEOS_DATABASE_URL=sqlite:////tmp/aneos-worker-final3.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core --cov-report=term-missing --cov-fail-under=30

COVERAGE_FILE=/tmp/aneos-worker-component2.coverage PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-worker-component2.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/test_network_job_worker.py \
  -q --tb=short --cov=aneos_api --cov-config=tests/coverage-network-worker.ini \
  --cov-report=term-missing --cov-fail-under=90
```

Final full suite: **989 passed, five network deselected, five warnings in 44.90s;
45.41% core coverage against unchanged 30% gate**. New worker suite: **40 passed,
93.73% worker coverage against added 90% gate**, 7.46s. Existing job-store85 and
population-report90 gates remain unchanged. No pre-existing tests were disabled.
The new test file is force-added because the repository ignores `test_*.py`.

The stopped/expired-blocked-provider tests advance the controlled attempt deadline
only after the source is actually blocked, and wait for actual SQL lease expiry;
they do not assume startup finishes in a sub-second scheduling window. Process-kill
recovery uses a shared controlled lease clock and real process termination/database
commits/readback. These are bounded process/transaction controls, not hardware crash
or astronomical-performance certification. Earlier exploratory runs are superseded.
The root experimental working tree/database and historical KY26 artifacts are not
modified. Final commit/PR and remote verification are recorded separately.
