# Durable primary-menu batch execution

Development work under #15/#2/#41; ADR-095. Not in published v2.2.0b2.
This qualifies a specific local file-SQLite batch path, not every legacy/API job,
production deployment or scientifically calibrated inference.

## Submit or resume

Use an explicit, separate writable file-SQLite database, as in the installation
guide. Primary menu option 3 first asks for a saved job ID or recovery key. Press
Enter to choose a new designation file; enter a saved token to resume/inspect the
original request without rereading a mutable file.

Before submission, the menu prints a recovery key. Retain it even if the following
acknowledgement fails. Submission, readback and queue admission must succeed before
provider work starts. A storage error has no unacknowledged in-session fallback.
Retry the same key/ID rather than assume an unconfirmed write did not commit.
Changed input/configuration under the same key is a conflict, not a replacement.

Cooperating processes must share the existing JPL transport state directory on a
filesystem with working POSIX locks; see [provider transport](jpl-provider-transport.md).
This is not multi-host provider coordination or full provider-policy certification.

The configured source profile, canonical policy/version, input bytes/lines, execution
budgets and submission time are frozen. Exact repeated labels share an acquisition
but remain separate request rows. Existing bounded concurrent batch scheduling is
preserved. Source/cache/retry behavior is not an exactly-once HTTP or freshness
promise; actual source metadata and the acquired evidence remain in each report.

Normal menu submission permits three **execution** attempts for interruption recovery
and one observed source outcome per object. Thus an acknowledged source outage is
reported as unavailable, not silently polled again or reclassified natural. Advanced
callers may explicitly request bounded source retries within that execution budget:

```python
from aneos_api.batch_jobs import default_runtime
from aneos_core.analysis.batch_reporting import read_input
runtime = default_runtime()  # explicit ANEOS_DATABASE_URL; starts no background loop
try:
    job = runtime.submit(read_input("objects.txt"), "operator-retained-key",
                         max_attempts=3, source_attempts=2)
    state = runtime.run(job["job_id"])
    print(state["status"], state["request_progress"])
finally:
    runtime.close()
```

A retry-wait state does not bypass its recorded deadline. Queue eligibility and
lease dates are displayed as execution UTC timestamps, never as observation dates.
A completed job means its processing result committed, not that all sources supplied
measurements or scientific validation passed. Terminal batch reports retain failed,
invalid, missing and successful requests with explicit counts.

## What survives failure

The registered `canonical-batch-job-v1` handler uses the existing `JobStore`, not a
new database or queue. It checkpoints each existing per-object report before that
result is acknowledged. A checkpoint callback failure is fatal; unlike a cosmetic
progress callback, it cannot be ignored. A pre-item callback checks current execution
authority and source configuration before starting more work.

Successful/inconclusive reports are immutable and reused after recovery. Transient
failure attempts are preserved separately; explicit retries select the appropriate
later outcome without overwriting an older one. After retry exhaustion, a processed
source failure remains an unavailable report. Interrupted work that has not processed
all requests cannot claim a complete batch; its partial checkpoints remain inspectable.

Final output must reproduce the request and committed reports. A separate immutable
assembly-clock checkpoint freezes the actual report-construction time. Submission,
report assembly, source retrieval and astronomical epochs are different roles. A
recovery does not substitute the current date for an earlier acquired measurement.

Job result/event/admission release commit transactionally in the existing store.
Leases fence request, kind, token, attempt and attempt budget. Owner-scoped inspection
also verifies checkpoint bindings. Targeted claims cannot accidentally run another
queued batch instead of the selected one. Fully checkpointed recovery and completed
readback do not initialize a provider.

## Read-only history and permissions

The menu uses a fixed trusted local application owner, not authentication or OS-user
isolation. File permissions and host administration remain the operator's responsibility.
The existing authenticated network API has its own principal scopes; this increment
does not introduce an unauthenticated batch REST endpoint.

Explicit `mode=ro` or recognized immutable-reader URIs inspect existing completed or
partial jobs without schema initialization. They cannot submit or resume provider
work. Ordinary writers require an explicit file-backed database and the existing
safe SQLite settings. Memory stores and unqualified PostgreSQL batch hosting are
refused before provider work. OS permissions, disk exhaustion and later commit
failures can still make storage unavailable.

Completed-job replay does not append another latest-object compatibility analysis
merely because it was viewed again. The authoritative job result and source history
remain in SQL. JSON/CSV exports are portable exclusive copies, not the sole archive.
A standalone `build_batch_report()` call still performs no persistence by itself.

## Limits

- The worker cannot forcibly abort an already started blocking provider operation.
  An in-flight object acquisition can include several adapters and may still retain
  source evidence. Cancellation prevents later job checkpoint/result commits; it
  does not erase already acquired evidence or reverse remote effects.
- Process death after acquisition but before checkpoint acknowledgement may cause
  unacknowledged work to repeat. Already committed checkpoints are reused. A source
  cache hit is not an independent new observation.
- Original input is limited to 1 MiB/1,000 rows; existing finite JSON/checkpoint/result
  limits also apply (16 MiB). This is an admission cap, not a latency/completion
  guarantee. Fencing currently rereads/verifies checkpoints; large-catalogue
  throughput has not been qualified. Large or repeatedly interrupted batches may exhaust attempts; preserve
  their checkpoint history and split follow-up work explicitly.
- Source fingerprints cover declared API settings, not provider authenticity or a
  complete environment lock. Drift blocks new acquisitions, not access to already
  frozen compatible results. Incompatible scientific policies are not silently
  upgraded during replay.
- Public execution failures retain codes and receipts rather than arbitrary provider
  exception strings. Raw scientific payloads have their existing retention/privacy
  rules; this is not a universal log-redaction or confidential-data guarantee.
- Checksums detect inconsistency relative to stored content, not initial authenticity,
  malicious administrator resistance or independently witnessed history. No backup,
  replication, operational supervision or recovery service is provisioned here.

## Evidence and checks

Synthetic controls exercise real file-SQLite, admission, partial checkpoints, targeted
claims, lost acknowledgement, cancellation/fencing, source outage, idempotency races,
read-only inspection and primary-menu restart. A real child process is killed after
one object commits while another blocks; a second process resumes without reacquiring
the saved object, and a third process reads/verifies the completed result. Controlled
lease clocks avoid scheduler-dependent subsecond expiry tests; no system clock is changed.

The shared report verifier also rejects rehashed object-label substitution and
unsupported validation/five-sigma permission. These are structural integrity checks,
not authentication or scientific calibration.

```bash
python -m pytest tests/test_durable_batch_jobs.py tests/test_network_job_worker.py \
  tests/test_durable_job_store.py -q
python -m pytest tests/ aneos_core/tests/ -m "not network" --cov=aneos_core --cov-fail-under=30
```

The existing network-worker coverage gate continues to cover the extracted shared
lease driver at its unchanged floor. A new scoped batch-worker gate supplements it.
Other job families, deployment backup/recovery, exhaustive menus and independent
scientific evaluation remain open; passing this increment does not close #15 or #2.
