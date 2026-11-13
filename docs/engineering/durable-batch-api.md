# Durable, owned REST batch processing

Development review, 2026-11-13; #15/#2/#6 and ADR-097. This extends the existing
registered SQL batch handler, not every analysis/export/automatic job in the suite.
It does not add scientific calibration or change canonical rankings.

## What changes

`POST /api/v1/analysis/analyze/batch` previously acknowledged an in-memory dictionary
and scheduled an unowned background callback. A restart lost its result lookup, and
optional legacy authentication allowed anonymous work. The current adapter commits
and verifies an owned SQL job before acknowledging it. Status replays its actual
immutable checkpoints. The legacy direct Python callback is explicitly withdrawn;
there is no fallback to an unacknowledged process-memory job.

Existing submit/status paths remain. `batch_id`, `status`, `total_neos` and
`status_url` are retained; `job_id`, native `execution_status` and storage/scientific
status are explicit. New jobs use opaque SQL job IDs, not timestamps. Old lost
process-memory IDs cannot be reconstructed; they return 404, not invented history.
No existing stored report or measurement is overwritten or deleted.

## Authentication and ownership

Use an operator-configured or active SQL-user API key. Placeholder/default keys,
development mock tokens, legacy unscoped JWTs, mixed credentials and duplicate
credential headers are rejected. Credentials and current account roles are checked
afresh, including token use after key revocation. Analyst/admin may submit/cancel;
readers must own the job. A viewer cannot create work. An administrator has no
implicit access to another owner's research results. Unknown/wrong-owner IDs both
return 404. API bodies cannot choose owner, policy or execution budgets.

A bearer token must be explicitly issued with `audience=batch-jobs-v1`. It does not
authorize network jobs or legacy administrator operations; network-jobs-v1 tokens
do not authorize batch work. Both audiences reuse the existing verified credential
store and signing implementation, not a second identity system. Keys remain subject
to existing plaintext-storage and provisioning limitations (#47); this is not
suite-wide security certification.

## Example: submit, read and optionally cancel

These commands use the network-enabled API. They are instructions, not recorded
KY26 observations or a claim that its origin is established. Configure a separate
file-SQLite database and provider state before starting the API; do not use the
checked-in research database as a disposable test store. No new launcher option is
introduced. Existing application startup owns schema initialization and the host.

```sh
curl -X POST 'http://localhost:8000/api/v1/analysis/analyze/batch' \
  -H "X-API-Key: $ANEOS_ANALYST_API_KEY" \
  -H 'Idempotency-Key: choose-and-retain-a-unique-client-request-id' \
  -H 'Content-Type: application/json' \
  -d '{"designations":["1998 KY26","1998 KY26"],"force_refresh":false,"include_raw_data":true}'

# Replace JOB_ID with the acknowledged batch_id; prefer the returned status_url.
curl 'http://localhost:8000/api/v1/analysis/batch/JOB_ID/status' \
  -H "X-API-Key: $ANEOS_ANALYST_API_KEY"
curl -X DELETE 'http://localhost:8000/api/v1/analysis/batch/JOB_ID' \
  -H "X-API-Key: $ANEOS_ANALYST_API_KEY"

# Optional scoped token exchange; requires configured signing support.
curl -X POST 'http://localhost:8000/api/v1/auth/token?audience=batch-jobs-v1' \
  -H "X-API-Key: $ANEOS_ANALYST_API_KEY"
```

Keep credentials out of URLs, examples, repositories and exported reports. Use one
credential mechanism per request. Idempotency-Key is optional but strongly advised:
without a client-retained key, a lost HTTP acknowledgement does not imply that no
job was committed. The key is owner/kind scoped and stored hashed. Identical retries
reuse the first frozen clock/configuration, even while the host is stopped; changed
designations/refresh/raw-response intent conflict with 409. Concurrent same-key
submissions admit one job. Full shared capacity returns 429 with Retry-After; respect
HTTP rate-limit and queue backoff. Poll at a modest interval, not a tight loop.

## Input and result semantics

The API accepts 1–100 designations, each 1–50 characters after trimming; control
characters and comment-only labels are refused. It serializes the ordered list to
canonical CSV and verifies exact round-trip identity before admission. Each
occurrence remains visible; exact duplicates reuse one report, not independent
observations. No alias/case identity is guessed. Refresh reaches the shared pipeline.
Unknown options, coerced boolean strings/numbers and non-null `progress_webhook` are
422 errors; webhooks are not implemented and cannot silently be promised.

The frozen job preserves source profile, original serialized input, execution and
retry policy, screening version and versioned HTTP response options. Existing
menu-v1 requests without HTTP options remain valid. Ordinary source failures produce
explicit unavailable results in one pass; crash recovery has a separate bounded
attempt budget. Ready checkpoints are not reacquired on recovery.

Status returns every request occurrence, including `not_assessed` entries with null
score/assessment, retained incomplete evidence and explicit source/invalid failures.
`completed` counts request rows with reports without execution errors; an
inconclusive scientific result can be processed successfully. `failed` counts
reported execution errors; `pending` counts rows without reports. These are not
precision, origin truth, independent-object counts or measurement-completeness
percentages. `request_progress` separately counts exact designation labels.

`status=completed` means the execution report was committed and verified. It does
not mean all measurements exist, all hypotheses were evaluated or any scientific
validation succeeded. Missing does not become zero or natural. Partial cancelled
jobs retain earlier reports but do not fabricate a final scientific report. Errors
use public codes/messages; leases, identities and arbitrary exception details are
not returned. Raw source payloads/logs are not universally redacted by this adapter.

With `include_raw_data=true`, status includes per-object typed source snapshots and
the complete verified `scientific_report` batch envelope, suitable for the existing
batch-report verifier/export/dashboard tools. With false, it omits those raw
sidecars and the full envelope; it does not return an edited envelope pretending to
retain the original hash. Normalized screening evidence/provenance remains visible.
The complete replay inputs remain preserved internally regardless of response
projection. Batch data is not copied into the legacy global designation cache;
use the owned status path, not unscoped `/results/{designation}` lookup.

Submission, queue eligibility, lease and terminal dates are UTC execution dates,
not observation/measurement/solution epochs. Original source chronology is unchanged.

## Execution host, shutdown and recovery

The existing app-owned host serves both registered job kinds against one SQL queue
and capacity limit, one network job then one batch job per scheduling round.
This is serial job dispatch, not concurrent job throughput or strict cross-kind FIFO.
A blocked source can delay both kinds. Internal canonical batch concurrency remains
bounded by frozen server configuration and existing provider policies.

`ANEOS_NETWORK_WORKER_ENABLED` and `ANEOS_BATCH_WORKER_ENABLED` independently enable
admission/processing (default true in the full API). The compatible NetworkRuntime
constructor leaves batch disabled unless explicitly enabled. No provider work starts
on import or OpenAPI generation. Disabled/dead/stopping hosts reject new work;
owned verified reads/cancellation remain available. Health reports both kinds.
A configured API host may recover pending menu batch jobs if they deliberately
share the same database; their fixed trusted local owner is not exposed as an HTTP
principal. Use separate configured research/test stores when that is not intended.

A process kill leaves SQL lease/checkpoints intact. After lease expiry, the new host
reuses committed reports and processes only unresolved items within the stored
attempt budget. Shutdown is cooperative, with an explicit join deadline. It cannot
forcibly cancel a remote request; in-flight evidence may be preserved after job
cancellation, but stale/cancelled job commits are fenced. No exactly-once provider-I/O,
external backup, multi-host throughput or full administrator-compromise guarantee.
File SQLite is the qualified backend here; other backend operation is not established.

## Verification

```sh
python -m pytest tests/test_durable_batch_api.py tests/test_durable_network_api.py -q \
  --cov=aneos_api --cov-config=tests/coverage-batch-api.ini --cov-fail-under=90
python -m pytest tests/test_durable_batch_jobs.py tests/test_network_job_worker.py -q
python -m pytest tests/test_screening_service.py tests/test_nongrav_evidence.py -q
python -m pytest tests/ aneos_core/tests/ -m 'not network' \
  --cov=aneos_core --cov-fail-under=30
```

Controls use owned temporary SQLite stores and non-sensitive synthetic accounts and
measurements, not live provider validation. A real API child process is killed after
A is committed while B is blocked; a second host acquires only B, and a third
execution-disabled host verifies the exact final result/ownership. Positive and
cross-owner denial controls, audience separation/revocation, shared capacity,
concurrent idempotency, lost commit acknowledgement, partial cancellation, corrupt
checkpoint denial, source drift and actual health/shutdown accompany numerical parity
and A3-preservation tests. The normal HTTP exception handler now preserves declared
response headers, including queue Retry-After; see FastAPI's
[error/header contract](https://fastapi.tiangolo.com/tutorial/handling-errors/#add-custom-headers).

ADR097 and DDD record existing ownership. README/API guide and committed OpenAPI
must match actual routing and validation. Other API/automatic/export jobs, operational
backup/recovery, performance and independent scientific calibration remain open.
