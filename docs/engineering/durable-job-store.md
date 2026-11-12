# Durable execution storage foundation

Date: 2026-11-12. Branch `codex/durable-job-store`.
Base `557b3b00d2a7f85235666e3de616d6f33721cb48` (draft #36).
Programme #2; execution #15, scientific result contract #6, verification #17.
Decision ADR-076. **Storage is implemented/tested; existing API workers are not yet
integrated.** This does not fix queued network sigma/p-value placeholders or claim
whole-application restart readiness.

## A. Engineering summary

Repository inspection found no durable execution store to extend. Existing network
jobs use daemon threads/in-memory state. File SQLite's shared StaticPool connection
was unsuitable for independent concurrent job transactions. The API package eagerly
assembled optional app/ML services; database import created schema. A standalone
component-coverage run terminated in a native Torch import before executing storage
tests. Storage tools should not initialize unrelated features or mutate a database.

Changes:

- `aneos_api/job_store.py` extends the existing SQLAlchemy Base with four additive
  tables: admission settings/counter, job records, immutable checkpoints and events.
  No external queue/provider, new database or new scientific engine is added.
- Atomic bounded admission (default 128 active jobs), owner-scoped reads/cancellation,
  owner/kind/idempotency-key reuse, and request/policy conflict refusal. Requests,
  checkpoints/results are strict finite JSON objects, max 16 MiB and depth 64; no
  silent Python type/key coercion. The same idempotency identity must be used after
  an unconfirmed submission; absence of confirmation is not proof nothing committed.
- Leased claims with attempt limits, retry deadlines, renewals and unique fencing
  tokens. Expired/reclaimed/cancelled or locally altered-request leases cannot write.
  Frozen checkpoints are append-only/idempotent; different content under the same
  key is rejected. Cancellation preserves acquired evidence and fences writes; it
  does not abort an in-flight provider request or reverse external side effects.
- Terminal result, event and admission release commit in one transaction. Public
  views omit lease tokens. Request/result, checkpoint-envelope and event-envelope
  checksums detect inconsistency. They are not signatures, authentication or proof
  that a supplied payload is scientifically correct.
- File SQLite uses independent connections. Job operations serialize short SQL
  write transactions with BEGIN IMMEDIATE (coherent reads use BEGIN) and require SQLite >=3.35, on-disk journaling
  and FULL/EXTRA synchronization. Ephemeral/memory/StaticPool stores and unsafe
  SQLite settings are refused. No network/calculation runs inside a DB transaction.
- API public exports are lazy, retaining their original names. Importing storage
  creates no database and imports no web app/Torch runtime. Existing schema setup
  remains available explicitly. Full API lifespan now confirms schema initialization
  before starting services; failed/unconfirmed setup cannot yield a successful start.
  Lifespan cleanup also runs after a body exception or partial service-startup error.
  This confirms the setup operation, not comprehensive migration/health of every
  pre-existing schema or optional backend.

Review regressions caught first-claim events mislabelled as recovery, altered local
lease requests still authorizing writes, unsafe SQLite settings accepted, and stale
pre-lock timestamps permitting late completion. Those were corrected, not accepted
as operational limitations. Lease freshness is checked **after** the row write lock.

## B. Scientific assessment

No new observations, scores, fits or statistical validation are produced. Queued and
failed storage records have null results; the store does not manufacture sigma,
p-values, natural-origin decisions or probability. Execution completion and scientific
acceptance are separate. A future handler must bind validated results to frozen
observations, effective configuration, policy/model/software versions and job identity.
A generic JSON result is not independently verified scientific evidence.

Historical KY26 inputs/results and dated source captures are unchanged. Independent
calibration, reference populations, measurement uncertainty, observational hypothesis
fits and five-sigma error control remain separate unfinished work.

## C. Verification and architecture

New controls use real SQL transactions, independent connections, threads and actual
child processes with synthetic payloads. They cover positive submit/claim/checkpoint/
complete/read-back; owner/cross-owner access and cancellation; capacity/idempotency
races; retries/attempt exhaustion; stale/expired/cancelled/altered-request fences;
immutable snapshots; configuration/JSON/type/version/durability refusal; request/
result/event/checkpoint corruption; terminal-write rollback; and a database lock wait
that outlives the old lease. Real process tests kill a worker after committing a frozen
input, recover it without replacing that input, and read completion in another process.
A separate kill after flush but before transaction commit leaves no checkpoint/event/
revision partial write. These do not simulate a full host crash or power failure.

Import purity is checked in a fresh interpreter with app/Torch imports blocked and
no database-file creation allowed. API export identity is retained. Explicit schema
startup has a positive control and False/None/truthy-string denial cases. The optional
full-app/ML cold-start path remains broader #17 work; native backend health is not
certified by making storage imports independent.

The full core suite retains its 30% coverage gate; an additional **85% job-store
component gate** is added to CI. No existing test is disabled or relabelled to hide a
failure. Metrics, commands and final revision are recorded below/with the PR. No new
bounded context is created. SQLite transaction/pooling decisions were checked against
[SQLAlchemy's SQLite documentation](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html).
[SQLite documents transactional atomicity](https://www.sqlite.org/transactional.html);
our measured evidence is narrower: the specified process/restart/rollback controls.

## D. Operations, limits and required next work

Library setup, explicitly using the existing application database:

```python
from aneos_api.database import SessionLocal
from aneos_api.job_store import JobStore
store = JobStore(SessionLocal, capacity=128)
store.initialize_schema()  # Explicit additive setup; no worker or API is activated.
```

Initialize once before workers, with agreed admission capacity. Existing schema/
capacity mismatches are refused rather than silently reset. Expected columns,
primary keys/idempotency constraint and admission count are checked. Missing admission
state beside existing work is not re-created as a zero count: explicit recovery is
required. Incompatible tables are not automatically altered or deleted. Repeated initialization
does not delete jobs/results or reset counters. The normal `init_database()` registers
all model tables, but does not initialize admission policy or start a worker. Configure
`ANEOS_DATABASE_URL` to a properly owned persistent deployment database; temporary
filesystems and in-memory URIs are not durable deployments. Do not pass credentials
in owner/worker/idempotency identifiers or request/checkpoint/result payloads.

Read-only SQLite reopens can inspect owned history/results without modifying the
store; attempts to submit there fail explicitly. All stores need fresh Engine-bound
sessionmaker sessions; caller-owned connections/
transactions are rejected. SQLite transactions are deliberately short/serialized.
PostgreSQL-compatible SQL is used but **no PostgreSQL runtime verification** has been
performed for this increment. Cross-host clocks/leases, full host/hardware/power loss,
backups, database-user privilege enforcement and migrations of an existing incompatible
schema are not certified. Use synchronized server clocks; expiry is not permission to
commit after another claimant has fenced the old token. Remote side effects are not
exactly-once. A lost commit acknowledgement can be ambiguous: inspect/reuse the same
idempotency identity rather than submitting unrelated duplicates.

There is no destructive cleanup/retention routine. Preserve original databases and
completed results. Storage/configuration/corruption failures are explicit, never an
empty successful queue or a scientific negative. Rollback is a code revert with job
writers stopped; preserve the additive tables/history. Do not drop tables/reset active
counts to make tests or startup succeed. Recovery/migration must be separately planned.

Before replacing the current API/daemon jobs, complete these dependencies:

1. Versioned network scientific report: frozen actual inputs and per-object shared
   screening; descriptive pattern evidence separated from calibrated inference. No
   minima-of-selected-p-values treated as independent streams, missing angles as zero,
   computed CAD epochs as astrometric observations, or queued numeric sigma/p defaults.
   Explicit requested/executed/unavailable module states; no silently ignored options.
2. Registered real handler: validate source identity/configuration, freeze successful
   inputs before dependent analysis, resume existing checkpoints, renew leases without
   holding DB transactions during network work, respect provider cooldowns and bounded
   retries, and stop committing after cancellation/lost lease. Validate the domain result
   and request/policy binding before `finish`; the generic store does not do that for it.
3. Managed worker/API lifecycle: explicit startup/shutdown and actual worker health,
   durable submission before acknowledgement, correct returned status URL and persistent
   polling/results. Derive owner scopes from verified principals, not caller strings;
   the existing optional `get_current_user` alone is not production authentication.
   Test authorized positives and cross-owner denial without reading unrelated data.
4. End-to-end controls: actual source-injected worker success and candidate preservation,
   missing/partial/invalid/outage outcomes, duplicate ordering, fresh-process API/status
   reads, process termination/restart/retry and stored-result integrity. Existing enqueue-
   only tests are not substitutes. Do not claim scientific calibration from synthetic jobs.

Current network/automatic/polling API jobs remain unchanged and still require #6/#15.
Case lifecycle/queryable evidence history is #13; provider policy/config/cache/coverage
is #16; broader startup/auth/legacy/documentation verification is #17. These remain
open; no five-sigma or whole-application reliability permission is granted.

## E. Reproducibility

```sh
PYTHONDONTWRITEBYTECODE=1 ANEOS_DATABASE_URL=sqlite:////tmp/aneos-job-store-final11.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core --cov-report=term-missing --cov-fail-under=30

COVERAGE_FILE=/tmp/aneos-job-store-component9.coverage PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-job-store-component9.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/test_durable_job_store.py \
  -q --tb=short --cov=aneos_api.job_store --cov-report=term-missing --cov-fail-under=85
```

Final full suite: **902 passed, five network deselected, five warnings in 40.23s;
45.06% core coverage against the unchanged 30% gate**. New coverage: **39 storage
controls plus six API bootstrap controls**. Targeted storage suite: **39 passed,
92.51% job-store coverage against the added 85% gate** (2.50s). The final commit and
remote CI are recorded separately with the PR. All
new tests use disposable database files/synthetic scopes, not the root experimental
database. Root working-tree research and original case artifacts are not overwritten.

CI follow-up: the initial runner correctly rejected an expired 300 ms test lease.
The two real process-termination tests now use a controlled shared lease clock,
advanced only after termination; process kill, transaction rollback and independent
process readback remain real. Production lease checks are unchanged.
