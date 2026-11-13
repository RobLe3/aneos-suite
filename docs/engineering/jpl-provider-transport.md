# Shared JPL provider transport

Date: 2026-11-12. Branch: `codex/jpl-provider-transport`.
Base: `2e2e9feb85acf28f9f43a43bb00147b5995678f2` (draft #33).
Programme #2; provider work #16; verification #17; impact dependency #32.
Decision: ADR-073. This is one provider-foundation increment, not completion of
those issues or of the KY26 investigation.

## Confirmed defects and correction

JPL requests had independent per-source sessions/limiters and direct call sites,
so concurrent jobs could overlap across endpoints. Retry-enabled Sessions could
send extra attempts inside a single apparent call. There was no shared persisted
cooldown. DataFetcher queried source health before every actual object query;
a failed probe could prevent a valid request. DataSourceBase treated a returned
failed FetchResult as circuit-breaker success.

The new `aneos_core/data/provider_http.py` explicitly wraps JPL GETs. Inventoried
call sites include the SBDB/base adapter, all Horizons query modes, DataFetcher CAD,
historical polling, PHA queries, dataset preparation, old menu diagnostics/history,
both standalone pollers and the test-interface diagnostic. Non-JPL URLs pass
through without JPL pacing or header/status policy. No global requests monkeypatch
is installed. Transport changes do not validate these tools' scientific algorithms.

A POSIX flock is held until the response body is buffered and the response closed.
A separate, persistent lock file prevents first-use initialization races. The
owner-private JSON state stores only scheduling metadata: schema version,
next-allowed time, consecutive transient failures and an in-flight quiet-window
marker. It stores no designations, query strings, bodies, credentials or secrets.

One guarded call makes at most one wire attempt. JPL-specific Session adapters
have automatic urllib3 retries disabled; other providers' adapters remain intact.
429/server/transport/invalid-response failures impose shared exponential backoff.
Fallback delay starts at two seconds and caps at sixty; a valid numeric or HTTP-date
Retry-After can require longer and is **not** capped. A longer wait than the call's
budget fails without sending. Permanent HTTP errors are failures, not empty data,
and do not create an outage cooldown. Redirects are not followed. Missing/boolean/
mock HTTP statuses are not treated as 200. JSON decode failures are not empty success.

The User-Agent uses the installed package version and project issue contact.
Queries are URL-encoded, and empty endpoints no longer invent a trailing slash.
Per-object health probes are removed; explicit probes remain guarded. The generic
base health probe may accept an expected 400; object-query adapters still require
a successful response. Failed source results now count as failures and are not
cached; a genuine recovered response can still advance.

## Authority and scope

JPL requires serialized requests, identifiable applications, error backoff and
restrained traffic. It also advises cache reuse and checking response versions;
formats and availability are not guaranteed. These requirements were checked
against the [official API policy](https://ssd-api.jpl.nasa.gov/) on 2026-11-12.
The local default 0.25-second interval is conservative application pacing, **not**
a published NASA quota. Existing source schemas remain their adapters' responsibility.

This change alone is not full provider-policy compliance. Endpoint version gates,
cache reuse/invalidation, source selection/configuration isolation, query coverage/
pagination, bounded queueing, consistent outage reporting and deployment-wide
coordination still require work under #16. No new Sentry adapter or impact model
is included; #32 remains open. No live scientific results, calibration metrics,
KY26 score, physical fit or five-sigma permission are produced by these tests.

## Configuration and operations

- `ANEOS_PROVIDER_STATE_DIR`: default `~/.cache/aneos/providers`. All participating
  aNEOS processes for a deployment **must use the same directory**. Docker Compose
  sets `/app/data/provider-state` on the existing shared data mount. Additional
  workers/containers must share that mount, same effective ownership and a
  filesystem whose POSIX flock behavior has been verified. Separate worktrees do
  not implicitly get separate production scheduling roots.
- `ANEOS_JPL_MIN_INTERVAL`: nonnegative finite seconds; default `0.25`. There is
  still only one active guarded request when this is zero. This is not permission
  for redundant polling; the caller must justify/cache its queries.
- `_max_wait`: adapter coordination/cooldown budget, default 30 seconds. Request
  connect/read timeouts are separate **inactivity** limits, not total deadlines.
  Outer retry decorators may consume additional time; durable bounded jobs are #15.
- Use asyncio `to_thread` for async integrations. Cancellation does not stop the
  worker thread or release its active provider lease. A second caller must wait.
- State/lock files are owner-private. Unsafe ownership/permissions, symlinks,
  invalid schemas or corrupt JSON fail closed. There is no silent reset or
  unguarded fallback on unsupported platforms, permissions or storage errors.
- `provider_status()` reads scheduling metadata, not endpoint health, data freshness
  or the user's NASA eligibility/quota. Unix wall-clock changes can alter cooldown
  duration; keep clocks synchronized. This is not a multi-host distributed lock.

After a worker is killed, the OS releases its lock but the persisted in-flight
marker delays the next call by the configured timeout-derived quiet window. This
is a recovery precaution; it cannot prove that a remote server stopped processing.
A crash during a state write can leave invalid JSON, deliberately blocking requests.
This is scheduling recovery, not durable analysis/job/result recovery.

For corrupt state: stop **all** participating workers, preserve the state file for
inspection, check ownership/storage and outage logs, and wait at least the largest
known retry deadline/quiet window before deliberate repair. If a retry deadline
cannot be recovered, obtain operational guidance rather than deleting state to
resume traffic. Never unlink the lock file while a worker may still hold it.
Do not delete observational caches, databases, case histories or original results.
Rollback is limited to reverting this branch's transport changes and approved
coordination artifacts with workers stopped; reverting removes these safeguards.

## Verification and reproducibility

`tests/test_jpl_provider_transport.py` uses synthetic senders with external sockets
denied. Positive controls are paired with blocked/error cases. Tests cover:

- strict provider host matching, other-provider pass-through, HTTPS/credential
  rejection, correct identification, query encoding and buffer/close ordering;
- six competing threads across endpoint families, and two actual child processes;
- a killed worker's persisted quiet window, and cancelled async work retaining its
  lease until the worker finishes;
- shared numeric/date Retry-After, long cooldown refusal without another request,
  fallback backoff, permanent errors, malformed state/permissions/symlinks, invalid
  statuses/JSON, close failures and redirect rejection;
- real Session retry configuration (JPL only), explicit configured pacing, actual
  SBDB/base transport and explicit health paths, plus failed-source/recovery controls.

Root `conftest.py` isolates scheduling files per test, preventing tests from changing
user/deployment state or inheriting a prior test's outage. Pacing tests explicitly
set their interval; other parser tests do not need wall-clock pacing sleeps.
Existing CAD/Horizons request fixtures now provide an explicit 200 and buffered body,
not an implicitly successful MagicMock status. The existing SBDB User-Agent test
moves from the obsolete aiohttp seam to the actual guarded requests seam. The PHA
query no longer needs aiohttp: its old missing-library exception test is replaced
by a synthetic positive transport control, not removed or skipped. An intermediate
integration run exposed that obsolete test could attempt a provider query; its
transport is now explicitly injected. No response from that run is scientific
validation or frozen reference evidence. No tests or coverage gates are disabled.

Run from this branch's worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-provider-final.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core \
  --cov-report=term-missing --cov-fail-under=30
```

Final local result: **768 passed, four deselected, five warnings in 34.29 seconds;
42.70% core coverage with the unchanged 30% gate**. The new transport file has
**36 controls**, including real subprocesses. The branch commit and remote CI state
are recorded with the PR. These counts are not detection-performance estimates.
No observational dataset is altered; the original historical-score limitations and
separate current SBDB capture remain unchanged. Scientific model fitting, independent
calibration, impact-source integration and durable case/job execution remain required.
