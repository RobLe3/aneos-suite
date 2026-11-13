# Durable authenticated network API

2026-11-13. Scope: existing population job submission/status, owned cancellation,
scoped credentials and app-managed worker lifecycle; #15/#17/#6. ADR-079.

## A. Engineering summary

Normal network routes no longer use the daemon/in-memory dictionary. API lifespan
confirms the existing database, initializes JobStore, starts NetworkRuntime and
closes it before service teardown. Provider/numerical work uses the registered
ADR-078 handler. Job submission commits before acknowledgement; actual SQL state
survives application processes. Missing observations remain missing, never zero.

Fresh credential checks accept configured or active SQL-user API keys of bounded
length, not placeholder/mock credentials or cached global grants. Keys need suitable
operator-provisioned entropy; length alone does not prove strength. Existing SQL key
storage is plaintext JSON and still needs broader credential-hardening review.
Configured role keys are shared service identities, not distinct human users.

Scoped signed tokens are explicitly requested from the existing token route with
`audience=network-jobs-v1`. They require configured signing support and a verified
current API key. Algorithm HS256, issuer/audience, expiration/issued-at/subject and
credential binding are checked; current SQL/config role and active key are rechecked.
A revoked key also revokes its scoped token. Administrative routes use this same
fresh verification to close the credential-provisioning bypass: legacy development
admin grants cannot mint persistent identities. Provision/list uses the runtime
database, not a potentially different global test/deployment session, and UUID
identifiers avoid same-second provisioning collisions. Legacy unscoped tokens and development
mocks do not authorize network jobs. General authentication/legacy token exchange and success-only user lifecycle
implementations remain unremediated scope under #47/#17; no suite-wide security claim.
Primary implementation reference: [python-jose JWT implementation](https://github.com/mpdavis/python-jose/blob/master/jose/jwt.py).

Owners are derived from verified principals. No arbitrary owner body field or admin
cross-owner exemption. Read/cancel of another owner's or unknown job returns404;
viewer submission/cancellation returns403. Inactive/unknown credentials return401;
ambiguous key or unavailable credential/storage state fails closed. Never weaken
permissions to obtain a positive test.

Idempotency-Key is owner/kind scoped and hashed for storage. A repeated equivalent
client intent reuses its original request/window, even after date/profile changes;
a different body conflicts409. Concurrent inserts retain store uniqueness/race
handling. Full queue returns429 and Retry-After, not an acknowledged unsaved job.
Without an idempotency key, an unconfirmed HTTP response must not be assumed absent;
use an explicit key for retryable clients. Polling exposes persisted attempts,
revision/checkpoint counts, native execution status, error codes and the fully
verified execution/scientific report, with null significance throughout.

Health observes, not initializes. Worker death/stop rejects admission; disabled mode
(`ANEOS_NETWORK_WORKER_ENABLED=false`) allows existing owned reads/cancellation but
not submission. Default true starts a managed worker; invalid flag fails startup.
Queue capacity/schema must match existing storage; never reset to make startup pass.
Cooperative stop is not forced provider cancellation: a blocked call can exceed the
shutdown budget, which raises/unconfirms shutdown and requires process supervision.
Old direct daemon/route Python helpers are controlled deprecations. Existing legacy
process-memory records are not automatically migrated or deleted; preserve/reconcile
old process research before deployment. Other job families are not made durable here.

## B. Scientific assessment

This integrates canonical screening/reporting, not calibration or physical-origin
fitting. Execution completion is not100% object data, scientific validity, natural
origin or artificial evidence. Source failures exhaust bounded retries visibly;
partial committed inputs survive without being silently reported as a complete
successful analysis. Valid observations with missing fields can remain inconclusive.
The historical KY26 baseline is unchanged; #7–#11/#25 remain open.

## C. Architecture / files

Existing app/endpoints/auth token/schema/JobStore compose new network_access,
network_routes and network_runtime modules. No new service/UI/detection engine/BC.
ADR079/DDD, README/REST/OpenAPI and this scoped report document integration. OpenAPI
is generated from the actual registered app. Existing prototype tests are migrated
to real authenticated SQL HTTP execution, not skipped or replaced by enqueue mocks.
Bootstrap tests explicitly isolate unrelated services; separate actual-runtime tests
exercise the network lifecycle. Full optional ML/backend startup is not certified.

## D. Remaining work

- Apply source/field completeness and local multi-source assertion/enrichment/history
  requirements #40; per-object uncertainties/provenance #25/#5 and provider policy #16.
- Full-menu render/readability matrix #41, verified object explanations/graphs #42,
  scientific suitability #43, README/how-to/example audit #44, synchronized version
  bump #45 and reproducible quality/CI comparison scorecard #46. These are user-required
  scope, not accomplishments implied by this API integration.
- General legacy authentication/token routes, credential hashing/provisioning,
  all other job families, full startup/security, PostgreSQL/cross-host clock handling,
  backups/power loss and broader production readiness remain open. TestClient/SQL
  process controls are not a deployed TLS/proxy/load/security certification.
- A model/version upgrade can invalidate replay of prior jobs; preserve compatible
  code/evidence and use explicit migration policy, never silently replace history.
- Avoid claiming100% knowledge: separate frozen required-data completeness, eligible
  source coverage, enrichment completion and operation-specific scientific readiness.

## E. Reproducibility / verification

Actual HTTP positive controls create a real stored candidate report. Tests cover
cross-owner read/cancel denial, key/token revocation and current roles, configured-key
rotation, bad scoped claims, duplicate/mixed credentials, queue/idempotency behavior,
missing/outage/wrong-object failures, storage uncertainty, disabled/stopped worker,
startup cleanup and unconfirmed shutdown. A child API process is killed after one
input checkpoint while its next provider blocks; a new API instance recovers only
the missing input and serves the completed report to its owner, not another account.
Provider fixtures/accounts are synthetic/non-sensitive; no live provider requests.

Commands/final counts and remote commit/CI references follow below/in the draft PR.

```sh
PYTHONDONTWRITEBYTECODE=1 ANEOS_DATABASE_URL=sqlite:////tmp/aneos-network-api-final3.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core --cov-report=term-missing --cov-fail-under=30

COVERAGE_FILE=/tmp/aneos-network-api-component2.coverage PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-network-api-component2.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/test_durable_network_api.py \
  tests/test_pattern_api.py tests/test_api_bootstrap_contract.py -q --tb=short \
  --cov=aneos_api --cov-config=tests/coverage-network-api.ini \
  --cov-report=term-missing --cov-fail-under=85
```

Prior integration run: **1,025 passed, five network deselected, five warnings in57.16s;
45.41% core coverage against unchanged30% gate**. Targeted API/lifecycle suite:
**53 passed, 92.27% combined access/routes/runtime coverage against new85% gate**,
14.22s. Existing storage85/report90/worker90 gates remain unchanged. New API test
file33 controls and three additional compatibility-route controls account for36
more full-suite cases; existing daemon tests are migrated to actual durable HTTP
worker positive/error cases. No tests or scientific gates were disabled/lowered.
Initial red run had11 missing-runtime errors; a subsequent helper import path error
was corrected. Full OpenAPI is regenerated from actual application registration,
not its test relaxed. Earlier exploratory runs are superseded by these receipts.
Root experimental research/database and frozen KY26 artifacts remain untouched.

Follow-up inspection found the legacy administrator could provision SQL identities.
The same verified-admin boundary now guards administration, and actual authorized
provisioning plus mock-grant denial and cross-owner controls test this dependency.
The initial positive fixture lacked the required password field; it was corrected
to a valid synthetic request, not the API validator relaxed. No claim is made that
legacy password/lifecycle implementation is complete; see #47. Updated final
verification receipts below supersede the prior integration counts above.

Final provisioning-boundary follow-up (supersedes preceding counts):

```sh
PYTHONDONTWRITEBYTECODE=1 ANEOS_DATABASE_URL=sqlite:////tmp/aneos-network-api-final5.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core --cov-report=term-missing --cov-fail-under=30

COVERAGE_FILE=/tmp/aneos-network-api-component3.coverage PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-network-api-component3.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/test_durable_network_api.py \
  tests/test_pattern_api.py tests/test_api_bootstrap_contract.py -q --tb=short \
  --cov=aneos_api --cov-config=tests/coverage-network-api.ini \
  --cov-report=term-missing --cov-fail-under=85
```

**1,027 passed, five network deselected, five warnings in58.76s; core45.41%/gate30.**
**55 targeted passed, access/routes/runtime92.38%/newgate85**, 13.74s. This branch
adds38 cases overall:35 new API tests plus three compatibility-helper controls.
The old direct FakeDB provisioning test was migrated to actual verified HTTP/SQL,
checking persisted user/key and legacy-map registration, not disabled. Historical
phase tests retain their distinct scope; library auth defaults are not certified.
