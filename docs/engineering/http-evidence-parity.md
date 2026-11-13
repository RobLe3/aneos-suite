# HTTP evidence parity

## A. Confirmed defects and correction

Actual HTTP regressions reproduced two adapter mismatches. The history response dropped
`native_epoch`; the raw screening schema rejected albedo1.08 although the canonical
service accepts nonnegative geometric albedo without a probability upper bound. Input
coercion could also turn a boolean/string into a measurement. Arbitrary a/e/diameter
caps differed from canonical domain/abstention rules. The core and HTTP conic checks
are now consistent; physically inconsistent combinations are rejected, not scored.

A separate full-application test reproduced NaN in a validation error causing HTTP500
because the framework attempted to echo a non-JSON-safe value. The new handler returns
422 with error type/location and generic explanation, without raw request input or
exception context. This also avoids echoing private input in validation errors. It is
not a review of every endpoint's errors, authentication or credential lifecycle.

The existing history route now preserves a replay-checked native epoch DTO, original
header, precision, scale/source/role, zero angles and explicit missing values. Numeric
coercion/nonfinite values, changed derived epochs, mismatched headers, wrong roles and
unsupported source fields are rejected. Schema validation is consistency checking,
not independent source authentication or physical verification.

## B. What a history response means

`GET /api/v1/analysis/history?designation=1998%20KY26&years=2` returns computed
osculating samples, not an observational history, detected maneuvers or a force-model
fit. Existing epoch/numeric fields remain, with these additional fields:

- `schema_version`: `orbital-history-v2`.
- `evidence_basis`: `computed_ephemeris_not_observations`.
- `scientific_status`: `not_independently_validated`.
- `returned_points`: the actual count, not the requested duration or completeness.
- `epoch_status`: native epochs retained, or explicitly legacy/unbound.
- `time_axis_status`: a numeric axis on one declared scale is available only when
  every point has a numeric native epoch and the same known scale. Mixed/unknown scales,
  calendar-only or unbound legacy points do not get an invented common axis.
- `coverage_status`: bounded computed samples, not full observation coverage.
- `query_evidence`: actual request parameters, attempt/retrieval times, signature/schema
  status and response-text hash when supplied by the current adapter.

A usable native numeric axis allows a client to plot samples without guessing UTC. It
does not establish frame compatibility with other solutions, covariance propagation,
physical-origin evidence or calibrated significance. Legacy list adapters can still
return explicitly unbound points; they do not acquire fabricated query provenance.

The list-compatible source query records available, empty, invalid and unavailable
states. The route uses a worker thread for synchronous provider access, retaining the
provider timeout and coordination. HTTP200 is a returned computed series;404 is an
empty series,502 is unsupported/invalid evidence,503 is source failure. Raw provider
exception text is not returned. Invalid request prerequisites fail422 before polling.
Endpoint worker threads are not a durable job/restart guarantee; #15 remains open.

[JPL's Horizons API documentation](https://ssd-api.jpl.nasa.gov/doc/horizons.html)
defines ELEMENTS time-scale, frame, origin and unit parameters. Requests explicitly
set TDB, ICRF/ecliptic, heliocentric center and AU-D, rather than relying on hidden
defaults. Native returned labels remain intact. ELEMENTS payload versions1.2/1.3 are
recognized; an absent signature is labeled unverified, not certified. Unknown versions,
provider error payloads or missing bounded element blocks are invalid evidence.

## C. Raw screening input compatibility

`POST /api/v1/analysis/detect` uses finite JSON numbers, not numeric strings or booleans.
Nonnegative catalogue albedo may exceed one; it is not an artificial-origin probability.
The scientific domain rationale and primary source are documented in
[atomic physical evidence](atomic-physical-evidence.md). Existing valid numeric
requests retain their ranking result. Long-period bound or hyperbolic inputs are not
rejected by arbitrary adapter caps; canonical screening either ranks eligible inputs
or explicitly abstains outside its prototype domain. No unsupported Gaussian sigma,
posterior probability or FDR is added.

For example, this is an explicitly **synthetic supplied-input** request, not a measured
KY26 dataset or a natural/artificial-origin conclusion:

```sh
curl -sS http://localhost:8000/api/v1/analysis/detect \
  -H 'Content-Type: application/json' \
  -d '{"designation":"SYNTHETIC","a":1.2,"e":0.2,"i":1.49,"albedo":1.08}'
```

Clients that relied on boolean/string coercion must send actual JSON numbers. Invalid
conics, negative albedo and nonfinite input remain rejected. The full-app validation
error keeps a `detail` array with type/location/message and supplies the project's
error/status/timestamp envelope; raw inputs/context are intentionally omitted.

## D. Architecture, tests and remaining work

ADR-083/DDD retain existing acquisition, API and canonical-analysis responsibilities.
No second detector, new endpoint/UI, external service or runtime dependency is added.
The registered OpenAPI snapshot is regenerated, not hand-edited into a competing API.

Tests use the actual router over HTTP with synthetic provider controls; separate
full-app clients exercise routing/error handlers without claiming startup certification.
Positive tests cover core/HTTP parity, actual provider parser→DTO→HTTP, zero values,
native precision, unknown/mixed scales, legacy compatibility and query binding. Negative
tests cover bad headers/roles/derived values, invalid schema, empty/outage distinction,
request prerequisites, nonfinite and coerced values, conics and redacted exceptions.
A thread check confirms that provider work is outside the event loop. Earlier OpenAPI
snapshot failure was fixed by regenerating the real application schema, not removing
the assertion. Existing quality gates remain; a new95% gate applies only to the history
schema component, not the entire API or its auth implementation.

The optional real-provider HTTP contract makes one Horizons request for1998KY26, checks
native/query transport and freezes canonical provider/HTTP JSON plus software/source/hash
metadata locally. It is one bounded transport check, not source accuracy, full observations,
independent validation, startup/auth isolation or universal availability.

Remaining: #40 normalized durable all-source object ledger/query attempts/partial-source
polling and identity; #6 legacy DB origin claims; #41 every menu/submenu/runtime prerequisite;
#43 physical fits/calibration/independent controls; #47 credential lifecycle. Existing
root WIP/research databases and historical captures are not reset or migrated here.

## E. Reproduction

```sh
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-http-check.db python -m pytest tests/test_api_evidence_parity.py -q
# Optional explicit live contract; not required for offline CI:
ANEOS_API_LIVE_EVIDENCE_DIR=/tmp/aneos-api-history-new python -m pytest tests/test_api_evidence_network.py -m network -q
```

Capture destinations must not exist. Offline tests deny live sockets and use disposable
databases. Source and HTTP snapshots are dated new captures, not replacements for the
November12 historical KY26 baseline. API transport correctness does not establish100%
scientific completeness or a five-sigma capability.

### Verified response example

[history-v2-synthetic.json](../api/examples/history-v2-synthetic.json) is captured from
the actual router using an explicitly synthetic parsed element block. A regression
compares it with the current HTTP output. Its query provenance is null because that
example supplies a list rather than a live acquisition envelope; it must not be cited
as a real KY26 observation or fabricated source receipt.

### Verification record

The final focused suite contains62 passing transport/parity tests; the history schema
component reaches100% statement coverage against its new95% gate. This does not measure
all API paths, scientific accuracy or independent-object performance. Existing gates
are retained. Full-suite and clean-head CI/live receipts are recorded on the PR;
intermediate OpenAPI and NaN failures are disclosed above rather than counted as passes.
