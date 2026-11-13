# Non-gravitational evidence increment

Programme #2; ingestion #4; end-to-end evidence #5; source integrity #16.
Stacked on baseline-preservation PR #18. No origin conclusion is claimed.

## Confirmed defect and correction

The former adapter sent undocumented `nongrav=1`, expected `nongrav_params`,
discarded sigmas/kinds/covariance, and could lose the SBDB result to a faster
source. The old API passed absolute A2 in AU/day² into a propulsion heuristic
whose threshold was described in m/s². The implementation now transports typed
fit evidence and exposes a descriptive sidecar instead of that scalar misuse.

Authoritative contract: [JPL SBDB 1.3](https://ssd-api.jpl.nasa.gov/doc/sbdb.html).
API versions outside the supported contract are explicit failures. Missing
signatures are flagged unverified; unsupported/missing units are never guessed
for SI conversion. Existing scalar records still deserialize as unverified
legacy evidence. Existing functions and impact interfaces remain available.

## Tests and scope

```sh
python3 -m pytest tests/test_nongrav_evidence.py -m 'not network' -q
python3 -m pytest tests/ aneos_core/tests/ -m 'not network' -q
# Optional, intentionally live:
python3 -m pytest tests/test_nongrav_evidence.py -m network -q
```

Before implementation, the first 20 regression controls failed against the
parent branch. After correction, they passed; additional path and malformed-data
controls were then added. Tests use synthetic provider-shaped data and exercise
all three components, signed values, sigma/units/kind, labelled correlations,
normalized PSD rejection, SI coefficient covariance, disk cache round-trip,
source timing, real menu/detector/API adapters and explicit unavailable cases.
Tests do not certify the reference population, posterior or physical hypotheses.

The separately captured KY26 response was exercised locally without a new network
request. Its fitted A3 value, literal uncertainty, nine-coordinate covariance and
distinct display/covariance epochs survive parsing and reporting. The raw payload
is not redistributed here pending review. The immutable historical baseline is
not modified or replaced by this current observation.

## Remaining work

- #6: remove unsupported statistical certainty across all current/legacy outputs.
- #8/#9: fit comparable observations, propagate to common epochs, compare physical
  hypotheses and run predictive checks. Coefficients alone cannot do that work.
- #10/#11: dependence-aware calibration and independent controls.
- #15: durable job/result acknowledgement and restart recovery.
- #16: provider-wide serialization/backoff and broader production configuration.

Do not equate this data-contract increment with completion of the whole goal.

## Measured verification for this increment

- Final full existing CI-style selection and unchanged coverage gate:
  **405 passed, 3 deselected, 5 warnings**; **35.18% coverage**, above the existing
  30% minimum. Command includes `--cov=aneos_core --cov-fail-under=30`.
- New module contributes 41 non-network controls. The optional live KY26 contract
  test passed separately (one request). The final parser and typed JSON replay
  also passed against the previously captured KY26 response, without refetching.
- Numerical review found that finite but impossible covariance entries could
  overflow normalization and bypass the PSD test; a failing regression was added
  before requiring finite normalized correlations. No failed test was disabled.
- Existing tests and coverage gates are unchanged. Five warnings remain, including
  upstream pytest/Starlette warnings and legacy NumPy-bool response coercion.

The full suite's `not network` selection is not a proof of global offline
independence: an existing legacy test performs a collection-time JPL reachability
probe and has an unmarked live case. This is explicitly tracked in issue #17/#11.
The new contract module has no collection-time network access.
