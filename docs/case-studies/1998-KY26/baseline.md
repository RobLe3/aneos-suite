# 1998 KY26 historical baseline

Status: preserved user report plus a separate rounded-input replay. Exact original
reproduction remains unresolved. This is not a new origin determination.

## What is preserved

`tests/fixtures/ky26_baseline.json` retains the supplied 2026-11-12 output:
SUSPICIOUS, displayed score 4.142, model-generated probability 3.67%, combined
p-value 0.0012976288, two evidence sources and reported completeness 85%.
Those labels are historical software outputs, not calibrated scientific claims.

The rounded inputs are a=1.230 AU, e=0.200, i=1.49°, diameter=11 m,
albedo=0.52, rotation=0.09 h and H=25.9. They are not asserted to be the exact
original catalogue response or the preferred literature measurements.

## Repository and reproduction boundary

At inspection on 2026-11-12, local HEAD and GitHub main both resolved to
`42cab0acd7a1e5e7781a26831f07b11a77f48ead`. Local engineering remediation was
uncommitted. The source decoded from the JSON archive is byte-for-byte from that revision,
not from the modified detector. JSON-escaped lines preserve original whitespace
without adding whitespace errors to the patch. The decoded-source SHA-256 is in
the baseline JSON; the manifest separately hashes the archive file.
The original reported run's exact revision is not known; the matching main
revision is the explicit reproduction candidate, not invented historical provenance.

A local, excluded WIP checkpoint captured the tracked diff and explicitly selected
new source/tests/docs with hashes. It excludes environment files, databases,
credentials, caches and unrelated pre-existing files. That operational checkpoint
is separate from the public scientific baseline; no secrets are required for replay.

## Offline replay

```sh
python3 scripts/reproduce_ky26_baseline.py
python3 -m pytest tests/test_ky26_baseline.py -q
```

The script executes only the hash-verified archived fixture. It makes no provider
requests and does not modify the baseline. `--output PATH` refuses an existing
file, so a new calculation cannot silently replace an old replay.

Reference execution: Python 3.11.9, NumPy 2.2.2, SciPy 1.15.1, 64-bit floating
point. The replay prints current versions and NumPy numerical-error settings.

| Quantity | Supplied historical output | Rounded-input replay |
|---|---:|---:|
| Legacy score | 4.142 | 4.141288008229254 |
| Uncalibrated origin probability | 3.67% | 3.674337652291626% |
| Combined heuristic p-value | 0.0012976288 | 0.0013052410308005413 |
| Orbital effect | 2.645 | 2.6451142085847565 |
| Physical effect | 3.904 | 3.902621655498992 |

This close replay does **not** prove exact reproduction. The mismatch is retained,
not eliminated by tuning diameter, magnitude, inclination or thresholds.

## Missing historical evidence

The original full-precision input, provider payload, orbit solution ID/epoch,
retrieval timestamp, physical measurement references and exact original execution
configuration were not supplied. Their fields remain null. Observation arc
1998–2024 is a reported baseline description, not recovered astrometric data.
No current JPL request has been substituted for these missing historical records.

The heuristic score is neither Gaussian-equivalent discovery significance nor a
probability of artificial origin. Calibration and operational FDR are not
established by the archived detector's toy consistency check.

Tracked under [baseline issue #3](https://github.com/RobLe3/aneos-suite/issues/3).
The baseline issue must remain open until its actual provenance/reproduction
requirements are resolved or explicitly bounded by scientific review.
