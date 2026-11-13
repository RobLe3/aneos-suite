# Atomic physical evidence

## A. Confirmed defects and correction

The SBDB adapter retained raw `phys_par` but discarded numeric sigmas when constructing
physical data. The common reconciler merged each dataclass field independently. A
primary diameter of0.011km without uncertainty could therefore acquire a secondary
provider's0.003km uncertainty belonging to another measurement. The regression also
covers equal rounded values: equality is not evidence of common measurement identity.

`aneos_core/data/physical_evidence.py` now defines a replayable provider assertion and
collection. Each record owns the native value, uncertainty syntax, units, reference,
notes and response context. Supported diameter/period units convert value and sigma
together. H, albedo and spectral labels use parameter-specific definitions; missing
length/time units are not inferred. H uncertainty is retained in the normalized record,
not invented as a new scalar model field. Unknown parameters/units, invalid values,
unsupported intervals and missing uncertainty have explicit status/issue fields.

[JPL's SBDB documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html#physical-parameters-phys_par-section)
describes physical parameters as collected external-source information and sigma as a
1-sigma quantity. This is a catalogue assertion, not independent measurement verification.
No measurement epoch, fitted/derived status or common physical covariance is inferred
from the containing orbital solution. A literature reference is retained literally,
not treated as a verified publication date.

The fetcher constructs a checked typed projection and selects diameter/albedo/rotation
value–uncertainty pairs atomically. Secondary uncertainty never fills a primary gap.
Original physical source snapshots and per-selection references remain available;
global reference/method fields are not merged into misleading object-wide provenance.
Within one response, duplicate native records are marked; alternatives remain separate
and produce scalar abstention instead of last-record-wins. Unsupported alternatives
also remain visible. Cross-source priority is not scientific proof of compatibility.

Reconstruction and canonical analysis check new assertions against their scalar
projections. A changed uncertainty or self-rehashed normalized record is rejected.
New fetches use `neo_data:atomic-physical-v1:<designation>`; old keys/analyses are not
overwritten or silently upgraded. Old snapshots without the contract are legacy,
not retroactively source-verified. Historical absence of an uncertainty is not corrected
by editing the old result; new acquisition produces a new snapshot.

## B. Scientific interpretation

The old[0,1] albedo restriction confused a physical quantity with a probability bound.
Geometric albedo can exceed one: NASA's
[Saturnian satellite fact sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html)
reports1.08 for Enceladus; this is a domain counterexample, not an NEO control object.
The data/canonical adapter now accepts finite nonnegative albedo without clipping;
negative/nonfinite inputs remain invalid. Bond albedo and geometric albedo are not
interchangeable, and a generic legacy field is not a thermophysical model contract.
No rank formula, weight, threshold, historical KY26 baseline or probability is tuned.

Uncertainty preservation does not mean uncertainty-aware fitting is implemented. The
screening rank is still uncalibrated and its fixed size population/dependent proxies
remain limitations. H/diameter/albedo are not independent origin measurements.
Alternative shape/spin solutions retain native data but do not yet have a complete
coordinate/epoch-aware model. Full physical likelihood/BIC and independent controls
remain required work. Unsupported uncertainty intervals are not assigned Gaussian
confidence or converted to zero. A known sigma alone does not establish scientific
readiness or100% data completeness.

## C. Integration and architecture

Changed: SBDB adapter; common fetcher/reconciler/cache key; NEO reconstruction and
canonical analysis guard; object inventory/Rich/plain details. Existing JSON/CSV and
scientific reports retain the metadata without a competing format. New optional
inventory fields are added only when atomic records are present; old review payloads
without them remain replayable. ADR-081/DDD use existing acquisition/analysis and
presentation responsibilities, not a new bounded context or database migration.

The canonical detail now displays each normalized physical assertion, uncertainty
convention, reference and ambiguity/unavailable status. Actual captures at80/120/160:
[physical-evidence/2.2.0.dev0](../qa/physical-evidence/2.2.0.dev0/), with explicitly
synthetic selected/ambiguous/unknown-unit cases. They are real Rich text/HTML/SVG
renders, not provider or scientific validation. Local raster inspection uses
rsvg-convert; no new runtime dependency.

## D. Remaining work

#25 is advanced, not closed: other providers need documented atomic adapters;
alternative spin/shape models need coordinate/epoch contracts; unsupported interval
formats need source-specific interpretation. #40 still requires a durable normalized
multi-source ledger, alias/conflict handling, explicit eligible-source/field polling
coverage and dated enrichment history. Existing file cache is not that ledger. A
source without orbital data can still be excluded by current orchestration; this
must be resolved before claiming complete enrichment. Observation-native TDB orbit
epochs remain in solution metadata, not the old datetime-only `OrbitalElements.epoch`;
do not pretend that absence means JPL did not supply an epoch. #5/#41 track that view.
Generic durable menu jobs, all-menu QA, likelihood fits, calibration and scientific
review remain open. No artificial/natural-origin conclusion follows from this repair.

## E. Reproduction

```sh
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-physical-check.db python -m pytest tests/test_atomic_physical_evidence.py -q
python scripts/qa_object_review.py --physical-cases --output /tmp/aneos-physical-render
# Optional explicit live check; never required for offline CI:
ANEOS_LIVE_EVIDENCE_DIR=/tmp/aneos-physical-live-new python -m pytest tests/test_physical_evidence_network.py -m network -q
```

Capture directories must not exist. The live test freezes canonical response JSON,
NEO snapshot, review and SHA-256 manifest; canonical JSON is not original HTTP bytes.
The offline suite uses provider-shaped synthetic controls and disposable databases,
with live sockets denied. Live provider availability is not an offline quality gate.

### Live contract evidence (not a new historical baseline)

A real SBDB1998KY26 acquisition/reconstruction/review contract passed on
2026-11-13 at01:11:06.380551UTC. Canonical response SHA-256:
`ec1a8939f6a967831292847bbb974b9746bc5e7cdd3ebb8b27ca765fe97f0ef0`.
The local, uncommitted evidence directory `/tmp/aneos-ky26-physical-live-20261113`
contains the canonical source response, typed snapshot, replayed review and hash
manifest. It is not the November12 baseline and does not overwrite it. Raw provider
responses are not bundled into this PR while redistribution review remains pending.
The optional contract now records software/file-hash metadata with future captures.

That response supplied diameter0.011±0.002km, rotation0.089193±0.000002h,
albedo0.52±0.08 and H25.91±0.03, with catalogue reference
“Santana-Ros et al.(2025) Nat.Commun.16,8275”. These are the adapter's dated catalogue
assertions, not newly verified measurements from the paper. Physical measurement
epoch/estimation method remain unavailable. They survive with their own uncertainties;
no calibrated artificial-origin probability or model-fit assessment follows.

The same response supplies native orbital epoch2461200.5TDB in solution metadata;
the old datetime-only orbit field remains empty. This is a known transport/view
limitation, not missing source evidence or permission to relabel TDB as UTC.
