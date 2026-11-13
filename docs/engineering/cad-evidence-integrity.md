# CAD encounter integrity and canonical screening

Programme #2; CAD defect #22; stacked on required-gate PR #24.
This is acquisition/control-flow remediation, not calibrated detection performance.

## Source contract and reproduced defects

[JPL CAD 1.5](https://ssd-api.jpl.nasa.gov/doc/cad.html), checked2026-11-12,
returns computed encounters with labelled columns. It does not provide orbital
elements. Its JD and calendar dates are TDB; encounter distances are au and
velocities km/s. Optional diameter and its sigma are km. `diameter=true` requests
output; it is not a filter excluding unknown diameters. A documented zero-count
response is valid, whereas a missing response is not evidence of an empty survey.

The old historical adapter inserted a=1,e=.1,i=10, default encounter values and
a discovery date from the query window. Its fallback misread positional columns
as orbital elements. Scoring added random bonuses; source exceptions became
empty successful surveys; chunks discarded every non-candidate record. Overlap
merging collapsed distinct encounters of the same object. Workflow integration
could label a failed PipelineResult successful.

The increment also reproduced two connected integrity defects: NEOData cache
serialization discarded physical uncertainties/other fields, and importing the
optional Gaia client contacted its remote status service before any analysis.

## Implemented boundary

`data/sources/cad.py` validates signature/version, labels, dimensions, counts,
numerical values and declared query completeness. It preserves raw fields/rows,
source-response content hash, native units/time, source/orbit ID, query scope and
available retrieval provenance. Missing values stay null; invalid/truncated
responses fail explicitly. An unknown base-adapter HTTP retrieval time stays
unknown, distinct from the time the application receives that adapter result.

`CloseApproach` retains native JD(TDB), calendar literal, time scale, body and
provenance across model/cache serialization. A naive TDB calendar may serve
legacy calendar consumers; it is never silently tagged UTC. No new TDB↔UTC
conversion or precision claim is supplied. Encounters cannot extend an observed
arc or establish discovery time.

Historical intake obtains a separate whole orbit through the existing DataFetcher
when needed, with per-object reuse and outage cooldown. It deep-copies source
objects before attaching encounters. Catalogue orbit provenance remains separate
from the CAD orbit ID used for an encounter prediction; they are not assumed to
be one fitted solution. Without an orbit, the encounter remains available but
screening abstains. Atomic physical measurement ingestion from SBDB remains #25;
this increment only preserves already-present physical fields through caches.

Poller scoring wrappers now delegate to ScreeningAnalysisService. No random
bonus, magic placeholder-value rejection or second artificial-origin classifier
remains there. A previous test expecting a separate e>.9 rule now checks the
canonical policy instead; positive retrograde/ranking controls still advance.
No scientific accuracy or coverage acceptance threshold is lowered.

All intake records survive into the next stage, not just candidates. Overlap
merging identifies encounters by object/body/native epoch, preserving distinct
events and source-version diagnostics. Reports distinguish encounter-record
counts, unique designations, scored cases, abstentions and source coverage.
Query failures cannot masquerade as successful negative detections. Workflow
status separates error/partial/success from scientific validation not established.

## Caches, reports and offline behavior

New `cad-intake-v1` caches are policy-bound, checksummed, TTL-limited and written
to unique exclusive files. Old/random-scored caches remain untouched but are not
trusted. Typed physical uncertainties, native encounter evidence and screening
snapshots roundtrip. Orbit-outage results are not cached as negative detections.
Checksums detect accidental content changes, not hostile tampering or source
truth. Normal historical reports retain every chunk's status/reason and avoid
replacing same-second history. Crash-safe publication/jobs remain #15.

Gaia import/client configuration is deferred until an actual Gaia query. A fresh
subprocess with network connections denied verifies offline validation imports;
an injected query-client positive control verifies that explicit query capability
was not merely disabled. Existing optional Gaia scientific calculations are not
certified by this import correction; legacy validator remediation remains #23.

Verification: 26 initial regressions failed before implementation, plus later
cache/workflow/import regressions. `tests/test_cad_evidence_integrity.py` has46
offline controls with a network-denial fixture and one optional, separately
marked live KY26 CAD positive control. The optional live control passed once;
it verifies the current transport/schema, not scientific origin or accuracy.

```sh
python -m pytest tests/test_cad_evidence_integrity.py -m 'not network' -q
python -m pytest tests/ aneos_core/tests/ -m 'not network' -q \
  --cov=aneos_core --cov-report=term-missing --cov-fail-under=30
# Optional: one bounded provider request, never required for offline CI
python -m pytest tests/test_cad_evidence_integrity.py -m network -q
```

## Limits that remain

- These are current catalogue computations over requested historical dates, not
  historical raw astrometry or an unbiased NEO population. Default acquisition is
  Earth encounters within .05au; a request limit is explicit. Overflow fails
  rather than silently claiming complete coverage. Adaptive splitting/pagination
  is not yet implemented.
- Supplemental DataFetcher/history queries intentionally retain a ten-record
  bound and report partial coverage when applicable. They are not complete
  30-year observation histories. Rich output exposes coverage/native time scale.
- [JPL's fair-use policy](https://ssd-api.jpl.nasa.gov/doc/index.php) requires
  application identification, caching, error backoff and no simultaneous API
  requests. This increment adds identifying headers and local retry/cooldown,
  but does not establish provider-wide/cross-worker scheduling or full compliance
  across the application; that remains #16.
- Canonical scientific validation remains inconclusive. ATLAS/network/legacy
  inference audit, observational likelihood/BIC fitting, physical hypotheses,
  independent calibration/evaluation and versioned case lifecycle remain open.
- Operational cache/report history is not a frozen, independently reproducible
  scientific investigation. No prior KY26 baseline is overwritten or retuned.
