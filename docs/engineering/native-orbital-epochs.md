# Native orbital epochs and computed history

## A. Engineering findings

Three defects were reproduced:

1. SBDB's native epoch survived only in metadata, while the typed orbit and detail view
   could report it unavailable. The unused calendar parser could not represent a JD/TDB
   epoch and must not be used to relabel it UTC.
2. Real Horizons history headers contain an equals sign. The parser skipped those
   headers and returned no bound rows. The old tests used calendar-only headers, so
   they did not cover the real form. A regression initially returned0 instead of2 rows.
3. Float conversion could make two different retained epoch literals compare equal,
   falsely indicating that a covariance applied to the displayed epoch.

`NativeEpoch` now retains the original numeric/calendar literal, representation, time
scale, source, role and header. It validates finite numeric values without converting
TDB/TT/TDT/UT/UTC. MJD can be expressed as JD by adding2400000.5 on the same scale with
Decimal arithmetic; that operation does not establish UTC or convert between scales.
Calendar labels remain opaque to numeric conversion. Precision preservation is not a
claim of physical accuracy. UNKNOWN is explicit; a new scale is not guessed.

The SBDB adapter carries its documented JD/TDB epoch into OrbitalElements. Horizons
object-data EPOCH headers and computed history rows carry explicitly labeled scales;
object-data provenance includes the response-text hash, header and retrieval context.
NEODyS preserves MJD and its returned scale token; TDT is not silently changed to TDB
or UTC. New SBDB/Horizons epochs are checked against retained source solutions during
reconstruction and canonical analysis. Source-priority orbit/epoch selection remains
atomic. Computed histories reject unbound, duplicated or nonfinite elements, and token
boundaries distinguish `A` from `MA`/`TA`. Calendar-only legacy headers still work.

Old orbit snapshots without the optional field serialize as before and are not silently
upgraded. New fetches use `neo_data:atomic-physical-v1-native-epoch-v1:<designation>`;
earlier cache entries are not overwritten or relabeled. Historical scientific output
is not regenerated in place. Source/software hashes now include the epoch module.

## B. Scientific meaning and primary sources

[JPL's SBDB orbit documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html#orbit-data-section)
defines the osculating epoch and covariance epoch as separate TDB Julian-day quantities.
The [Horizons manual](https://ssd.jpl.nasa.gov/horizons/manual.html) describes explicit
output time scales and element/ephemeris conventions. Those definitions justify native
transport, not a UTC timestamp or a claim that an epoch is an observation date.

The details show the original orbital epoch and separate covariance epoch. A mismatch
is not repaired by assuming independence, propagating a matrix implicitly, or borrowing
a different source's solution. Equal compatible-scale numeric epochs are necessary
information, not proof of frame/solution compatibility. Cross-scale or calendar-only
comparison remains unavailable. Legacy covariance/display equality now compares native
numeric literals without float rounding; historical rare false-equality reports should
be retained as legacy, not edited to a new result.

The catalogue field-presence profile still has15 fixed fields. It is version2 when the
epoch field accepts an explicit native record, so its changed representation criterion
is not hidden. A native epoch with UNKNOWN scale is present data, not scientific
readiness. Overall/source completeness remains unestablished. No scoring formula,
threshold, prior, likelihood ratio, discovery significance or origin assessment changes.
The historical KY26 baseline and raw observations are not replaced.

## C. Integration and architecture

ADR-082/DDD keep the change inside existing acquisition, evidence, analysis and
presentation responsibilities. No new bounded context, database migration, external
service or runtime dependency is required. New native time is not a replacement for
the datetime needed by older calendar consumers. CAD's existing native TDB encounter
evidence remains separate; orbital epochs cannot become encounter/observation dates.
Actual JSON/CSV roundtrips retain the original literal/time scale. The scientific
report software manifest includes the new module.

The real Rich views are captured at80/120/160 columns for synthetic native-TDB,
unknown-scale and distinct-covariance controls:
[native-epochs/2.2.0.dev0](../qa/native-epochs/2.2.0.dev0/). JSON, text, HTML, SVG and
SHA-256/source inventories are included. These are actual terminal renders, not
physical propagation, independent scientific validation or all-menu certification.

## D. Verification and limits

Regressions cover source→typed model→reconstruction→canonical review/export, exact JD/
MJD representation arithmetic, cross-scale abstention, old snapshot serialization,
source/time-scale tampering, native covariance differences below float resolution,
real-format multi-epoch Horizons headers, calendar-only compatibility, missing/bad
values, duplicate/unbound/nonfinite elements and separate computed encounters.

The initial NEODyS test used the wrong existing method signature, and an export test
used the report-level sigma key instead of the canonical assessment's `sigma_confidence`;
both tests were corrected to exercise the actual interfaces. A draft self-consistency
test also wrongly expected authentication from an internally consistent changed scale;
the replacement distinguishes derived-field replay from the actual source-binding
check. No verification is presented as source authentication. No failing scientific
assertion or existing quality gate is disabled.

CI retains all previous thresholds. Object-detail coverage additionally selects these
new epoch-view tests because the presentation now consumes the new contract; its85%
gate remains unchanged. Native epoch code has a separate90% gate. Offline tests use
synthetic provider-shaped controls, denied live sockets and disposable databases.
Optional real SBDB/KY26 and Horizons history contracts are explicit network tests.
The latter uses the existing two-year history API, not an invented CLI/API option;
it returned usable native-TDB rows in the bounded check. It is computed ephemeris,
not an observed acceleration or validation population.

Still open: #40 full normalized local ledger/all eligible polling sources and identity,
observation-date precision/history, physical fitting/propagation across epochs/frames,
independent calibration, full menu/submenu prerequisite QA, durable non-network jobs
and complete user documentation. This repair supplies a prerequisite, not scientific
readiness or100% completeness.

## E. Reproduction

```sh
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-epochs-check.db python -m pytest tests/test_native_orbital_epochs.py -q
python scripts/qa_object_review.py --epoch-cases --output /tmp/aneos-native-epoch-renders
# Optional real contracts; not an offline CI requirement:
python -m pytest tests/test_physical_evidence_network.py -m network -q
```

Render/capture destinations must not already exist. The SBDB optional contract can
write a new frozen canonical response, typed snapshot, review and software/hash manifest
using `ANEOS_LIVE_EVIDENCE_DIR`; it refuses existing destinations. That capture is a new
dated assessment, never the missing original full-precision November12 baseline.

### Local verification record

Final functional offline suite:1157 passed,7 explicit network tests deselected;
core47.70% against unchanged30% gate. The native component gate remains90%; the
object-detail gate remains85% and now includes the epoch-view regressions. Provider
contracts are reported separately, not pooled into independent scientific validation.
The two-year live Horizons history check passed for one object; no general availability,
full historical coverage or observation accuracy claim is made. Exact commit/CI and
clean-head provider-capture receipts are recorded in the PR.
