# Object chronology: know what each date means

Application **2.2.0.dev0**, unreleased draft stack. Addresses #65 and the date
presentation part of #40/#41. This is not whole-application or scientific certification.

## The reproduced defect

The enrichment compatibility row used a local commit timestamp as `first_discovered`.
A real KY26 export therefore showed a 2026 intake date under that label, alongside
1998 observations. SQLite also discarded the offset from that compatibility
`DateTime` value. Local intake does not establish discovery, and first observation
may be precovery. Neither may substitute for a sourced discovery date.

New rows no longer populate `first_discovered` from the local clock. Existing
discovery-column values are left intact; prior row values are archived before
legacy enrichment. Compatibility caches such as `last_updated` still receive
normal cache updates, so their previous values may live in the immutable legacy
archive rather than the current row. None is silently reassigned to discovery or
a different time zone.

## Reading the new fields

| Field or role | Meaning | What it does not establish |
|---|---|---|
| `discovery_date` | Selected sourced calendar day, or null | A midnight UTC instant or an inference from first observation |
| First/last observation in fit | Catalogue observation-window labels, including partial dates | Actual discovery, complete observation coverage or an exact time instant |
| Orbital-element epoch | Native JD/MJD/calendar representation and declared scale | Retrieval time or conversion to UTC |
| Covariance epoch | Native epoch from the existing typed TDB covariance contract | Propagation to the displayed orbit epoch or a new fit |
| Orbit fit produced | Provider's fit-production label | Its dynamical epoch or a UTC instant by assumption |
| Adapter receipt | Frozen adapter timestamp and its explicit offset, if present | Observation time or a synchronized clock certificate |
| `first_ledger_recorded_at` / `last_ledger_recorded_at` | Verified-format UTC append receipts in ledger sequence order | First-ever intake before this ledger, discovery, last observation or data freshness |
| Compatibility dates | Preserved legacy/cache values with unverified semantics/time basis | Trusted scientific dates or permission to retag naive values as UTC |

The new top-level `chronology` contains replayable, referenced inputs, role-separated
events, source discovery assertions, ledger receipt information, preserved legacy
values and explicit limitations. Full literal precision remains in JSON. UTC receipts
may display `Z` instead of the equivalent `+00:00`; this is notation, not an astronomical
time-scale conversion. No mixed UTC/TDB/TDT/Pacific-local timeline is invented.

### Controlled compatibility change

The ambiguous exported aliases `first_discovered` and `last_updated` are now null,
with `deprecated_timestamp_fields` explaining their replacements. Use the sourced
`discovery_date` or explicit ledger fields, according to the question being asked.
Raw compatibility date values remain under `chronology.legacy_dates`; existing
rows are not rewritten as if their historical meaning were known.

`discovery_data` now exposes a selected source-native discovery section when one
is available and unambiguous. Previous unverified compatibility content remains
in `legacy_discovery_data` and, on import, the original legacy archive. Native
source assertions and historical analyses remain accessible.

Internal compatibility `DateTime` columns are still timezone-lossy caches. The
append-only ledger stores authoritative receipt strings with explicit UTC offsets;
this change does **not** migrate or certify every timestamp column in the suite.
Re-polling predicates still need the separate source-freshness repair in #66.

## What the provider actually supplies

The existing SBDB request now also asks for the documented `discovery=true` section;
there is no separate request, endpoint or detection engine. Its native section,
response hash, requested object and source context remain an atomic replayable
assertion. The parser accepts the documented `YYYY-MMM-DD` layout, validates the
calendar date without a locale-dependent month parser, and retains malformed or
unavailable source values without selecting a date. Distinct discovery claims
remain ambiguous rather than first/last-record-wins. Discovery does not enter the
heuristic rank or establish technological origin.

**Material provider limits:** JPL documents discovery circumstances only for numbered
asteroids and numbered comets. A missing section does not mean the object was never
discovered; an unnumbered object may need an independent literature/catalogue source.
JPL also documents `soln_date` as **Pacific local time**, not UTC or TDB, and permits
`??` in first/last observation months or days. These declarations are retained;
no DST offset, missing month/day or midnight UTC is guessed. See the primary
[JPL SBDB API documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html).

A typed covariance can exist without non-gravitational acceleration components.
Chronology reads its native epoch from the source metadata when necessary; it does
not invent A1/A2/A3 values or transform the matrix. The older field inventory may
not link that epoch, so it explicitly directs readers to source-specific chronology.

## Storage, replay and preservation

The existing SQL repository still owns acquisition transactions. Chronology is a
pure derived view over verified source assertions and local receipt records, not
a second lifecycle database. Ledger append sequence defines first/last receipt;
clock regression is reported, not hidden by wall-clock sorting. Unverified receipt
offsets/layouts or precision are retained but cannot become verified UTC fields.
Earlier local intake may predate the ledger.

A new acquisition normalization context avoids reusing cached omissions of discovery
data. Existing frozen queries, input datasets and review-v1 assessment IDs are not
rewritten. A synthetic review generated with the preceding commit is checked for
unchanged replay. The added chronology display derives from its frozen source
snapshot; it does not alter the recorded assessment or provide new evidence.

Source assertions, projections, local row-value archives and derived views are not
a byte-preserving database backup. Compatibility updates, cache expiry and explicit
cache clearing have their existing effects. Raw source bytes/JSON and historical
results must be retained separately where their licensing and storage rules allow.

## Reproduce the bounded QA

```sh
python -m pytest tests/test_object_chronology.py -q
python scripts/qa_object_chronology.py --output /tmp/aneos-chronology-new-capture
```

The output directory must not exist. The generator exercises the actual public
fetch → file-SQLite → stored-object detail/export path with **synthetic** sourced
discovery, partial observation and legacy-unverified cases. It captures text/HTML/SVG
at 80/120/160 columns, complete JSON exports and source/artifact hashes. The legacy
case uses only the old compatibility table, proving read-only date presentation
without pretending an uninitialized new ledger is a valid store. Missing scientific
values remain missing. Captures live under `docs/qa/object-chronology/2.2.0.dev0/`.

Tests include sourced discovery versus an earlier precovery day, native epoch
precision, Pacific-local fit labels, partial dates, unknown/invalid/missing time
bases, clock regression, legacy archive and engine-restart preservation, replay
rejection, control/Markdown escaping, real menu/plain output and capture checks.
The frame/source comparison was bounded to the actual defect and primary provider
documentation; method self-review is not independent scientific certification.

## Remaining work

#66/#40: source freshness must use original adapter receipts and real installed-source
mappings, not naive compatibility dates or local rehydration time. Automatic partial
cache backfill remains pending. #41/#44: every remaining menu, document and example
still requires scope-matched QA. #7–11/#43: complete dated KY26 source investigation,
physical model fits, uncertainty propagation, conditional populations and independent
calibration remain necessary. No five-sigma or artificial-origin claim follows from
correctly labelled dates or complete response-profile coverage.

## Delivery evidence

Final local offline suite: **1,566 passed**, eight network tests excluded; core
coverage **50.98%** against the unchanged 30% floor. The **71 new cases** reach
**96.73%** chronology component coverage against the new 90% floor. Every existing
CI gate remains. Final capture/source hashes and all nine terminal-width bounds
were verified, with sourced, partial and legacy views visually inspected. These
are bounded engineering checks, not an independent scientific review. The
published revision's complete CI/Docker results and live receipts are reported
in its PR; none is inferred from this local run.
