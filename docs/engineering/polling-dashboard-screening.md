# Candidate screening dashboard and results browser

**Application:** 2.2.0b2 engineering beta.
**Issues:** #72 (dashboard), #71 (retention). **Decisions:** ADR-090, ADR-091.

The dashboard now answers “Which supplied records meet the follow-up ranking policy?”
It no longer treats a legacy probability field as evidence that an object is artificial,
or a missing score as evidence that it is natural.

## Reading a result

| Outcome | Meaning |
|---|---|
| Follow-up candidate | The uncalibrated rank meets its recorded policy threshold. Investigate the evidence and competing physical explanations. |
| Not flagged | The supplied eligible orbit falls below that ranking threshold. This does not establish natural origin or ordinary physical behavior. |
| Inconclusive | Required inputs are absent, invalid, unverified or outside the prototype domain. Missing evidence is not a negative finding. |

Raw rank is displayed in ranking units, never sigma. Origin probability, confidence,
p-value and false-discovery estimates remain unavailable. The summary graph counts
records in these outcomes; it is not a probability distribution over origins.

The dashboard reports input records and distinct exact designation labels separately.
Two records with the same designation are not automatically fused or independent
observations. Catalogue presence uses the existing fixed 15-field profile only when
a typed inventory is available. It does not measure all scientific knowledge, source
accuracy or independent evidence. There is no manufactured aggregate quality score.

## Inputs and provenance

Supported inputs include:

- A complete `NEOData.to_dict()` snapshot inside `neo_data`, with matching designation.
- A canonical `screening_assessment` and its replayable evidence snapshot.
- A standalone canonical assessment or frozen object-evidence review.
- A current polling-screening record or dashboard export, verified on import.

When both a typed snapshot and an assessment are supplied, they must match, including
additional evidence. Units and date roles use the shared model contracts. Missing
created/retrieval fields are not filled in just to admit an incomplete serialized
snapshot. The model's old orbital-only `completeness` cache may be recomputed; the
original is retained in input, and it is not the dashboard's scientific quality metric.

Legacy rows—including unschematized orbital dictionaries, `artificial_score`, old
confidence and probability fields—remain available as archived assertions. They do
not become measured inputs merely because their keys look plausible. The detail view
labels their original JSON as unverified and does not adopt its origin claims. Declared
synthetic/mock provenance is not promoted through a canonical container. This does
not authenticate unmarked data or detect every undisclosed fabrication.

Typed/replayed data are still not independently authenticated observations. A correct
checksum demonstrates internal consistency, not source truth or scientific accuracy.
Malformed, nonfinite, unserializable or oversized input is explicitly rejected. When
finite JSON retention itself is impossible, the record says so rather than claiming
to preserve the original bytes. Existing input files are never edited by import.

## Existing menu controls

The legacy dashboard menu and Norton-style results browser now use these same records:

- Browse all records, follow-up candidates or inconclusive records.
- Move between pages, select a global record ID for detail, filter, sort, inspect
  counts and export the current page. Empty filtered views remain navigable.
- Filter by ranking outcome, typed inventory availability or multiple recorded source
  labels. “Multiple sources” does not mean statistically independent evidence.
- Source-listed impact filtering requires a replayable, designation-bound Sentry
  report. The dashboard does not fit an impact model or infer risk from origin rank.
- Sigma-filter options explain that calibration is unavailable; they do not convert
  a score or probability to significance.
- Sort by raw rank, screening outcome, declared field presence, exact designation or
  local screening receipt. That receipt is not discovery or observation time.

Historical method names remain import-compatible. Faulty temporal-origin, unitless
physical-origin and round-number/orbital-origin helper calls raise documented
`NotImplementedError`; use the typed object review and source chronology instead.
Name-based Tesla/generic orbit and physical defaults are no longer supplied.

## Files, exports and examples

The existing launcher can open a current source-bearing result or older file:

```bash
python artificial_neo_dashboard_launcher.py --file /path/to/result.json
```

Older files are inspected without rewriting them. Unsupported old probability claims
stay unverified; they do not silently become current classifications. The legacy
results browser discovers completed new exports as well as its older known formats.

Each export creates a new directory under `dashboard_results/` containing:

- `report.json`: raw input, canonical review, record identifiers, frozen per-record
  policies and software/source hashes.
- `report.md`: the same scientific interpretation and detailed object explanations.
- `complete.json`: hashes written after both report files succeed.

Record and report hashes are replay checks, not signatures. Import preserves each
record's frozen policy, including different recorded ranking thresholds, without
claiming those thresholds are statistically calibrated. There is no overwrite of
previous reports. An interrupted/failed export can leave a partial directory; it is
not marked complete or automatically listed as a completed bundle. A standalone JSON
can still be explicitly inspected on its own merits. This is not a filesystem backup
or a guarantee against later manual edits, deletion, synchronization or cache eviction.

The legacy menu file reader has a 64 MiB inspection bound. Dashboard input is bounded
to 10,000 explicit records; canonical finite-JSON snapshots/reports use the existing
16 MiB serialization bound. Exceeding a bound fails explicitly rather than truncating
an investigation or inventing completeness.

Generate the synthetic display examples without polling a provider:

```bash
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-dashboard-qa.db \
  python scripts/qa_polling_dashboard.py --output /tmp/aneos-new-dashboard-captures

ANEOS_DATABASE_URL=sqlite:////tmp/aneos-dashboard-tests.db \
  python -m pytest tests/test_polling_dashboard_screening.py -q \
  --cov=aneos_core --cov-config=tests/coverage-polling-dashboard.ini --cov-fail-under=90
```

The capture command refuses an existing destination. The committed examples in
`docs/qa/polling-dashboard/2.2.0.dev0/` exercise actual dashboard, browser and detail
render paths for typed follow-up, typed incomplete and legacy-unverified controls at
80/120/160 columns. They include JSON, text, HTML, SVG and source/artifact hashes.
Synthetic controls exercise engineering behavior; they do not establish astronomical
accuracy or independent detection performance.

## What remains open

The root `neo_poller.py` is a different module from the core poller. Its source helpers
still have confirmed native-schema, missing-as-zero and synthetic-score defects (#73).
The legacy candidate workflow also discards a second source payload before enrichment
(#71). The corrected dashboard retains such rows as unverified rather than endorsing
them; it does not repair upstream acquisition. Use the ordinary typed per-object
acquisition path for source-bearing evidence while those integrations are addressed.

Automatic selective backfill (#40), remaining menu/graphical QA (#41/#69), complete
scientific documentation (#44), the exact original KY26 baseline (#3), physical model
fits, independent calibration and search-wide significance control remain necessary.
No natural/artificial origin or five-sigma discovery is established by this change.

## Engineering delivery evidence

The local offline suite passed **1,721 tests**, with eight network tests excluded;
core coverage **53.01%** against the unchanged 30% floor. The **82 new cases** reach
**98.32%** statement coverage of the dashboard/browser/view component against its new
90% gate. All previous CI gates remain. Tests include typed positive ranking and
entry-point parity, missing/invalid/legacy inputs, source-listed Sentry positive and
identity-negative controls, frozen multi-policy round trips, export failure, actual
navigation/filter/sort options, and standalone file loading.

All 27 actual render cases have checked source/artifact hashes, terminal cell bounds
and coloured SVG geometry. Selected overview, browser and detailed layouts were
visually inspected at the advertised widths. These are engineering checks, not an
independent scientific review or a certification of every application menu. Exact
published-head CI/Docker and archived live-input replay receipts are recorded in the
PR; they are not inferred from the local run.

### Source payload retention (#71)

The legacy workflow now sends **all returned records** to the existing transactional
ledger before presentation. Previously, two sources for one designation could leave
only the first payload plus both source names. The ledger now receives both native
payloads; exact repeats are handled by its existing session-aware idempotency.
Conflicts and differing epochs remain in history. Rendering still does not promote
unverified legacy fields into scientific measurements.

The workflow reports returned records separately from distinct exact designation
labels. The dashboard may contain multiple records for one label; these are not
independent observations or additional evidence streams. Database views retain the
existing per-object projections and source history. A failed/unacknowledged commit
stops before dashboard publication. Returned, empty and unavailable sources are
separate outcomes, not a claim that polling achieved 100% completeness. Chunked
collection without per-source receipts remains explicitly unverified.

The beta's root polling quarantine remains in place (#73). These regression tests
exercise synthetic provider boundaries with the actual workflow and a disposable
file-SQLite ledger; they do not establish restored live bulk polling. Run:

```sh
python -m pytest tests/test_polling_source_retention.py -q
```
