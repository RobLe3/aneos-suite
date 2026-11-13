# Durable, correlated local enrichment

Development review, 2026-11-13. Scope: #54 and the first persistence building block of
#40. This is not a completed all-source acquisition system or scientific validation.

## What failed, and what changed

A disposable-SQL audit reproduced three defects: a reported successful MPC update lost
source/session tracking and the consolidated albedo; a later null origin output caused
rollback; and the earlier larger heuristic origin output was retained as if current.
Plain JSON list mutation, incomplete physical merging and max-score logic caused these
failures. The audit used synthetic data, not the user's research database.

`EnrichedNEOService.enrich_neo_data` now stores native payloads, normalized source
snapshots, dated receipts and replayable projections in one transaction. The shared
reconciler also serves DataFetcher, keeping orbital solutions whole and physical values
paired with their own uncertainty/reference. A physical-only provider can contribute
without invented orbital elements. Exact duplicate encounter records collapse in the
projection; different records and their provenance remain available. Unknown native fields
remain in history without guessed units or physical interpretation.

The acquisition cache namespace adds `source-projection-v2` so older reconciled
cache entries cannot silently stand in for this policy; old cache files are not deleted.

The existing legacy polling enrichment call and database-detail view use the service.
The detail view shows the named catalogue inventory, physical selection provenance and
dated source history. JSON export contains full assertions and projection inputs, not only
rounded display values. Stored history is not a source-authentication guarantee.

## Storage and preservation contract

- Explicit additive initialization creates `neo_source_assertions_v1` and
  `neo_object_projections_v1` on the existing Base. It does not reset databases.
- Only file-SQLite is validated/accepted by this initializer. Memory and other database
  backends require separate implementation and evidence; do not assume portability.
- The service requires a fresh Session and owns its transaction. It does not commit or
  roll back unrelated caller changes. Writes serialize with `BEGIN IMMEDIATE`.
- Assertion, projection and compatibility row commit together. Failure returns
  `not_committed` and zero success counters. Concurrent identical receipts are idempotent.
- Receipt identity includes designation, source, session, kind and native payload.
  Same payload in another session remains history and is labelled an exact repeat.
  A changed source-priority policy produces a new projection, not new observations.
- Before the first ledger write to an existing object, its old row values and source slots
  are archived as legacy-unverified evidence. The compatibility row is then updated.
  This preserves recorded values, **not original database bytes or an external backup**.
  Back up research databases separately before enabling new writes. Reads alone do not
  migrate legacy rows. Original acquisition timestamps cannot be reconstructed if absent.
- SQLite UPDATE/DELETE triggers and hash/replay checks resist ordinary accidental edits.
  A database administrator can bypass them. They are not signatures, source authenticity,
  access isolation, a backup system or a compliance claim.
- A process killed before commit leaves no partial object; after commit the projection is
  readable by a new process. This is tested for process termination, not every storage,
  filesystem, power-loss or disaster-recovery failure.

## Scientific interpretation

Current artificial-origin probability, scientific completeness, source coverage and
precision remain unavailable. A legacy heuristic output survives in history with its
uncalibrated label, not as a maximum/current posterior. Missing values are not zero.
The fixed 15-field catalogue-presence inventory is not readiness for an observational fit.

Latest receipt per source is the current projection policy. It does not establish the
latest observation/solution epoch. Distinct provider names are not independent evidence.
Alternative values are not automatically averaged. Alias resolution, physical interval
interpretation, spin-coordinate compatibility, correlated measurement uncertainties and
observational likelihood fitting still need work. No five-sigma or origin conclusion follows.

## Reproduce a local example

Use a **new disposable file path**, not an existing research database. From the repository
root, with the documented dependencies installed:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from aneos_api.database import EnrichedNEOService
from aneos_api.enrichment_store import initialize_schema

engine = create_engine('sqlite:////tmp/aneos-example-new.sqlite')
initialize_schema(engine, include_all=True)  # explicit requested schema write
sessions = sessionmaker(bind=engine)
with sessions() as db:
    outcome = EnrichedNEOService(db).enrich_neo_data([
        {'designation': 'SYNTHETIC', 'data_source': 'SBDB',
         'physical_properties': {'diameter_km': 0.011}},
        {'designation': 'SYNTHETIC', 'data_source': 'MPC',
         'physical_properties': {'albedo': 0.52}},
    ], 'synthetic-example')
    assert outcome['commit_status'] == 'committed'
with sessions() as db:  # fresh read after commit
    service = EnrichedNEOService(db)
    evidence = service.get_enriched_neo('SYNTHETIC')
    assert evidence['physical_properties']['albedo'] == 0.52
    assert evidence['artificial_probability'] is None
    service.export_enriched_json('SYNTHETIC', '/tmp/aneos-example-new.json')
engine.dispose()
```

Export uses exclusive creation and refuses an existing output file. Typed NEOData can
be supplied with `snapshot_schema: neo-data-v1` and `neo_data: neo.to_dict()` alongside
the designation/data_source. Source identity must match the supplied object; ambiguous
catalogue aliases are not resolved by this call. Query attempts can separately record
available/missing/unavailable/invalid/ambiguous/not_supported outcomes. No fixed source
plan is inferred from the subset that happened to respond.

## Tests and presentation artifacts

Run the actual SQL/restart/concurrency/detail suite:

```sh
python -m pytest tests/test_durable_enrichment.py -q --tb=short \
  --cov=aneos_api --cov-config=tests/coverage-enrichment-ledger.ini \
  --cov-report=term-missing --cov-fail-under=90
python scripts/qa_enriched_object.py --output /tmp/aneos-detail-new-capture
```

The generator refuses an existing directory and uses only its own temporary synthetic
SQLite database. Committed 80/120/160-column text, HTML and SVG renders plus JSON and
source/artifact hashes are in `docs/qa/enriched-object/2.2.0.dev0/`. These show one actual
SQL-backed detail scenario, not certification of every menu or real provider. Process-kill,
restart, rollback, concurrent initialization/writers, duplicate, missing, legacy migration,
integrity failure and whole-value/uncertainty regression tests use real disposable SQLite.

## Remaining work / review gates

#40 still requires ordinary query-to-ledger integration, a frozen eligible-source plan,
pollability and rate-limit tests for every provider, identity/alias ambiguity, and all-source
coverage/readiness. #41 requires every menu/submenu's authorized positive and failure
paths. #43 requires defensible physical fits, dependencies and independent calibration.
#47 still tracks broader legacy authentication. Other job families are not made durable
by this change. Exact original KY26 baseline reproduction and literature/model comparison
remain open (#3/#7–11). Do not close those issues because storage tests pass.

Compatibility note: DatabaseManager retains existing initialization for other backends,
excluding the new ledger tables. That preserves the old startup path; it is not a new
PostgreSQL runtime validation. Enrichment remains explicitly unavailable there.

## Engineering/scientific review record

The focused suite passes 45 cases; enrichment-store coverage is 96.97% against a new
90% CI gate. The process-kill cases cover both sides of commit and a fresh-process read;
they are bounded failure evidence, not a universal durability guarantee. The documented
Python example was executed with fresh temporary paths. Render artifact/source hashes
were checked, and raster previews of the actual detail layout were visually inspected.
Existing component gates are retained. CI results and exact reviewed commit are recorded
on the linked draft PR; a draft/green CI is not release approval.

No new KY26 physical fit or updated origin assessment is produced in this change. The
original exact historical baseline remains unresolved (#3). This work preserves typed
A1/A2/A3, uncertainty and native epoch evidence through a synthetic SQL/export regression;
it does not independently validate the latest provider measurements. The Phobos 1,
natural-force and orbit-fit-artifact explanations remain untested here. Artificial-origin
probability, FDR and discovery significance remain uncalibrated/unavailable.

Changed responsibilities/files: `enrichment_store.py` owns SQL receipts/projections;
`data/enrichment.py` owns normalization/replay; `data/reconciliation.py` shares source
selection; `database.py`, `fetcher.py` and `aneos_menu.py` integrate their respective
boundaries. ADR084 and the DDD addendum document the contracts. Tests, the capture script,
versioned QA artifacts, README/changelog and the additional CI gate record verification.
No issue is automatically closed and no prior scientific result is overwritten by this
report. All-source acquisition, whole-menu QA, scientific validation and independent
review remain release blockers under programme #2.
