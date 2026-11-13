# Object acquisition: source coverage is not scientific completeness

Status: implemented in the draft reliability stack; not an independently validated
scientific release. Application version: **2.2.0.dev0**.

An ordinary `DataFetcher.fetch_neo_data()` query now records a fixed source plan,
each source outcome and the accepted per-source evidence in the existing local
file-SQLite evidence store. A result is returned only after that query commits.
This extends the enrichment ledger; it does not introduce another detection engine.

## What a query records

The installed base profile names SBDB, NEODyS, MPC, Horizons and CAD. Initialization
failure does not remove a source from the denominator. Outcomes distinguish
`available`, `missing`, `unavailable`, `invalid`, `ambiguous` and `not_supported`.
Unknown configured sources remain visible, not silently successful. This is a
named profile, **not a claim that every scientifically useful catalogue or observation
service has been integrated**. CAD is a bounded computed-encounter query, not a new
independent astrometric observation.

Each query retains its plan, requested designation, software/source fingerprints,
UTC acquisition receipt times, normalized assertions, adapter packets and native
responses where available. Failed legacy adapters may not expose a response body;
that absence is not replaced by invented data. Parameter units, source epochs and
physical uncertainties retain their existing atomic evidence contracts. Native
calendar labels without a declared time scale remain labels, not fabricated UTC
observation instants.

The receipt confirms a local commit. It does not authenticate a provider, establish
independence between catalogues, or prove a measurement is correct. Exact repeated
native-body hashes are labelled as repeats, not independent observations.

## Reading the detail view

The existing object review and stored-object detail show source status, fixed
counts and acquisition timestamps. Source contribution, parsed/explicit-empty
responses, the existing 15-field presence inventory and scientific readiness are
separate quantities. Even **5/5 contributing sources and 15/15 fields do not establish
100% scientific completeness**. Suitable observations, covariance, model validity,
selection effects and independent calibration still matter.

The stored projection uses the latest received assertion per source. It can contain
an older source assertion after a later query fails for that source. Detail output
identifies which selected assertions belong to the latest query. It does not
silently retimestamp them. The immediate query result contains only the evidence
admitted by that query; the local history remains available separately.

Use the existing `EnrichedNEOService.export_enriched_json(designation, path)` for
an exclusive-create JSON export containing the verified query/source history.
The screen limits its history display to the last 20 entries; export is not limited
to those entries.

## Cache and persistence behavior

A cache envelope binds the whole frozen query to its source plan, exact requested
identity and acquisition context. A hit reuses the original query and dates, not a
new observation or an extended TTL. The repository acknowledges the same query
idempotently; a cache replay cannot reapply an old global projection policy.
Corrupt envelopes are ignored and reacquired. A partial cached query currently
requires `force_refresh=True` to retry missing sources before its TTL expires;
automatic selective backfill remains open in #40.

Whole-key SHA-256 filenames replace truncated/path-normalized cache keys. Existing
legacy cache entries are ignored, not silently adopted. Construction no longer
deletes unrelated pickle files. Cache expiry and explicit clear still evict cache
files: this is **not a backup or preservation guarantee**. Durable scientific
history belongs in the evidence store, not the cache.

The default adapter uses the configured application database. Unsupported storage
or schema prerequisites fail before polling. A later commit failure returns an
explicit persistence failure, not a usable scientific success or cache entry. All
source assertions, correlated projection, compatibility row and query receipt
commit together. All-failed queries retain attempts without manufacturing a NEO.
File-SQLite is the tested backend. Schema preparation may add tables/triggers even
when subsequent acquisition fails; it is not a byte-preserving database backup.
Legacy row values are archived; original database bytes are not promised unchanged.
There is no durable checkpoint of a request killed before its commit.

## MPC identity correction

A provisional designation such as `1998 KY26` is queried as a designation, never
as asteroid number 1998. A bounded two-row lookup detects ambiguity; native
identifiers must match before adopting a row. SDK and legacy HTTP paths share that
binding rule. Unavailable elements remain missing, including on the legacy HTTP
path; a real zero magnitude remains zero. MPC's transfer delta-v field is not a
measured maneuver or non-gravitational acceleration. See the primary
[Astroquery MPC API](https://astroquery.readthedocs.io/en/stable/api/astroquery.mpc.MPCClass.html).
These controlled tests do not establish current MPC service availability.

## Reproduce the bounded QA

From the repository root with the project dependencies installed:

```sh
python -m pytest tests/test_object_acquisition_ledger.py -q
python scripts/qa_object_acquisition.py --output /tmp/aneos-acquisition-new-capture
```

The second command requires a nonexistent output directory. It runs actual public
query, disposable SQLite and stored-detail paths with **synthetic** complete,
partial and unavailable sources. It creates text/HTML/SVG at 80/120/160 columns,
full JSON exports and a hash manifest, not live-provider scientific validation.
Committed captures are under `docs/qa/object-acquisition/2.2.0.dev0/`.

## Remaining work

#40/#16 retain all-source eligibility and provider configuration, selective backfill,
canonical cross-query aliases, ambiguous NEODyS resolution, observation/history
roles and other source integrations. The exact request key is not a global object
identity registry. Custom endpoint settings use an instance-scoped cache context;
this is not authorization isolation or proof every provider honors overrides.
#41 retains every-menu QA; these captures cover the stored-object acquisition view.
#43/#8–11 retain observational fits, covariance-aware model comparison, conditional
populations and independent calibration. No artificial-origin probability or
Gaussian-equivalent significance is established by this delivery. The exact
unrounded historical KY26 baseline remains unresolved in #3.

## Delivery evidence — 2026-11-13

The offline suite passed 1,422 tests (eight network tests excluded); core coverage
was 49.63% against the unchanged 30% floor. The new acquisition/presentation and
repository gates require 90%; the measured repository coverage was 90.98%.
The 78 focused tests include rollback, concurrent admission, process death around
commit, cache replay, identity ambiguity, unavailable sources and actual renders.
CI remains the authority for the published revision's complete gate results.

A separate live `1998 KY26` query used a disposable SQLite database and cache under
`/tmp/aneos-ky26-acquisition-live`, not the project database. SBDB and Horizons
contributed; MPC returned an explicit empty result, CAD no bounded encounters and
NEODyS was unavailable. Thus contribution was 2/5 and typed/explicit-empty responses
4/5; scientific completeness remained unavailable. Query ID:
`e76c1173-2116-4b27-9cda-bc8f1b881d66`. Fresh-session export SHA-256:
`363b260f047a237f177eba9209b5fb46b1063af9faeaf66e0a488f10918b47f0`.
SBDB native response SHA-256 was
`ec1a8939f6a967831292847bbb974b9746bc5e7cdd3ebb8b27ca765fe97f0ef0`.

The initial external verification script accessed the wrong metadata attribute
after the application had returned. Correcting that harness and replaying the
committed cached query returned the same query and byte-identical export without
repolling. This was a harness error, not an additional observation or an application
failure. Live raw responses remain outside Git pending redistribution review.
Neither this run nor the synthetic controls validate physical-origin hypotheses.

Final admission review also reproduced and repaired empty-NG/uncertainty-only
contribution inflation and loss of node/periapsis-only partial data (#62). Counts
require actual values, not an empty evidence container or an uncertainty alone.
