# Bounded CAD candidate queries and local evidence

**Post-beta2 development foundation:** this Python acquisition service is not in the
published v2.2.0b2 tag. The legacy root/menu bulk-polling route remains quarantined
under #73 until its controller and enrichment integration are verified.

A candidate query is different from a complete object investigation. CAD provides
computed close approaches selected by a date window, body and distance—not all NEOs,
new discoveries or observations. Its encounter JD/calendar are TDB; distance bounds
and formatted time uncertainty are 3-sigma, while optional diameter sigma is 1-sigma.
These conventions come from [JPL's CAD documentation](https://ssd-api.jpl.nasa.gov/doc/cad.html).

## Query into a separate database

Use the core dependency profile and a fresh development checkout. Select a separate
runtime database before importing the API module; do not replace a research database.
For example, from that checkout:

```sh
mkdir -p beta-runtime/candidate-preview
export ANEOS_DATABASE_URL="sqlite:///$PWD/beta-runtime/candidate-preview/evidence.sqlite"
python - <<'PY'
from aneos_core.data.candidate_discovery import CADCandidateService

result = CADCandidateService().poll("2026-11-12", "2026-11-13", limit=3)
r = result["receipt"]
print("Query:", r["query_id"], "Storage:", result["commit_status"])
print("Outcome:", r["outcome"], "Rows:", r["native_count"], "Matching total:", r["native_total"])
for packet in result["records"]:
    print(packet["designation"], "encounters:", len(packet["neo_data"]["close_approaches"]))
PY
```

This performs **one shared-transport invocation**, with explicit Earth/NEO/0.05-au
selection. It does not query five providers per object, page automatically or promise
a new observational measurement. The application accepts a 1–1,000 encounter-row
limit; object count can be smaller because one object can have multiple encounters.
The dates are the requested native calendar literals, not UTC observation dates.
The source may be unavailable. A storage prerequisite failure stops before querying;
a failed commit/readback raises `EvidencePersistenceError`, never a success result.
A query-specific persistence error carries `query_id` for recovery inspection.

## Read the committed query

```python
from aneos_api.candidate_repository import CandidateQueryRepository

repository = CandidateQueryRepository()
saved = repository.load(query_id)  # None means this ID is not recorded.
entity_bytes = repository.load_native_body(query_id)  # bytes, or None if transport supplied no body
```

`load_native_body` raises `KeyError` for an unknown ID. Readback verifies the frozen
request/body-derived records, source assertion identities/normalization, storage
bindings and original projection inputs. A newer query for the same object does not
rewrite the older receipt. Source snapshots use the existing local assertion ledger;
there is no separate analysis engine. All rows, raw fields, native IDs and source
versions remain traceable. Receipt `software` captures the application/Git/dirty state,
package versions, floating-point settings and relevant source hashes using the same
read-only metadata utility as scientific reports. It is not a complete environment
lock or proof of loaded bytecode; dirty source must be preserved separately. Older
metadata-free receipts are not retroactively relabeled.

Within-response exact repeats and alternatives are reported rather than
counted as independent observations. Cross-query bodies retain hashes for comparison;
automatic cross-query deduplication/alias resolution remains integration work.
Current per-source projections are not deletion
of older evidence.

## Interpret outcomes

| Outcome | Meaning |
|---|---|
| `available` | Valid admitted rows; the source declares all matching query rows returned. |
| `partial` | The source declares additional matching rows, including a zero-row partial response. No hidden next-page request occurs. |
| `missing` | A valid zero-count response for this selection—not proof no NEOs exist. |
| `unavailable` | HTTP/transport failure. Native body bytes are retained when the transport delivered them. |
| `invalid` | JSON/schema/identity/selection checks failed. The source response is archived, not normalized into an object. |
| `not_processed` | Body exceeds the JSON parse budget. Full entity bytes are archived, but no object measurements are admitted. |

`native_count`/`native_total` stay `None` when they cannot be established; they are
not replaced with zero. `object_count` counts admitted packets only. Scientific
completeness and independent-observation count remain unavailable for every outcome.
The native response can include unit-labelled H/diameter/sigma fields; these remain
in the per-object native rows pending the atomic physical-evidence adapter. They
are not silently promoted into an independently verified physical measurement.
A CAD-only packet has no fitted orbit and cannot establish an anomaly or origin.

## Preservation and limits

The receipt stores exact buffered/decompressed response entity bytes up to 1 MiB.
Larger complete entities are stored in an immutable SQL BLOB, with a prefix in the
JSON receipt; `body_in_receipt_complete` describes that JSON copy only.
Readback checks the full archive hash and length; `native_body_status` reports
`complete_entity_verified` or `not_available_from_transport`. This preserves the
committed entity, **not compressed network packets**. The parse/prefix budget is not
a receive or memory bound: shared transport buffers first. Inactivity timeouts are
not absolute end-to-end deadlines. Storage/resource failure produces no success
acknowledgement; it is not a guarantee that arbitrary-size responses can be retained.

These are append-only normal ingestion/read paths, not a backup or migration.
Administrative DDL/reset, direct file editing/replacement, cleanup and deletion can
change or remove the database. Cache eviction is not used for this archive, but
important research still needs a separate backup. Checksums establish consistency,
not external authenticity or protection from a database administrator. Local UTC
receipt/write times are distinct from native TDB encounter times and astronomical
discovery dates; clock-order anomalies are retained rather than repaired by guessing.

## Tests

```sh
python -m pytest tests/test_candidate_discovery.py -q
ANEOS_CAD_NETWORK_TEST=1 ANEOS_CAD_TEST_START=2026-11-12 ANEOS_CAD_TEST_END=2026-11-13 \
  python -m pytest tests/test_candidate_network.py -m network -q
```

The first command is offline and uses synthetic native-shaped controls with real
file-SQLite transactions. It includes authorized insert positives, normal SQL denial
checks, fresh-process replay and separately labeled administrator fault injection in
owned disposable databases. The optional second command makes one public request
into a disposable test database; when explicitly enabled, an unavailable source is a
failed check, not silently skipped success. Neither establishes scientific validation,
full source eligibility, physical fits, an artificial-origin probability or five-sigma
performance. Root restoration and broader enrichment remain #73 work.

### Declared read-only SQLite URIs

Fresh collection and ordinary named-object acquisition reject `mode=ro` before
provider work, even when a fully initialized schema already exists. For the fresh
write profile, `immutable` must be omitted or explicitly false (`0`, `false`,
`off`, `no`, or empty; words are case-insensitive). Enabled or nonstandard values
are not assumed writable. Normal file paths and `mode=rw`/`rwc` remain supported.
These configuration checks do not prove OS permissions, custom VFS behavior,
locking, disk space or future commit success.

Historical query readback/replay remains available through read-only connections;
no evidence is overwritten or copied to a fallback database. SQLite's
[URI documentation](https://www.sqlite.org/uri.html) specifies that `immutable=1`
also disables locking/change detection: use it only for genuinely fixed snapshots,
not as an authenticity guarantee or on a database another process might modify.
