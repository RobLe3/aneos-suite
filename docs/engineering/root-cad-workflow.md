# Bounded CAD collection through the root poller

**Post-beta2 development; not a new scientific validation claim.** This restores the
NASA_CAD root/menu path using shared acquisition, the existing transactional evidence
ledger and canonical screening. Other root bulk discovery choices remain unavailable.
Named-object acquisition through the shared adapters is a different operation.

## Collect a small sample

From an installed checkout, use a file-backed project SQLite database:

```sh
export ANEOS_DATABASE_URL="sqlite:////absolute/path/to/research.sqlite"
python neo_poller.py --api NASA_CAD --period 1d --max-results 10
```

This explicitly performs one CAD collection request for NEO Earth encounters within
0.05 au, sorted by date. `--max-results` limits **native encounter rows**, not unique
objects, to 1..1000. No automatic pagination occurs. The requested dates are printed
in the committed receipt; calendar periods are anchored to today's UTC calendar date,
while request/output astronomical dates remain native TDB. These are predicted
encounters, not discovery dates or direct observations. The supported period is
positive and at most 200 calendar years; a provider may reject a window it cannot
serve, which remains a recorded unavailable outcome rather than invented data.

The summary distinguishes available, partial, valid empty, invalid and unavailable
queries. The compatibility `fetch_cad_data` helper now has the same explicit
1..1000 row budget. Over-budget callers must request smaller declared windows or
remain unavailable; no silent clamping pretends to deliver a complete historical
chunk. Broader historical/source restoration remains #22/#73. A limit smaller than the provider's matching total remains **partial**.
Successful JSON/Markdown exports are exclusive bundles under `dashboard_results/`;
empty/failed collections retain their query evidence in invocation context, without
a fake object row. Invalid/unavailable direct CLI outcomes return status 2; export
failure returns 1. Plain and Rich menu routes use the same controller. No interactive
browser opens in direct mode unless `--browse` is explicitly supplied.

## Enrich selected objects explicitly

```sh
python neo_poller.py --api NASA_CAD --period 1d --max-results 10 --enrich-objects 2
```

The default additional-object budget is zero. `--enrich-objects` accepts 0..20 and
selects up to that many distinct objects in native first-appearance order, not a
scientifically calibrated priority. Objects outside the budget stay visible.

Each selected object uses the existing named-object `DataFetcher`, configured provider
plan, cache/backoff and retry limits. This is an **object invocation budget**, not a
claim that exactly one HTTP request is made per object. CAD is replayed from the
already committed collection, not requested again or counted as another observation.
The optional source-plan profile records that reference and preserves the original
read timestamp. Other provider outcomes remain available/missing/unavailable/invalid
or ambiguous; `no_new_source_data` means no non-CAD adapter contributed new data.

The collector announces its committed query before waiting on optional providers,
then reports progress for requested objects only. A live one-object check on
2026-11-13 waited roughly 75 seconds for MPC's failed request: provider latency is
not an instant-completion guarantee, and the default additional-object budget stays
zero. This one check is not a universal latency bound or availability estimate.

Object acquisition must commit and read back successfully before analysis. The
controller then reads the existing verified combined ledger projection; original
candidate snapshots and all encounter rows remain separately preserved. Provider
failures cannot become measured zeros, natural-origin labels or claimed completeness.
A persistence/identity/binding failure returns no completed analysis; the committed
collection query ID remains available for recovery. Source aliases and conflicting
measurements still have the limits documented by #40/#59 and source-specific contracts.

## Physical assertions and interpretation

Native labelled H, diameter (km) and its documented 1-sigma uncertainty (km) are admitted
through the existing atomic physical-evidence aggregate. H has no invented uncertainty.
Diameter and sigma always come from the same row. Native strings, row/field labels,
indices and parent response hash are preserved. Equal native catalogue quantities
with the same orbit reference share a representative across encounters; this does
not establish independent measurement replication. Alternative values/sigmas/references
remain ambiguous and are not averaged. Missing measurement dates/references are not
replaced with encounter or retrieval dates. No albedo, orbit, acceleration or
artificial-origin probability is fabricated.

CAD alone is not an orbit source, so screening may remain inconclusive even after a
successful query. An uncalibrated ranking does not establish Gaussian significance,
physical origin or impact risk. Query coverage is not scientific completeness.

## Read-only collection replay

```python
from neo_poller import NEOPoller
poller = NEOPoller()
records = poller.replay_candidate_query("cad-<existing-32-character-hex-id>")
poller.display_results(records)
```

Use an actual ID printed by the collector. Replay reads the same configured database,
verifies the source/packet/projection bindings, and does not poll, append records,
enrich or reinterpret the frozen physical policy. It replays the **collection**, not
an implicit latest scientific assessment. Enriched report bundles separately retain
the object acquisition receipt, frozen combined projection and shared assessment;
the existing dashboard report loader can inspect those without a network operation.

Old receipts without the optional `cad-physical-v1` policy still replay their original
raw-row-only packets and hashes. Neither old research inputs nor beta2 tags are
rewritten. Hashes prove consistency with saved evidence, not initial authenticity
or resistance to a fully compromised source/administrator.

## Offline checks

```sh
python -m pytest tests/test_root_native_cad.py tests/test_candidate_discovery.py \
  tests/test_object_acquisition_ledger.py tests/test_atomic_physical_evidence.py -q
```

Controls use native-shaped synthetic responses, temporary file-SQLite databases,
network denial, genuine acquisition/persistence/replay, plain/Rich menus, CLI exit
codes and 80/120/160-column views. They are engineering evidence, not independent
scientific calibration. Optional live checks must remain explicitly bounded and
provider-safe; normal CI stays offline.
