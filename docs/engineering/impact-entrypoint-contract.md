# Impact entry points, conditional scenarios and local evidence archive

Date: 2026-11-12. Branch `codex/impact-entrypoint-contract`.
Base `1386c89c9219b69e3a57fabb431a5a6170238396` (draft #35, including CI dependency
correction). Programme #2; impact #32; shared reporting #6/#12; persistence/cases
#15/#13; verification #17. ADR-075. This is a scoped increment, not scientific
readiness or a complete durable job/research-case implementation.

## A. Engineering summary

Confirmed defects:

- V2 impact analysis passed catalogue data to an unfitted calculator and invented
  a 30-day missing observation arc. Browser/display defaults turned absent values
  into zero and estimated a Torino class. Flat exports lost source evidence.
- A daemon thread wrote impact probability into an origin anomaly-score DB field,
  silently swallowed failures, and could stop before saving. Export fallback could
  overwrite a historical file after an exclusive-create failure.
- ImpactEnhancedPipeline inferred observation arc/uncertainty from quality scores,
  promoted origin scores to propulsion/risk and converted heuristic results into
  a scientific-looking record. Missing domain records defaulted to negligible risk.
- The direct calculator labelled hand-set geometry/arc factors as probability,
  confidence and intervals. Documentation overstated local Earth/Moon capabilities.

Changes:

- V2 menu, browser, export and enhanced-impact wrapper use the same replay/primary-
  identity-bound Sentry report as REST. Explicit source lookup does not depend on
  origin rank or geometric risk proxies. Screening is preserved separately. The wrapper's `impact_assessment` is now the
  replayable source-report dictionary, not a fabricated scalar ImpactAssessment.
  Consumers must use that versioned report contract. Old heuristic threshold
  configuration is read-only rather than silently accepted as an active policy;
  `include_impact_assessment=False` explicitly disables the lookup. Invalid
  evidence, provider outage, unexpected failure, not-requested state and unconfirmed
  persistence are not scientific negatives or zero-risk results.
- Former calculator/Torino entry-point names remain, but fail with explicit
  controlled-deprecation messages. Private legacy formulas remain available through
  an explicitly named diagnostic replay requiring a supplied scenario arc. Its
  geometric/lunar/quality scalars are labelled **not probabilities or intervals**.
- `calculate_conditional_impact_energy` retains useful classical physics with explicit
  mass kg and speed km/s, no guessed density/diameter/arc or collision/damage inference.
  It states the nonrelativistic assumption and rejects invalid/nonfinite/superluminal
  inputs. TNT conversion follows [NIST SP 811's energy-equivalent factor](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b9):
  one ton is 4.184e9 J, hence one megaton is 4.184e15 J.
- Empty legacy domain records round-trip null/not-assessed, not zero/negligible risk.
  Legacy catalogue UI cannot promote origin/geometric proxies to HIGH/MED/LOW risk.
- Synchronous replay-verified, no-overwrite local snapshot archiving replaces impact
  daemon DB writes. Source probability is not stored as an origin score. Menu JSON/
  CSV retains complete source/report evidence; export uses exclusive file creation
  and no overwriting fallback. Other Exporter callers retain their existing defaults.
- README current mission/plain-English/impact descriptions and example are corrected;
  a high statistical rank cannot establish human or extraterrestrial identity.
  Remaining historical examples/capability tables still require the broader #17 audit.

## B. Scientific assessment

No new measurements or fits were obtained for this increment. Synthetic controls
and the previously captured real MC source-format fixture exercise transport and
interpretation, not independent impact/origin calibration. The dated KY26 not-listed
lookup, historical screening baseline and source uncertainties are unchanged.

Sentry remains an external model output with listed-event scope and material source
uncertainty. It supplies no local fitted impact probability, lunar likelihood,
confidence interval, aNEOS risk class, origin evidence or discovery significance.
Conditional kinetic energy says what follows **if the supplied mass/speed apply**;
it does not predict impact, atmospheric coupling, fragmentation, crater or damage.
Legacy diagnostic numbers retain reproducibility, not scientific acceptance.

## C. Verification and architecture

`tests/test_impact_entrypoint_contract.py`: **33 new controls**. Coverage includes
actual menu source/display/browser/JSON/CSV flows; valid available MC evidence as
well as absent/invalid/wrong-object/outage paths; actual core/enhanced screening
plus source integration; explicit disabled/failure/persistence outcomes; conditional
energy and legacy diagnostic contracts; no origin-to-risk mapping; archive copies,
concurrent publication, existing corruption, changed dated snapshots, fresh-process
read-back and killing a writer before publication. No scoring threshold was tuned.

The archive is in the existing Reporting boundary, not a new bounded context or UI.
Data Acquisition still owns Sentry parsing/transport; Impact Assessment owns scoped
interpretation. ADR-075/DDD/API/scientific notes and deployment configuration are
updated. The existing API schema remains unchanged by this entry-point increment.

Existing tests requiring heuristic menu calls, a default missing arc and locally
invented Torino classes are migrated to the corrected contract—not skipped or
weakened. A vacuous `assert ... or True` export fallback test is replaced by an
actual no-overwrite/no-false-success check. The impact-persistence test now reads
back exact evidence instead of asserting an import/string in source code.

A stale old impact test seam initially allowed one unmarked live `99942` Sentry
lookup; none of that response was adopted as baseline/validation evidence. To prevent
recurrence, the entire selected offline suite now denies unmarked DNS/socket calls
and fails teardown even when application code catches them. The stronger gate
exposed pre-existing unmocked localhost health probes and real daemon population
workers escaping transport-only tests. Health fixtures now inject their actual
probe; network API tests explicitly cover enqueue/status only with an injected
worker. Their real worker/analysis/durability is not certified (#6/#15). A child
pytest demonstrates a caught forbidden attempt **fails**, while an explicitly
network-marked synthetic localhost exchange succeeds. No external provider is used
by these controls. No test is disabled, no network mark added to hide a failure,
and no coverage gate lowered.

## D. Archive operations and remaining work

`ANEOS_EVIDENCE_ARCHIVE` defaults to `neo_data/evidence`. Compose sets
`/app/data/evidence` on the existing persistent data mount. Configure a durable,
owner-controlled local filesystem for deployed use. Report IDs are checksums, not
cryptographic source authentication. The archive stores normalized report JSON
including the exact source UTF-8 string; source bytes are recoverable without
altering their hash. It does not preserve arbitrary external file metadata.

A put verifies the report, writes/fsyncs a private staging file, atomically hard-
links to a new report-ID filename, fsyncs the directory, then reads/replays the
published report before confirmation. Same-ID saves are idempotent. Different dated
reports remain separate. An existing invalid/conflicting entry blocks the save;
there is no destructive repair or overwrite. Archive read-back survives process
restart. Failed save confirmation is explicit; valid source evidence remains in
memory. If a write was published before a later confirmation failure, retry the
same verified ID rather than deleting history.

This requires working POSIX local-filesystem hard-link/fsync semantics. Network/
distributed filesystems, hardware loss, full host crash behavior and backups are
not certified. A killed staging writer can leave `.pending-*` bytes; they are not
published reports and are not silently cleaned. Stop writers and inspect/retain
such files before any separately approved maintenance. Never delete existing
reports or the user's scientific database to obtain a clean run. Rollback is a code
revert with writers stopped; preserve archive history. Export exclusive creation
prevents normal overwrites but is not itself crash-safe archiving.

Still required: durable API/polling/network/automatic jobs and restart recovery,
API-wide report persistence, a queryable append-only case lifecycle/history,
backup/retention/recovery policy, and the broader source/config/cache and legacy
interface audit. Legacy population jobs currently initialize numeric sigma/p-value
placeholders while queued and use daemon/in-memory execution; that remains #6/#15.
This archive does not solve those jobs. #32 stays open pending a wider impact/science
claim audit; #17's whole-project documentation audit remains incomplete. Physical
uncertainty ingestion, KY26 literature/observational fitting, independent calibration/
controls and 2031 predictions remain open. No five-sigma permission is granted.

## E. Reproducibility

```sh
PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-impact-entry-final5.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core \
  --cov-report=term-missing --cov-fail-under=30
```

Final local result: **857 passed, five network deselected, five warnings in 35.63s;
45.06% core coverage, unchanged 30% gate**. The selected suite now enforces offline
isolation as described above. The PR records the final revision/remote CI separately.
No repository database or original research result is changed by this branch.

### Clean-runner offline fixture correction

The first remote #36 run had **857 passing tests and two offline-guard teardown
errors**. Both were pre-existing NEODyS provisional-designation tests calling the
real MPC resolver without an injected response. Their weak `result != year` checks
could pass even after a failed live lookup. Local behavior had not exposed the same
cold-runner path. The tests now inject explicit synthetic MPC record replies and
assert exact returned identifiers **and** query arguments. Those synthetic numbers
are not object catalogue identifications. The offline gate remains unchanged;
no network marker, skip, xfail or weaker acceptance is introduced.
