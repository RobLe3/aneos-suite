# Discovery, historical integrity and calibration: roadmap review

**Evidence date:** 2026-11-13. **Reviewed main:** `3dd47f193d866da828358cb906f09d111759ca98`.
**Programme:** [#2](https://github.com/RobLe3/aneos-suite/issues/2).

## Decision and scope

Extend the existing trajectory, not the number of engines. Preserve the canonical
ranking policy and its 5.0 **ranking-unit** threshold. Restore qualified exploratory
diagnostics separately from origin inference. Add historical-integrity assessment
through existing acquisition, normalization and evidence persistence. No new bounded
context, database, classifier or network service is justified by this review.

This is a repository-wide **dependency and gap assessment**, not an exhaustive runtime
certification of every menu, provider or scientific algorithm. Main source, cumulative
ADRs/DDD, issue bodies, release/PR inventory and CI were inspected; targeted offline
baseline and canonical-boundary tests were rerun. Unexercised historical utility math,
provider archive access and independent scientific validation remain unverified.
No production-readiness or artificial-origin accuracy claim follows.

### Status vocabulary

- **Implemented:** code exists; not necessarily called by the application.
- **Integrated:** a specific application path calls it; scope must be named.
- **Independently validated:** external, appropriately designed evaluation supports
  a stated scientific claim. Passing our tests is not this status.
- **Experimental:** callable research utility without qualified inference.
- **Withdrawn:** previously promoted inference is deliberately excluded or unavailable.
- **Unavailable:** prerequisites or implementation are absent; not a zero measurement.

## A. Current implementation and gaps

| Area / source locator | Evidence-backed status | Remaining gap / owner |
|---|---|---|
| `analysis/pipeline.py:AnalysisPipeline.analyze_neo`, `analysis/screening.py:ScreeningAnalysisService` | Integrated shared screening; `legacy_indicators_run=[]`; ranking, not calibrated significance | Exploratory coverage #86; inference separation #6 |
| `pipeline.py:_initialize_indicators`, `_run_indicators`; `analysis/indicators/` | 18 utilities initialized; runner retained but not called by canonical analysis (ADR-069) | Qualified prerequisites, availability states, safe descriptions and runner #86 |
| `analysis/scoring.py:ScoreCalculator`, `advanced_scoring.py:AdvancedScoreCalculator` | Legacy heuristic utilities remain callable, not an independently validated origin engine | Do not reconnect old classifications; caller/bypass audit #21/#23/#30 |
| `analysis/enhanced_pipeline.py:analyze_neo_with_advanced_scoring`, `pipeline/automatic_review_pipeline.py:_apply_xviii_swarm_scoring` | Legacy stage-to-clue/proxy fusion explicitly withdrawn; canonical wrappers retained | Individual physical models #8/#9, qualified grouping #10 |
| `validation/multi_stage_validator.py` | Required review infrastructure; comparative fits, known-object exclusion and calibrated significance can return unavailable | A review gate does not install missing science #8/#9/#11 |
| `validation/delta_bic_analysis.py:analyze_delta_bic` | Callable legacy orbital-prior comparison, BIC called with n=1 and k=3 | Not an observational fit; keep out of validated physical inference #8 |
| `data/sources/sbdb.py`, `data/models.py`, native NG/epoch/physical contracts | Typed A1/A2/A3/uncertainty/units/covariance and source-specific evidence plumbing implemented with tests | Full acceptance by source/path, historical solution compatibility, scientific force fits #4/#5/#25/#43 |
| `data/enrichment.py`, `aneos_api/acquisition_repository.py`, `aneos_api/database.py` | Existing transactional assertions, replayable projections, acquisition receipts and source-payload retention; #71 closed within its scope | Archive eligibility, independent capture authenticity, historical reconciliation #40/#87 |
| Root `neo_poller.py`, legacy menu acquisition | Unsafe discovery paths remain quarantined; readable stored-result views landed in PR82 | Actual bulk discovery/controller and physical admission #73/#83/#84; do not equate view repair with acquisition repair |
| `docs/case-studies/1998-KY26/`, fixtures and replay script | Frozen historical display and rounded-input replay preserved; source register exists | Original full-precision input provenance #3; complete dated scientific/historical case #7 |
| Rich/REST/dashboard/report/export adapters | Shared contract integration and scoped display repairs landed | Exhaustive actual option/width/field/route coverage still #12/#41/#42; no universal menu QA claim |
| `ROADMAP.md`, DDD glossary and historical scientific docs | Older promises conflict with current unavailable/withdrawn inference | Active-status overrides and this dependency map; preserve dated design history #17/#44/#46 |

Paths above are relative to `aneos_core/` unless prefixed otherwise. File/symbol
locators identify inspectable evidence; they are not a claim that every code branch
was runtime-exercised. In particular, a source import or class name does not establish
that a detector is integrated or physically valid.

### Historical indicator inventory

All 18 below are **implemented, dormant in canonical analysis, not independently
validated**. The existing `IndicatorResult` conflates missing/error outcomes with
zero scores and hand-selected confidence. `PipelineConfig` also reuses one mutable
default `IndicatorConfig` across indicators. Both must be addressed before a new
qualified runner; do not simply invoke `_run_indicators` again.

| Utility (`analysis/indicators/`) | Required evidence / qualification before use |
|---|---|
| `orbital.EccentricityIndicator` | e, uncertainty and conditional population; remove near-circular/high-e artificiality language and fixed-confidence interpretation |
| `orbital.InclinationIndicator` | i/frame/reference plane and conditional selection model; low i or retrograde alone is not engineering |
| `orbital.SemiMajorAxisIndicator` | a/epoch/units and population context; orbital location alone is not origin evidence |
| `orbital.OrbitalResonanceIndicator` | Period uncertainties and resonance definition; simple period ratios are descriptive, dynamical resonance needs a qualified model |
| `orbital.OrbitalStabilityIndicator` | Propagation/force model, covariance and temporal baseline; instantaneous orbital heuristics do not establish active maintenance |
| `velocity.VelocityShiftIndicator` | Compatible velocity definitions/geometry and actual observations; different encounters are not repeated velocity measurements of one state |
| `velocity.AccelerationIndicator` | Replace relative encounter-speed differences/time with compatible residual dynamics for any acceleration claim; never equate it with measured A1/A2/A3 |
| `velocity.VelocityConsistencyIndicator` | Native velocity uncertainty/coverage, geometry and natural comparator; consistency alone is not control |
| `velocity.InfinityVelocityIndicator` | Native hyperbolic-excess definition/body/units; slow encounters/co-orbital dynamics are not spacecraft confirmation |
| `temporal.CloseApproachRegularityIndicator` | Encounter series origin, selection/window and dynamical comparator; computed encounters are not independent observations or targeting |
| `temporal.ObservationGapIndicator` | Actual observation dates/cadence and survey coverage; encounter gaps cannot substitute for observation gaps |
| `temporal.PeriodicityIndicator` | Adequate dated measurements, sampling/window aliases, red-noise/null model, trials correction and withheld prediction |
| `temporal.TemporalInertiaIndicator` | Existing approach-distance change proxy cannot establish suppressed orbital evolution; requires physical predictions and compatible observations |
| `geographic.SubpointClusteringIndicator` | Valid body-fixed geometry/time/frame plus selection baseline; geographic coincidence is not intentional targeting |
| `geographic.GeographicBiasIndicator` | Same geometric prerequisites and survey/site bias controls; no geopolitical or intentional inference |
| `physical.DiameterAnomalyIndicator` | Atomic diameter/uncertainty/method and size-conditioned population; spacecraft-scale dimensions are common to natural small bodies |
| `physical.AlbedoAnomalyIndicator` | Atomic albedo uncertainty/model and size/H dependence; high reflectivity is not engineered material |
| `physical.SpectralAnomalyIndicator` | Native taxonomy, measurement/uncertainty and orbital-population context; unknown or unusual labels are not composition proof |

**Not supplied by this suite of 18:** fitted residual acceleration/change-point
analysis with covariance, compatible historical orbit-solution discontinuities,
spin evolution, density/area-to-mass inference, radar polarization or thermophysical
model qualification. Some typed fields and research validation modules exist; that
is not an integrated observational model for these tasks. Explicit prerequisite
checks must report unavailable until qualified (#5/#8/#9/#43/#86).

### Historical acquisition: authority and access limits

- [JPL SBDB documentation](https://ssd-api.jpl.nasa.gov/doc/sbdb.html) describes
  current solutions, solution-epoch output, covariance and alternate orbits. Its
  `soln-epoch` option is not a historical-snapshot selector; alternate orbits are not
  a general solution-history archive. A current request at an old epoch cannot
  establish what a catalogue published then. A separate historical archive needs
  evidence of capture and identity. JPL also uses MPC observations: two providers
  are not automatically independent observations.
- [JAXA's 2020-09-15 briefing](https://www.hayabusa2.jaxa.jp/enjoy/material/press/Hayabusa2_Press_20200915_ver9_en2.pdf)
  is an accessible dated primary publication (p.7 contains the then-reported KY26
  properties). It is a historical interpretation, not raw astrometry. Text access
  succeeded; the attempted visual PDF capture timed out. Exact table transcription
  and archival capture verification remain #7 work, not completed by finding a URL.
- MPC data access returned HTTP403 in this review; the NEODyS site returned an
  access-denied/Anubis page. This establishes a limitation of these checks, not that
  historical records do not exist. Do not bypass access controls or silently use
  fabricated replacements. Archive eligibility, supported access and redistribution
  must be settled through the provider under #16/#40/#87.
- Horizons-derived historical trajectories, archive mirrors and literature tables
  need separate evidence-type labels. Availability/authenticity of eligible historic
  solutions, MPC circulars/observations and independent captures is not yet established
  comprehensively. No actual catalogue manipulation was substantiated by this review.

### KY26 baseline and current scientific status

The historical **displayed** 4.142 remains archived. Frozen rounded inputs reproduce
**4.141288008229254**, with legacy combined p-value **0.0013052410308005413**. The
original unrounded inputs/source responses are missing; do not call this exact
reproduction of the recorded 4.142/p=0.0012976288 result. These are preserved legacy
outputs, not validated significance/probability. The canonical ranking remains a
separate versioned result, not a retroactive replacement.

`tests/test_ky26_baseline.py` checks frozen hashes/numerical replay/non-overwrite;
`tests/test_core_analysis_contract.py` checks shared screening, candidate and
not-flagged positive paths, unavailable/error abstention and adapters. **38 tests
passed** on this review base with Python3.11.9; this is bounded engineering evidence.
The three added inventory controls also passed (41 targeted tests total); all 18
retained utilities had zero calls while canonical payload parity was preserved.
The full unchanged offline core gate passed: **1,773 tests, 8 network tests excluded,
53.11% core coverage against the existing 30% floor**. Component gates and Docker
still require the pushed-head CI run for this PR.
No physical-origin ranking, historical tampering finding or independently calibrated
significance has been established. Natural explanations, the human Phobos1 hypothesis
and measurement/model artifacts still require compatible observational fits (#8/#9).

## B. Architecture impact

Existing ADR-065/069 own ranking/inference separation and the canonical pipeline;
ADR-068 owns withdrawal of placeholder physical validation; ADR-084/086/088/089/091
own ledger, acquisition, chronology, freshness and payload handoff. These remain
valid. DDD needs an active-status glossary so historical sigma/posterior/SWARM design
language cannot override the current contract. No new bounded context is needed.

Before incompatible implementation, #86 should amend the core-analysis contract
and #87 should amend acquisition/normalization lineage and integrity comparison.
An ADR is warranted for those material contracts after review; this assessment does
not assign the next ADR number (PR85 already proposes ADR-092). Reporting consumes
these contracts; it must not independently infer scores or initiate network work.

| Responsibility | Existing owner / boundary |
|---|---|
| Historic retrieval and source eligibility | Data acquisition / adapters (#16/#40) |
| Native records, lineage and checkpoints | Existing evidence ledger / persistence (#13/#40/#87) |
| Compatible historical reconciliation | Normalization/source history, with validation for physical transforms (#87/#8) |
| Individual exploratory indicators | Core analysis (#86) |
| Dependence-aware aggregation/follow-up | Canonical analysis policies (#6/#10) |
| Fitted likelihood and model complexity | Scientific validation (#8/#9) |
| Brier/evaluation and independent controls | Validation/evaluation (#10/#11) |
| Evolving scientific case | Existing case persistence/reporting (#7/#13) |
| Rich/REST/dashboard/export | Existing reporting adapters (#12/#41/#42) |

## C. Issue reconciliation

No existing issue is closed by this plan. Closed #71 retains its source-payload
scope; it does not certify historical authenticity or independently backed-up storage.
Retained development branches are not necessarily active: many draft increments were
integrated through PR76. Issue bodies/comments can describe older snapshots; reconcile
against the current commit, not against the word “draft” in an old comment.

| Action | Issues / specific addition |
|---|---|
| Create | #86 qualified exploratory evidence/runner; #87 historical lineage/comparison and compromise limits. Two deliverables, no new issue per indicator |
| Update / link | #2 programme; #6 six separate outputs and frozen ranking; #10 dependency groups/conditional populations; #11 defined prospective Brier targets and object/time splits |
| Update / link | #13 append-only source/case lineage; #40 source-eligibility/history/duplicates; #16 safe historic access; #7 KY26 evidence chronology |
| Update / link | #43 operation-specific scientific fitness; #12 existing presentation contracts; #17/#44/#46 architecture/readiness/documentation evidence |
| Preserve near-term ownership | #73 bulk legacy polling, #83 receipt foundation (PR85), #84 bounded controller/physical admission; do not block all acquisition on future science |
| Leave scope unchanged | #3 exact baseline, #4/#5 native NG, #8/#9 physical fits, #14 predictions, #15 durability, #25 atomic physical assertions, #41 menu QA, #42 explanation/graphics, #45 version consistency and open defect issues |
| Defer implementation, not accountability | Independent signed archives/external replication; probabilistic prediction service; significance/origin claims until prerequisites and authority exist |

No GitHub release milestones existed at inspection. Do not invent milestone dates
from horizon labels. Priority/scope/risk are recorded in this map and #86/#87;
project-wide milestone restructuring is unnecessary for the first increment.

## D. Dependency-aware roadmap and release gates

### Short term: safe acquisition, preservation and contract review

1. Preserve beta2 and canonical/baseline numerical contracts. Finish qualified #83
   receipt foundation; #84 then restores a bounded actual root/menu/CLI CAD path.
   Unsupported discovery sources remain explicit, not fake samples (#73).
2. Inventory source/path/field retention, provider eligibility and historical captures
   (#16/#40/#71); distinguish query completion from scientific measurement completeness.
3. Review #86 indicator and #87 lineage/integrity contracts; add owned synthetic
   legitimate-revision and corruption controls before promoting classifiers.
4. Publish current-status docs, source matrix, unresolved acceptance evidence and
   actual-menu scorecards (#17/#41/#44/#46). Fix prerequisite defects in small PRs.

**Release gate:** existing CI/component floors unchanged; native positive plus
empty/missing/invalid/outage/restart/readback and relevant actual-entrypoint tests;
no history loss or newly implied calibrated significance/origin accuracy. These
engineering changes need not wait for a full physical-origin investigation.

### Medium term: useful, qualified investigation

- #86 reviewed primitives -> individually qualified diagnostic groups -> #10
  dependency-aware, conditional-population follow-up policies. Preserve legacy rank
  as a named policy, never add correlated feature “votes”.
- #40/#13 source lineage -> #87 compatibility-gated historical comparisons -> #7
  KY26 historical consistency case. Parallel work may establish source eligibility
  and frozen fixtures while the bounded acquisition controller is integrated.
- #5/#7 compatible observations -> #8 valid fitted likelihood/residual diagnostics ->
  #9 natural/human/artifact model comparisons. #11 hard natural and confirmed human
  controls validate actual deployed policies, not just toy accuracy.
- #12/#42 present separate exploratory, quality, residual, priority, origin and
  integrity results in existing interfaces, including unavailable states.

**Acceptance:** measured reproducibility/coverage improvement and false-positive
control by stratum, not a larger anomaly count. Useful unresolved outcomes remain.

### Long term: independently tested predictions and durable operation

- Freeze prediction targets/windows/outcome adjudication -> independent object/time
  splits -> simple baselines -> prospective evaluation/Brier/reliability and
  precision-recall with uncertainty (#10/#11). Unknown origin is not a natural label.
- Search-wide significance needs a defensible null, dependence/trials correction and
  independent validation. No sample count or threshold alone enables “five sigma”.
- Expand eligible historical monitoring, independently witnessed checkpoints,
  separately administered recovery/security/scalability tests (#87/#15/#47). Do not
  provision or activate external monitoring/backups merely because they are planned.
- Maintain longitudinal reassessments and frozen prospective predictions (#13/#14),
  including eventual direct observations, without editing past forecasts.

**Acceptance:** independent generalization and reviewed scientific evidence within
explicit limits; authenticated provenance and recoverable histories within a stated
threat model. A fully compromised source/ingestion/administration chain remains a
limitation unless an independent witness can contradict it.

**ETA:** horizons are dependencies, not calendar promises. Beta2 is already
published; main has newer scoped changes. PR85 is green at `dfa404d`; #84 has not
been implemented or estimated sufficiently to promise the next integrated polling
release. Estimate that release after #84's bounded path and unresolved source
prerequisites are assessed. Do not hold a qualified engineering increment for the
entire scientific programme, and do not call it a scientifically validated release.

## E. Proposed contracts — design only, not installed schemas

All structures below need version, policy/software identifiers, stable evidence
references and serializable unavailable/error states. Review backward-compatible
extensions first. No migration or incompatible API is implemented by this PR.

1. **Indicator evidence:** id/version, prerequisites, availability, native measurements
   and uncertainty, evidence basis, reference population/version, effect (nullable),
   dependency group, data-quality flags, explanation status and limitations. Anomaly
   states: NOT_ASSESSED, INSUFFICIENT_DATA, CONSISTENT_WITH_REFERENCE, UNUSUAL,
   REPRODUCED_ANOMALY, CONFLICTING_EVIDENCE, INVALID_INPUT. “Reproduced” must say
   whether it means replay of the same data or independent new observations.
2. **Historical lineage:** source/assertion/parent IDs, designation/aliases, native
   payload hash, capture/observation/publication/retrieval times and their basis,
   witness/signature when available, solution/epoch/time scale/frame/origin/model,
   units/uncertainties/ordered covariance, correction/retraction/publication links,
   parser/software/schema and redistribution status. Distinguish capture authenticity
   from a locally consistent hash. Missing capture dates remain missing.
3. **Integrity assessment:** comparison IDs, event type (duplicate/revision/physical
   measurement/reinterpretation/missing/withdrawn/conflict), compatibility gates,
   differences and uncertainty, candidate explanations, independent witnesses,
   diagnostic evidence and reviewer/rationale. States: CONSISTENT,
   EXPLAINED_REVISION, UNRESOLVED_CONFLICT, SUSPECTED_CORRUPTION,
   POSSIBLE_TAMPERING, INSUFFICIENT_HISTORY, SOURCE_UNAVAILABLE. POSSIBLE_TAMPERING
   needs explicit unauthorized-modification evidence or authenticated independent
   historical contradiction after alternatives are investigated; it is never an
   origin classification or an attribution of intent/actor.
4. **Follow-up priority:** reproducible policy/version and observation budget, candidate
   set, group contributions/dependence treatment, availability/quality, selection
   rationale and tie rule. Preserve the canonical historical rank separately. No
   hidden bonuses or probability-like interpretation of the priority value.
5. **Calibration evaluation:** prediction target/event/window and outcome protocol,
   dataset/split hashes, object/time grouping, independently adjudicated labels,
   frozen model/policy, simple comparator, counts/uncertainty, Brier/reliability,
   precision-recall/errors/abstentions, selection/trials corrections and limitations.
   Ineligible or unobserved outcomes cannot silently become negative labels. No
   Brier-derived artificiality score without an independently established target.

## F. Verification plan and measurable gates

| Unit / sample | Comparator and pass gate | Required evidence |
|---|---|---|
| Frozen KY26 rounded fixture | Historical hash manifest; numeric replay relative tolerance 1e-12; no file replacement | Existing replay/manifest tests; original missing inputs remain #3 |
| Canonical boundary: candidate, not-flagged, missing/invalid | Same typed input and policy gives identical shared payload; no legacy execution; no rank/threshold change | Existing core parity plus explicit 18-utility spy inventory test |
| Each qualified diagnostic | Predeclared natural/reference and unusual positive controls plus missing/invalid/zero cases; all specified cases pass, whole counts reported | Per-indicator inputs, source hashes, uncertainty/model/prerequisite tests; independent data needed for scientific calibration |
| Historical pair comparisons | At least one fixture for each event/state and legitimate updated fit, improved measurement, model/epoch mismatch, malformed covariance, withdrawal vs outage | Expected state and abstention frozen before implementation; every fixture correct; actual tampering remains unsubstantiated |
| Integrity/recovery | Owned temporary DB: authorized insert/replay succeeds; modification/deletion/broken binding detected relative to sealed record; independent-witness and full-compromise limitations explicit | Failure injection, fresh-process read/recovery, preserved original bytes and append-only correction history; no real evidence tampering |
| Dependency-aware follow-up | Duplicated provider/derived feature cannot gain independent weight; missing does not reduce anomaly as a measured zero; deterministic budget/ties | Frozen policies, ablations, provenance graph and matched comparison by group |
| Probabilistic target | Outcome/window fixed before prediction; no object/time/parameter leakage; compare each condition against simple baseline | Counts, Brier/reliability/PR, error uncertainty and independent review; no invented universal sample threshold |
| Source/path and presentation | Source-by-field matrix and each actual eligible menu/route: native positive, empty, partial, stale/conflicting/unavailable, narrow/wide output and export | #40/#41/#46 scorecards; no network during scoring/rendering; no universal completeness from a single successful query |

“100%” may describe a specified query's eligible responses or a tested fixture set,
with an explicit denominator. It cannot promise every scientific measurement exists.
Coverage and scientific fitness remain separate; cannot fill unknowns to hit a target.

Reproduction on this base:

```sh
python -m pytest tests/test_ky26_baseline.py tests/test_core_analysis_contract.py \
  tests/test_exploratory_boundary_inventory.py -q -o addopts=''
python scripts/reproduce_ky26_baseline.py --output /tmp/ky26-review-new.json
# Output creation is exclusive: choose a new path, do not overwrite an old replay.
```

Use the pinned project environment/dependency manifests and record interpreter,
commit and dataset hashes. Normal CI remains the authority for engineering gates,
not an independent scientific validation service. Main CI43411880886 passed at
3dd47f1. PR85 CI43421988273 passed both tests and Docker at dfa404d; its 1,846 tests
include scoped acquisition evidence, not proof that #73/#84 or all sources are done.

## G. Smallest first PR and review disposition

**First new increment:** this roadmap/DDD-status reconciliation, #86/#87 issue
ownership, and an offline boundary inventory regression. It changes no runtime
contract, dependency, ranking, schema or release version. It is low risk because
it makes the intended work testable without reactivating unsafe science.

**Existing increment:** PR85 can proceed through normal review as the acquisition
foundation for #84. It uses the existing ledger/transaction engine and an explicit
collection-query receipt, not a parallel persistence system. Keep its limitations:
one bounded CAD request, partial results explicit, native entity retained, no
automatic multi-provider enrichment or physical-origin classification. Buffered
transport means the payload parse/archive budget is not a streaming receive cap.

Then implement #84's smallest bounded native controller/admission path with actual
root/menu/CLI controls. Review #86/#87 contracts before their runtime implementation.

**Disposition:** YES, the existing architecture can accommodate the proposed work
within the named boundaries; NO, current tests/docs support neither universal
historical authenticity nor calibrated artificial-origin inference; MAYBE, historic
archive completeness, withdrawn utility scientific validity and full menu/source
coverage pending the named evidence. This bounded review is delivered, not the
completion of programme #2. Independent scientific/security review remains pending.
