# Scientific reliability and KY26 work programme

Status: active. No scientific origin conclusion or release readiness is claimed.

Authoritative issue index: [Programme: Scientific reliability and hypothesis-neutral 1998 KY26 investigation](https://github.com/RobLe3/aneos-suite/issues/2).

The issue bodies preserve acceptance checklists, dependencies, scientific safeguards and evidence requirements. All are open; local implementation is not remote completion.

| Order | Workstream | Tracking |
|---|---|---|
| 1 | P0: Freeze the 1998 KY26 historical baseline and code/data provenance | [#3](https://github.com/RobLe3/aneos-suite/issues/3) |
| 2 | P0: Correct SBDB model_pars ingestion and validate provider schemas | [#4](https://github.com/RobLe3/aneos-suite/issues/4) |
| 3 | P0: Propagate A1/A2/A3 uncertainty, covariance and provenance end to end | [#5](https://github.com/RobLe3/aneos-suite/issues/5) |
| 4 | P0: Separate ranking scores, calibrated significance, probability and FDR | [#6](https://github.com/RobLe3/aneos-suite/issues/6) |
| 5 | P1: Build the versioned 1998 KY26 source and scientific case register | [#7](https://github.com/RobLe3/aneos-suite/issues/7) |
| 6 | P1: Implement observational likelihood fitting and valid Delta-BIC | [#8](https://github.com/RobLe3/aneos-suite/issues/8) |
| 7 | P1: Compare natural, artifact and Phobos 1 hypotheses without origin bias | [#9](https://github.com/RobLe3/aneos-suite/issues/9) |
| 8 | P1: Develop conditional population calibration and dependency-aware evidence fusion | [#10](https://github.com/RobLe3/aneos-suite/issues/10) |
| 9 | P2: Validate deployed policy on independent difficult control populations | [#11](https://github.com/RobLe3/aneos-suite/issues/11) |
| 10 | P2: Integrate uncertainty-aware Rich/REST/JSON/Markdown scientific reports | [#12](https://github.com/RobLe3/aneos-suite/issues/12) |
| 11 | P2: Add append-only research case lifecycle and reassessment history | [#13](https://github.com/RobLe3/aneos-suite/issues/13) |
| 12 | P3: Freeze falsifiable Hayabusa2 2031 prediction register | [#14](https://github.com/RobLe3/aneos-suite/issues/14) |
| 13 | P1: Make analysis jobs and results durable across workers, restart and outage | [#15](https://github.com/RobLe3/aneos-suite/issues/15) |
| 14 | P1: Enforce provider-safe acquisition and production configuration contracts | [#16](https://github.com/RobLe3/aneos-suite/issues/16) |
| 15 | P1: Reconcile ADR/DDD/BC, CI gates and programme-wide verification | [#17](https://github.com/RobLe3/aneos-suite/issues/17) |

## Delivery rules

Freeze the historical baseline before dependent scientific changes. Ship small reviewable PRs in dependency order. No automatic merging. Keep the full programme intact when an individual step remains blocked by missing observations.

Do not tune KY26 scores toward any origin, treat human technology as extraterrestrial evidence, equate smaller residuals with proof, fabricate unavailable values/probabilities/intervals, double-count correlated features, replace history, or weaken tests/CI gates. Preserve current functions unless a documented defect justifies controlled deprecation; retain planetary-defense coverage.

Each increment must inspect current implementation/issues; identify defects/limits; define minimal change; add a failing regression; implement; run targeted/full relevant tests; review numerical/scientific correctness; update ADR/DDD where warranted; link issue/PR/test evidence; report before dependent work.

## Scientific case and completion boundary

The case must contain (1) sources/observations, (2) physical measurements, (3) orbital/acceleration history, (4) hypothesis definitions, (5) fit comparisons, (6) sensitivity/uncertainty, (7) supporting evidence, (8) contradicting evidence, (9) unresolved questions and (10) a reproducible assessment.

H0 gravity; H1 radiation/Yarkovsky/thermal/spin; H2 outgassing/dark-comet; H3 specific Phobos 1 human-spacecraft model; H4 measurement/orbit-fit artifact. Compatible observations, covariance-aware fitted likelihoods, complexity penalties and withheld predictive checks are required for comparative claims. Without them return insufficient evidence, not a preferred origin.

Permitted conclusions include natural consistency, unusual but physically explainable, insufficient evidence, strong support for a specified model and independent confirmation required. A scientifically credible negative result is useful. Unusualness alone never establishes engineering.

## Final report contract

- Engineering: defects/root causes, actual changes, passed/failed tests, remaining limitations.
- Science: historical replay, updated ranking, A1/A2/A3 evidence, physical fits, uncertainty/calibration, Phobos hypothesis and natural alternatives.
- Architecture: changed files, ADR/DDD/BC changes and issues/PRs created/updated/closed.
- Remaining work: defects, missing observations, independent validation and next priorities.
- Reproducibility: commits, data versions/timestamps, frozen hashes, commands and artifact paths.

## Current measured state

Baseline commit/main match: `42cab0acd7a1e5e7781a26831f07b11a77f48ead`. Rounded KY26 replay: score 4.141288008229254, not exactly the supplied 4.142. Full-precision historical provenance remains missing. See [baseline](../case-studies/1998-KY26/baseline.md).

The first engineering tranche had 393 passing tests and 2 exclusions in a disposable copy. The subsequent uncommitted service-contract migration initially had 26 failures/367 passes; 16 new targeted shared-service tests passed. These runs are different snapshots and must not be pooled into a green release claim. No scientific validation has been completed.

The baseline PR deliberately contains no production scoring, API, menu or planetary-defense changes. Existing broad WIP remains separate until tested and divided into reviewable increments.

## Review increments — 2026-11-12

- Draft PR #18 freezes the supplied KY26 historical output and rounded-input
  replay. Original full-precision execution provenance remains missing; #3 is
  not complete. GitHub test and Docker checks passed.
- Draft PR #19 repairs typed non-gravitational ingestion, cache/provenance and
  descriptive reporting, with ADR-064. GitHub test and Docker checks passed.
  Covariance propagation and physical-model fitting are not implemented.
- The screening-result-contract increment (ADR-065, stacked on #19) withdraws
  unsupported inference and integrates a shared ranking service across the
  specified adapters. See `screening-result-contract.md` for its exact coverage,
  controlled test/contract migration and deliberately unfinished surfaces.

All increments require review; none is merged or a scientific certification.
Independent calibration and production-readiness acceptance remain unmet.
