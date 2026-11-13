# aNEOS documentation

Start here to install the engineering beta, investigate an object and understand
what the result does—and does not—support. Application identity: **2.2.0b2**.
The published tag and the development checkout have different capabilities.

## Start using the suite

1. [Installation](user-guide/installation.md): isolated Python environment and database.
2. [Quick start](user-guide/quick-start.md): query, save, inspect and interpret a result.
3. [Menu guide](user-guide/menu-system.md): actual option labels and prerequisites.
4. [Beta release guide](releases/2.2.0b2.md): workflows supported by the published tag.

The development-only [root CAD workflow](engineering/root-cad-workflow.md) is not
in v2.2.0b2. Follow its explicit request and enrichment budgets when testing that
checkout. Other root bulk discovery routes remain unavailable.

## Interpret evidence and results

| Question | Scoped contract reference |
|---|---|
| What does the ranking mean? | [Screening result semantics](engineering/screening-result-contract.md) |
| Which application paths share analysis? | [Core analysis](engineering/core-analysis-contract.md) and [legacy adapters](engineering/legacy-menu-contract.md) |
| What was queried and what is missing? | [Acquisition receipts and coverage](engineering/object-acquisition-ledger.md) |
| Do value and uncertainty belong together? | [Atomic physical evidence](engineering/atomic-physical-evidence.md) |
| How are A1/A2/A3 represented? | [Non-gravitational evidence](engineering/nongrav-evidence-contract.md) |
| What do dates and epochs mean? | [Object chronology](engineering/object-chronology.md) and [native orbital epochs](engineering/native-orbital-epochs.md) |
| How is local history retained? | [Evidence ledger](engineering/durable-enrichment-ledger.md) |
| Can I replay a saved report? | [Dashboard/report loading](engineering/polling-dashboard-screening.md) |
| What impact claims are supported? | [External Sentry evidence](engineering/sentry-impact-contract.md) and [entry-point limits](engineering/impact-entrypoint-contract.md) |
| Which statistical claims are withheld? | [Validation integrity](scientific/VALIDATION_INTEGRITY.md) |

Some engineering notes retain their original branch/version labels. They describe
scoped contracts, not the installation status of every checkout; use the release
guide and actual installed behavior to determine availability.

No current score alone establishes origin, calibrated discovery significance or
impact risk. Completeness percentages need an explicit denominator; field presence
is not scientific fitness. See the [README](../README.md#understand-a-result) for
a plain-language interpretation of candidate, not-flagged and inconclusive results.

## API and configuration

Start the local API with an explicit loopback host as described in the
[quick start](user-guide/quick-start.md). Its running `/docs` and `/openapi.json`
describe the installed routes. They are not guarantees of authentication or science
readiness. The checked-in [REST reference](api/rest-api.md) and
[OpenAPI snapshot](api/openapi.json) may lag the running application; verify the
route before integrating it. Do not infer deployment readiness from a schema.

- [Configuration reference](reference/configuration-reference.md)
- [Policy inspection and controlled configuration](engineering/truthful-policy-menu.md)
- [Primary-menu durable batch jobs](engineering/durable-batch-execution.md)
- [Owned durable REST batches](engineering/durable-batch-api.md)
- [Statistical utility inputs and limits](engineering/statistical-utility-contract.md)
- [Durable job storage](engineering/durable-job-store.md) and
  [resumable network worker](engineering/resumable-network-worker.md): scoped families,
  not proof that every legacy task survives failure.

## Research and architecture

- [1998 KY26 historical baseline](case-studies/1998-KY26/baseline.md)
- [Programme #2](https://github.com/RobLe3/aneos-suite/issues/2)
- [Current roadmap](../ROADMAP.md) and
  [discovery/integrity review](engineering/discovery-integrity-roadmap-review-2026-11-13.md)
- [Architecture decisions](architecture/ADR.md) and [DDD](architecture/DDD.md)

Architecture documents include historical designs and future contracts. Read their
status notes before treating a design as implemented or independently validated.
The roadmap keeps exploratory anomalies, data integrity, follow-up ranking and
physical-origin inference separate.

## Historical and not-yet-reconciled pages

The older [scientific methodology](scientific/scientific-documentation.md),
[theory](scientific/theory.md), [full user reference](user-guide/user-guide.md),
[profession guide](user-guide/profession-guide.md),
[deployment guide](deployment/deployment-guide.md) and
[current-state summary](current-state-summary.md) have **not** completed the current
claim-by-claim review. Their probability, sigma, maturity, route and deployment
claims are not current validation evidence. Prefer the scoped references and release guide above.
These pages remain available for history and the continuing documentation audit.

The [entry-guide archive](archive/entry-guides-2026-11-13/README.md) preserves the
superseded README, index and introductory guides. The
[review record](engineering/entry-documentation-review-2026-11-13.md) states the
checked scope, example results and remaining gaps; it does not certify every page.

## Contribute or report a problem

Follow [CONTRIBUTING.md](../CONTRIBUTING.md). Include the version/commit, exact
command or menu option, expected/actual behavior and source availability. Redact
credentials and private observations. Link changed examples to executable tests;
label synthetic controls and network-dependent steps explicitly.
