# Engineering milestone and development pause — 2026-11-13

Development is paused at the user's request to conserve tokens. Resume only after
an explicit request. This is an engineering checkpoint, not completion of the
scientific reliability programme or a new published release.

## Code included in main

The qualified code is `c080f8ca95c12f86fd0f9889e65e2a93f55164b8` plus this documentation
checkpoint. It consolidates PR85 and PR88–95 without the unfinished Monte Carlo work.
[CI43485017915](https://github.com/RobLe3/aneos-suite/actions/runs/43485017915) passed
2,237 tests (9 live-network tests excluded), all component gates and Docker builds.
Integration/main checks must also pass; GitHub check state is authoritative.

Included: bounded CAD acquisition/root integration; provider-safe read-only
preflight; evidence-linked documentation and batch request accounting; restart-safe
primary-menu and owner-scoped REST batches through the existing SQL queue; explicit
statistical-utility prerequisites and unavailable empirical error rates. ADR092–097
and their engineering guides describe the individual scopes and limits. No parallel
scoring/persistence system or new scientific-origin classifier was introduced.

The latest published tag remains **v2.2.0b2**. Main contains newer engineering work;
its application identity still says 2.2.0b2 and is not evidence of tag equivalence.
Record the commit when testing. No new tag/release or scientific certification is
part of this pause. Use an isolated checkout, explicit disposable database and
provider configuration; preserve research data and archived reports.

## Work intentionally excluded

Branch `codex/policy-evaluation`, checkpoint `e2e8bd9`, preserves unqualified Monte
Carlo remediation and its own resume note. Four initial regressions reproduced
incorrect candidate counting, failure-as-negative behavior, empty fabricated rates
and a generator index/formatting error. Those four pass after partial edits, but
full coverage, full-suite CI, adversarial/parallel/replay/export checks and design
review are unfinished. That code and proposed ADR098 are **not** part of main.
The retained main Monte Carlo harnesses must not support scientific FPR/five-sigma
claims; the frozen canonical screening contract remains separate and uncalibrated.

## Remaining horizons

- **Short term:** review outstanding issue acceptance against integrated evidence;
  finish legacy Monte Carlo remediation, remaining menu/feature QA and operational
  qualification. Other API/export/automatic job families and backup/recovery are
  not made durable merely by the batch work. No parent issue is closed by this note.
- **Medium term:** #86 qualified diagnostics, #10 conditional/dependency-aware
  ranking, #87 compatible historical comparisons, #7 KY26 chronology, #8/#9 actual
  observational fits and #11 independent controls. Integration does not supply
  missing measurements or prove historical authenticity.
- **Long term:** independently evaluated prediction targets/calibration, search-wide
  significance where supportable, witnessed archives, longitudinal reassessment and
  frozen future predictions (#13/#14). No completion date is substantiated.

The programme remains incomplete. Missing data stays missing; unknown origin is
not natural truth. Synthetic mechanics, workflow passes and green software CI do
not establish independent scientific performance or a five-sigma discovery.

## Resume checklist

1. Inspect current main, this checkpoint, programme #2 and actual open issues/PRs.
2. Inspect the separate WIP branch before any further implementation; do not merge
   it on the basis of four tests or reuse obsolete Monte Carlo claims.
3. Use a fresh isolated worktree/database. The original user checkout contains
   independent uncommitted work and must not be reset or cleaned.
4. Continue small changes with positive/negative/regression controls and unchanged
   CI gates. Preserve historical results and distinguish implemented from validated.
