# Unqualified Monte Carlo work checkpoint

Paused at the user's request to conserve tokens. Base: c080f8ca95c12f86fd0f9889e65e2a93f55164b8
(PR95, CI43485017915 passed 2,237 tests and Docker). This branch is NOT the qualified
main milestone and must not be merged or advertised as working scientific validation.

Confirmed defects: retained Monte Carlo code reads unavailable is_artificial as if
it were a negative; processing failures can become False; generator loop overwrites
its integer index with inclination and returns an empty population after formatting
fails; global RNG state changes; unsupported five-sigma recommendations remain.
Four initial regressions failed; after the current partial edits those four pass.
No full suite, complete new-contract coverage or current-head CI has qualified this WIP.

Partial changes: ADR098/DDD proposal, shared policy_evaluation orbital-scenario
mechanics/generator, two legacy facades, exclusive experiment export. Still required:
complete adversarial/parallel/export/replay tests; validate report shapes and provenance
limits; review memory/process budgets and generator assumptions; verify all legacy
contracts/callers; user documentation, CI gate and full relevant tests. No independently
curated real-world control set, calibrated model or source authenticity is supplied.

Resume only after explicit user instruction. Start from issue11/23/6, inspect this
branch and the current main before continuing. Keep original archived results and
the user's dirty original checkout untouched. Do not call four tests scientific
validation or use this unfinished work to alter ranking thresholds.
