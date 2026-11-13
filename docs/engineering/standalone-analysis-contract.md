# Standalone/simple/advanced analysis contract correction

Date: 2026-11-12. Branch: `codex/standalone-analysis-contract`.
Base: `08b1e8b34a01bf8cce6b56a45760268ca3ef400d` (draft #31).
Programme #2; analysis/reporting #6/#12; standalone #30; impact follow-up #32;
full verification #17. Decision: ADR-072.

## A. Engineering summary

Confirmed defects:

- SimpleNEOAnalyzer selected prototype detector fallbacks; `test` returned .234,
  other names/designation patterns generated fixed probabilities, and Python's
  name hash generated orbital elements. Errors could become zero/natural results.
- Assumed priors/ratios and a probability cap were described as calibrated;
  mock impact orbits/approaches/arcs produced calculated/negligible-risk language.
- Advanced analysis misread DataFetcher's NEOData return as a success/neo_data
  wrapper and generated mock data on errors. It used a second heuristic score,
  invented enrichment/temporal inputs and promoted origin scores to threat levels.
- Capability imports/counts advertised advanced maturity, coverage and statistical
  validation without a validated model. Standalone CLI defaulted to the test demo
  and formatted unavailable scientific outputs as numeric probability/significance.

Changes:

- Simple/advanced library analysis, aliases and actual script commands now use the
  typed shared core and replayable report. There is no source-error/SDK/core-failure
  fallback detector, artificial probability, generic orbit or natural-origin verdict.
  Actual input records, A3 sigma/units/model status and covariance survive transport.
- Synchronous methods retain their names; async counterparts support an active event
  loop and refresh without nesting asyncio.run or accidentally creating an unawaited
  coroutine. The script keeps `single/analyze/status` commands; analysis requires an
  explicit designation and shares CLI flags/exits, including source failures.
- Advanced caller-provided mode is not mixed with remote defaults. Zero-valued
  elements are retained, missing fields remain missing, units/aliases are checked,
  rotation must be finite positive hours, and supplied epochs must be explicitly
  timezone-aware dates rather than guessed/native-JD conversions. Caller records
  retain their unverified role; invalid/nonfinite/non-JSON records are rejected
  rather than serialized as valid observations or silently normalized to zero.
- Per-call source filtering was not implemented. It now raises NotImplementedError
  rather than silently claiming the requested providers. This does not implement
  provider-wide configuration/cache/rate-limit isolation (#16).
- Probability scalars, name/hash/mock inputs and defective prototype inference/
  enrichment retain function names but raise controlled-deprecation errors. The
  calibration-status helper returns not-calibrated/null without assumed priors.
  Actual data getters have explicit source/type/identity failures, not mock records.
- Unified result/report confidence, origin probability, sigma, quality and risk/threat
  are explicitly nullable/unassessed. Compatibility keys are present as null instead
  of allowing a caller's `.get(key, 0)` to fabricate certainty. A scalar advanced
  anomaly-score accessor is shared raw ranking units, not bounded probability.
- Capability inventory reports implemented candidate screening and importable typed
  acquisition separately from provider health or scientific validation. Statistical
  validation/artificial detection remain unavailable; feature counts/installation
  cannot create maturity or coverage probabilities.
- Impact bridges call an actual-input prerequisite inventory, not the unfitted risk
  calculator. Presence of catalogue covariance or a dated arc is not propagated
  impact uncertainty, fitted residuals or an encounter model. Missing prerequisites
  yield null probability and `not_assessed`, never negligible/zero risk. Core impact/
  scenario research modules are not deleted or newly validated; their broader
  REST/menu/core contract is a separate confirmed finding (#32).

## Verification

`tests/test_standalone_analysis_contract.py`: **41 new external-network-denied
controls**. Initial 18 failed before correction. These exercise actual typed core/
shared-service candidate positives and unflagged controls; all library aliases;
known names/test with failed sources; missing core/type/identity; explicit scalar/
mock deprecation; raw-input non-supplementation and falsey-invalid containers;
finite units/rotation/epoch checks; async/refresh and nested-loop rejection;
non-mutating enrichment; A3 sigma/covariance preservation and report replay;
actual dated arc without quality/probability; honest capability counts; and actual
standalone subprocesses with provider sockets denied and injected synthetic source
records. Synthetic controls are not observed objects or independent model validation.

An intermediate strict-JSON change exposed invalid raw metadata leaking NaN/Infinity;
the regression now rejects that input with a structured invalid-evidence outcome.
It does not drop values into a valid fitted model. No existing tests were modified,
skipped, xfailed or disabled; no quality gate was reduced.

Final full suite selected by `-m 'not network'`: **732 passed, four deselected,
five warnings; 41.98% coverage** against the unchanged 30% gate, 33.52 seconds.
`git diff --check` is clean. No newly retrieved astronomical data was used as a
validation set, source fixture or replacement for historical KY26 inputs.

## B. Scientific assessment

This is evidence-transport/interpretation repair, not updated KY26 significance,
posterior, physical-origin evidence, Phobos 1 test or natural-model fit. Historical
baseline fixtures/thresholds are unchanged and included in the regression suite.
A3/covariance transport is not force identification or covariance epoch propagation.
No independent sensitivity/FPR, calibrated origin probability or five-sigma claim
is established. Publication/scientific review remains necessary.

For the separate impact follow-up, [JPL's Sentry API documentation](https://ssd-api.jpl.nasa.gov/doc/sentry.html)
was checked on 2026-11-12. It distinguishes method-specific impact results and
warns that even its estimated probabilities can differ by several-fold, sometimes
an order of magnitude or more. Removed/not-found states do not provide a numeric
probability. A future authoritative adapter must preserve those limits beside its
output, not replace local fit uncertainty with an unqualified precise number.
No specific object's Sentry result was fetched or inferred here.

## C. Architecture impact

Production: `simple_neo_analyzer.py`, `analysis/advanced_aneos_analyzer.py`,
`interfaces/unified_analysis.py`, `analysis/reporting.py` and new
`analysis/impact_evidence.py` under `aneos_core` except the standalone script.
New tests above; README, ADR-072, DDD, API/scientific notices updated. No new
scientific engine, UI, REST route or bounded context. Large deletions are retired
prototype bodies; method names are retained and core scientific libraries unchanged.

## D. Remaining work

Issue #32: public impact REST/menu/core probability/confidence claims still use
heuristic/default assumptions and need their own observational/scenario contract
and regression review. This facade guard is not evidence that those surfaces are
safe or validated. Wider legacy admin/reanalysis/pattern/network/export claims remain
#6/#12/#17. Actual measurements/uncertainties, dated KY26 source case, astrometric
fits and independent conditional-population calibration remain #25/#7–#11.
Provider controls #16, durable jobs/cases #15/#13 and REST scientific Markdown
integration #12 remain unfinished. Coding-agent self-review is not independent
scientific certification; no issue is complete merely because a branch is green.

## E. Reproduction

Base/branch above; final commit and CI are recorded on the draft PR. Existing frozen
KY26 data references/hashes are unchanged. New inputs are labeled synthetic controls.
Reports retain source/review/policy/report IDs and runtime source hashes with their
existing limits; they are not authentication or a full environment lock.

```sh
PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-standalone-review.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core \
  --cov-report=term-missing --cov-fail-under=30
```

Implementation/testing used an isolated Git worktree and disposable DBs. The dirty
original checkout/database were not overwritten; no archival analysis was deleted.
