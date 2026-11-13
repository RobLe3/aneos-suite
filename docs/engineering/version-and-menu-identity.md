# Application identity and menu-entry QA

2026-11-13. Issues45/41/44/46, programme2. This is a development/review change,
not completion of the requested whole-menu/data/scientific/documentation programme.

## A. Engineering summary

The repository exposed conflicting application labels: README/core1.2.2,
API package1.0.0, dashboard1.0.0, main menu “v2”, REST contract2.0.0 and health
fallback0.7.0. Local and GitHub tag inventories containv2.1.6 andv1.2.0; v2.1.6
points to f1f6f31 (2025-08-11). No GitHub release/tag is created by this change.

`aneos_version.py` is the dependency-free application authority: **2.2.0.dev0,
development**. This advances beyond the known existing2.1.6 tag without claiming a
stable/scientifically validated release. Core/API/dashboard package exports, active
defaults, launcher and both menu headers, application HTTP metadata and newly built
reproduction reports use it. `python aneos.py --version` works outside the repository
without importing science/API/storage runtimes or starting services.

The REST contract stays2.0.0. Root/health compatibility `version` and X-API-Version
remain contract identifiers; explicit application_version/X-Application-Version and
OpenAPI x-application-version identify the application build. Independent model,
schema, provider and nested component versions are not relabelled. Dashboard default
metadata uses application identity; explicit custom dashboard configuration remains
operator-controlled. This repository currently has no pyproject/setup package build
metadata to update; no pip-installed-distribution version is invented.

README's active badge and CHANGELOG identify the unreleased review build. Historical
changelog/case/source versions remain intact. Human reports display their **recorded**
software version, not the current constant substituted into an old assessment.
The custom OpenAPI metadata now uses the actual app description rather than an old
artificial-detection claim, and removes the placeholder example.com logo.

## Menu entry and help

A frozen MenuOption catalogue binds the fifteen primary feature IDs plus exit to
labels, purpose/basis and real handler names. Entry display and actual run-loop
routing use the same catalogue. It does not install scientific prerequisites or
assert that all handlers/results are validated. Legacy “121-option full coverage”
and live network-sigma/threshold assertions are removed from the primary entry.

The entry uses responsive columns, clear groups, literal Rich Text labels, no
emoji-only meaning and readable un-dimmed explanatory text. Plain output has no
Rich markup leakage. Existing feature IDs/navigation are preserved. Scientific Help
now renders the current `docs/scientific/screening-guide.md`, not keyword-selected
prototype text containing unsupported causal/significance claims. The prototype
file remains accessible with a historical warning. Current help explains measured/
fitted/proxy evidence, NOT_FLAGGED versus normal, missing versus zero, completeness
versus readiness, correlation prerequisites and unavailable origin significance.

No provider data, fitted model, ranking threshold or historical KY26 result is
changed. User-facing per-object insights/graphs and full detailed views remain42;
new metadata/readability is not scientific calibration.

## B. Actual QA evidence and limits

Artifacts: [menu-entry/2.2.0.dev0](../qa/menu-entry/2.2.0.dev0/).
Each before/after width uses the same installed Rich renderer; the preceding source
is frozen at1ae3fe2da299331843620041eae7100fe7fc4285. Actual text, HTML and SVG
exports and hashes accompany the inventory. These are Rich-generated terminal
renderings, not desktop screenshots or proof of an interactive scientific result.
Font declarations requiring a CDN are removed from the export template; local
monospace fallback keeps the artifacts usable without a font service. Raster views
were inspected locally using rsvg-convert; no new dependency is added to the app.

| Width | Before lines / max columns | After lines / max columns |
| --- | --- | --- |
|80|33 /70|35 /80|
|120|33 /70|30 /120|
|160|33 /70|30 /160|

All sixteen choice IDs appear exactly once at each tested width. Actual loop tests
intercept each of fifteen feature handlers to prove navigation/dispatch, not provider
execution. Real current-help rich/plain captures are included at each width; the
legacy header is captured separately. The narrative/basis column was made brighter
after visual inspection of a low-contrast iteration. This is a design judgment,
not independent user-study evidence or a universal terminal/font guarantee.

The inventory explicitly leaves each feature's prerequisite, full behavior and
result QA **pending**. Static discovery finds23 legacy candidate method names and
158 literal option tuples, not158 verified reachable features; duplicated/dead/
nested choices and runtime reachability still need review. Do not substitute the
old121 claim or these static counts for a checked feature denominator.

First capture exposed Rich's record buffer being cleared before HTML export;
the generator now preserves it, and a regression requires actual HTML/SVG/text
content and hashes. It refuses to overwrite an existing capture directory. Earlier
failed/low-contrast captures were moved to separate temporary locations, not used
as accepted artifacts. The old source-string test for option15 is migrated to an
actual catalogue/loop probe; no existing feature test is disabled.

## C. Architecture impact

No new bounded context or scientific engine. Dependency-free application identity
is separate from API/analysis versions; the menu catalogue is existing presentation
navigation responsibility. The source/dispatch contract and QA captures expose
coverage, not scientific approval. No ADR proliferation is needed for this metadata/
UI correction. DDD receives the identity distinction; substantial data/behavior
boundaries remain subject to the existing ADR/DDD programme and #40–#47.

## D. Remaining work

- Every actual menu/submenu/fallback: prerequisite enforcement, positive/invalid/
  missing/denied/outage/stale cases, side effects and actual result/export values.
  Captured entry labels and intercepted dispatch are not that proof (#41).
- Full detailed object explanations/graphics42; source-by-field normalized local
  enrichment/identity/conflict ledger40 and atomic physical uncertainty25.
- Scientific operation readiness43, observational fits/calibration7–11; no version,
  number of rows, attractive plot or test count grants five-sigma/origin inference.
- User docs/examples44 and executable quality scoreboard46. The historical
  scientific-documentation archive is warned, not fully scientifically rewritten.
- Known menu follow-ups include the proximity scan's same-sign A2 “correlation”
  wording, feature-specific coverage/no-result claims, generic fallback formatting
  and legacy prerequisite declarations. Review actual implementations before
  assigning dispositions or changing scientific contracts.
- No merge/tag/release, root WIP reset, live source query, unrelated user-data read
  or research database restore is part of this change. Live root DB byte integrity
  remains unverified after prior independent metadata changes; do not claim it from
  these isolated tests. Preserve existing research and dated source artifacts.

## E. Reproduction

Use a new output directory; existing captures are not overwritten:

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/qa_menu_entry.py \
  --output /tmp/new-menu-entry-capture \
  --baseline-ref 1ae3fe2da299331843620041eae7100fe7fc4285
python aneos.py --version
python -m pytest tests/test_application_identity.py -q --tb=short
```

The QA command requires its explicit baseline Git object locally. CI checks out
full history for this comparison; tests do not fetch missing history themselves.
The new tests use disposable databases and the existing offline network guard;
no provider access is required. Optional local PNG inspection:
`rsvg-convert after-120.svg -o after-120.png` (not an app/CI prerequisite).
The exact verified commands and final results follow below/in the draft PR.

```sh
PYTHONDONTWRITEBYTECODE=1 ANEOS_DATABASE_URL=sqlite:////tmp/aneos-identity-final1.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core --cov-report=term-missing --cov-fail-under=30

PYTHONDONTWRITEBYTECODE=1 ANEOS_DATABASE_URL=sqlite:////tmp/aneos-identity-component.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/test_application_identity.py -q --tb=short
```

Final full suite: **1,057 passed, five network deselected, five warnings in59.87s;
45.41% core coverage against unchanged30% gate**. Targeted identity/render suite:
**30 passed**, 6.39s. Existing storage85/report90/worker90/API85 gates remain;
CI adds the actual identity/render test command. The initial red suite had11 failures
before the identity contract existed. One later source-string route assertion was
replaced by a real navigation probe rather than restoring duplicated dispatch maps.
Capture tests check meaningful HTML content, no remote font declarations, hashes
and refusal to overwrite. No scientific tests or thresholds were relaxed.

No provider observations or new scientific conclusions were produced. Remote CI,
commit and draft PR are recorded separately; local success is not remote green.
