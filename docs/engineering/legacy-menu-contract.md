# Primary legacy menu, launcher and report contract correction

Date: 2026-11-12. Branch: `codex/legacy-menu-contract`.
Base: `936c64f66dd2752d383047d0a7cb4cbb4cdd76d4` (draft #29).
Programme #2; analysis/reporting #6/#12; preservation/durability #13/#15;
provider/verification #16/#17; remaining standalone analyzer #30. ADR-071.

## A. Engineering summary

Confirmed defects: Rich/plain single/batch differed; names selected demo orbits
and invented course history; null artificial-origin status became NATURAL; null
sigma/p/FDR crashed rendering. Predefined workflows and statistical helpers
manufactured scores, probabilities, comparisons, risk/readiness and publication
claims. Launcher analysis interpolated designation text into executable Python
and preferred a separate analyzer. The raw physical adapter returned .37 km as
`diameter`, while the detector contract uses metres (370 m). Menu startup silently
deleted cache JSON and altered environment/defaults, unrelated to an analysis request.

Corrections:

- Primary legacy single/batch/interactive, spectral/dynamics/cross-reference/
  statistical/advanced suite and predefined comprehensive/rapid/deep/publication/
  comparative routes now use actual AnalysisPipeline and shared screening. Every
  ordered batch outcome remains, including failures and duplicates. No hidden demo
  or name-only bypass; missing input cannot become a natural/zero result.
- Neutral Rich/plain presentation replays canonical inputs and drops unsupported
  claim aliases. Candidate positive controls remain visible, not forced natural.
  Raw proxy rank, uncertainty, significance, origin and dependency groups remain
  distinct. No high-rank artificial confirmation or publication-readiness rule.
- Retain compatibility names. Withdraw defective probability/sigma conversion,
  sigma filters, fixed workflow outputs and prototype statistical/cross-validation
  helpers using explicit NotImplementedError. Threshold/filter names cannot be
  silently reinterpreted as ranking thresholds. Real scientific core libraries
  and planetary-defense modules are unchanged, not newly certified by this work.
- Preserve actual computed Horizons history separately. Empty/failed history does
  not create a fictional observed epoch-zero record; computed osculating elements
  are not manoeuvre evidence. Coverage/raw-response archival remains unverified;
  the sidecar is not an observational fitting dataset.
- Primary launcher `analyze/analysis/a/simple/detect` aliases call the real service
  directly, not subprocess-interpolated code or SimpleNEOAnalyzer. Integrate and
  document `--json`, `--force-refresh`, `--output`; defaults are Markdown and source
  cache use. Sources/report failures return nonzero. Output creates an exclusive
  new file; it cannot normally overwrite a historical report. It is not atomic,
  crash-safe storage or protection from later editing/deletion (#15).
- `scientific-screening-report-v1` owns a strict-JSON snapshot: canonical assessment,
  full normalized NEO input, metadata, review-input ID and report hash. Verify
  current-policy replay, raw-input binding and full report checksum. Native A3
  sigma/units/status and covariance stay intact. Hashes are not authentication.
- Include optional Git revision/dirty status, selected package versions, float
  settings and key source-file hashes at report time. These do not prove loaded
  bytecode, freeze dirty source or constitute a full environment/dependency lock.
- Entering the legacy menu no longer initializes DB, cleans caches or changes
  environment. Explicit maintenance functions remain separate; this change does
  not certify all later cleanup/delete operations. Headers no longer equate module
  imports with scientifically validated readiness. Diagnostics are explicitly
  synthetic workflow controls, not measured detection performance or saved reports.

## Verification and controlled test migration

`tests/test_legacy_menu_contract.py`: 46 new external-network-denied controls.
Initial 27 cases failed before implementation. Controls exercise actual initialized
core/service behavior, Rich/plain entrypoint parity, candidate positives, source
failures, duplicates, alias non-bypass, computed history, native A3 sigma/covariance,
report replay/hash/raw binding, file non-overwrite, CLI flags/exit codes and a
malicious quoted designation passed as data rather than executed Python. Diagnostics
retain labeled synthetic positive/negative/missing controls.

One existing test required known names to return hard-coded orbital data without
fetching. That contract contradicts the source-integrity requirement. It is replaced
with a stronger assertion that the source seam is called and missing source evidence
raises even for a known alias. The physical-property fixture now uses actual typed
NEOData/OrbitalElements/PhysicalProperties instead of permissive MagicMocks. Its
unit assertion is corrected from .37 to 370 with an explicit metres marker; albedo
and rotation preservation assertions remain. The unknown-designation and live-
source-seam tests are retained. No tests were skipped, disabled or xfailed; no
scientific/CI acceptance gate was lowered.

An initial full run exposed obsolete test mocks that permitted acquisition attempts
for unknown/name/99942 inputs. The seam/fixtures were corrected. No retrieved data
was adopted as scientific validation or substituted into the frozen KY26 baseline.
New controls deny external sockets; the full suite is selected by `-m 'not network'`,
not presented as a proof that every historic test is network-independent.

Final suite: **691 passed, four network tests deselected, five warnings; 40.65%
coverage**, unchanged 30% minimum, 29.33 seconds. `git diff --check` is clean.

## B. Scientific assessment

No updated calibrated KY26 significance/probability, observational fit, valid BIC
ranking, Phobos 1 finding or natural-origin conclusion is supplied. Frozen historical
results and scoring thresholds are unchanged. Correct input units are not threshold
tuning. Native acceleration transport/description is not force-model evidence or
covariance epoch propagation. Missing values remain missing; five-sigma permission
is false. Independent performance and uncertainty-aware physical comparisons remain
#7–#11/#25, not inferred from UI/control tests.

## C. Architecture impact

Production changes: `aneos.py`, `aneos_menu.py`, new
`aneos_core/analysis/reporting.py` and `aneos_core/analysis/cli.py`. Tests above and
the controlled `test_menu_v2.py` migration. ADR-071/DDD, README, CLI/API/scientific
notices describe implemented versus unavailable capability. No new engine, UI or
bounded context. Numerous prototype method bodies are retired, while their names
and unrelated maintenance/menu functionality remain; prior code remains in Git.

## D. Remaining work

The standalone `simple_neo_analyzer.py` still generates hash/name-based orbits and
probabilities and invokes mock impact/validation paths (#30). Its library/script
API is not fixed by changing the primary launcher aliases. Wider legacy admin,
reanalysis, display/export, pattern/network and standalone-model claims remain
#6/#12/#17; this is not a whole-app readiness certification. New report CLI support
is not REST Markdown export, append-only case storage or recoverable jobs (#12/#13/#15).
Provider-wide limits, configuration isolation, calibration/independent evaluation
and actual physical fitting remain unfinished. No independent reviewer certified
this coding-agent change.

## E. Reproduction

Base/branch above; the draft PR records final commit and CI. Existing frozen KY26
fixture/baseline references and hashes remain unchanged; new test inputs are
synthetic controls, not observed astronomy. Runtime reports include their own
current source/input hashes and explain the limits of those fingerprints.

```sh
PYTHONDONTWRITEBYTECODE=1 \
ANEOS_DATABASE_URL=sqlite:////tmp/aneos-menu-review.db \
/Users/roble/.venvs/aneos-suite/bin/python -m pytest tests/ aneos_core/tests/ \
  -m 'not network' -q --tb=short --cov=aneos_core \
  --cov-report=term-missing --cov-fail-under=30
```

Work was isolated in a separate Git worktree. The dirty original checkout/database
were not overwritten. Tests use disposable DB paths. No archival source response
or existing analysis file was deleted as part of this correction.
