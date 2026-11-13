# Entry-documentation reconciliation, 2026-11-13

Programme #2; documentation #44; presentation #12/#41/#42; verification #17/#46.
Source review base: `00932e7004b5e903788ded285716cea6ec1403a0`, the local integration
of #85/#88/#89/#90. Those changes remain separate from the published v2.2.0b2 tag.
This document records a bounded documentation repair, not programme completion.

## Scope and decision

The base contained **99 tracked Markdown files** (`git ls-files '*.md'`). This
increment rewrites five entry documents: root README, documentation index,
installation, quick start and menu guide. The remaining **94 original Markdown
files are not claimed reconciled** by this pass. Linked contracts were consulted
as evidence, not thereby certified as fully current. New archive/review documents
are additions, not extra baseline coverage. Launcher help is also corrected.

Audience: a researcher installing a separate beta, querying an object, inspecting
provenance and interpreting results without mistaking a rank for discovery or origin.
Acceptance was fixed before tests: current executable routes, explicit network and
write effects, frozen historical preservation, honest unavailable outputs, no new
inference or release-readiness claims. Mutations are authorized by the user's
explicit documentation/remediation request; research databases are out of scope.

## Confirmed contradictions and repair

The old README's opening caveat was contradicted by later “working” Bayesian
calibration, six-independent-stream/Fisher claims, Gaussian-sigma examples and
“ARTIFICIAL VALIDATED” results. It also mixed v1.2.2, 360-test and current beta
identities. The index and introductory guides promoted those methods as current,
used obsolete menu paths and included unsupported output/timing examples.

The rewrite removes those active claims rather than relying on an opening warning.
It teaches current labels and uncertainties, separates tag/development capability,
uses actual routes and labels network steps. Exact old bytes remain in
[the archive](../archive/entry-guides-2026-11-13/README.md), with paths, counts and
SHA-256 in its manifest. Nothing edits historical scientific fixtures, source
responses, older tags or result artifacts.

## Claim/example to evidence matrix

| Reader-facing statement or action | Source inspected | Executable evidence / limit |
|---|---|---|
| Application 2.2.0b2 differs from API/model/dataset identity | `aneos_version.py`, release guide | Existing `test_application_identity.py`; current `--version` command |
| Named-object JSON/Markdown flags and exclusive output | `aneos.py`, `analysis/cli.py` | New actual CLI-backend tests with explicitly synthetic typed acquisition; both outputs, null inference and failed replacement checked |
| Rank/labels are not sigma, probability or natural ground truth | `analysis/screening.py`, current screening contract | Existing core parity/missingness controls; new report assertions check actual payload, not an invented schema key |
| Primary menu has 16 choices including exit | `aneos_menu_catalog.py`, `aneos_menu_v2.py` | New 16-row exact catalogue match plus existing render/dispatch tests; not exhaustive backend QA |
| Local API example specifies loopback | `aneos.py`, `start_api.py` | Exact dispatch argument test without starting an exposed server; existing API identity/schema tests; not deployment qualification |
| Dashboard file command does not poll and preserves input | File launcher, dashboard loader | Actual subprocess with network/DNS denied; original bytes unchanged and completed report bundle required as positive control |
| Dashboard loading writes exports | File launcher passes `save=True` | Completed exclusive bundle checked in temporary directory; no blanket backup guarantee |
| Legacy KY26 replay is offline and inexact historically | Replay script, fixture, manifest | Actual subprocess, numeric tolerance 1e-12, historical-inexact flag and rejected replacement; baseline manifest tests |
| Five source profile, limited field presence, atomic measurements and history | Fetcher/acquisition, ledger, physical evidence and chronology contracts | Existing source/ledger/covariance tests; source availability or universal scientific completeness not asserted |
| Development-only bounded CAD differs from tagged beta | Root controller/guide and PR inventory | Existing root/CAD/read-only tests; no new source polling in this increment |
| All five original entry documents remain accessible | Archive manifest and source commit | SHA-256, lengths and exact `git show` byte comparison for each of five originals |
| Entry links name existing local targets | Five rewritten entry documents | Local-path existence tests; external destination/anchor correctness needs separate review |
| Isolated Python installation | Core requirements, beta guide, current Python 3.11 test environment | Fresh installation not repeated by this increment; no Windows/hardware/time guarantee |

`analysis/` paths above are relative to `aneos_core/`. Tests using synthetic
measurements demonstrate report plumbing, not provider correctness or observed
KY26 physics. No synthetic fixture is published as an observed new assessment.

## Verification record

New controls: `tests/test_entry_documentation.py`. First run: **16 passed, 3 failed**:
two links pointed to this not-yet-created review and one test assumed a non-existent
payload key. Added the intended review and corrected the test to the inspected
actual contract; no scientific output or quality gate was changed.

Final targeted/full results and exact reviewed head are recorded in the associated
PR/issue evidence after execution, avoiding a self-referential commit identifier.

```bash
python -m pytest tests/test_entry_documentation.py tests/test_application_identity.py \
  tests/test_legacy_menu_contract.py tests/test_ky26_baseline.py -q -o addopts=''
python -m pytest tests/ aneos_core/tests/ -m "not network" \
  --cov=aneos_core --cov-fail-under=30
```

The new CLI/dashboard/replay subprocess controls deny network and DNS and use
temporary paths. The documented live named-object query is **not run against
providers in this documentation increment**; its syntax/backend are exercised with
injected synthetic acquisition. Installation/downloads and live API serving are
not disguised as offline tests. Existing gates remain unchanged.

## Architecture and compatibility

No ADR/DDD/BC change is needed: no new ownership, scientific schema, persistence,
engine, dependency or routing is introduced. Only launcher help text changes at
runtime. The application remains 2.2.0b2 until a separately qualified release is
prepared; rewriting documentation is not a reason to relabel historical results.

The documentation index now distinguishes current contracts from historical or
not-yet-reconciled theory, full user/profession references, deployment and old
current-state pages. These older documents are retained; their individual claims
remain work under #44, not silently accepted by this repair.

## Review disposition and remaining work

Bounded YNM-1/documentation review in portable mode: **NO**, the previous entry
pages accurately represented the current inference contract; source and outputs
contradicted that proposition. **YES**, the specified new route/archive examples
can be checked by the named controls, within their exact scopes. **MAYBE**, all
user documentation and full scientific/menu readiness: implementation/reproduction
and independent user evidence remain missing. No independent reviewer or external
scientific certification was obtained.

Reader-context pass checks whether someone can install, identify network/write
effects, inspect the result and distinguish unavailable inference. It is a maintainer
review, not an astrophysicist usability study. Semantic claim review complements
mechanical link/tests; a clean keyword scan cannot validate science or prose.

Next documentation increments should reconcile the remaining 94 original files by
role, especially the direct-access scientific methodology, profession/user guide,
API reference/snapshot, deployment and old state summary. Execute each retained
example against the right tag/configuration; record failed/not-run cases. Do not
close #44 or claim all menus are qualified from this entry-point repair.
