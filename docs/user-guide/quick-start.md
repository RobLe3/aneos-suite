# Query, save and inspect one object

This walkthrough uses **1998 KY26** to exercise the per-object engineering beta.
It does not demonstrate artificial origin or calibrated significance. Start with
an [isolated installation](installation.md), activate its environment and set
`ANEOS_DATABASE_URL` to the separate beta file-SQLite database.

## 1. Check the entry point — offline

```bash
python aneos.py --version
python aneos.py analyze --help
```

The analysis help lists a designation, `--json`, `--force-refresh` and `--output`.
These help commands do not fetch observations.

## 2. Query the object — network required

```bash
python aneos.py analyze "1998 KY26" --json --output beta-runtime/ky26.json
```

The configured source plan records provider outcomes and admitted evidence before
returning a report. Some providers may fail, return no record or lack a requested
measurement. A successful source response does not mean every field is available.
A query may take time because provider-safe retries and timeouts still apply.

Use a new output path each time. `--force-refresh` bypasses the application source
cache; it does not establish that a provider bypassed its own cache. Do not edit
an older report to make it resemble a newer query.

## 3. Understand the report

Inspect these parts together:

- **Identity and provenance:** designation, source outcomes, evidence references
  and acquisition metadata. A retrieval date is not an observation date.
- **Inputs:** native orbital/physical values, units, epochs, uncertainty and
  covariance where supplied. Missing fields remain unavailable.
- **Screening assessment:** raw ranking, policy identity, individual contributions
  and limitations. Contributions can share underlying evidence.
- **Scientific interpretation:** `FOLLOW_UP_CANDIDATE` calls for investigation;
  `NOT_FLAGGED` is not proven natural origin; `INCONCLUSIVE` is not zero anomaly.

The default threshold of 5.0 is in **ranking units**, not Gaussian sigma. Artificial
origin probability is not calibrated. Neither a source count nor a basic-field
percentage measures whether the evidence is sufficient for a physical hypothesis.
The [screening contract](../engineering/screening-result-contract.md) defines the
separate outputs and the remaining legacy ranking assumptions.

Exit status 0 means a report without recorded errors, not a scientific finding.
Status 1 means the report contains errors; status 2 means analysis or report
creation failed. Preserve the error and source outcomes when reporting a problem.

## 4. Inspect the saved report — no new acquisition

```bash
python artificial_neo_dashboard_launcher.py --file beta-runtime/ky26.json
```

The existing dashboard loader reads the saved report and saves an exclusive bundle
under `dashboard_results/` in the current directory. It leaves the input report
unchanged in the tested loading path. Tables/graphics are views of
that evidence, not additional observations or statistical validation. Keep the
original file and its references. Do not expect a historical legacy-replay JSON
file to have the same schema as a current screening report.

For a human-readable Markdown assessment, make a separately recorded query:

```bash
python aneos.py analyze "1998 KY26" --output beta-runtime/ky26.md
```

That command does not convert the saved JSON and can use a different cache/source
state. Compare evidence and policy identifiers, not just displayed scores.

## 5. Explore the interface

```bash
python aneos.py
```

Choose 1 to screen one object, 2 to inspect screening evidence, 9 to browse results,
10 to export, or 14 to read scientific help. The [menu guide](menu-system.md)
explains the remaining options and their limits. An option's presence is not proof
that its scientific prerequisites are available.

To inspect the local API schema instead:

```bash
python aneos.py api --dev --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Keep the explicit loopback address; the server's
underlying default is `0.0.0.0`. Do not publish this development server or interpret
its generated schema as an authentication/deployment certification.

## Historical regression — offline, development checkout

```bash
python scripts/reproduce_ky26_baseline.py --output beta-runtime/ky26-legacy-replay.json
```

This executes the verified frozen historical implementation against rounded inputs,
without polling providers. Its legacy field names are archived behavior, not current
origin inference. The rounded replay is 4.141288008229254; the original recorded
4.142 display cannot be reproduced exactly without missing full-precision inputs.
See the [baseline](../case-studies/1998-KY26/baseline.md). Choose a new output path;
the script refuses replacement.

## What to investigate next

For any candidate, verify identity, coverage, uncertainty and source conflicts
before interpreting the rank. Seek compatible observations and compare natural
forces and measurement/model artifacts before considering an origin hypothesis.
Useful outcomes include a natural explanation or an explicitly unresolved case.
Complete physical-model comparison and independent calibration remain programme
work, not features enabled by a higher score.
