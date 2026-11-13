# Root poller: screening views and explicit exports

This is a post-2.2.0b2 implementation follow-up for #81, not a change to the published
beta-2 tag. Root bulk polling remains temporarily unavailable under #73.

The root `neo_poller.NEOPoller` keeps its existing callable entrypoints:

| Entry point | Behavior |
|---|---|
| `display_results(records)` | Shared screening summary and count graph; returns the dashboard. No polling or export. |
| `_interactive_object_explorer(dashboard)` | Shared browser, filters, sorting, global record details and explicit page export. |
| `_show_detailed_analysis(record)` | Existing source-bound object review and calculation explanation, or labeled unverified input. |
| `launch_analysis_dashboard(records)` | Async-compatible display and optional browser; does not automatically save. |
| `save_results(records, source, period)` | Explicit exclusive JSON/Markdown export, including empty results; returns the new directory or `None` on failure. |

A legacy 0.999 score is not adopted as an origin probability. Missing probability
stays unavailable; it is not zero and no longer causes formatting/comparison errors.
Typed computational inputs use the same screening service as the existing views.
Frozen records keep their own verified policies rather than being silently ranked
under today's policy. Not flagged does not establish natural origin.

## Inspect an existing file without querying sources

The standalone results launcher remains the usual CLI entrypoint:

```sh
python artificial_neo_dashboard_launcher.py --file beta-runtime/ky26.json
```

For existing Python integrations using the root helpers:

```python
import json
from pathlib import Path
from neo_poller import NEOPoller
from aneos_core.analysis.artificial_neo_dashboard import records_from_file

records = records_from_file(json.loads(Path("beta-runtime/ky26.json").read_text()))
poller = NEOPoller()
poller.display_results(records)  # read-only, including when console is None
output = poller.save_results(records, "file-import", "not-applicable")
if output is None:
    raise RuntimeError("No completed export was verified")
```

`source` and `period` are retained as **requested invocation context**, not proof of
which sources answered or what observational dates the records cover. That context
appears in `report.json` and `report.md`, bound by the report fingerprint. Do not put
credentials or private notes into it. Each explicit export creates a new directory
under `dashboard_results/` with JSON, Markdown and `complete.json` checksums. Old
`neo_poll_*.json` files remain readable through the shared file decoder and are not
rewritten. Re-import does not migrate or back up the original file. As before,
manual edits, cleanup and deletion can change local files; preserve important
research separately. Interrupted exports may leave a partial directory without a
completion marker, never a verified completed export.

## Verification and limits

```sh
python -m pytest tests/test_root_poller_presentation.py -q
python scripts/qa_root_poller.py --output /tmp/aneos-root-poller-qa-new
```

The capture destination must not already exist. The script captures actual root
summary, browser and typed/legacy detail methods at 80/120/160 columns, with text,
HTML, SVG, input snapshots and source/artifact hashes. These are synthetic controls,
not provider observations. Tests deny network connections and exercise nullable,
empty, positive, identity-mismatched, edited, export-failure and navigation paths.

This does not certify every root menu option, restore bulk acquisition, authenticate
provider data or calibrate a physical-origin/significance model. Record counts and
field presence are not scientific completeness. Remaining source restoration and
whole-menu QA stay tracked separately.
