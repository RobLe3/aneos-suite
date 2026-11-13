# Batch screening: requests are not independent objects

Development increment under #41/#12/#15/#44; ADR-094. Not in the published
v2.2.0b2 tag. The primary menu's option 3 uses the existing bounded
`AnalysisPipeline.analyze_batch` through the [SQL job host](durable-batch-execution.md). It does not restore a legacy origin classifier.

## Input and execution

Select a UTF-8 `.txt` file containing one designation per non-comment line, or a
one-column `.csv` with an optional `designation` header. Blank/comment rows do not
become requests. Limits are explicit: 1 MiB and 1,000 request rows, with no silent
clamping. Malformed, unreadable or unsupported files fail before acquisition.
Original bytes (including comments) are retained locally in the exported envelope,
so review the file for private material before sharing an export.

Each distinct **exact designation label** enters the shared batch pipeline once.
Repeated labels reuse its frozen report but retain their original request index and
source line. Case variants and aliases are not merged without identity evidence.
Provider concurrency/retries follow the existing pipeline/source controls; the
object count is not a bound on total HTTP attempts or latency.

Every completed batch accounts for all input occurrences: valid ranks, insufficient
orbital data, source unavailability, invalid identity and other explicit failures.
A candidate is a follow-up ranking result, not an origin finding. Missing data never
become zero or natural ground truth. A whole-pipeline or report-building failure
publishes no falsely completed batch; source acquisition may already have committed
its own receipts. Recoverable background batch execution remains #15.

## Read, export and replay

The summary displays request count, distinct exact labels, repeated rows, execution
status and screening classification separately. Ranks use nine significant digits
in this view; full JSON retains the original numerical values. Option 13 keeps batch run/request/exact-label counts separate from latest-object
views; its legacy export now refuses replacement rather than silently overwriting.
Option 9 browses a
batch run and individual request report, including failure details. It does not poll.

Option 10 offers a separate **batch request history** export scope. Latest-object
views are not that history: they can omit failed requests and collapse repeated
object selections. Batch JSON is `screening-batch-report-v1`; each ordered request
embeds the existing `scientific-screening-report-v1`. The file/line binding, report
identities, summary and exact-repeat reuse are checked during replay.

CSV contains one row per request, full `report_json` per row and a complete
`batch_envelope_json` in the first row. An empty rank cell means unavailable, not
zero; status and canonical JSON preserve the distinction. Formula-like designation
display cells are escaped with a leading apostrophe; the canonical JSON identity
is unchanged. No legacy `verified_count=0` or artificial-origin summary is invented.

Export is exclusive: an existing filename is refused. Export errors are visible;
there is no overwrite fallback. This is not an atomic multi-file backup or a guarantee
that disk failure cannot leave a partial output file. Validate a saved report before
using it. The shared envelope helper also rejects JSON over its existing 16 MiB
bound; split large investigations rather than silently dropping records.

```python
import json
from aneos_core.analysis.batch_reporting import verify_batch_report, render_batch_report
with open("exports/my-batch.json", encoding="utf-8") as stream:
    saved = verify_batch_report(json.load(stream))
render_batch_report(saved)
```

For the existing graphical/report dashboard loader:

```bash
python artificial_neo_dashboard_launcher.py --file exports/my-batch.json
```

This reads saved evidence and creates a new exclusive dashboard bundle. It does
not acquire observations. Request context and the original frozen report remain
in each imported record. Dashboard counts distinguish input records from distinct
exact labels; repeats are not independent evidence.

The current development primary-menu path acknowledges a SQL batch job before
execution and materializes only a committed result. Resume using its job ID/recovery
key after restart; completed replay does not poll. Standalone report construction
does not persist anything. Existing best-effort object-analysis summaries are not
the authoritative batch archive. Other job families and operational recovery remain #15.
Checksums establish consistency, not initial source authenticity or tamper-proofing.

## Verification and remaining limits

Controls use typed **synthetic** source-shaped measurements, not new KY26 observations.
They exercise actual menu acquisition orchestration, missing/invalid/unavailable and
duplicate outcomes, browser/export, concurrency, frozen report validation and
80/120/160-column Rich/plain displays. Recomputed checksums cannot authorize a
failure report claiming successful screening: the shared verifier checks execution
and classification against the actual assessment. Legitimate `service_unavailable`
reports remain supported; a full-suite regression caught that existing status.

```bash
python -m pytest tests/test_batch_menu_accounting.py tests/test_menu_v2_e2e.py -q \
  --cov=aneos_core --cov-config=tests/coverage-batch-report.ini --cov-fail-under=95
python scripts/qa_batch_menu.py --output /tmp/aneos-batch-capture-new
```

See the [QA record](../qa/batch-menu/README.md). Normal tests deny network/DNS or
inject typed acquisition, while ledger/provider correctness has its own tests.
No live-provider reliability, independent detection accuracy, full menu completion,
new calibrated inference or restart guarantee is claimed by this increment.
