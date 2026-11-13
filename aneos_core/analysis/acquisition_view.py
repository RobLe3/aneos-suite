"""Recorded source-plan/outcome presentation, never scientific completeness."""

from aneos_core.data.acquisition import SCHEMA, STATUSES, plan, coverage
from aneos_core.data.enrichment import clone, identifier
from .object_review import literal


def verify_summary(summary):
    s = clone(summary)
    if s.get("schema_version") != SCHEMA or s["source_plan"] != plan(
        s["source_plan"]["source_priority"],
        context_id=s["source_plan"].get("context_id"),
    ):
        raise ValueError("Unsupported acquisition summary")
    identifier(s["query_id"], "query id")
    if [o["source"] for o in s["outcomes"]] != s["source_plan"]["sources"]:
        raise ValueError("Acquisition summary denominator mismatch")
    if any(o.get("status") not in STATUSES for o in s["outcomes"]):
        raise ValueError("Unknown source outcome")
    if coverage(s["source_plan"], s["outcomes"]) != s["coverage"]:
        raise ValueError("Acquisition summary counts mismatch")
    receipt = s["storage_receipt"]
    if (
        receipt.get("commit_status") != "committed"
        or receipt.get("query_id") != s["query_id"]
        or receipt.get("bundle_id") != s["bundle_id"]
    ):
        raise ValueError("Acquisition summary lacks a matching committed receipt")
    return s


def acquisition_markdown(summary):
    s = verify_summary(summary)
    c = s["coverage"]
    lines = [
        "## Recorded acquisition",
        f"Query: `{literal(s['query_id'])}`; declared committed local query receipt.",
        f"{c['contributing_sources']}/{c['planned_sources']} planned adapters contributed data; {c['validated_response_sources']}/{c['planned_sources']} returned typed data or an explicit empty response.",
        "| Source | Outcome | Recorded at (UTC) |",
        "| --- | --- | --- |",
    ]
    for o in s["outcomes"]:
        lines.append(
            "| "
            + " | ".join(
                literal(o[k]).replace("|", "\\|")
                for k in ("source", "status", "completed_at")
            )
            + " |"
        )
    lines.append(
        "These are fixed installed-profile response counts, not field completeness, independent observations, source authenticity or scientific readiness. The declared local query record is required for full replay; this summary is not a replay of every native packet."
    )
    return "\n".join(lines)


def render_acquisition_summary(summary, *, console=None):
    s = verify_summary(summary)
    c = s["coverage"]
    if console is None:
        print(acquisition_markdown(s))
        return
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    table = Table(expand=True)
    for name in ("Source", "Outcome", "Recorded at (UTC)", "Meaning"):
        table.add_column(name, overflow="fold")
    meanings = {
        "available": "Typed data contribution; not independently verified",
        "missing": "Explicit empty result in this query; not zero physical risk",
        "unavailable": "Adapter/result unavailable; no substitute data",
        "invalid": "Response not admitted as evidence",
        "ambiguous": "Identity unresolved; not adopted",
        "not_supported": "Configured adapter not supported",
    }
    for o in s["outcomes"]:
        meaning = meanings.get(o["status"], "Unknown outcome")
        if o["source"] == "CAD" and o["status"] == "available":
            meaning = "Computed encounters, not observations"
        if o["reason_code"] == "SOURCE_INITIALIZATION_FAILED":
            meaning = "Source could not initialize"
        if o["reason_code"] == "MPC_DEPENDENCY_UNAVAILABLE":
            meaning = "Optional MPC dependency unavailable"
        timestamp = o["completed_at"].replace("T", "\n", 1).replace("+00:00", "Z")
        table.add_row(
            Text(literal(o["source"])),
            Text(o["status"]),
            Text(timestamp),
            Text(meaning),
        )
    console.print(
        Panel(
            Group(
                Text(
                    f"Data contributed: {c['contributing_sources']}/{c['planned_sources']} planned adapters. Typed data or explicit empty responses: {c['validated_response_sources']}/{c['planned_sources']}."
                ),
                table,
                Text(
                    "Installed catalogue + bounded CAD profile only. Counts are not total completeness, independent evidence, source authenticity or scientific readiness."
                ),
                Text(
                    "Recorded times are adapter-result receipts, not observation epochs. Native/adapter packets where available and query IDs remain in local JSON history."
                ),
                Text("Query: " + s["query_id"]),
            ),
            title="Recorded acquisition outcomes",
            border_style="cyan",
        )
    )
