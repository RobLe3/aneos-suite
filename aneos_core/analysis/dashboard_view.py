"""Shared Rich/plain/Markdown views; screening, field presence and origin stay distinct."""

import html
import json
from aneos_version import APPLICATION_VERSION
from .artificial_neo_dashboard import CATEGORY_LABELS, LIMITS, safe_classification
from .object_review import render_object_review, object_review_markdown


def visible(value):
    return "".join(
        f"\\x{ord(c):02x}" if ord(c) < 32 or ord(c) == 127 else c for c in str(value)
    )


def message(text, console=None):
    text = visible(text)
    if console is None:
        print(text)
    else:
        from rich.text import Text

        console.print(Text(text))


def rank_text(c):
    return "Unavailable" if c.screening_score is None else f"{c.screening_score:.6g}"


def presence_text(c):
    if c.review and c.review["inventory"]:
        v = c.review["inventory"]["coverage"]["basic_catalogue"]
        return f"{v['present']}/{v['required']} fields"
    return "Not assessed"


def render_rows(objects, console=None, *, start=0):
    rows = []
    for i, value in enumerate(objects, start + 1):
        c = safe_classification(value)
        rows.append(
            [
                str(i),
                visible(c.designation),
                CATEGORY_LABELS[c.category],
                rank_text(c),
                presence_text(c),
            ]
        )
    if console is None:
        for row in rows:
            print(" | ".join(row))
    else:
        from rich.table import Table
        from rich.text import Text

        table = Table(
            title="Screening records — ranking units, not sigma",
            show_lines=True,
            expand=True,
        )
        for label in (
            "ID",
            "Object",
            "Screening outcome",
            "Raw rank",
            "Catalogue presence",
        ):
            table.add_column(label, overflow="fold")
        for row in rows:
            table.add_row(*(Text(v) for v in row))
        console.print(table)
    if not rows:
        message("No screening records in this view.", console)


def render_dashboard(objects, metrics, console=None):
    from .artificial_neo_dashboard import rank_key

    summary = (
        f"Input records: {metrics.total_records}; distinct exact designation labels: {metrics.total_objects}; "
        f"typed field inventories: {metrics.records_with_inventory}. "
        "Repeated designations are not automatically independent evidence or fused observations."
    )
    if console is None:
        print(
            f"aNEOS {APPLICATION_VERSION} — Candidate screening dashboard\n" + summary
        )
    else:
        from rich.panel import Panel
        from rich.text import Text

        console.print(
            Panel(
                Text(summary),
                title=f"aNEOS {APPLICATION_VERSION} — Candidate screening dashboard",
            )
        )
    counts = {
        "follow_up_candidate": metrics.follow_up_candidates,
        "not_flagged": metrics.not_flagged,
        "inconclusive": metrics.inconclusive,
    }
    for key, count in counts.items():
        bar = (
            "━" * round(20 * count / metrics.total_records)
            if metrics.total_records
            else ""
        )
        message(
            f"{CATEGORY_LABELS[key]}: {count}/{metrics.total_records} records {bar}",
            console,
        )
    render_rows(sorted(objects, key=rank_key), console)
    message(LIMITS, console)


def render_detail(value, console=None):
    c = safe_classification(value)
    message(f"{c.designation} — {CATEGORY_LABELS[c.category]}", console)
    basis = {
        "legacy_unverified": "Legacy assertions — unverified; not used as measurements",
        "typed_snapshot_not_authenticated": "Typed source snapshot — consistency checked, source not authenticated",
        "replayed_assessment_not_authenticated": "Frozen screening evidence — replay checked, source not authenticated",
        "replayed_review_not_authenticated": "Frozen object review — replay checked, source not authenticated",
        "declared_synthetic_unverified": "Declared synthetic input — not observational evidence",
        "invalid_input": "Invalid or inconsistent input — screening unavailable",
    }
    message("Input basis: " + basis.get(c.input_status, "Unverified input"), console)
    message(
        "Source-format consistency/replay is not provider authentication or independent validation.",
        console,
    )
    if c.review:
        render_object_review(c.review, console=console, verbose=True)
    else:
        message(c.analysis_details["interpretation"], console)
        message(
            "Legacy or invalid input below is archived, not adopted as measurements or an origin conclusion.",
            console,
        )
        text = json.dumps(c.input_record, indent=2, allow_nan=False)
        if console is None:
            print(text)
        else:
            from rich.text import Text

            console.print(Text(text, overflow="fold"))
    message(LIMITS, console)


def dashboard_markdown(objects):
    lines = ["# Candidate screening dashboard", "", LIMITS, ""]
    for value in objects:
        c = safe_classification(value)
        lines.extend(
            [
                "## " + html.escape(visible(c.designation)),
                "",
                CATEGORY_LABELS[c.category] + "; rank " + rank_text(c) + ".",
                "Input status: " + c.input_status + ".",
                "",
            ]
        )
        if c.review:
            lines.append(object_review_markdown(c.review))
        else:
            raw = (
                json.dumps(c.input_record, indent=2, allow_nan=False)
                .replace("`", "\\u0060")
                .replace("<", "\\u003c")
                .replace(">", "\\u003e")
            )
            lines.extend(
                [
                    "Legacy/invalid input — archived claims are not adopted:",
                    "```json",
                    raw,
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"
