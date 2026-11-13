"""Readable source-receipt review; no terminal markup from stored object labels."""


def safe(value):
    return "".join(
        f"\\x{ord(c):02x}" if ord(c) < 32 or ord(c) == 127 else c for c in str(value)
    )


def read_label(value):
    literal = safe(value or "Unknown")
    if literal.endswith("+00:00"):
        literal = literal[:-6] + "Z"
    return literal.replace("T", "\n", 1)


def render_freshness(report, console=None):
    title = "Source receipt review"
    summary = (
        f"Objects checked: {report['checked_count']}; need review: {report['outdated_count']}; "
        f"{report['eligible_count']} eligible for an explicit whole-plan refresh."
    )
    limits = (
        "Adapter receipt age is not observation age or proof of fresh network data. "
        "Unknown legacy timestamps are not assigned UTC. Runtime provider admission still applies."
    )
    rows = []
    for obj in report["objects"]:
        for source in obj["sources"]:
            rows.append(
                [
                    safe(obj["designation"]),
                    safe(source["source"]),
                    safe(source["latest_outcome"].replace("_", " ")),
                    safe(source["freshness"].replace("_", " ")),
                    read_label(source["last_adapter_read_at"]),
                    safe(source["retry_reason"].replace("_", " ")),
                ]
            )
    if console is not None:
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console.print(Panel(Text(summary), title=title))
        table = Table(show_lines=True, expand=True)
        for column in (
            "Object",
            "Source",
            "Last outcome",
            "Receipt age status",
            "Adapter read",
            "Retry review",
        ):
            table.add_column(column, overflow="fold")
        for row in rows:
            table.add_row(*(Text(x) for x in row))
        if rows:
            console.print(table)
        console.print(Text(limits))
        if report["truncated"]:
            console.print(
                Text(
                    f"Object limit {report['object_limit']} reached; more objects were not reviewed."
                )
            )
    else:
        print(title + "\n" + summary)
        for row in rows:
            print(" | ".join(row))
        print(limits)
        if report["truncated"]:
            print(
                f"Object limit {report['object_limit']} reached; more objects were not reviewed."
            )


def render_refresh(report, console=None):
    lines = [
        "Explicit object refresh",
        f"{report['recorded_count']} committed, replay-verified queries.",
    ]
    for row in report["results"]:
        line = safe(row["designation"]) + ": " + row["status"].replace("_", " ")
        if "coverage" in row:
            c = row["coverage"]
            line += f"; {c['contributing_sources']}/{c['planned_sources']} adapters contribute"
        if row.get("blocked_sources"):
            line += "; deferred sources: " + ", ".join(
                safe(s) for s in row["blocked_sources"]
            )
        lines.append(line)
    if report.get("cleanup_status") == "failed":
        lines.append(
            "Cleanup failed; recorded query evidence remains available. Review diagnostics before another refresh."
        )
    lines.append(report["limits"])
    if console is not None:
        from rich.panel import Panel
        from rich.text import Text

        console.print(Panel(Text("\n".join(lines)), title="Refresh outcome"))
    else:
        print("\n".join(lines))
