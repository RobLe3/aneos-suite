"""Existing legacy browser operations on canonical screening records, not origin scores."""

from aneos_version import APPLICATION_VERSION
from .artificial_neo_dashboard import (
    ArtificialNEODashboard,
    CATEGORIES,
    CATEGORY_LABELS,
    safe_classification,
    rank_key,
    records_from_file,
)
from .dashboard_view import message, render_rows, render_detail, visible


def ask(menu, label, choices=None):
    if menu.console is not None:
        from rich.prompt import Prompt

        return Prompt.ask(label, choices=choices, console=menu.console)
    return input(label + ": ").strip()


def categorized(objects):
    groups = {key: [] for key in CATEGORY_LABELS}
    for obj in objects:
        c = safe_classification(obj)
        groups[c.category].append(c)
    return groups


def summary(groups):
    return " | ".join(f"{CATEGORY_LABELS[k]}: {len(v)}" for k, v in groups.items())


def show_groups(menu, groups):
    message(
        summary(groups) + " (record counts, not confirmed origins or impact risk)",
        menu.console,
    )


def filter_menu(menu, objects):
    choices = {
        "1": "Follow-up candidates (uncalibrated ranking)",
        "2": "Not flagged (not natural)",
        "3": "Source-listed impact events (requires a replayable Sentry report)",
        "4": "Inconclusive records",
        "5": "All records with an available rank",
        "6": "Records with a typed field inventory (not verified high quality)",
        "7": "More than one recorded source label (not independent evidence)",
        "8": "Unavailable: calibrated five-sigma filter",
        "9": "Unavailable: calibrated two-sigma filter",
        "0": "Cancel",
    }
    for key, label in choices.items():
        message(key + " — " + label, menu.console)
    choice = ask(menu, "Select filter", list(choices))
    items = [safe_classification(o) for o in objects]
    if choice == "0":
        return None
    if choice in ("8", "9"):
        message(
            "No calibrated significance model is installed. Rank/probability cannot be converted into sigma; filter was not applied.",
            menu.console,
        )
        return None
    if choice == "1":
        return [c for c in items if c.category == "follow_up_candidate"]
    if choice == "2":
        return [c for c in items if c.category == "not_flagged"]
    if choice == "3":
        return [c for c in items if menu._has_impact_risk_indicators(c)]
    if choice == "4":
        return [c for c in items if c.category == "inconclusive"]
    if choice == "5":
        return [c for c in items if c.screening_score is not None]
    if choice == "6":
        return [c for c in items if c.basic_presence is not None]
    if choice == "7":
        return [c for c in items if len(c.data_sources) > 1]
    message("Unknown filter; no filter applied.", menu.console)
    return None


def sort_menu(menu, objects):
    choices = {
        "1": "Raw rank (available first; not origin probability)",
        "2": "Screening outcome",
        "3": "Declared catalogue field presence (unknown last)",
        "4": "Local screening receipt (not discovery date)",
        "5": "Exact designation",
        "6": "Typed field inventory availability (not scientific completeness)",
        "0": "Cancel",
    }
    for key, label in choices.items():
        message(key + " — " + label, menu.console)
    choice = ask(menu, "Select sort", list(choices))
    items = [safe_classification(o) for o in objects]
    keys = {
        "1": rank_key,
        "2": lambda c: (list(CATEGORY_LABELS).index(c.category), c.designation),
        "3": lambda c: (
            c.basic_presence is None,
            -c.basic_presence if c.basic_presence is not None else 0,
            c.designation,
        ),
        "4": lambda c: (-c.analysis_timestamp.timestamp(), c.designation),
        "5": lambda c: c.designation,
        "6": lambda c: (c.basic_presence is None, c.designation),
    }
    if choice not in keys:
        return None, ""
    return sorted(items, key=keys[choice]), choices[choice]


def browse(menu, objects, title="Screening records"):
    items = sorted([safe_classification(o) for o in objects], key=rank_key)
    page = 0
    size = 12
    sort_label = "Raw rank"
    while True:
        total_pages = max(1, (len(items) + size - 1) // size)
        page = max(0, min(page, total_pages - 1))
        message(
            f"aNEOS {APPLICATION_VERSION} | "
            + visible(title)
            + f" — page {page+1}/{total_pages}; {len(items)} records; sort: {sort_label}",
            menu.console,
        )
        show_groups(menu, categorized(items))
        rows = items[page * size : (page + 1) * size]
        render_rows(rows, menu.console, start=page * size)
        message(
            "ID detail | P previous | N next | HOME/END | G<number> page | F filter | S sort | C counts | E export page | Q return",
            menu.console,
        )
        command = ask(menu, "Browser command").strip().upper()
        if len(command) > 12:
            message(
                "Browser command is too long; use an ID or a listed command.",
                menu.console,
            )
            continue
        if command in ("Q", "QUIT", "EXIT", "0"):
            return
        if command in ("P", "PREV", "PREVIOUS", "LEFT", "←"):
            page = max(0, page - 1)
        elif command in ("N", "NEXT", "RIGHT", "→"):
            page = min(total_pages - 1, page + 1)
        elif command in ("HOME", "FIRST"):
            page = 0
        elif command in ("END", "LAST"):
            page = total_pages - 1
        elif command.startswith("G") and command[1:].isdecimal():
            target = int(command[1:]) - 1
            if 0 <= target < total_pages:
                page = target
            else:
                message("Page is outside this result set.", menu.console)
        elif command == "F":
            selected = filter_menu(menu, items)
            if selected is not None:
                items = selected
                page = 0
        elif command == "S":
            selected, label = sort_menu(menu, items)
            if selected is not None:
                items = selected
                page = 0
                sort_label = label
        elif command == "C":
            show_groups(menu, categorized(items))
        elif command == "E":
            menu._export_current_view(rows, f"page_{page+1}")
        elif command.isdecimal():
            idx = int(command) - 1
            if 0 <= idx < len(items):
                menu._show_detailed_neo_view(items[idx])
            else:
                message("Record ID is outside this result set.", menu.console)
        else:
            message("Unknown browser command.", menu.console)


def interactive(menu, dashboard):
    labels = {
        "1": "Browse all screening records",
        "2": "Follow-up candidates (not orbital violations)",
        "3": "Follow-up ranking candidates (not confirmed artificial)",
        "4": "Inconclusive records (origin unresolved)",
        "5": "Display screening summary",
        "6": "Browse stored source evidence",
        "0": "Return",
    }
    while True:
        for key, label in labels.items():
            message(key + " — " + label, menu.console)
        choice = ask(menu, "Dashboard option", list(labels))
        if choice == "0":
            return
        items = [safe_classification(c) for c in dashboard.classifications]
        if choice == "1":
            browse(menu, items, "All screening records")
        elif choice in ("2", "3"):
            browse(
                menu,
                [c for c in items if c.category == "follow_up_candidate"],
                "Follow-up candidates",
            )
        elif choice == "4":
            browse(
                menu,
                [c for c in items if c.category == "inconclusive"],
                "Inconclusive records",
            )
        elif choice == "5":
            dashboard.display_dashboard(save_results=False)
        elif choice == "6":
            menu._browse_enriched_database()
