"""Human-readable role-separated chronology; no invented shared UTC plot axis."""

import json
import html
from aneos_core.data.chronology import verify_chronology

LABELS = {
    "first_observation": "First observation in fit",
    "last_observation": "Last observation in fit",
    "orbit_epoch": "Orbital-element epoch",
    "covariance_epoch": "Covariance epoch",
    "orbit_fit_date": "Orbit fit produced",
    "adapter_receipt": "Adapter receipt",
    "discovery": "Discovery day",
}


def _visible(value):
    return "".join(
        f"\\x{ord(c):02x}" if ord(c) < 32 and c != "\n" or ord(c) == 127 else c
        for c in str(value)
    )


def _value(e):
    value = e["value"]
    if value is None:
        return "not available"
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, allow_nan=False)
    )
    if len(text) > 180:
        text = text[:180] + "… (full value in JSON)"
    if e["time_scale"] == "UTC" and text.endswith("+00:00"):
        text = text[:-6] + "Z"
    if e["representation"] == "ISO timestamp":
        text = text.replace("T", "\n", 1)
    elif e["representation"] in ("JD", "MJD"):
        text += "\n" + e["representation"]
    return _visible(
        text + "\n" + e["time_scale"].replace("not_declared", "scale not declared")
    )


SHORT_MEANINGS = {
    "first_observation": "Earliest observation in fit; not discovery.",
    "last_observation": "Latest observation in fit; not discovery.",
    "orbit_epoch": "Element reference epoch; not converted or propagated.",
    "covariance_epoch": "Covariance reference epoch; not propagated.",
    "orbit_fit_date": "Fit-production label; no UTC/DST offset inferred.",
    "adapter_receipt": "Adapter read time; not an observation.",
    "discovery": "Catalogue day label, not a time instant.",
}


def _rows(c):
    rows = []
    for e in c["events"]:
        if e["status"] == "not_available":
            continue
        meaning = SHORT_MEANINGS.get(e["role"], e["meaning"])
        if e["status"] == "partial_calendar_label":
            meaning = "Month/day partly unknown; not filled in."
        elif e["status"] == "invalid":
            meaning = "Invalid/unsupported value; raw data retained."
        elif e["status"] == "time_scale_unverified":
            meaning = "Time basis unverified; not assumed UTC."
        rows.append(
            (LABELS.get(e["role"], e["role"]) + "\n" + e["source"], _value(e), meaning)
        )
    for label, key in [
        ("First ledger receipt", "first_ledger_recorded_at"),
        ("Last ledger receipt", "last_ledger_recorded_at"),
    ]:
        value = c["local_receipts"][key]
        if value is not None:
            e = {"value": value, "representation": "ISO timestamp", "time_scale": "UTC"}
            rows.append((label, _value(e), "Local ledger append, not discovery."))
    return rows


def _notes(c):
    d = c["discovery"]
    note = "Discovery day: " + (
        d["selected"]["value"] + " (catalogued day, not an instant)."
        if d["selected"]
        else (
            "ambiguous; no scalar date selected."
            if d["status"] == "ambiguous"
            else "not available; not inferred from observations or local intake."
        )
    )
    notes = [note]
    if c.get("reference_scope") != "chronology_input_json_pointers":
        notes.append(
            "Legacy references identify snapshot context, not guaranteed direct value pointers; preserved compact inputs remain available."
        )
    if any(s["discovery_evidence"] is not None for s in c["inputs"]["sources"]):
        notes.append(
            "SBDB documents discovery circumstances only for numbered asteroids/comets; an omitted date is not evidence of non-discovery."
        )
    if not c["local_receipts"]["records"]:
        notes.append(
            "No verified ledger receipt history supplied in this view; source timestamps are not a substitute."
        )
    if c["local_receipts"]["unverified_count"]:
        notes.append(
            str(c["local_receipts"]["unverified_count"])
            + " ledger timestamp(s) have invalid/unverified time bases; raw values remain in JSON."
        )
    if c["local_receipts"]["clock_order"] == "clock_regression_sequence_order_retained":
        notes.append(
            "Receipt clocks regress. Append sequence, not wall-clock sorting, defines ledger order; no elapsed age is inferred."
        )
    notes.append(
        "UTC receipts may display Z instead of +00:00; original literals and complete meanings remain in JSON."
    )
    notes += c["limitations"]
    return notes


def _md(value):
    text = html.escape(_visible(value), quote=True).replace("\\", "\\\\")
    for character in "|`[]()*_~!":
        text = text.replace(character, "\\" + character)
    return text.replace("\n", "<br>")


def chronology_markdown(document):
    c = verify_chronology(document)
    lines = (
        ["## Object chronology — distinct date roles", ""]
        + _notes(c)
        + ["", "| Role / source | Recorded value / scale | Meaning |", "|---|---|---|"]
    )
    for row in _rows(c):
        lines.append("| " + " | ".join(_md(value) for value in row) + " |")
    legacy = [
        (key, r) for key, r in c["legacy_dates"].items() if r["value"] is not None
    ]
    if legacy:
        lines += [
            "",
            "### Compatibility dates — unverified semantics",
            "These preserved values are not accepted discovery dates or UTC receipts.",
        ]
        for key, r in legacy:
            lines.append("- " + _md(key) + ": " + _md(r["value"]) + "; " + r["status"])
    return "\n".join(lines) + "\n"


def render_chronology(document, *, console=None):
    c = verify_chronology(document)
    if console is None:
        print(chronology_markdown(c))
        return
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich.console import Group

    table = Table(expand=True)
    for name in ("Role / source", "Recorded value / scale", "Meaning"):
        table.add_column(name, overflow="fold")
    rows = _rows(c)
    for row in rows:
        table.add_row(*(Text(_visible(value)) for value in row))
    parts = [Text(text) for text in _notes(c)]
    parts.append(
        table
        if rows
        else Text(
            "No sourced dates or verified local receipts are available in this view."
        )
    )
    legacy = [
        (key, r) for key, r in c["legacy_dates"].items() if r["value"] is not None
    ]
    if legacy:
        parts.append(
            Text(
                "Compatibility dates — semantics/time basis unverified; preserved, not silently relabelled UTC.",
                style="yellow",
            )
        )
        old = Table(expand=True)
        old.add_column("Compatibility field", overflow="fold")
        old.add_column("Stored value (not discovery)", overflow="fold")
        for key, r in legacy:
            old.add_row(Text(key), Text(_visible(r["value"]).replace("T", "\n", 1)))
        parts.append(old)
    console.print(
        Panel(
            Group(*parts),
            title=Text("Object chronology — distinct date roles"),
            border_style="cyan",
        )
    )
