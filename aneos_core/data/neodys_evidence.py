"""Replayable OEF2.0 normalization. Native covariance is not a Keplerian covariance."""

import math
import re
from hashlib import sha256

from .epoch_evidence import NativeEpoch, SCALES
from .physical_evidence import build_neodys_evidence, physical_projection


def catalogue_identifier(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Unsupported catalogue identifier")
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+", text):
        if int(text) <= 0:
            raise ValueError("Catalogue numbers must be positive")
        return str(int(text))
    compact = "".join(text.upper().split())
    if re.fullmatch(r"(?:1[89][0-9]{2}|2[012][0-9]{2})[A-Z]{2}[0-9]*", compact):
        return compact
    raise ValueError("A verified number or unpacked provisional identifier is required")


def equinoctial_to_keplerian(a, esinLP, ecosLP, hLN, kLN, mean_L):
    if any(
        type(v) not in (int, float) or not math.isfinite(v)
        for v in (a, esinLP, ecosLP, hLN, kLN, mean_L)
    ):
        raise ValueError("Invalid equinoctial element")
    e = math.hypot(esinLP, ecosLP)
    if a <= 0 or e >= 1:
        raise ValueError("Only finite elliptic EQU states are supported")
    lp = math.atan2(esinLP, ecosLP)
    node = math.atan2(hLN, kLN)
    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "inclination": math.degrees(2 * math.atan(math.hypot(hLN, kLN))),
        "ra_of_ascending_node": math.degrees(node) % 360,
        "arg_of_periapsis": math.degrees(lp - node) % 360,
        "mean_anomaly": (mean_L - math.degrees(lp)) % 360,
    }


def parse_oef(text, identifier):
    """Admit one matching elliptic ECLM/J2000 EQU state; account for every data line."""
    identifier = catalogue_identifier(identifier)
    if not isinstance(text, str) or len(text.encode("utf-8")) > 2_000_000:
        raise ValueError("Invalid or oversized OEF text")
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.strip() == "END_OF_HEADER"]
    if len(headers) != 1:
        raise ValueError("Expected exactly one OEF header")
    end = headers[0]
    header = {}
    for line in lines[:end]:
        line = line.split("!", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError("Invalid OEF header")
        key, value = line.split("=", 1)
        key = key.strip()
        if key in header:
            raise ValueError("Ambiguous OEF header")
        header[key] = value.strip().strip("'\"")
    if (
        header.get("format") != "OEF2.0"
        or header.get("rectype") != "ML"
        or header.get("refsys") != "ECLM J2000"
    ):
        raise ValueError("Unsupported OEF format, record type or reference frame")
    records = [
        (i + 1, line, line.split("!", 1)[0].split())
        for i, line in enumerate(lines)
        if i > end
    ]
    records = [(i, line, tokens) for i, line, tokens in records if tokens]
    if (
        not records
        or len(records[0][2]) != 1
        or catalogue_identifier(records[0][2][0]) != identifier
    ):
        raise ValueError("OEF native object identity does not match request")
    native_identifier = records[0][2][0]
    data = records[1:]
    by_tag = {}
    for line_number, line, tokens in data:
        if len(tokens) == 1:
            try:
                catalogue_identifier(tokens[0])
            except ValueError:
                pass
            else:
                raise ValueError("Multiple native object identifiers in one file")
        by_tag.setdefault(tokens[0], []).append((line_number, line, tokens))
    if len(by_tag.get("EQU", [])) != 1 or len(by_tag.get("MJD", [])) != 1:
        raise ValueError("Expected exactly one EQU state and labelled epoch")
    equ = by_tag["EQU"][0][2]
    mjd = by_tag["MJD"][0][2]
    if len(equ) != 7 or len(mjd) != 3:
        raise ValueError("Unsupported EQU/epoch dimensions")
    values = [float(t.replace("D", "E").replace("d", "e")) for t in equ[1:]]
    result = equinoctial_to_keplerian(*values)
    compatibility_epoch = float(mjd[1].replace("D", "E").replace("d", "e"))
    if not math.isfinite(compatibility_epoch):
        raise ValueError("Unsupported nonfinite epoch")
    native_epoch = NativeEpoch(
        mjd[1].replace("D", "E").replace("d", "e"),
        "MJD",
        mjd[2] if mjd[2] in SCALES else "UNKNOWN",
        "NEODyS",
    )
    result["native_epoch"] = native_epoch.to_dict()
    result["_epoch_mjd"] = compatibility_epoch
    context = {
        "native_file_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "native_identifier": native_identifier,
        "reference_frame": header["refsys"],
        "orbital_epoch": native_epoch.to_dict(),
        "epoch_role": "orbit_solution_not_physical_measurement_date",
        "source_documentation": "https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html",
    }
    physical = build_neodys_evidence([r[1] for r in by_tag.get("MAG", [])], context)
    result["_physical"] = physical_projection(physical)
    if result["_physical"].get("absolute_magnitude_h") is not None:
        result["_H"] = result["_physical"]["absolute_magnitude_h"]
    if len(by_tag.get("MAG", [])) == 1:
        tokens = by_tag["MAG"][0][2]
        # Compatibility scalars only; original lines and all alternatives remain native.
        for index, key in ((2, "_G"),):
            try:
                value = float(tokens[index].replace("D", "E").replace("d", "e"))
                if math.isfinite(value):
                    result[key] = value
            except (IndexError, ValueError, OverflowError):
                pass
    accounting = []
    for number, line, tokens in data:
        tag = tokens[0]
        if tag in ("EQU", "MJD"):
            status, reason = (
                "normalized",
                "named_state_or_native_epoch; no covariance transformation",
            )
        elif tag == "MAG":
            status, reason = (
                (
                    "partly_normalized"
                    if result["_physical"].get("absolute_magnitude_h") is not None
                    else "not_normalized"
                ),
                "H selected only when valid and unambiguous; G native, no typed phase-law model or fitting-status assertion",
            )
        elif tag in ("COV", "COR", "NOR", "LSP"):
            status, reason = (
                "native_not_normalized",
                "native matrix/model record; dimensions, basis, correlations or fit status not inferred",
            )
        else:
            status, reason = (
                "unsupported_native",
                "unknown data record retained without guessed semantics",
            )
        accounting.append(
            {
                "line_number": number,
                "tag": tag,
                "native_line": line,
                "status": status,
                "reason": reason,
            }
        )
    result["_evidence_metadata"] = {
        "physical_evidence": physical,
        "native_oef_header": header,
        "native_file_sha256": context["native_file_sha256"],
        "identity_assessment": {
            "status": "matched_native_identifier_not_authenticated",
            "native_identifier": native_identifier,
            "requested_catalogue_identifier": identifier,
            "basis": "exact_oef_native_identifier",
        },
        "record_accounting": accounting,
        "normalization_limits": "EQU to classical elliptic elements only. Native covariance remains native; orbital epoch is not a photometry timestamp. No NG acceleration or origin inference from LSP.",
    }
    return result
