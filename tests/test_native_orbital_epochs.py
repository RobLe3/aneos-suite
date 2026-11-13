"""Native astronomical times are not UTC datetimes or observation events."""

import json
import logging
from unittest.mock import AsyncMock
import pytest
from aneos_core.config.settings import APIConfig
from aneos_core.data.models import NEOData, OrbitalElements
from aneos_core.data.sources.sbdb import SBDBSource
from aneos_core.data.sources.horizons import HorizonsSource
from aneos_core.data.fetcher import DataFetcher


def payload(epoch="2461200.50000000001"):
    return {
        "signature": {"version": "1.3", "source": "synthetic-control"},
        "object": {"des": "SYNTHETIC"},
        "orbit": {
            "epoch": epoch,
            "orbit_id": "synthetic-orbit",
            "model_pars": [],
            "elements": [
                {"name": k, "value": v}
                for k, v in [("a", "1.2"), ("e", ".2"), ("i", "1.49")]
            ],
        },
    }


def acquired(data):
    source = SBDBSource(APIConfig())
    source._http_get = AsyncMock(return_value=data)
    fetcher = DataFetcher.__new__(DataFetcher)
    fetcher.logger = logging.getLogger("epoch-control")
    return fetcher._fetch_from_source("SBDB", source, "SYNTHETIC")


def test_sbdb_native_epoch_survives_real_source_model_cache_and_review_path():
    from aneos_core.analysis.object_review import (
        build_object_review,
        verify_object_review,
    )

    n = acquired(payload())
    assert n is not None
    epoch = n.orbital_elements.native_epoch
    assert epoch.native_value == "2461200.50000000001"
    assert epoch.time_scale == "TDB"
    assert n.orbital_elements.epoch is None
    restored = NEOData.from_dict(json.loads(json.dumps(n.to_dict())))
    assert restored.orbital_elements.native_epoch.to_dict() == epoch.to_dict()
    review = verify_object_review(build_object_review(restored))
    row = next(
        r for r in review["inventory"]["measurements"] if r["key"] == "orbit_epoch"
    )
    assert row["value"] == "2461200.50000000001" and row["units"] == "JD TDB"
    assert row["status"] == "present"
    assert (
        review["inventory"]["coverage"]["profile_id"]
        == "basic-catalogue-presence-15-v2"
    )


def test_actual_jd_equals_calendar_history_headers_yield_complete_separate_rows():
    text = """$$SOE
2451545.000000001 = A.D. 2000-Jan-01 12:00:00.0001 TDB
 EC= .2 QR= .9 TP= 2451500.0
 OM= 20 W = 30 IN= 1.5
 A= 1.2 MA= 40
2451546.000000001 = A.D. 2000-Jan-02 12:00:00.0001 TDB
 EC= .21 QR= .9 TP= 2451501.0
 OM= 21 W = 31 IN= 1.6
 MA= 41 A= 1.21
$$EOE"""
    rows = HorizonsSource.__new__(HorizonsSource)._parse_elements_table(text)
    assert len(rows) == 2
    assert rows[0]["native_epoch"]["native_value"] == "2451545.000000001"
    assert rows[0]["native_epoch"]["time_scale"] == "TDB"
    assert rows[1]["a"] == 1.21 and rows[1]["M"] == 41
    assert rows[0]["i"] == 1.5 and rows[1]["i"] == 1.6


def test_native_jd_mjd_equality_preserves_digits_without_time_scale_conversion():
    from aneos_core.data.epoch_evidence import NativeEpoch

    a = NativeEpoch("51544.500000000001", "MJD", "TT", "SYNTHETIC")
    b = NativeEpoch("2451545.000000000001", "JD", "TT", "SYNTHETIC")
    assert a.to_dict()["julian_day_same_scale"] == "2451545.000000000001"
    assert a.same_julian_instant(b) is True
    assert (
        b.same_julian_instant(
            NativeEpoch("2451545.000000000002", "JD", "TT", "SYNTHETIC")
        )
        is False
    )
    assert (
        b.same_julian_instant(NativeEpoch(b.native_value, "JD", "TDB", "SYNTHETIC"))
        is None
    )
    assert (
        b.same_julian_instant(NativeEpoch(b.native_value, "JD", "UNKNOWN", "SYNTHETIC"))
        is None
    )
    with pytest.raises(TypeError):
        b.same_julian_instant(None)


@pytest.mark.parametrize("value", [True, None, "NaN", "inf", "banana", "1e1000", {}])
def test_bad_native_numeric_epoch_is_rejected_without_zero_default(value):
    from aneos_core.data.epoch_evidence import NativeEpoch, decimal_value

    with pytest.raises(ValueError):
        decimal_value(value)
    with pytest.raises(ValueError):
        NativeEpoch(value, "JD", "TDB", "SYNTHETIC")


def test_sbdb_invalid_epoch_is_source_failure_not_default_or_epoch_zero():
    assert acquired(payload("not-a-number")) is None
    n = acquired(payload(None))
    assert n is not None
    assert n.orbital_elements.native_epoch is None
    assert "native_epoch" not in n.orbital_elements.to_dict()


def test_legacy_snapshot_serialization_is_not_rewritten_to_new_time_scale():
    from datetime import datetime

    oe = OrbitalElements(semi_major_axis=1.2, epoch=datetime(2020, 1, 1))
    data = oe.to_dict()
    assert "native_epoch" not in data
    assert OrbitalElements.from_dict(data).to_dict() == data
    assert oe.epoch.tzinfo is None


def test_rehashed_native_epoch_projection_and_source_mismatch_are_rejected():
    import copy
    from aneos_core.data.epoch_evidence import NativeEpoch

    n = acquired(payload())
    original = n.to_dict()
    changed = copy.deepcopy(original)
    changed["orbital_elements"]["native_epoch"]["julian_day_same_scale"] = "0"
    with pytest.raises(ValueError):
        NEOData.from_dict(changed)
    changed = copy.deepcopy(original)
    changed["orbital_elements"]["native_epoch"] = NativeEpoch(
        "2461201.5", "JD", "TDB", "SBDB"
    ).to_dict()
    with pytest.raises(ValueError, match="source solution"):
        NEOData.from_dict(changed)
    from aneos_core.analysis.screening import ScreeningAnalysisService

    n.orbital_elements.native_epoch = NativeEpoch("2461201.5", "JD", "TDB", "SBDB")
    with pytest.raises(ValueError, match="source solution"):
        ScreeningAnalysisService().analyze_neo(n)


def test_obj_data_epoch_header_is_parsed_without_inventing_utc():
    s = HorizonsSource.__new__(HorizonsSource)
    for suffix, scale in [
        (" ! 2023-Feb-25.0 (TDB)", "TDB"),
        (" ! unknown calendar", "UNKNOWN"),
    ]:
        data = s._parse_obj_data(
            "EPOCH= 2460000.500000001" + suffix + "\nEC= .2 A= 1.2 IN= 1.5\n"
        )
        assert data["native_epoch"]["native_value"] == "2460000.500000001"
        assert data["native_epoch"]["time_scale"] == scale
        assert "epoch" not in data


def test_history_calendar_header_is_retained_but_not_guessed_to_jd():
    rows = HorizonsSource.__new__(HorizonsSource)._parse_elements_table(
        "$$SOE\n2020-Jan-01 00:00:00.0000 TDB\nEC= .2 A= 1.2 IN= 2\n$$EOE"
    )
    assert rows[0]["native_epoch"]["format"] == "calendar"
    assert rows[0]["native_epoch"]["julian_day_same_scale"] is None
    assert rows[0]["epoch"] == "2020-Jan-01 00:00:00.0000 TDB"


@pytest.mark.parametrize(
    "text",
    [
        "$$SOE\nEC= .2 A= 1.2\n$$EOE",
        "$$SOE\n2451545.0 = A.D. 2000-Jan-01 12:00:00 TDB\nA= 1.2\nA= 1.3\n$$EOE",
    ],
)
def test_unbound_or_duplicated_elements_cannot_be_silently_merged(text):
    with pytest.raises(ValueError):
        HorizonsSource.__new__(HorizonsSource)._parse_elements_table(text)


def test_neodys_mjd_tdt_is_retained_not_mislabeled_tdb_or_utc():
    from aneos_core.data.sources.neodys import NEODySSource

    src = NEODySSource.__new__(NEODySSource)
    data = src._parse_eq0(
        "format = OEF2.0\nrectype = ML\nrefsys = ECLM J2000\nEND_OF_HEADER\n123\n EQU 1.2 .1 .1 .01 .01 20\n MJD 51544.500000000001 TDT\n MAG 20 .15\n",
        123,
    )
    assert data["native_epoch"]["format"] == "MJD"
    assert data["native_epoch"]["time_scale"] == "TDT"
    assert data["native_epoch"]["julian_day_same_scale"] == "2451545.000000000001"


def test_actual_horizons_fetch_source_and_reconstruction_preserve_header(monkeypatch):
    from unittest.mock import Mock
    import aneos_core.data.sources.horizons as module

    text = "EPOCH= 2460000.500000001 ! 2023-Feb-25.0 (TDB)\n EC= .2 A= 1.2 IN= 1.5"
    response = Mock()
    response.json.return_value = {
        "signature": {"source": "synthetic-control"},
        "result": text,
    }
    monkeypatch.setattr(module, "provider_get", Mock(return_value=response))
    f = DataFetcher.__new__(DataFetcher)
    f.logger = logging.getLogger("epoch-horizons-control")
    n = f._fetch_from_source("Horizons", HorizonsSource(APIConfig()), "SYNTHETIC")
    assert n is not None
    assert n.orbital_elements.native_epoch.native_value == "2460000.500000001"
    assert n.evidence_metadata["solution"]["result_text_sha256"]
    assert NEOData.from_dict(n.to_dict()).orbital_elements.epoch is None


def test_native_calendar_is_opaque_but_not_arbitrary_nondate():
    from aneos_core.data.epoch_evidence import NativeEpoch

    with pytest.raises(ValueError):
        NativeEpoch("banana", "calendar", "TDB", "SYNTHETIC")
    a = NativeEpoch("2020-Jan-01 00:00:00 TDB", "calendar", "TDB", "SYNTHETIC")
    assert a.same_julian_instant(a) is None


def test_covariance_native_epoch_is_distinct_and_not_propagated_or_relabelled():
    from aneos_core.data.models import NonGravitationalParameters
    from aneos_core.data.orbit_evidence import OrbitalCovariance
    from aneos_core.analysis.object_review import (
        build_object_review,
        render_object_review,
    )
    import io
    from rich.console import Console

    n = acquired(payload("2461200.5"))
    n.nongrav = NonGravitationalParameters(
        covariance=OrbitalCovariance(
            ["e", "q", "tp", "node", "peri", "i"],
            [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)],
            "2457098.5",
        ),
        solution={"epoch_jd_tdb": "2461200.5"},
    )
    r = build_object_review(n)
    comparison = r["inventory"]["epoch_comparison"]
    assert comparison["same_julian_instant"] is False
    assert comparison["covariance_epoch"]["native_value"] == "2457098.5"
    out = io.StringIO()
    render_object_review(r, console=Console(file=out, width=80, color_system=None))
    assert "2457098.5" in out.getvalue() and "2461200.5" in out.getvalue()
    assert "not propagated" in out.getvalue()


@pytest.mark.parametrize(
    "format,scale,source,label",
    [
        ("epoch", "TDB", "x", None),
        ("JD", {}, "x", None),
        ("JD", "TDB", "", None),
        ("JD", "TDB", "x", {}),
    ],
)
def test_invalid_native_contract_fields_are_rejected(format, scale, source, label):
    from aneos_core.data.epoch_evidence import NativeEpoch

    with pytest.raises(ValueError):
        NativeEpoch("2451545.0", format, scale, source, label=label)


def test_epoch_dictionary_has_no_unchecked_derived_fields():
    from aneos_core.data.epoch_evidence import NativeEpoch

    for record in (
        None,
        {},
        NativeEpoch("2451545.0", "JD", "TDB", "SYNTHETIC").to_dict()
        | {"julian_day_same_scale": "0"},
    ):
        with pytest.raises(ValueError):
            NativeEpoch.from_dict(record)


def test_actual_epoch_capture_script_labels_scale_and_covariance_at_three_widths(
    tmp_path,
):
    import subprocess, sys
    from pathlib import Path
    from hashlib import sha256

    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "epoch-qa"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/qa_object_review.py"),
            "--epoch-cases",
            "--output",
            str(directory),
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    inventory = json.loads((directory / "inventory.json").read_text())
    for width in (80, 120, 160):
        text = (directory / f"different-covariance-{width}.txt").read_text()
        assert (
            "2461200.500000001" in text
            and "2457098.5" in text
            and "not propagated" in text
        )
        assert (
            "Time scale not provided"
            in (directory / f"unknown-scale-{width}.txt").read_text()
        )
        assert (
            inventory["cases"]["different-covariance"]["widths"][str(width)][
                "max_columns"
            ]
            <= width
        )
    for name, digest in inventory["artifact_sha256"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest


def test_mismatched_scale_on_sbdb_snapshot_does_not_become_utc_evidence():
    from aneos_core.data.epoch_evidence import NativeEpoch

    n = acquired(payload())
    data = n.to_dict()
    value = data["orbital_elements"]["native_epoch"]["native_value"]
    data["orbital_elements"]["native_epoch"] = NativeEpoch(
        value, "JD", "UTC", "SBDB"
    ).to_dict()
    with pytest.raises(ValueError, match="source solution"):
        NEOData.from_dict(data)


def test_sub_float_precision_epoch_difference_is_not_false_covariance_match():
    from aneos_core.data.models import NonGravitationalParameters
    from aneos_core.data.orbit_evidence import OrbitalCovariance

    ng = NonGravitationalParameters(
        covariance=OrbitalCovariance(["e"], [[1.0]], "2451545.000000000001"),
        solution={"epoch_jd_tdb": "2451545.000000000002"},
    )
    assert float(ng.covariance.epoch_jd_tdb) == float(ng.solution["epoch_jd_tdb"])
    assert ng.evidence_report()["covariance_applies_to_display_epoch"] is False


def test_native_epoch_merge_never_imports_secondary_covariance_or_epoch():
    from aneos_core.data.epoch_evidence import NativeEpoch

    a = acquired(payload("2461200.5"))
    b = acquired(payload("2457098.5"))
    f = DataFetcher.__new__(DataFetcher)
    f.source_priority = ["primary", "secondary"]
    m = f._reconcile_sources({"secondary": b, "primary": a})
    assert m.orbital_elements.native_epoch.native_value == "2461200.5"
    assert (
        m.evidence_metadata["conflicts"]["orbital_elements:secondary"]["native_epoch"][
            "native_value"
        ]
        == "2457098.5"
    )
    assert m.nongrav is None
    assert (
        NEOData.from_dict(m.to_dict()).orbital_elements.native_epoch.native_value
        == "2461200.5"
    )


@pytest.mark.parametrize("fmt,suffix", [("1", "json"), ("2", "csv")])
def test_actual_menu_export_preserves_native_epoch_literal_and_scale(
    tmp_path, fmt, suffix
):
    from aneos_menu_v2 import ANEOSMenuV2
    from unittest.mock import Mock

    menu = ANEOSMenuV2.__new__(ANEOSMenuV2)
    menu.console = None
    n = acquired(payload())
    result = menu._run_detection("SYNTHETIC", n)
    assert result is not None
    menu._detection_results = {"SYNTHETIC": result}
    menu._impact_results = {}
    path = tmp_path / f"epoch.{suffix}"
    menu._ask = Mock(side_effect=[fmt, str(path)])
    menu._export_results()
    if fmt == "1":
        r = json.loads(path.read_text())["neos"][0]["object_review"]
    else:
        import csv

        with path.open() as stream:
            r = json.loads(
                next(
                    csv.DictReader(line for line in stream if not line.startswith("#"))
                )["object_review_json"]
            )
    epoch = r["inventory"]["source_snapshot"]["orbital_elements"]["native_epoch"]
    assert (
        epoch["native_value"] == "2461200.50000000001" and epoch["time_scale"] == "TDB"
    )
    assert r["assessment"]["sigma_confidence"] is None


def test_native_orbit_epoch_is_not_a_cad_encounter_or_observation_date():
    from aneos_core.data.models import CloseApproach
    from aneos_core.analysis.object_review import build_object_review

    n = acquired(payload())
    n.close_approaches = [
        CloseApproach(
            "SYNTHETIC", epoch_jd_tdb=2450000.5, time_scale="TDB", body="Earth"
        )
    ]
    r = build_object_review(n)
    assert (
        r["inventory"]["orbital_epoch_evidence"]["native_value"]
        == "2461200.50000000001"
    )
    assert r["inventory"]["encounters"][0]["epoch_jd_tdb"] == 2450000.5
    assert r["inventory"]["encounters"][0]["basis"] == "computed_encounter"
    assert n.first_observation is None and n.last_observation is None


def test_history_rejects_nonfinite_elements_instead_of_presenting_them():
    text = "$$SOE\n2451545.0 = A.D. 2000-Jan-01 12:00:00 TDB\nA= 1e999\n$$EOE"
    with pytest.raises(ValueError, match="Nonfinite"):
        HorizonsSource.__new__(HorizonsSource)._parse_elements_table(text)


def test_history_rejects_duplicate_same_line_tokens():
    text = "$$SOE\n2451545.0 = A.D. 2000-Jan-01 12:00:00 TDB\nA= 1.2 A= 1.3\n$$EOE"
    with pytest.raises(ValueError, match="Duplicate"):
        HorizonsSource.__new__(HorizonsSource)._parse_elements_table(text)
