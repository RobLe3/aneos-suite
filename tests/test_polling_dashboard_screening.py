"""Actual dashboard/browser safety and positive canonical screening controls."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock
import pytest
from aneos_core.analysis.artificial_neo_dashboard import ArtificialNEODashboard
from aneos_core.data.models import NEOData, OrbitalElements, PhysicalProperties


def neo(designation="SYNTHETIC"):
    stamp = datetime(2026, 11, 13, 8, tzinfo=timezone.utc)
    return NEOData(
        designation,
        orbital_elements=OrbitalElements(
            semi_major_axis=1.23, eccentricity=0.2, inclination=1.49
        ),
        physical_properties=PhysicalProperties(
            diameter_km=0.011, albedo=0.52, absolute_magnitude_h=25.9
        ),
        fetched_at=stamp,
        created_at=stamp,
        updated_at=stamp,
        sources_used=["SBDB"],
    )


@pytest.mark.parametrize("probability", ["missing", 0, 0.99, None])
def test_legacy_probability_never_manufactures_origin_or_confidence(probability):
    row = {"designation": "SYNTHETIC"}
    if probability != "missing":
        row["artificial_probability"] = probability
    dashboard = ArtificialNEODashboard()
    dashboard.console = None
    result = asyncio.run(dashboard._classify_neo(row))
    assert result.category == "inconclusive"
    assert result.confidence is None
    assert result.artificial_probability is None
    assert result.analysis_details["data_completeness"] is None
    assert result.screening_score is None


def test_typed_positive_uses_same_service_not_an_all_abstaining_dashboard():
    from aneos_core.analysis.screening import ScreeningAnalysisService

    n = neo()
    expected = ScreeningAnalysisService().analyze_neo(n).to_payload()
    d = ArtificialNEODashboard()
    d.console = None
    result = asyncio.run(
        d._classify_neo({"designation": n.designation, "neo_data": n.to_dict()})
    )
    assert result.screening_score == expected["screening_score"]
    assert result.screening_score is not None
    assert (
        result.analysis_details["assessment"]["evidence_id"] == expected["evidence_id"]
    )
    assert result.artificial_probability is None


def test_legacy_raw_orbit_is_retained_but_not_promoted_as_verified_measurements():
    row = {
        "designation": "SYNTHETIC",
        "orbital_elements": {"a": 1.23, "e": 0.2, "i": 1.49},
        "artificial_score": 0.99,
    }
    d = ArtificialNEODashboard()
    d.console = None
    result = asyncio.run(d._classify_neo(row))
    assert result.screening_score is None
    assert result.input_record == row
    assert result.input_status == "legacy_unverified"


@pytest.fixture
def menu():
    from aneos_menu import ANEOSMenu

    m = ANEOSMenu.__new__(ANEOSMenu)
    m.console = None
    m.wait_for_input = Mock()
    m.display_header = Mock()
    return m


def classes():
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    a = neo("FOLLOW_UP")
    a.orbital_elements = OrbitalElements(
        semi_major_axis=2.5, eccentricity=0.9, inclination=170
    )
    b = neo("NOT_FLAGGED")
    return [
        classify_record(a),
        classify_record(b),
        classify_record({"designation": "UNKNOWN", "artificial_probability": 0.999}),
    ]


def test_canonical_raw_assessment_replays_without_adopting_edited_origin_fields():
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    p = ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    expected = p["screening_score"]
    p["artificial_probability"] = 0.999
    p["classification"] = "ARTIFICIAL"
    c = classify_record({"designation": "SYNTHETIC", "screening_assessment": p})
    assert c.screening_score == expected and c.artificial_probability is None
    assert c.category != "artificial"
    assert c.input_record["screening_assessment"]["artificial_probability"] == 0.999


@pytest.mark.parametrize(
    "change",
    ["identity", "missing_field", "bad_number", "source_assessment_conflict", "policy"],
)
def test_invalid_typed_or_replay_inputs_abstain_and_retain_original(change):
    from aneos_core.analysis.artificial_neo_dashboard import classify_record
    from aneos_core.analysis.screening import ScreeningAnalysisService
    import copy

    n = neo()
    row = {"designation": n.designation, "neo_data": n.to_dict()}
    if change == "identity":
        row["designation"] = "OTHER"
    if change == "missing_field":
        row["neo_data"].pop("created_at")
    if change == "bad_number":
        row["neo_data"]["orbital_elements"]["a"] = "bad"
    if change == "source_assessment_conflict":
        row["screening_assessment"] = {"artificial_probability": 0.99}
    if change == "policy":
        row = {
            "designation": n.designation,
            "screening_assessment": ScreeningAnalysisService()
            .analyze_neo(n)
            .to_payload(),
        }
        row["screening_assessment"]["policy_id"] = "not-current"
    old = copy.deepcopy(row)
    c = classify_record(row)
    assert c.input_status == "invalid_input" and c.screening_score is None
    assert c.input_record == old and row == old


@pytest.mark.parametrize("raw", [None, [], True, 7, "text"])
def test_nonobject_inputs_are_retained_as_invalid_not_scored(raw):
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    c = classify_record(raw)
    assert c.input_status == "invalid_input"
    assert c.input_record == {"unverified_input": raw}


@pytest.mark.parametrize(
    "raw",
    [
        {"designation": "SYNTHETIC", "p": float("nan")},
        {"designation": "SYNTHETIC", "opaque": object()},
    ],
)
def test_unserializable_input_is_explicitly_not_archived_as_finite_evidence(raw):
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    c = classify_record(raw)
    assert c.input_status == "invalid_input"
    assert (
        c.input_record["retention_error"]
        == "NONFINITE_UNSERIALIZABLE_OR_OVERSIZE_INPUT"
    )


@pytest.mark.parametrize("where", ["packet", "metadata"])
def test_explicit_synthetic_marker_is_not_promoted_to_measurements(where):
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    n = neo()
    if where == "metadata":
        n.evidence_metadata["synthetic"] = True
    row = {"designation": n.designation, "neo_data": n.to_dict()}
    if where == "packet":
        row["is_synthetic"] = True
    c = classify_record(row)
    assert (
        c.screening_score is None and c.input_status == "declared_synthetic_unverified"
    )


def test_old_class_constructor_cannot_reintroduce_origin_defaults():
    from aneos_core.analysis.artificial_neo_dashboard import NEOClassification

    c = NEOClassification("SYNTHETIC", "artificial", 0.95, 0.999)
    assert (
        c.category == "inconclusive"
        and c.confidence is None
        and c.artificial_probability is None
    )
    assert c.input_record["category"] == "artificial"


def test_model_round_trip_and_mutation_guard():
    from aneos_core.analysis.artificial_neo_dashboard import (
        classify_record,
        verify_record,
        safe_classification,
    )

    c = classify_record(neo())
    p = c.to_payload()
    assert classify_record(p).to_payload() == p
    p["screening_score"] = 999
    with pytest.raises(ValueError):
        verify_record(p)
    c.input_record["edited"] = True
    assert safe_classification(c).input_status == "invalid_input"


def test_review_tampering_does_not_reach_sort_or_metrics():
    from aneos_core.analysis.artificial_neo_dashboard import (
        safe_classification,
        rank_key,
        classify_record,
    )

    c = classify_record(neo())
    c.review["assessment"]["artificial_probability"] = 0.99
    assert safe_classification(c).screening_score is None
    assert rank_key(c)[0] is True


def test_metric_denominators_and_unavailable_quality_confidence_are_explicit():
    d = ArtificialNEODashboard()
    items = classes()
    m = d._calculate_metrics(items + [items[0]])
    assert m.total_records == 4 and m.total_objects == 3
    assert m.follow_up_candidates == 2 and m.not_flagged == 1 and m.inconclusive == 1
    assert (
        m.artificial_confirmed is None
        and m.natural_objects is None
        and m.data_quality_score is None
    )
    assert m.records_with_inventory == 3
    assert d._calculate_metrics([]).total_records == 0


@pytest.mark.parametrize(
    "helper",
    [
        "_detect_orbital_anomalies",
        "_analyze_temporal_patterns",
        "_analyze_physical_characteristics",
    ],
)
def test_false_origin_helper_has_controlled_deprecation(helper):
    with pytest.raises(NotImplementedError):
        getattr(ArtificialNEODashboard(), helper)({})


def test_menu_helpers_never_use_name_based_defaults_or_probabilities(menu):
    items = classes()
    assert len(menu._check_threshold_violations(items)) == 1
    groups = menu._categorize_objects_enhanced(items)
    assert set(groups) == {"follow_up_candidate", "not_flagged", "inconclusive"}
    old = menu._convert_polling_to_classifications(
        [{"designation": "Tesla", "artificial_score": 0.99}]
    )[0]
    assert menu._extract_orbital_elements_from_object(old) == {}
    assert menu._extract_physical_data_from_object(old) == {}
    assert menu._get_data_quality_score(old) == "Not assessed"
    assert menu._get_data_sources_summary(old) == "Not verified"
    assert menu._get_physical_property_unit("diameter") == "not declared"
    assert menu._get_physical_property_unit("diameter_km") == "km"
    assert "fields" in menu._get_enriched_data_summary(items[0])
    assert menu._assess_impact_risk(old) == "NOT_ASSESSED"


@pytest.mark.parametrize(
    "choice,count",
    [
        ("1", 1),
        ("2", 1),
        ("3", 0),
        ("4", 1),
        ("5", 2),
        ("6", 2),
        ("7", 0),
        ("8", None),
        ("9", None),
        ("0", None),
        ("bad", None),
    ],
)
def test_each_filter_option_positive_missing_and_unsupported_is_readable(
    menu, monkeypatch, choice, count
):
    monkeypatch.setattr("builtins.input", lambda _: choice)
    result = menu._enhanced_filter_menu(classes())
    assert (len(result) if result is not None else None) == count


@pytest.mark.parametrize("choice", ["1", "2", "3", "4", "5", "6", "0", "bad"])
def test_each_sort_option_handles_unavailable_fields_and_correct_date_role(
    menu, monkeypatch, choice, capsys
):
    monkeypatch.setattr("builtins.input", lambda _: choice)
    rows, label = menu._enhanced_sort_menu(classes())
    if choice in ("0", "bad"):
        assert rows is None
    else:
        assert len(rows) == 3
        if choice == "1":
            assert rows[-1].screening_score is None
        if choice == "4":
            assert "not discovery" in label
    assert "Artificial Probability" not in capsys.readouterr().out


def test_actual_browser_navigation_filters_sort_details_export_and_empty_view(
    menu, monkeypatch
):
    actions = iter(
        [
            "N",
            "P",
            "HOME",
            "END",
            "G1",
            "G999",
            "C",
            "1",
            "999",
            "bad",
            "E",
            "S",
            "1",
            "F",
            "4",
            "0",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(actions))
    menu._show_detailed_neo_view = Mock()
    menu._export_current_view = Mock()
    menu._norton_style_browser(classes(), "Synthetic workflow")
    assert menu._show_detailed_neo_view.call_count == 1
    assert menu._export_current_view.call_count == 1
    monkeypatch.setattr("builtins.input", lambda _: "Q")
    menu._norton_style_browser([])


def test_actual_interactive_dashboard_dispatch(menu, monkeypatch):
    d = ArtificialNEODashboard()
    d.classifications = classes()
    d.display_dashboard = Mock()
    menu._browse_enriched_database = Mock()
    responses = iter(["1", "Q", "2", "Q", "3", "Q", "4", "Q", "5", "6", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    menu._dashboard_interactive_menu(d)
    assert d.display_dashboard.call_count == 1
    assert menu._browse_enriched_database.call_count == 1


@pytest.mark.parametrize("width", [80, 120, 160])
def test_actual_dashboard_rich_and_browser_detail_fit_terminal_width(
    width, menu, monkeypatch
):
    from rich.console import Console
    from rich.cells import cell_len
    import io

    stream = io.StringIO()
    console = Console(file=stream, width=width, height=25, color_system=None)
    d = ArtificialNEODashboard(console=console)
    d.classifications = classes()
    d.display_dashboard(save_results=False)
    menu.console = console
    menu._show_detailed_neo_view(d.classifications[0])
    responses = iter(["C", "Q"])
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: next(responses))
    menu._norton_style_browser(d.classifications)
    output = stream.getvalue()
    assert max(map(cell_len, output.splitlines())) <= width
    assert "Not flagged does not mean natural" in " ".join(output.split())
    assert "Confirmed Artificial" not in output and "σ-Prob" not in output
    assert "ranking units" in output
    from aneos_version import APPLICATION_VERSION

    assert APPLICATION_VERSION in output


def test_safe_labels_and_legacy_raw_details_are_explicit_not_terminal_markup(
    menu, capsys
):
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    row = {
        "designation": "[red]SYNTHETIC[/red]",
        "artificial_probability": 0.99,
        "text": "\x1b[31m```<script>",
    }
    c = classify_record(row)
    menu._show_object_details(c)
    text = capsys.readouterr().out
    assert "\x1b" not in text
    assert "not adopted" in text
    from aneos_core.analysis.dashboard_view import dashboard_markdown

    md = dashboard_markdown([c])
    assert "<script>" not in md and "\\u0060" in md


def test_exclusive_exports_roundtrip_raw_inputs_and_reproducibility(
    menu, tmp_path, monkeypatch
):
    from aneos_core.analysis.artificial_neo_dashboard import records_from_file
    import json

    monkeypatch.chdir(tmp_path)
    d = ArtificialNEODashboard()
    d.console = None
    d.classifications = classes()
    first = d._save_dashboard_results("dashboard_results")
    before = (first / "report.json").read_bytes()
    second = d._save_dashboard_results("dashboard_results")
    assert first != second and (first / "report.json").read_bytes() == before
    assert (first / "complete.json").exists() and (first / "report.md").exists()
    data = json.loads(before)
    assert len(records_from_file(data)) == 3
    from hashlib import sha256

    for name, digest in json.loads((first / "complete.json").read_text()).items():
        assert sha256((first / name).read_bytes()).hexdigest() == digest
    menu._norton_style_browser = Mock()
    menu._load_and_browse_results(first / "report.json")
    assert menu._norton_style_browser.call_count == 1
    assert menu._norton_style_browser.call_args.args[0][0].screening_score is not None
    result = menu._export_current_view(classes(), "page_1")
    assert result and (result / "complete.json").exists()
    data["records"][0]["artificial_probability"] = 0.99
    with pytest.raises(ValueError):
        records_from_file(data)


@pytest.mark.parametrize(
    "data",
    [
        [{"designation": "SYNTHETIC", "artificial_probability": 0.99}],
        {
            "classifications": [
                {
                    "designation": "SYNTHETIC",
                    "category": "artificial",
                    "confidence": 0.95,
                    "artificial_probability": 0.99,
                    "analysis_timestamp": "2000-01-01",
                }
            ]
        },
        {"results": [{"designation": "SYNTHETIC"}]},
        {"objects": [{"designation": "SYNTHETIC"}]},
    ],
)
def test_legacy_file_shapes_preserve_claims_without_adopting_or_modifying_them(data):
    from aneos_core.analysis.artificial_neo_dashboard import records_from_file
    import copy

    old = copy.deepcopy(data)
    items = ArtificialNEODashboard().classify_records(records_from_file(data))
    assert items[0].category == "inconclusive"
    assert data == old


def test_incomplete_typed_data_retains_inventory_without_origin_or_rank():
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    n = neo()
    n.orbital_elements = OrbitalElements(semi_major_axis=1.23)
    c = classify_record(n)
    assert c.category == "inconclusive" and c.screening_score is None
    assert c.basic_presence is not None


def test_source_and_assessment_matching_positive():
    from aneos_core.analysis.artificial_neo_dashboard import classify_record
    from aneos_core.analysis.screening import ScreeningAnalysisService

    n = neo()
    p = ScreeningAnalysisService().analyze_neo(n).to_payload()
    c = classify_record(
        {
            "designation": n.designation,
            "neo_data": n.to_dict(),
            "screening_assessment": p,
        }
    )
    assert c.screening_score == p["screening_score"]


@pytest.mark.parametrize("stamp", ["2026-11-13T00:00:00", "2026-11-13T00:00:00+02:00"])
def test_unverified_or_non_utc_record_receipt_does_not_replay(stamp):
    from aneos_core.analysis.artificial_neo_dashboard import (
        classify_record,
        verify_record,
    )

    p = classify_record(neo()).to_payload()
    p["analysis_timestamp"] = stamp
    with pytest.raises(ValueError):
        verify_record(p)


def test_foreign_review_identity_and_record_schema_are_rejected():
    from aneos_core.analysis.artificial_neo_dashboard import (
        NEOClassification,
        classify_record,
        verify_record,
    )

    c = classify_record(neo())
    with pytest.raises(ValueError):
        NEOClassification("OTHER", review=c.review)
    with pytest.raises(ValueError):
        verify_record({})


@pytest.mark.parametrize("value", [{}, None, "text", {"results": None}])
def test_records_from_file_rejects_wrong_container_or_returns_unverified_object(value):
    from aneos_core.analysis.artificial_neo_dashboard import records_from_file

    if value == {}:
        assert records_from_file(value) == [{}]
    else:
        with pytest.raises(ValueError):
            records_from_file(value)


def test_invalid_file_is_not_modified_and_failed_export_does_not_announce_success(
    menu, tmp_path, monkeypatch, capsys
):
    import json

    path = tmp_path / "legacy.json"
    path.write_text("bad json")
    before = path.read_bytes()
    menu._load_and_browse_results(path)
    assert path.read_bytes() == before
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    original = Path.write_text

    def fail_markdown(self, *a, **kw):
        if self.name == "report.md":
            raise OSError("secret-fixture")
        return original(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", fail_markdown)
    assert menu._export_current_view(classes(), "page") is None
    assert not list((tmp_path / "dashboard_results").glob("*/complete.json"))
    assert "secret-fixture" not in capsys.readouterr().out


def test_display_empty_and_compatibility_render_methods(capsys):
    d = ArtificialNEODashboard()
    d.console = None
    d.display_dashboard(save_results=False)
    d._display_category_details()
    d._display_risk_assessment()
    assert "Input records: 0" in capsys.readouterr().out
    with pytest.raises(ValueError):
        d.classify_records({})


def test_real_integration_factory_does_not_acquire_or_require_live_sources(monkeypatch):
    from aneos_core.analysis.artificial_neo_dashboard import (
        create_dashboard_from_polling,
    )

    monkeypatch.setattr(
        "aneos_core.data.fetcher.DataFetcher",
        Mock(side_effect=AssertionError("renderer cannot query providers")),
    )
    d = asyncio.run(
        create_dashboard_from_polling(
            [{"designation": "SYNTHETIC", "neo_data": neo().to_dict()}],
            display=False,
            save=False,
        )
    )
    assert d.classifications[0].screening_score is not None
    assert d.metrics.artificial_confirmed is None


def test_standalone_file_launcher_uses_same_records_and_new_bundle_discovery(
    tmp_path, monkeypatch
):
    from artificial_neo_dashboard_launcher import (
        launch_dashboard_from_file,
        find_recent_polling_results,
    )
    import json

    d = ArtificialNEODashboard()
    d.console = None
    d.classifications = classes()
    report = d._save_dashboard_results(tmp_path / "dashboard_results")
    assert str(report / "report.json") in find_recent_polling_results(str(tmp_path))
    # Keep real file decode + classify, suppress only rendering/export side effects.
    original = ArtificialNEODashboard.display_dashboard
    monkeypatch.setattr(
        ArtificialNEODashboard, "display_dashboard", lambda *a, **kw: None
    )
    loaded = asyncio.run(launch_dashboard_from_file(str(report / "report.json")))
    assert loaded.classifications[0].screening_score is not None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert asyncio.run(launch_dashboard_from_file(str(bad))) is None


def test_recomputed_legacy_completeness_cache_is_not_quality_or_missing_primary_data():
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    n = neo()
    n.orbital_elements = OrbitalElements(semi_major_axis=1.23)
    original = n.to_dict()
    original["completeness"] = 0.999
    c = classify_record({"designation": n.designation, "neo_data": original})
    assert c.input_status == "typed_snapshot_not_authenticated"
    assert c.input_record["neo_data"]["completeness"] == 0.999
    assert c.basic_presence is not None and c.basic_presence != 0.999
    assert c.analysis_details["data_completeness"] is None


def test_source_listed_filter_has_authorized_positive_and_identity_negative(
    menu, monkeypatch
):
    from aneos_core.data.sources.sentry import parse_sentry_response
    from aneos_core.analysis.artificial_neo_dashboard import classify_record
    import json

    native = {
        "signature": {"version": "2.0", "source": "NASA/JPL Sentry Data API"},
        "summary": {
            "des": "1998 KY26",
            "fullname": "(1998 KY26)",
            "method": "IOBS",
            "ip": "1e-7",
            "n_imp": "1",
            "first_obs": "1998-05-31",
            "last_obs": "2024-05-06",
            "cdate": "2024-05-07 13:14:15",
            "pdate": "2024-05-07 20:14:15",
            "diameter": ".011",
            "mass": "1000",
            "energy": ".02",
            "ts_max": "0",
            "ps_cum": "-6",
        },
        "data": [
            {
                "date": "2050-01-01.25",
                "ip": "1e-7",
                "energy": ".01",
                "ps": "-7",
                "ts": "0",
                "sigma_vi": "1.2",
            }
        ],
    }
    report = parse_sentry_response(
        json.dumps(native).encode(),
        designation="1998 KY26",
        retrieved_at="2026-11-13T08:00:00+00:00",
    )
    assert report["status"] == "available"
    positive = classify_record({"designation": "1998 KY26", "impact_evidence": report})
    negative = classify_record({"designation": "OTHER", "impact_evidence": report})
    monkeypatch.setattr("builtins.input", lambda _: "3")
    assert menu._enhanced_filter_menu([positive, negative]) == [positive]
    assert positive.artificial_probability is None


def test_current_qa_captures_actual_three_views_and_all_terminal_geometries(tmp_path):
    from scripts.qa_polling_dashboard import capture_all
    from hashlib import sha256

    directory = tmp_path / "capture"
    manifest = capture_all(directory)
    assert len(manifest["widths"]) == 27
    for entry in manifest["widths"].values():
        assert entry["actual_text_max_cells"] <= entry["requested_columns"]
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        capture_all(directory)


def test_typed_report_preserves_additional_evidence_across_entry_points():
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.analysis.artificial_neo_dashboard import classify_record

    n = neo()
    extra = {
        "orbital_history": [
            {"a": 1.23, "e": 0.2, "i": 1.49, "basis": "computed_ephemeris"}
        ]
    }
    assessment = ScreeningAnalysisService().analyze_neo(n, extra).to_payload()
    c = classify_record(
        {
            "designation": n.designation,
            "neo_data": n.to_dict(),
            "screening_assessment": assessment,
        }
    )
    assert c.input_status == "typed_snapshot_not_authenticated"
    assert c.analysis_details["assessment"] == assessment


def test_existing_standalone_assessment_and_frozen_object_review_are_supported():
    from aneos_core.analysis.artificial_neo_dashboard import classify_record
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.analysis.object_review import build_object_review

    n = neo()
    p = ScreeningAnalysisService().analyze_neo(n).to_payload()
    assert classify_record(p).screening_score == p["screening_score"]
    r = build_object_review(n)
    c = classify_record(r)
    assert c.review == r and c.input_record == r
    assert c.artificial_probability is None


@pytest.mark.parametrize("form", ["assessment", "review"])
def test_explicit_synthetic_provenance_is_not_bypassed_by_a_canonical_container(form):
    from aneos_core.analysis.artificial_neo_dashboard import classify_record
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.analysis.object_review import build_object_review

    n = neo()
    n.evidence_metadata["synthetic"] = True
    value = (
        ScreeningAnalysisService().analyze_neo(n).to_payload()
        if form == "assessment"
        else build_object_review(n)
    )
    result = classify_record(value)
    assert result.screening_score is None
    assert result.input_status == "declared_synthetic_unverified"
    assert result.input_record == value


def test_frozen_multiple_policies_round_trip_without_current_policy_overwrite(tmp_path):
    from aneos_core.analysis.screening import ScreeningPolicy
    from aneos_core.analysis.object_review import build_object_review
    from aneos_core.analysis.artificial_neo_dashboard import (
        classify_record,
        records_from_file,
    )
    import json

    n = neo()
    default = classify_record(n)
    frozen = classify_record(
        build_object_review(n, policy=ScreeningPolicy(candidate_threshold=1.0))
    )
    assert (
        frozen.category == "follow_up_candidate" and default.category == "not_flagged"
    )
    d = ArtificialNEODashboard()
    d.console = None
    d.classifications = [default, frozen]
    path = d._save_dashboard_results(tmp_path / "reports")
    data = json.loads((path / "report.json").read_text())
    assert len(data["policies"]) == 2
    records = records_from_file(data)
    assert [r["category"] for r in records] == ["not_flagged", "follow_up_candidate"]
    assert all(r["artificial_probability"] is None for r in records)


def test_default_factory_display_and_save_is_a_real_positive_path(
    tmp_path, monkeypatch
):
    from aneos_core.analysis.artificial_neo_dashboard import (
        create_dashboard_from_polling,
        records_from_file,
    )
    import json

    monkeypatch.chdir(tmp_path)
    result = asyncio.run(
        create_dashboard_from_polling(
            [{"designation": "SYNTHETIC", "neo_data": neo().to_dict()}]
        )
    )
    assert result.classifications[0].screening_score is not None
    files = list((tmp_path / "dashboard_results").glob("*/report.json"))
    assert len(files) == 1 and (files[0].parent / "complete.json").exists()
    loaded = records_from_file(json.loads(files[0].read_text()))
    assert loaded[0]["artificial_probability"] is None


def test_browser_rejects_overlong_and_non_decimal_number_commands(
    menu, monkeypatch, capsys
):
    responses = iter(["9" * 5000, "G²", "²", " Q "])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    menu._norton_style_browser(classes())
    assert "too long" in capsys.readouterr().out
