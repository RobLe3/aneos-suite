"""Reachable policy menu and file actions, not static wording-only tests."""

import io
from unittest.mock import Mock
import pytest
from rich.console import Console


def menu():
    from aneos_menu import ANEOSMenu

    ui = ANEOSMenu.__new__(ANEOSMenu)
    ui.console = Console(file=io.StringIO(), width=120, color_system=None)
    ui.wait_for_input = Mock()
    ui.show_error = Mock()
    return ui


def test_actual_policy_menu_does_not_claim_calibration_or_natural_origin(monkeypatch):
    from rich.prompt import Confirm, Prompt

    monkeypatch.setattr(Confirm, "ask", lambda *a, **k: False)
    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: "0")
    ui = menu()
    ui.configure_analysis()
    text = ui.console.file.getvalue()
    assert "99.99994%" not in text
    assert "artificial classification" not in text
    assert "natural classification" not in text
    assert "uncalibrated" in text.lower()
    assert "policy" in text.lower()


def test_legacy_alpha_helper_cannot_claim_to_apply_throwaway_settings(monkeypatch):
    from rich.prompt import FloatPrompt

    monkeypatch.setattr(FloatPrompt, "ask", lambda *a, **k: 0.01)
    ui = menu()
    config = {"alpha_level": 0.05}
    ui._configure_alpha_level(config)
    assert config == {"alpha_level": 0.05}
    assert "not applied" in ui.console.file.getvalue().lower()


def test_export_uses_explicit_exclusive_path_and_real_active_policy(tmp_path):
    ui = menu()
    path = tmp_path / "policy.json"
    result = ui._export_configuration({"alpha_level": 0.0001}, path=path)
    assert result["status"] == "exported"
    original = path.read_bytes()
    assert ui._export_configuration({}, path=path)["status"] == "not_exported"
    assert path.read_bytes() == original


def test_policy_description_is_actual_assessment_identity_and_owned_copy():
    from aneos_core.analysis.screening import ScreeningAnalysisService

    service = ScreeningAnalysisService()
    before = service.analyze_raw("synthetic-policy", {"a": 1.2, "e": 0.2, "i": 20.0})
    description = service.describe_policy()
    assert description["policy_id"] == before.policy_id
    description["reference_parameters"]["physical_properties"]["diameter"]["mean"] = 1
    assert (
        service.describe_policy()["reference_parameters"]["physical_properties"][
            "diameter"
        ]["mean"]
        == 800
    )
    assert (
        service.analyze_raw(
            "synthetic-policy", {"a": 1.2, "e": 0.2, "i": 20.0}
        ).policy_id
        == before.policy_id
    )


def test_archived_custom_policy_verifies_but_is_never_applied():
    from aneos_core.analysis.screening import ScreeningAnalysisService, ScreeningPolicy
    from aneos_core.analysis.policy_inspection import (
        build_policy_snapshot,
        verify_policy_snapshot,
    )

    current = ScreeningAnalysisService()
    before = current.describe_policy()
    archived = build_policy_snapshot(
        ScreeningAnalysisService(policy=ScreeningPolicy(candidate_threshold=7))
    )
    result = verify_policy_snapshot(archived, service=current)
    assert result["comparison"] == "differs_from_active"
    assert result["runtime_configuration_changed"] is False
    assert current.describe_policy() == before


def test_real_synthetic_controls_have_authorized_positive_and_missing_abstention():
    ui = menu()
    result = ui._test_validated_detector()
    assert result["control_passed"] is True
    assert len(result["controls"]) == 3
    positive, ordinary, missing = result["controls"]
    assert positive["candidate"] and not ordinary["candidate"]
    assert missing["score"] is None and missing["classification"] == "INCONCLUSIVE"
    assert not result["five_sigma_claim_permitted"]
    assert "not observations" in ui.console.file.getvalue()


def test_synthetic_control_failure_is_not_reported_as_success(monkeypatch):
    from aneos_core.analysis import policy_inspection

    class Broken:
        def analyze_raw(self, *a, **k):
            raise RuntimeError("private detail")

    monkeypatch.setattr(policy_inspection, "ScreeningAnalysisService", Broken)
    ui = menu()
    result = ui._test_validated_detector()
    assert not result["control_passed"] and result["status"] == "unavailable"
    assert "private detail" not in ui.console.file.getvalue()


@pytest.mark.parametrize("option", list("123456"))
def test_reachable_configuration_actions_dispatch_with_no_false_application(
    monkeypatch, option, tmp_path
):
    from rich.prompt import Prompt

    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: option)
    ui = menu()
    ui.get_input = Mock(return_value="")
    result = ui.configure_analysis()
    assert (
        result["status"]
        in {"unsupported_not_applied", "read_only_not_applied", "cancelled"}
        if option != "4"
        else result["control_passed"]
    )
    text = ui.console.file.getvalue()
    assert "99.99994%" not in text and "Thresholds updated" not in text
    assert not ui.show_error.called
    ui.wait_for_input.assert_called_once()


def test_plain_terminal_actual_menu_supports_inspection_and_controls(
    monkeypatch, capsys
):
    monkeypatch.setattr("builtins.input", lambda *a: "4")
    ui = menu()
    ui.console = None
    result = ui.configure_analysis()
    assert result["control_passed"]
    assert "uncalibrated" in capsys.readouterr().out.lower()


def test_main_menu_failure_is_explicit_and_does_not_prompt_or_leak(monkeypatch):
    from aneos_core.analysis import policy_inspection

    monkeypatch.setattr(
        policy_inspection,
        "build_policy_snapshot",
        Mock(side_effect=RuntimeError("private paths")),
    )
    ui = menu()
    result = ui.configure_analysis()
    assert result["status"] == "unavailable"
    assert "private paths" not in ui.console.file.getvalue()
    ui.wait_for_input.assert_called_once()


def test_threshold_and_bic_helpers_leave_actual_policy_and_required_config_unchanged():
    from aneos_core.analysis.screening import ScreeningAnalysisService

    ui = menu()
    config = {"enable_delta_bic": True, "required_stages": [1, 2, 3, 4, 5]}
    service = ScreeningAnalysisService()
    before = service.describe_policy()
    assert ui._toggle_delta_bic(config)["runtime_configuration_changed"] is False
    assert (
        ui._configure_detection_thresholds()["runtime_configuration_changed"] is False
    )
    assert config == {"enable_delta_bic": True, "required_stages": [1, 2, 3, 4, 5]}
    assert service.describe_policy() == before


def test_snapshot_export_verify_actual_menu_file_actions_restart(tmp_path):
    import json, subprocess, sys, os

    path = tmp_path / "policy.json"
    ui = menu()
    export = ui._export_configuration({"secret": "never exported"}, path=path)
    assert export["status"] == "exported"
    assert "secret" not in path.read_text()
    result = ui._import_configuration(path=path)
    assert (
        result["status"] == "verified_snapshot_not_applied"
        and result["comparison"] == "matches_active"
    )
    code = """
from aneos_core.analysis.policy_inspection import parse_snapshot_json
import json,sys
_,result=parse_snapshot_json(open(sys.argv[1]).read())
print(json.dumps(result))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert json.loads(proc.stdout)["comparison"] == "matches_active"
    assert "Not imported or applied" in ui.console.file.getvalue()


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        "[]",
        "null",
        "not json",
        '{"a":1,"a":2}',
        '{"exported":"old","validation_config":{"alpha_level":0.01}}',
        "x" * 65537,
        '{"candidate_threshold":NaN}',
    ],
)
def test_invalid_or_legacy_imports_never_apply(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    ui = menu()
    assert ui._import_configuration(path=path)["status"] == "not_verified"
    assert path.read_text() == payload


def test_missing_file_and_invalid_utf8_do_not_apply(tmp_path):
    ui = menu()
    assert (
        ui._import_configuration(path=tmp_path / "missing")["status"] == "not_verified"
    )
    p = tmp_path / "binary"
    p.write_bytes(b"\xff")
    assert ui._import_configuration(path=p)["status"] == "not_verified"


def test_snapshot_mutation_is_rejected_even_when_rehashed():
    from aneos_core.analysis.policy_inspection import (
        build_policy_snapshot,
        verify_policy_snapshot,
        fingerprint,
    )

    snapshot = build_policy_snapshot()
    snapshot["capabilities"]["gaussian_significance"] = "validated"
    snapshot["snapshot_id"] = fingerprint(
        {k: v for k, v in snapshot.items() if k != "snapshot_id"}
    )
    with pytest.raises(ValueError, match="capability"):
        verify_policy_snapshot(snapshot)


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_threshold", 0),
        ("candidate_threshold", -1),
        ("candidate_threshold", True),
        ("candidate_threshold", "5"),
        ("version", ""),
        ("version", None),
    ],
)
def test_invalid_policy_fields_rejected_after_consistent_checksum(field, value):
    from aneos_core.analysis.policy_inspection import (
        build_policy_snapshot,
        verify_policy_snapshot,
        fingerprint,
    )

    snapshot = build_policy_snapshot()
    d = snapshot["policy_description"]
    d["policy"][field] = value
    d["policy_id"] = fingerprint({k: v for k, v in d.items() if k != "policy_id"})
    snapshot["snapshot_id"] = fingerprint(
        {k: v for k, v in snapshot.items() if k != "snapshot_id"}
    )
    with pytest.raises(ValueError):
        verify_policy_snapshot(snapshot)


def test_cancelled_file_actions_do_not_create_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ui = menu()
    ui.get_input = Mock(return_value="")
    assert ui._export_configuration({})["status"] == "cancelled"
    assert ui._import_configuration()["status"] == "cancelled"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "legacy"),
        ("application_version", None),
        ("created_at", None),
        ("created_at", "2026-11-13T00:00:00"),
        ("created_at", "2026-11-13T01:00:00+01:00"),
        ("policy_description", []),
        ("snapshot_id", "bad"),
    ],
)
def test_top_level_snapshot_contract_is_checked(field, value):
    from aneos_core.analysis.policy_inspection import (
        build_policy_snapshot,
        verify_policy_snapshot,
        fingerprint,
    )

    s = build_policy_snapshot()
    s[field] = value
    if field != "snapshot_id":
        s["snapshot_id"] = fingerprint(
            {k: v for k, v in s.items() if k != "snapshot_id"}
        )
    with pytest.raises(ValueError):
        verify_policy_snapshot(s)


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy", {}),
        ("reference_parameters", []),
        ("engine_version", ""),
        ("policy_id", "bad"),
    ],
)
def test_policy_description_contract_is_checked(field, value):
    from aneos_core.analysis.policy_inspection import (
        build_policy_snapshot,
        verify_policy_snapshot,
        fingerprint,
    )

    s = build_policy_snapshot()
    s["policy_description"][field] = value
    s["snapshot_id"] = fingerprint({k: v for k, v in s.items() if k != "snapshot_id"})
    with pytest.raises(ValueError):
        verify_policy_snapshot(s)


def test_size_finite_and_duplicate_key_contracts():
    from aneos_core.analysis.policy_inspection import canonical, parse_snapshot_json

    with pytest.raises(ValueError):
        canonical({"oversize": "x" * 65537})
    with pytest.raises(ValueError):
        canonical({"invalid": float("nan")})
    with pytest.raises(ValueError):
        parse_snapshot_json(None)
    with pytest.raises(ValueError):
        parse_snapshot_json('{"nested":{"x":1,"x":2}}')


def test_wrong_synthetic_result_is_failure_not_only_exception():
    from aneos_core.analysis.policy_inspection import synthetic_controls
    from aneos_core.analysis.screening import ScreeningAnalysisService, ScreeningPolicy

    result = synthetic_controls(
        ScreeningAnalysisService(policy=ScreeningPolicy(candidate_threshold=100))
    )
    assert result["control_passed"] is False
    assert result["controls"][0]["passed"] is False


def test_invalid_plain_action_is_not_success(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a: "999")
    ui = menu()
    ui.console = None
    assert ui.configure_analysis()["status"] == "invalid_action"
    assert "nothing changed" in capsys.readouterr().out


def test_export_io_failure_does_not_claim_success_or_remove_partial_file(
    monkeypatch, tmp_path
):
    import os

    monkeypatch.setattr(
        os, "fsync", Mock(side_effect=OSError("private filesystem detail"))
    )
    ui = menu()
    p = tmp_path / "partial.json"
    result = ui._export_configuration({}, path=p)
    assert result["status"] == "not_exported"
    assert p.exists()
    assert "partial new file" in ui.console.file.getvalue()
    assert "private filesystem detail" not in ui.console.file.getvalue()


def test_export_does_not_follow_existing_symlink(tmp_path):
    target = tmp_path / "original"
    target.write_text("retained")
    link = tmp_path / "alias"
    link.symlink_to(target)
    assert menu()._export_configuration({}, path=link)["status"] == "not_exported"
    assert target.read_text() == "retained"


def test_export_path_markup_and_control_characters_are_displayed_literally(tmp_path):
    ui = menu()
    path = tmp_path / "[bold]policy\x1b.json"
    assert ui._export_configuration({}, path=path)["status"] == "exported"
    text = ui.console.file.getvalue()
    assert "[bold]policy\\x1b.json" in text
    assert "\x1b" not in text


def test_actual_snapshot_capture_hashes_and_widths(tmp_path):
    import subprocess, sys, json
    from pathlib import Path
    from hashlib import sha256

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "capture"
    command = [
        sys.executable,
        str(root / "scripts/qa_policy_menu.py"),
        "--output",
        str(out),
    ]
    subprocess.run(command, cwd=root, capture_output=True, check=True)
    manifest = json.loads((out / "inventory.json").read_text())
    for mode in ("inspection", "controls", "unavailable"):
        for width in (80, 120, 160):
            text = (out / f"{mode}-{width}.txt").read_text()
            assert manifest["metrics"][f"{mode}-{width}"]["max_columns"] <= width
            assert "99.99994%" not in text
            if mode == "inspection":
                assert "not Gaussian sigma" in " ".join(text.split())
            if mode == "controls":
                assert "positive" in text and "missing" in text
            if mode == "unavailable":
                assert "unavailable" in text
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256((out / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["source_sha256"].items():
        assert sha256((root / name).read_bytes()).hexdigest() == digest
    assert subprocess.run(command, cwd=root, capture_output=True).returncode != 0


def test_documented_explicit_file_precedence_is_measured_not_assumed(
    monkeypatch, tmp_path
):
    import json
    from aneos_core.config.settings import ConfigManager

    monkeypatch.setenv("ANEOS_REQUEST_TIMEOUT", "99")
    p = tmp_path / "configuration.json"
    p.write_text(json.dumps({"api": {"request_timeout": 7}}))
    manager = ConfigManager(config_file=str(p))
    assert manager.config.api.request_timeout == 7
    p.write_text("invalid JSON")
    with pytest.raises(ValueError):
        ConfigManager(config_file=str(p))


def test_invalid_general_config_update_is_atomic():
    from aneos_core.config.settings import ConfigManager

    manager = ConfigManager.__new__(ConfigManager)
    from aneos_core.config.settings import ANEOSConfig

    manager._config = ANEOSConfig()
    with pytest.raises(ValueError):
        manager.update_config(max_workers=-1)
    assert manager.config.max_workers == 10


def test_explicit_missing_file_does_not_fall_back_to_environment(tmp_path, monkeypatch):
    from aneos_core.config.settings import ConfigManager, ANEOSConfig

    monkeypatch.setenv("ANEOS_REQUEST_TIMEOUT", "99")
    with pytest.raises(FileNotFoundError):
        ConfigManager(config_file=str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        ANEOSConfig.from_file(str(tmp_path / "missing.json"))


def test_rejected_reload_preserves_prior_configuration_identity(tmp_path, monkeypatch):
    from aneos_core.config.settings import ConfigManager

    p = tmp_path / "valid.json"
    p.write_text('{"max_workers":3}')
    manager = ConfigManager(config_file=str(p))
    before = manager.config
    p.write_text('{"max_workers":-1}')
    with pytest.raises(ValueError):
        manager.reload()
    assert manager.config is before and manager.config.max_workers == 3
    p.write_text("bad JSON")
    with pytest.raises(ValueError):
        manager.reload()
    assert manager.config is before


def test_valid_configuration_update_and_reload_preserve_separate_loading_contract(
    tmp_path,
):
    from aneos_core.config.settings import ConfigManager

    p = tmp_path / "valid.json"
    p.write_text('{"max_workers":3}')
    manager = ConfigManager(config_file=str(p))
    before = manager.config
    manager.update_config(max_workers=4)
    assert manager.config.max_workers == 4
    assert before.max_workers == 3
    assert p.read_text() == '{"max_workers":3}'
    manager.reload()
    assert manager.config.max_workers == 3


def test_rejected_nested_or_unknown_update_keeps_previous_object():
    from aneos_core.config.settings import ConfigManager, ANEOSConfig, APIConfig

    manager = ConfigManager.__new__(ConfigManager)
    manager._config = ANEOSConfig()
    before = manager.config
    with pytest.raises(ValueError):
        manager.update_config(api=APIConfig(request_timeout=-1))
    assert manager.config is before and manager.config.api.request_timeout == 10
    with pytest.raises(ValueError):
        manager.update_config(unknown_option=1)
    assert manager.config is before


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "4", 4.5])
def test_worker_update_rejects_nonfinite_coerced_or_nonpositive_values_atomically(
    value,
):
    from aneos_core.config.settings import ConfigManager, ANEOSConfig

    manager = ConfigManager.__new__(ConfigManager)
    manager._config = ANEOSConfig()
    before = manager.config
    with pytest.raises(ValueError):
        manager.update_config(max_workers=value)
    assert manager.config is before


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, "1"])
def test_invalid_api_retry_field_is_rejected_atomically(value):
    from aneos_core.config.settings import ConfigManager, ANEOSConfig, APIConfig

    manager = ConfigManager.__new__(ConfigManager)
    manager._config = ANEOSConfig()
    before = manager.config
    with pytest.raises(ValueError):
        manager.update_config(api=APIConfig(max_retries=value))
    assert manager.config is before


def test_dict_api_cannot_replace_typed_configuration_or_mutate_prior_state():
    from aneos_core.config.settings import ConfigManager, ANEOSConfig

    manager = ConfigManager.__new__(ConfigManager)
    manager._config = ANEOSConfig()
    before = manager.config
    with pytest.raises(ValueError):
        manager.update_config(api={"request_timeout": 7})
    assert manager.config is before


@pytest.mark.parametrize(
    "text", ["[]", "null", '{"typo_max_workers":7}', '{"api":{"unknown":1}}']
)
def test_unknown_or_nonobject_explicit_configuration_is_rejected(tmp_path, text):
    from aneos_core.config.settings import ConfigManager

    p = tmp_path / "invalid.json"
    p.write_text(text)
    with pytest.raises(ValueError):
        ConfigManager(config_file=str(p))


def test_actual_legacy_submenu_route_reaches_policy_view_without_core_claims(
    monkeypatch,
):
    from rich.prompt import Prompt

    choices = iter(["9", "0", "0"])  # detection submenu -> policy -> return -> parent
    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: next(choices))
    ui = menu()
    ui.display_header = Mock()
    ui.neo_detection_menu()
    text = ui.console.file.getvalue()
    assert (
        "Analysis Policy" in text
        and "Active default canonical screening policy" in text
    )
    assert "99.99994%" not in text


def test_actual_menu_export_and_verify_prompts_use_real_file_actions(
    monkeypatch, tmp_path
):
    from rich.prompt import Prompt

    path = tmp_path / "actual-menu.json"
    ui = menu()
    choices = iter(["5", str(path)])
    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: next(choices))
    assert ui.configure_analysis()["status"] == "exported"
    choices = iter(["6", str(path)])
    assert ui.configure_analysis()["status"] == "verified_snapshot_not_applied"


def test_launcher_help_does_not_invent_legacy_feature_coverage(capsys):
    import aneos

    aneos.print_help()
    text = capsys.readouterr().out
    assert "121-option" not in text
    assert "feature QA remains pending" in text


@pytest.mark.parametrize("keyword", ["config_file", "config_path"])
def test_explicit_blank_configuration_path_is_not_silently_treated_as_default(keyword):
    from aneos_core.config.settings import ConfigManager

    with pytest.raises(ValueError):
        ConfigManager(**{keyword: ""})
