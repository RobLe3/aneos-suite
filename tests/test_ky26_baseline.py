"""Offline historical preservation; no current scorer or provider required."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def replay():
    spec = importlib.util.spec_from_file_location('ky26_replay', ROOT / 'scripts/reproduce_ky26_baseline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reproduce()


def test_frozen_source_is_identified_and_rounded_replay_is_stable():
    fixture = json.loads((ROOT / 'tests/fixtures/ky26_baseline.json').read_text())
    result = replay()
    expected = fixture['reproduction']['rounded_input_expected']
    for name in ['sigma_confidence', 'bayesian_probability', 'combined_p_value']:
        assert result['result'][name] == pytest.approx(expected[name], rel=1e-12)
    assert result['result']['is_artificial'] is False
    for actual, frozen in zip(result['result']['evidence'], expected['evidence']):
        assert actual['type'] == frozen['type']
        for key in ['effect_size', 'p_value', 'quality']:
            assert actual[key] == pytest.approx(frozen[key], rel=1e-12)
    assert result['detector_sha256'] == fixture['reproduction']['source_sha256']


def test_replay_does_not_overwrite_or_claim_exact_historical_data():
    path = ROOT / 'tests/fixtures/ky26_baseline.json'
    before = path.read_bytes()
    result = replay()
    assert path.read_bytes() == before
    assert result['exact_historical_reproduction'] is False
    baseline = json.loads(before)
    assert baseline['historical_record']['displayed_anomaly_score'] == 4.142
    assert baseline['historical_source_metadata']['orbit_solution_id'] is None
    assert result['result']['sigma_confidence'] != baseline['historical_record']['displayed_anomaly_score']


def test_manifest_detects_baseline_or_source_changes():
    manifest = json.loads((ROOT / 'docs/case-studies/1998-KY26/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_cli_refuses_to_replace_existing_replay(tmp_path):
    import subprocess
    import sys
    target = tmp_path / 'preserved.json'
    target.write_text('preserve-this-result')
    run = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_ky26_baseline.py'),
                          '--output', str(target)], capture_output=True, text=True)
    assert run.returncode != 0
    assert target.read_text() == 'preserve-this-result'
