"""Actual root entrypoints; synthetic data, no provider or research DB access."""
import asyncio
import copy
import io
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from rich.console import Console
from rich.text import Text
from aneos_core.data.models import NEOData, OrbitalElements
from aneos_core.analysis.artificial_neo_dashboard import records_from_file
from neo_poller import NEOPoller


@pytest.fixture
def poller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = NEOPoller.__new__(NEOPoller)
    p.console = None
    return p


def rows():
    neo = NEOData('SYNTHETIC', orbital_elements=OrbitalElements(
        semi_major_axis=1.23, eccentricity=.2, inclination=1.49))
    return [{'designation': neo.designation, 'neo_data': neo.to_dict()},
            {'designation': 'UNKNOWN', 'artificial_probability': None},
            {'designation': 'LEGACY', 'artificial_probability': .999}]


def test_actual_result_summary_preserves_nullable_outputs_and_typed_positive(poller, capsys):
    data = rows(); original = copy.deepcopy(data)
    dashboard = poller.display_results(data)
    assert dashboard.metrics.total_records == 3
    assert dashboard.metrics.not_flagged == 1
    assert dashboard.metrics.inconclusive == 2
    assert dashboard.classifications[0].screening_score is not None
    assert all(c.artificial_probability is None for c in dashboard.classifications)
    assert data == original
    text = capsys.readouterr().out
    assert 'uncalibrated' in text and 'Inconclusive' in text
    assert 'Highly suspicious' not in text
    assert list(Path.cwd().iterdir()) == []


def test_actual_detail_accepts_unavailable_origin_without_formatting_error(poller, capsys):
    from aneos_core.analysis.artificial_neo_dashboard import classify_record
    poller._show_detailed_analysis(classify_record(rows()[1]))
    text = capsys.readouterr().out
    assert 'UNKNOWN' in text and 'unverified' in text
    assert list(Path.cwd().iterdir()) == []


def test_actual_export_is_exclusive_and_retains_raw_inputs_and_invocation(poller):
    data = rows(); original = copy.deepcopy(data)
    old = Path('neo_poll_nasa_cad_1d_historical.json'); old.write_text('historical original')
    first = poller.save_results(data, 'NASA_CAD', '1d')
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    second = poller.save_results(data, 'NASA_CAD', '1d')
    assert first != second
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}
    assert old.read_text() == 'historical original' and data == original
    packet = json.loads((first/'report.json').read_text())
    assert packet['invocation_context']['requested_source'] == 'NASA_CAD'
    assert packet['invocation_context']['requested_period'] == '1d'
    assert len(records_from_file(packet)) == 3
    assert [r['input_record'] for r in packet['records']] == data
    for p in (first, second):
        hashes = json.loads((p/'complete.json').read_text())
        import hashlib
        assert all(hashlib.sha256((p/n).read_bytes()).hexdigest() == h for n,h in hashes.items())


@pytest.mark.parametrize('width', [80,120,160])
def test_actual_browser_navigation_and_detail_use_shared_outcomes(poller, width):
    from aneos_core.analysis.artificial_neo_dashboard import ArtificialNEODashboard
    dashboard = ArtificialNEODashboard(); dashboard.classifications = dashboard.classify_records(rows())
    stream = io.StringIO(); poller.console = Console(file=stream, width=width, height=25, force_terminal=True)
    with patch('rich.prompt.Prompt.ask', side_effect=['1','Q']):
        poller._interactive_object_explorer(dashboard)
    text = Text.from_ansi(stream.getvalue()).plain
    assert 'Inconclusive' in text and 'Not flagged' in text
    assert 'Confirmed natural objects' not in text and 'High confidence artificial objects' not in text
    assert 'Input basis:' in text
    assert all(Text(line).cell_len <= width for line in text.splitlines())
    assert list(Path.cwd().iterdir()) == []


def test_empty_summary_and_export_are_explicit_not_missing(poller, capsys):
    d = poller.display_results([])
    assert d.metrics.total_records == 0
    path = poller.save_results([], 'NASA_CAD', '1d')
    assert records_from_file(json.loads((path/'report.json').read_text())) == []
    assert '0/0' in capsys.readouterr().out


def test_actual_async_dashboard_launch_does_not_implicitly_export(poller):
    d = asyncio.run(poller.launch_analysis_dashboard(rows()))
    assert d.metrics.total_records == 3
    assert list(Path.cwd().iterdir()) == []


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        raise AssertionError('Rendering/export must not query providers')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'getaddrinfo', deny)


def test_actual_plain_browser_detail_filter_sort_and_authorized_page_export(poller, monkeypatch):
    from aneos_core.analysis.artificial_neo_dashboard import ArtificialNEODashboard
    d = ArtificialNEODashboard(); d.classifications = d.classify_records(rows())
    sequence = iter(['1', 'S', '5', 'F', '4', 'C', 'E', 'Q'])
    monkeypatch.setattr('builtins.input', lambda _: next(sequence))
    poller._interactive_object_explorer(d)
    paths = list(Path('dashboard_results').glob('*/report.json'))
    assert len(paths) == 1
    packet = json.loads(paths[0].read_text())
    assert packet['invocation_context']['selected_page'] == 'page_1'
    assert len(packet['records']) == 2
    assert all(r['category'] == 'inconclusive' for r in packet['records'])


def test_actual_rich_async_launch_uses_requested_console_and_no_implicit_export(poller):
    stream = io.StringIO(); poller.console = Console(file=stream, width=120, color_system=None)
    with patch('rich.prompt.Confirm.ask', return_value=True) as confirm, patch('rich.prompt.Prompt.ask', return_value='Q'):
        d = asyncio.run(poller.launch_analysis_dashboard(rows()))
    assert confirm.call_args.kwargs['console'] is poller.console
    assert d.metrics.total_records == 3
    assert 'Root poller screening records' in stream.getvalue()
    assert list(Path.cwd().iterdir()) == []


def test_export_context_is_finite_bound_and_escaped_in_markdown(poller):
    from aneos_core.analysis.artificial_neo_dashboard import ArtificialNEODashboard
    d = ArtificialNEODashboard(); d.console = None
    d.classifications = d.classify_records(rows())
    with pytest.raises(ValueError):
        d._save_dashboard_results('invalid-list', invocation_context=[])
    with pytest.raises(ValueError):
        d._save_dashboard_results('invalid-number', invocation_context={'bad': float('nan')})
    assert not Path('invalid-list').exists() and not Path('invalid-number').exists()
    context = {'note': '</script>```', 'source': 'requested, not established'}
    path = d._save_dashboard_results('reports', invocation_context=context)
    context['note'] = 'changed caller alias'
    packet = json.loads((path/'report.json').read_text())
    assert packet['invocation_context']['note'] == '</script>```'
    markdown = (path/'report.md').read_text()
    assert '</script>' not in markdown and '\\u003c' in markdown and '\\u0060' in markdown
    packet['invocation_context']['note'] = 'tampered'
    with pytest.raises(ValueError, match='fingerprint'):
        records_from_file(packet)


def test_source_listed_filter_has_positive_identity_and_modified_negative_controls():
    from aneos_core.analysis.poller_presentation import _Browser
    from aneos_core.analysis.artificial_neo_dashboard import classify_record
    from aneos_core.data.sources.sentry import parse_sentry_response
    # Synthetic source-shape control, not actual impact data for this designation.
    native = {'signature': {'version': '2.0', 'source': 'NASA/JPL Sentry Data API'},
              'summary': {'des': '1998 KY26', 'fullname': '(1998 KY26)', 'method': 'IOBS',
                          'ip': '1e-7', 'n_imp': '1', 'first_obs': '1998-05-31', 'last_obs': '2024-05-06',
                          'cdate': '2024-05-07 13:14:15', 'pdate': '2024-05-07 20:14:15',
                          'diameter': '.011', 'mass': '1000', 'energy': '.02', 'ts_max': '0', 'ps_cum': '-6'},
              'data': [{'date': '2050-01-01.25', 'ip': '1e-7', 'energy': '.01', 'ps': '-7', 'ts': '0', 'sigma_vi': '1.2'}]}
    report = parse_sentry_response(json.dumps(native).encode(), designation='1998 KY26', retrieved_at='2026-11-13T08:00:00+00:00')
    browser = _Browser(None)
    assert browser._has_impact_risk_indicators(classify_record({'designation':'1998 KY26','impact_evidence':report}))
    assert not browser._has_impact_risk_indicators(classify_record({'designation':'101955','impact_evidence':report}))
    report['collision_probability'] = .5
    assert not browser._has_impact_risk_indicators(classify_record({'designation':'1998 KY26','impact_evidence':report}))
    assert not browser._has_impact_risk_indicators(object())


def test_initialization_identifies_current_version_and_quarantine(capsys):
    from aneos_version import APPLICATION_VERSION
    p = NEOPoller()
    try:
        assert APPLICATION_VERSION in p.session.headers['User-Agent']
        text = capsys.readouterr().out
        assert APPLICATION_VERSION in text and 'temporarily unavailable' in text
        assert 'artificial NEO detection' not in text
    finally:
        p.session.close()


def test_capture_actual_root_views_widths_and_source_bound_artifacts(tmp_path):
    from scripts.qa_root_poller import capture_all
    import hashlib
    directory = tmp_path/'capture'
    manifest = capture_all(directory)
    assert len(manifest['widths']) == 12
    for name, expected in manifest['artifact_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected
    for name in manifest['widths']:
        text = (directory/(name+'.txt')).read_text()
        assert 'High confidence artificial objects' not in text and 'Confirmed natural objects' not in text
    with pytest.raises(FileExistsError):
        capture_all(directory)


@pytest.mark.parametrize('invalid', [None, {}, [None]*10001])
def test_root_display_rejects_nonlist_or_excess_records(poller, invalid):
    with pytest.raises(ValueError):
        poller.display_results(invalid)
    assert list(Path.cwd().iterdir()) == []


def test_failed_export_is_explicit_and_leaves_existing_report_unchanged(poller, monkeypatch, capsys):
    import aneos_core.analysis.dashboard_view as view
    first = poller.save_results(rows(), 'NASA_CAD', '1d')
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    capsys.readouterr()
    def fail(*args):
        raise OSError('synthetic write failure')
    monkeypatch.setattr(view, 'dashboard_markdown', fail)
    assert poller.save_results(rows(), 'NASA_CAD', '1d') is None
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}
    assert len(list(Path('dashboard_results').glob('*/complete.json'))) == 1
    text = capsys.readouterr().out
    assert 'Export failed' in text and 'Screening report saved' not in text


def test_actual_root_main_reports_export_failure_to_caller(poller, monkeypatch):
    import neo_poller
    import sys
    Path('dashboard_results').write_text('existing original file')
    monkeypatch.setattr(neo_poller, 'NEOPoller', lambda: poller)
    monkeypatch.setattr(poller, 'poll_and_analyze', lambda *args: rows())
    monkeypatch.setattr(sys, 'argv', ['neo_poller.py','--api','NASA_CAD','--period','1d'])
    assert neo_poller.main() == 1
    assert Path('dashboard_results').read_text() == 'existing original file'
