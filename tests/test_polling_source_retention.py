"""Synthetic provider boundaries; real legacy workflow and file-SQLite ledger."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from aneos_api.database import Base, EnrichedNEO
from aneos_api.enrichment_store import initialize_schema, records


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    import aneos_api.database as database
    import aneos_core.analysis.artificial_neo_dashboard as dashboard
    import neo_poller
    from aneos_menu import ANEOSMenu

    engine = create_engine('sqlite:///' + str(tmp_path / 'retention.sqlite'))
    Base.metadata.create_all(engine)
    initialize_schema(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(database, 'db_manager', SimpleNamespace(
        init_database=lambda: True, get_db=sessions))
    rows = {}

    def poll(self, source, *args):
        value = rows.get(source, [])
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(neo_poller.NEOPoller, 'poll_and_analyze', poll)
    displayed = []
    original = dashboard.create_dashboard_from_polling

    async def render(*, polling_results, **kwargs):
        displayed.append(copy.deepcopy(polling_results))
        return await original(polling_results, display=False, save=False)

    monkeypatch.setattr(dashboard, 'create_dashboard_from_polling', render)
    menu = ANEOSMenu.__new__(ANEOSMenu)
    menu.console = None
    menu._check_outdated_database_entries = AsyncMock(return_value={})
    yield menu, rows, sessions, displayed
    engine.dispose()


def row(source, **kwargs):
    return dict(designation='SYNTHETIC', data_source=source, **kwargs)


def run(workflow):
    return asyncio.run(workflow[0]._run_polling_dashboard_workflow('ALL_SOURCES', '1y', 10))


def test_complementary_sources_survive_actual_workflow_without_label_union(workflow):
    _, rows, sessions, displayed = workflow
    cad = row('NASA_CAD', close_approach_data={'dist': '0.1'})
    sbdb = row('NASA_SBDB', orbital_elements={'a': 1.2, 'e': .2, 'i': 2},
               physical_properties={'absolute_magnitude': 25.9})
    rows.update(NASA_CAD=[cad], NASA_SBDB=[sbdb])
    original = copy.deepcopy(rows)
    run(workflow)
    with sessions() as db:
        history = records(db, 'SYNTHETIC')
        assert [r['payload'] for r in history] == [cad, sbdb]
        n = db.scalar(select(EnrichedNEO))
        assert set(n.data_sources) == {'CAD', 'SBDB'}
        assert n.orbital_elements['semi_major_axis'] == 1.2
    assert displayed == [[cad, sbdb]]
    assert rows == original


@pytest.mark.parametrize('second', [
    row('NASA_SBDB', orbital_elements={'a': 1.3, 'e': .2, 'i': 2}),
    row('NASA_SBDB', orbital_elements={'a': 1.2, 'e': .2, 'i': 2, 'epoch': '2026-01-02'}),
    row('MPC', physical_properties={'diameter_km': .02}),
])
def test_distinct_conflicts_and_epochs_retained_not_deduped_by_object(workflow, second):
    _, rows, sessions, _ = workflow
    first = row('NASA_SBDB', orbital_elements={'a': 1.2, 'e': .2, 'i': 2},
                physical_properties={'diameter_km': .01})
    rows['NASA_SBDB'] = [first, second]
    run(workflow)
    with sessions() as db:
        assert [r['payload'] for r in records(db, 'SYNTHETIC')] == [first, second]


def test_exact_repeat_dedup_is_owned_by_ledger_not_source_name_merging(workflow):
    _, rows, sessions, displayed = workflow
    r = row('NASA_SBDB', orbital_elements={'a': 1.2})
    rows['NASA_SBDB'] = [r, copy.deepcopy(r)]
    result = run(workflow)
    with sessions() as db:
        assert len(records(db, 'SYNTHETIC')) == 1
    assert len(displayed[0]) == 2  # Returned record count, not independent observations.
    assert result['enrichment']['duplicate_assertions'] == 1


@pytest.mark.parametrize("orbital", [["invalid"], {"a": 1.2, "semi_major_axis": 1.3}])
def test_invalid_payload_rolls_back_and_does_not_publish_success_dashboard(workflow, capsys, orbital):
    _, rows, sessions, displayed = workflow
    rows['NASA_CAD'] = [row('NASA_CAD', orbital_elements={'a': 1.2})]
    rows['NASA_SBDB'] = [row('NASA_SBDB', orbital_elements=orbital)]
    result = run(workflow)
    assert displayed == []
    assert result['status'] == 'not_committed'
    with sessions() as db:
        assert records(db, 'SYNTHETIC') == []
        assert db.scalar(select(EnrichedNEO)) is None
    assert 'Database enrichment complete' not in capsys.readouterr().out


def test_partial_source_failure_is_not_complete_source_success(workflow):
    _, rows, _, _ = workflow
    rows['NASA_CAD'] = [row('NASA_CAD')]
    rows['NASA_SBDB'] = RuntimeError('synthetic outage')
    result = run(workflow)
    assert result['status'] == 'partial'
    assert result['source_outcomes']['NASA_SBDB']['status'] == 'unavailable'
    assert result['source_outcomes']['MPC']['status'] == 'empty'
    assert result['returned_records'] == 1
    assert result['distinct_designation_labels'] == 1


def test_all_empty_outcomes_do_not_claim_committed_enrichment(workflow):
    result = run(workflow)
    assert result['status'] == 'no_results'
    assert result['enrichment'] is None
    assert workflow[3] == []


def test_sql_failure_rolls_back_all_assertions_before_dashboard(workflow):
    from sqlalchemy import text
    _, rows, sessions, displayed = workflow
    with sessions() as db:
        db.execute(text("CREATE TRIGGER synthetic_write_failure BEFORE INSERT ON neo_object_projections_v1 "
                        "BEGIN SELECT RAISE(ABORT, 'synthetic disk failure'); END"))
        db.commit()
    rows['NASA_CAD'] = [row('NASA_CAD')]
    rows['NASA_SBDB'] = [row('NASA_SBDB', orbital_elements={'a': 1.2})]
    result = run(workflow)
    assert result['status'] == 'not_committed'
    assert displayed == []
    with sessions() as db:
        assert records(db, 'SYNTHETIC') == []
        assert db.scalar(select(EnrichedNEO)) is None


def test_unacknowledged_commit_does_not_render_or_review_receipts(workflow):
    menu, rows, _, displayed = workflow
    menu._enrich_database_with_polling_results = AsyncMock(return_value={})
    rows['NASA_CAD'] = [row('NASA_CAD')]
    result = run(workflow)
    assert result['status'] == 'not_committed'
    assert displayed == []
    menu._check_outdated_database_entries.assert_not_awaited()


def test_all_source_failures_report_unavailable_without_enrichment(workflow):
    _, rows, _, displayed = workflow
    rows.update({s: RuntimeError('synthetic unavailable')
                 for s in ('NASA_CAD', 'NASA_SBDB', 'MPC', 'NEODyS')})
    result = run(workflow)
    assert result['status'] == 'no_results'
    assert {o['status'] for o in result['source_outcomes'].values()} == {'unavailable'}
    assert result['enrichment'] is None and displayed == []


def test_chunked_handoff_retains_payloads_without_inventing_source_success(workflow):
    menu, _, sessions, displayed = workflow
    payloads = [row('NASA_CAD', native_epoch='one'), row('NASA_SBDB', native_epoch='two')]
    menu._run_chunked_polling = AsyncMock(return_value=payloads)
    result = asyncio.run(menu._run_polling_dashboard_workflow('ALL_SOURCES', '20y', None))
    assert result['status'] == 'partial'
    assert {o['status'] for o in result['source_outcomes'].values()} == {'not_verified'}
    with sessions() as db:
        assert [r['payload'] for r in records(db, 'SYNTHETIC')] == payloads
    assert displayed == [payloads]


@pytest.mark.parametrize('width', [80, 120, 160])
def test_rich_workflow_counts_records_not_objects_or_live_freshness(workflow, width):
    import io
    from unittest.mock import Mock
    from rich.console import Console
    from rich.text import Text
    menu, rows, _, _ = workflow
    stream = io.StringIO()
    menu.console = Console(file=stream, width=width, force_terminal=True, height=25)
    menu._dashboard_interactive_menu = Mock()
    rows['NASA_CAD'] = [row('NASA_CAD')]
    rows['NASA_SBDB'] = [row('NASA_SBDB')]
    result = run(workflow)
    output = Text.from_ansi(stream.getvalue()).plain
    assert result['returned_records'] == 2
    assert result['distinct_designation_labels'] == 1
    assert 'Live API' not in output
    assert 'Analyzing for artificial signatures' not in output
    assert 'complete source coverage' not in output
    assert all(Text(line).cell_len <= width for line in output.splitlines())
