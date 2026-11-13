"""Opt-in single bounded public CAD request; never required by offline CI."""
import os
import pytest
from sqlalchemy import create_engine


@pytest.mark.network
def test_real_bounded_cad_query_commits_and_replays(tmp_path):
    if os.environ.get('ANEOS_CAD_NETWORK_TEST') != '1':
        pytest.skip('Set ANEOS_CAD_NETWORK_TEST=1 and explicit start/end dates')
    from aneos_core.data.candidate_discovery import CADCandidateService
    from aneos_api.candidate_repository import CandidateQueryRepository
    engine = create_engine('sqlite:///' + str(tmp_path/'network-candidates.sqlite'))
    try:
        repo = CandidateQueryRepository(engine)
        result = CADCandidateService(repo).poll(os.environ['ANEOS_CAD_TEST_START'],
            os.environ['ANEOS_CAD_TEST_END'],limit=3)
        assert result['commit_status'] == 'committed'
        assert result['receipt']['outcome'] in ('available','missing','partial')
        assert repo.load(result['receipt']['query_id']) == result
    finally:
        engine.dispose()
