"""Beta safety gate: unsafe legacy inference fails; the typed path still ranks."""
from unittest.mock import Mock
import pytest

@pytest.mark.parametrize('name,args',[
    ('poll_and_analyze',('NASA_CAD','1d',1)),
    ('analyze_sbdb_data_for_artificial_signatures',({'object':{'des':'SYNTHETIC'}},)),
    ('analyze_cad_record_for_artificial_signatures',(['SYNTHETIC'],)),
    ('fetch_mpc_recent_neos',(1,)),('fetch_neodys_recent_neos',(1,)),
])
def test_quarantined_root_methods_never_poll_or_fabricate(name,args):
    from neo_poller import NEOPoller
    client=NEOPoller.__new__(NEOPoller);client.session=Mock()
    with pytest.raises(NotImplementedError,match='temporarily unavailable'):
        getattr(client,name)(*args)
    assert not client.session.mock_calls


def test_beta_has_a_typed_positive_not_only_disabled_capabilities():
    from aneos_core.analysis.screening import ScreeningAnalysisService
    from aneos_core.data.models import NEOData,OrbitalElements
    result=ScreeningAnalysisService().analyze_neo(NEOData('SYNTHETIC',orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,inclination=1.49)))
    assert result.screening_score is not None
    assert result.artificial_probability is None and result.is_artificial is None


def test_beta_core_installs_documented_api_runtime():
    from pathlib import Path
    assert 'uvicorn>=' in (Path(__file__).parents[1]/'requirements-core.txt').read_text()


def test_api_launcher_keeps_the_active_python_environment(monkeypatch):
    import aneos,sys
    monkeypatch.setattr(sys,'argv',['aneos.py','api','--host','127.0.0.1','--port','18765'])
    run=Mock();monkeypatch.setattr(aneos.subprocess,'run',run)
    aneos.main()
    assert run.call_args.args[0][0]==sys.executable
    assert run.call_args.args[0][-2:]==['--port','18765']


def test_release_archives_exclude_local_config_and_mutable_database():
    from pathlib import Path
    attrs=(Path(__file__).parents[1]/'.gitattributes').read_text()
    assert '/.env export-ignore' in attrs
    assert '/aneos.db export-ignore' in attrs
