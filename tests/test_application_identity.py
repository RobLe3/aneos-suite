"""Application identity is not an API/model version or readiness certificate."""
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).parents[1]


def test_application_version_bumps_known_tags_without_changing_contracts():
    from aneos_version import APPLICATION_VERSION,API_CONTRACT_VERSION,BUILD_CHANNEL
    from packaging.version import Version
    assert Version(APPLICATION_VERSION)>Version('2.1.6')
    assert Version(APPLICATION_VERSION).is_prerelease
    assert Version(APPLICATION_VERSION).pre==('b',1)
    assert not Version(APPLICATION_VERSION).is_devrelease
    assert API_CONTRACT_VERSION=='2.0.0'
    assert BUILD_CHANNEL=='beta'


def test_package_exports_use_one_application_identity():
    from aneos_version import APPLICATION_VERSION
    import aneos_core,aneos_api,aneos_dashboard
    assert aneos_core.__version__==aneos_api.__version__==aneos_dashboard.__version__==APPLICATION_VERSION


def test_dependency_free_identity_and_cli_outside_repository(tmp_path):
    script='''
import sys,runpy
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('numpy','scipy','torch','fastapi','aneos_api','aneos_core'):
   raise AssertionError('Version lookup must not initialize optional science/API/storage')
sys.meta_path.insert(0,Block())
sys.path.insert(0,sys.argv[1]);sys.argv=[sys.argv[1]+'/aneos.py','--version']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    done=subprocess.run([sys.executable,'-c',script,str(ROOT)],cwd=tmp_path,capture_output=True,text=True,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=10)
    from aneos_version import APPLICATION_VERSION
    assert done.returncode==0,done.stderr
    assert done.stdout.strip()==f'aNEOS {APPLICATION_VERSION} (beta)'
    assert not list(tmp_path.iterdir())


def test_readme_and_release_note_identify_development_not_scientific_readiness():
    from aneos_version import APPLICATION_VERSION
    readme=(ROOT/'README.md').read_text();change=(ROOT/'CHANGELOG.md').read_text()
    assert f'**v{APPLICATION_VERSION} | Beta preview' in readme
    assert f'## [{APPLICATION_VERSION}] - 2026-11-13' in change
    assert 'v1.2.2' in change  # Historical release identity is not rewritten.


@pytest.mark.parametrize('width',[80,120,160])
def test_real_v2_banner_readable_and_no_false_significance(width):
    from rich.console import Console
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_version import APPLICATION_VERSION
    output=io.StringIO();menu=ANEOSMenuV2();menu.console=Console(file=output,width=width,color_system=None)
    menu._print_banner();text=output.getvalue()
    assert APPLICATION_VERSION in text and 'beta' in text
    assert 'Network sigma' not in text and 'network sigma' not in text and 'σ thresholds' not in text
    assert '[cyan]' not in text and '[uncalibrated' not in text
    assert max(map(len,text.splitlines()))<=width
    assert '15' in text and 'Exit' in text
    import re
    from aneos_menu_catalog import MENU_OPTIONS
    for item in MENU_OPTIONS:
        assert len(re.findall(r'^│\s+'+item.choice+r'\s',text,re.M))==1


def test_plain_banner_has_no_markup_leakage(capsys):
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_version import APPLICATION_VERSION
    menu=ANEOSMenuV2();menu.console=None;menu._print_banner()
    text=capsys.readouterr().out
    assert APPLICATION_VERSION in text
    assert '[bold]' not in text and '[cyan]' not in text and '[dim]' not in text


def test_legacy_header_identifies_application_without_guessed_module_readiness():
    from rich.console import Console
    from aneos_menu import ANEOSMenu
    from aneos_version import APPLICATION_VERSION
    menu=ANEOSMenu.__new__(ANEOSMenu);out=io.StringIO();menu.console=Console(file=out,width=120,color_system=None)
    menu.display_header()
    assert APPLICATION_VERSION in out.getvalue() and 'uncalibrated' in out.getvalue()


def test_api_contract_application_header_and_payload_are_distinct():
    from fastapi.testclient import TestClient
    from aneos_api.app import create_app
    from aneos_version import APPLICATION_VERSION,API_CONTRACT_VERSION
    with_source=create_app();client=TestClient(with_source)
    root=client.get('/')
    assert root.status_code==200
    assert root.json()['application_version']==APPLICATION_VERSION
    assert root.json()['api_contract_version']==API_CONTRACT_VERSION
    assert root.headers['X-Application-Version']==APPLICATION_VERSION
    assert root.headers['X-API-Version']==API_CONTRACT_VERSION
    assert with_source.openapi()['info']['version']==API_CONTRACT_VERSION


def test_reproducibility_metadata_records_application_identity():
    from aneos_version import APPLICATION_VERSION
    from aneos_core.analysis.reporting import _software_metadata
    assert _software_metadata()['application_version']==APPLICATION_VERSION


def test_current_help_renders_actual_guide_not_historical_origin_claims(capsys):
    from aneos_menu_v2 import ANEOSMenuV2
    menu=ANEOSMenuV2();menu.console=None;menu._show_help()
    text=capsys.readouterr().out
    assert 'NOT_FLAGGED' in text and 'Perfectly normal' in text
    assert 'What it does not establish' in text and 'No valid physical-origin fit' in text
    assert 'Natural NEOs experience only gravitational accelerations' not in text


def test_catalogue_option_ids_and_dispatch_are_one_contract():
    from aneos_menu_catalog import MENU_OPTIONS
    from aneos_menu_v2 import ANEOSMenuV2
    choices=[item.choice for item in MENU_OPTIONS]
    assert set(choices)==set(map(str,range(16))) and len(choices)==len(set(choices))
    menu=ANEOSMenuV2()
    assert all(callable(getattr(menu,item.handler)) for item in MENU_OPTIONS if item.handler)


@pytest.mark.parametrize('choice',[str(i) for i in range(1,16)])
def test_actual_menu_loop_dispatch_matches_advertised_choice(choice,monkeypatch):
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_menu_catalog import MENU_OPTIONS
    menu=ANEOSMenuV2();calls=[];sequence=iter([choice,'0'])
    monkeypatch.setattr(menu,'_print_banner',lambda:None)
    monkeypatch.setattr(menu,'_prompt_choice',lambda:next(sequence))
    monkeypatch.setattr(menu,'wait_for_input',lambda:None)
    for item in MENU_OPTIONS:
        if item.handler:monkeypatch.setattr(menu,item.handler,lambda h=item.handler:calls.append(h))
    menu.run()
    assert calls==[next(item.handler for item in MENU_OPTIONS if item.choice==choice)]


def test_capture_preserves_actual_svg_html_text_and_refuses_overwrite(tmp_path):
    target=tmp_path/'capture'
    command=[sys.executable,str(ROOT/'scripts/qa_menu_entry.py'),'--output',str(target),
             '--baseline-ref','1ae3fe2da299331843620041eae7100fe7fc4285']
    done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=20,
                        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert done.returncode==0,done.stderr
    assert 'Rank candidates' in (target/'after-120.html').read_text()
    assert 'Rank candidates' in (target/'after-120.txt').read_text()
    svg=(target/'after-120.svg').read_text()
    assert 'Rank' in svg
    assert '@font-face' not in svg and 'cdnjs.cloudflare.com' not in svg
    assert 'NOT_FLAGGED' in (target/'help-120.html').read_text()
    inventory=json.loads((target/'inventory.json').read_text())
    assert inventory['primary_option_count']==16 and inventory['feature_count_excluding_exit']==15
    assert all(row['feature_prerequisite_qa']=='pending' for row in inventory['primary_options'])
    import hashlib
    for name,digest in inventory['artifact_sha256'].items():
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
    failed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert failed.returncode!=0


def test_markdown_uses_recorded_software_version_not_current_for_historical_input():
    from aneos_core.analysis.reporting import render_screening_markdown
    report={'designation':'synthetic','status':'test_only','software':{'application_version':'historical-marker'}}
    text=render_screening_markdown(report)
    assert 'Application version: historical-marker' in text
