"""Evidence-bound explanations; synthetic controls are not scientific calibration."""
import copy
import io
import json
import math
from datetime import datetime,timezone
import pytest
from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties,NonGravitationalParameters,CloseApproach
from aneos_core.analysis.screening import ScreeningAnalysisService,ScreeningPolicy


def neo(name='SYNTHETIC',inclination=1.49):
    return NEOData(designation=name,orbital_elements=OrbitalElements(
        semi_major_axis=1.23,eccentricity=.2,inclination=inclination,
        ra_of_ascending_node=0.,arg_of_periapsis=0.,mean_anomaly=0.),
        physical_properties=PhysicalProperties(diameter_km=.011,albedo=.52,
            rotation_period_hours=.089193,absolute_magnitude_h=25.9),
        sources_used=['synthetic-provider'])


def review_module():
    from aneos_core.analysis import object_review
    return object_review


def test_rank_trace_is_actual_computation_not_changed_assessment_schema():
    s=ScreeningAnalysisService();n=neo();a=s.analyze_neo(n);before=a.to_payload()
    trace=a.ranking_trace()
    assert math.sqrt(sum(t['weighted_square'] for t in trace['terms']))==a.screening_score
    assert trace['policy']['candidate_threshold']==5.
    assert trace['manual_inclination_enhancement'] is True
    assert a.to_payload()==before and 'ranking_trace' not in before
    trace['terms'].clear();assert a.ranking_trace()['terms']


def test_complete_orbit_is_not_complete_object_or_source_coverage():
    n=neo();n.completeness=1.0
    d=review_module().build_data_inventory(n)
    assert d['coverage']['screening_orbit']['present']==3
    assert d['coverage']['screening_orbit']['required']==3
    assert d['coverage']['basic_catalogue']['present']<d['coverage']['basic_catalogue']['required']
    assert d['coverage']['overall_data_completeness'] is None
    assert d['coverage']['eligible_source_coverage'] is None
    assert any(x['key']=='fetched_at' for x in d['missing_fields'])


def test_review_bound_to_source_policy_assessment_and_replayed():
    m=review_module();n=neo();before=copy.deepcopy(n.to_dict())
    r=m.build_object_review(n)
    assert n.to_dict()==before
    assert m.verify_object_review(json.loads(json.dumps(r)))==r
    assert r['assessment']==ScreeningAnalysisService().analyze_neo(n).to_payload()
    assert r['inventory']['source_snapshot']==before
    assert r['scientific_status']=='not_independently_validated'


def test_positive_candidate_explanation_not_artificial_or_normal():
    m=review_module();r=m.build_object_review(neo(inclination=180))
    assert r['assessment']['candidate'] is True
    assert r['decision']['status']=='follow_up_ranking'
    assert 'not a physical-origin finding' in r['decision']['text']
    assert r['assessment']['artificial_probability'] is None


def test_unflagged_and_inconclusive_are_not_natural_findings():
    m=review_module()
    a=m.build_object_review(neo(),policy=ScreeningPolicy(candidate_threshold=100))
    assert a['decision']['status']=='not_flagged_by_policy'
    assert 'does not establish a natural origin' in a['decision']['text']
    b=m.build_object_review(NEOData(designation='missing'))
    assert b['decision']['status']=='inconclusive'
    assert b['assessment']['screening_score'] is None
    assert all(t['share_of_squared_rank'] is None for t in b['ranking_trace']['terms'])


def test_small_diameter_and_manual_bonus_described_as_prototype_assumptions():
    r=review_module().build_object_review(neo())
    text=' '.join(x['text'] for x in r['insights'])
    assert '800' in text and 'diameter' in text
    assert 'hand-set inclination' in text
    assert 'not origin evidence' in text
    assert all(x['basis'] in ('heuristic_proxy','input_inventory','fitted_parameter_inventory') for x in r['insights'])
    assert all(x['references'] for x in r['insights'])


def test_zero_angles_are_present_and_missing_is_explicit():
    n=neo();n.orbital_elements.inclination=0.
    data=review_module().build_data_inventory(n)
    row=next(x for x in data['measurements'] if x['key']=='inclination')
    assert row['value']==0. and row['status']=='present'
    assert next(x for x in data['measurements'] if x['key']=='orbit_epoch')['status']=='not_available'


def test_observation_timestamps_are_not_sliced_or_guessed_utc():
    n=neo();n.first_observation=datetime(2024,1,1,13,14,15)
    d=review_module().build_data_inventory(n)
    row=next(x for x in d['measurements'] if x['key']=='first_observation')
    assert row['value']=='2024-01-01T13:14:15'
    assert row['time_status']=='timezone_unspecified'


def test_same_sign_scalar_a3_is_inventory_not_propulsion_or_temporal_correlation():
    n=neo();n.nongrav=NonGravitationalParameters(a3=1e-12)
    r=review_module().build_object_review(n)
    assert r['inventory']['non_gravitational']==n.nongrav.evidence_report()
    assert 'no force-model attribution' in ' '.join(i['text'] for i in r['insights'])


def test_zero_distance_and_native_tdb_encounters_remain_computed():
    n=neo();n.close_approaches=[CloseApproach(designation=n.designation,distance_au=0.,
        epoch_jd_tdb=2450000.5,time_scale='TDB',body='Earth')]
    d=review_module().build_data_inventory(n)
    assert d['encounters'][0]['distance_au']==0.
    assert d['encounters'][0]['epoch_jd_tdb']==2450000.5
    assert d['encounters'][0]['basis']=='computed_encounter'


def test_retained_source_differences_are_not_silently_resolved_or_averaged():
    n=neo();n.evidence_metadata={'field_sources':{'orbital_elements':'source-a'},
        'conflicts':{'orbital_elements:source-b':{'semi_major_axis':2.0,'epoch':'other'}}}
    r=review_module().build_object_review(n)
    assert r['inventory']['source_differences']==n.evidence_metadata['conflicts']
    assert any('compatible epochs' in i['text'] for i in r['insights'])


def test_raw_physical_sigma_stays_raw_not_promoted_to_verified_uncertainty():
    n=neo();raw={'name':'diameter','value':'.011','sigma':'.002','units':'km'}
    n.evidence_metadata={'source_solutions':{'synthetic':{'physical_measurements':[raw]}}}
    d=review_module().build_data_inventory(n)
    assert d['raw_physical_assertions'][0]['record']==raw
    assert d['raw_physical_assertions'][0]['status']=='raw_not_normalized_or_reconciled'


@pytest.mark.parametrize('field,value',[('decision',{'status':'confirmed_artificial'}),
    ('ranking_trace',{'total_squared_rank':0}),('scientific_status','validated')])
def test_rehashed_fabricated_review_refused(field,value):
    m=review_module();r=m.build_object_review(neo());r[field]=value
    r['review_id']=m.review_hash(r)
    with pytest.raises(ValueError):m.verify_object_review(r)


def test_supplied_canonical_assessment_and_source_must_match():
    m=review_module();n=neo();p=ScreeningAnalysisService().analyze_neo(n).to_payload()
    other=neo();other.physical_properties.diameter_km=1.
    with pytest.raises(ValueError):m.build_assessment_review(p,neo=other)
    bad=copy.deepcopy(p);bad['classification']='CONFIRMED_ARTIFICIAL'
    with pytest.raises(ValueError):m.build_assessment_review(bad,neo=n)


def test_assessment_only_has_explicit_missing_full_source_not_fabricated_data():
    m=review_module();p=ScreeningAnalysisService().analyze_neo(neo()).to_payload()
    r=m.build_assessment_review(p)
    assert r['inventory'] is None and r['scope']=='assessment_only'
    assert m.verify_object_review(r)==r


@pytest.mark.parametrize('width',[80,120,160])
def test_actual_detail_render_is_readable_literal_and_bound(width):
    from rich.console import Console
    n=neo('[red]SYNTHETIC[/red]');m=review_module();r=m.build_object_review(n)
    out=io.StringIO();console=Console(file=out,width=width,color_system=None)
    m.render_object_review(r,console=console)
    text=out.getvalue()
    assert '[red]SYNTHETIC[/red]' in text
    assert '100% data complete' not in text
    assert 'not independently validated' in text
    assert '5.35158' in text and 'unavailable' in text
    assert max(map(len,text.splitlines()))<=width


def test_recorded_application_identity_survives_metadata_only_release_change():
    m=review_module();r=m.build_object_review(neo())
    r['application_version']='older-recorded-build';r['review_id']=m.review_hash(r)
    assert m.verify_object_review(r)['application_version']=='older-recorded-build'


def test_data_control_characters_are_visible_not_terminal_commands(capsys):
    m=review_module();r=m.build_object_review(neo('SYNTHETIC\x1b[2J'))
    m.render_object_review(r,console=None,verbose=False)
    text=capsys.readouterr().out
    assert '\x1b' not in text and '\\x1b' in text


def test_real_menu_review_uses_actual_canonical_source_and_rejects_mutation(capsys):
    from aneos_menu_v2 import ANEOSMenuV2
    menu=ANEOSMenuV2();menu.console=None;n=neo()
    result=menu._run_detection(n.designation,n)
    assert result is not None
    menu._display_neo_data(n.designation,n)
    menu._display_detection_result(n.designation,result,verbose=True)
    text=capsys.readouterr().out
    assert 'Basic catalogue field presence' in text and 'overall_data_completeness' in text
    assert 'not_flagged_by_policy' in text or 'This does not establish a natural origin' in text
    result.analysis['assessment']['classification']='CONFIRMED_ARTIFICIAL'
    menu._display_detection_result(n.designation,result)
    assert 'modified or unbound' in capsys.readouterr().out

@pytest.mark.parametrize('fmt,suffix',[('1','json'),('2','csv')])
def test_actual_menu_export_preserves_full_unrounded_review_and_exclusive_create(tmp_path,fmt,suffix):
    from aneos_menu_v2 import ANEOSMenuV2
    from unittest.mock import Mock
    m=ANEOSMenuV2.__new__(ANEOSMenuV2);m.console=None
    n=neo('Named object');result=m._run_detection(n.designation,n)
    assert result is not None
    m._detection_results={n.designation.upper():result};m._impact_results={}
    path=tmp_path/f'review.{suffix}'
    m._ask=Mock(side_effect=[fmt,str(path)]);m._export_results()
    if fmt=='1':r=json.loads(path.read_text())['neos'][0]['object_review']
    else:
        import csv
        with path.open() as f:r=json.loads(next(csv.DictReader(line for line in f if not line.startswith('#')))['object_review_json'])
    assert review_module().verify_object_review(r)==result.object_review
    original=path.read_bytes()
    m._ask=Mock(side_effect=[fmt,str(path)]);m._export_results()
    assert path.read_bytes()==original
    result.object_review['decision']['text']='forged'
    other=tmp_path/f'forged.{suffix}';m._ask=Mock(side_effect=[fmt,str(other)])
    m._export_results();assert not other.exists()


def test_named_object_detail_can_be_opened_from_uppercase_browser_key(capsys):
    from aneos_menu_v2 import ANEOSMenuV2
    m=ANEOSMenuV2.__new__(ANEOSMenuV2);m.console=None
    n=neo('Named object');result=m._run_detection(n.designation,n)
    m._display_detection_result(n.designation.upper(),result)
    out=capsys.readouterr().out
    assert 'Evidence-grounded' in out or '## Explanation' in out
    assert 'unbound' not in out


def test_detail_capture_generator_actual_artifacts_hashes_and_no_overwrite(tmp_path):
    import subprocess,sys
    from hashlib import sha256
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    directory=tmp_path/'captures'
    command=[sys.executable,str(root/'scripts/qa_object_review.py'),'--output',str(directory)]
    subprocess.run(command,cwd=root,check=True,capture_output=True,text=True)
    inventory=json.loads((directory/'inventory.json').read_text())
    assert len(inventory['cases'])==3
    for case,info in inventory['cases'].items():
        for width in (80,120,160):
            assert info['widths'][str(width)]['max_columns']<=width
            assert 'SYNTHETIC' in (directory/f'{case}-{width}.txt').read_text()
            assert 'SYNTHETIC' in (directory/f'{case}-{width}.html').read_text()
            svg=(directory/f'{case}-{width}.svg').read_text()
            assert '@font-face' not in svg
    for name,digest in inventory['artifact_sha256'].items():
        assert sha256((directory/name).read_bytes()).hexdigest()==digest
    original=(directory/'inventory.json').read_bytes()
    assert subprocess.run(command,cwd=root,capture_output=True).returncode!=0
    assert (directory/'inventory.json').read_bytes()==original


def test_real_rich_data_and_result_views_preserve_acceleration_inventory():
    from rich.console import Console
    m=review_module();n=neo();n.nongrav=NonGravitationalParameters(a3=1.2e-12)
    review=m.build_object_review(n)
    output=io.StringIO();console=Console(file=output,width=80,color_system=None)
    m.render_data_inventory(review['inventory'],console=console)
    m.render_object_review(review,console=console)
    text=output.getvalue()
    assert all(name in text for name in ('A1','A2','A3','1.2e-12'))
    assert 'not thrust' in text
    assert 'not established' in text
    assert max(map(len,text.splitlines()))<=80


def test_inventory_render_refuses_rehashed_display_count():
    m=review_module();inventory=m.build_data_inventory(neo())
    inventory['coverage']['basic_catalogue']['present']=15
    with pytest.raises(ValueError,match='does not reproduce'):
        m.render_data_inventory(inventory)
