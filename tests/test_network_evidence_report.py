"""Synthetic evidence tests, not validation of population significance."""
import copy
import json
import pytest
from aneos_core.data.models import NEOData, OrbitalElements, CloseApproach, NonGravitationalParameters
from aneos_core.pattern_analysis.session import NetworkAnalysisSession, PatternAnalysisConfig


def neo(name='A', a=1.0, node=0.0, peri=0.0):
    return NEOData(designation=name, orbital_elements=OrbitalElements(
        semi_major_axis=a, eccentricity=.2, inclination=5.,
        ra_of_ascending_node=node, arg_of_periapsis=peri))


def test_real_screening_positive_and_no_inference():
    from aneos_core.analysis.screening import ScreeningAnalysisService
    n=neo(); n.orbital_elements.inclination=180.; n.dynamic_anomaly_score=9999
    result=NetworkAnalysisSession().run([n])
    assert result['object_assessments'][0]['assessment']==ScreeningAnalysisService().analyze_neo(n).to_payload()
    assert result['object_assessments'][0]['assessment']['candidate'] is True
    assert result['network_sigma'] is None and result['combined_p_value'] is None
    assert result['analysis_metadata']['five_sigma_claim_permitted'] is False


def test_freeze_replay_and_input_immutability():
    from aneos_core.pattern_analysis.session import verify_network_report
    n=neo(); before=copy.deepcopy(n.to_dict())
    report=NetworkAnalysisSession().run([n])
    assert n.to_dict()==before
    assert verify_network_report(json.loads(json.dumps(report)))==report
    report['object_assessments'][0]['assessment']['candidate']=True
    with pytest.raises(ValueError): verify_network_report(report)


def test_actual_positive_group_and_order_invariance():
    objects=[neo('C',node=359),neo('A',node=0),neo('B',node=1),neo('far',a=4)]
    cfg=PatternAnalysisConfig(harmonics=False)
    a=NetworkAnalysisSession(cfg).run(objects)
    b=NetworkAnalysisSession(cfg).run(list(reversed(objects)))
    assert a['clusters']==b['clusters']
    assert a['clusters'][0]['members']==['A','B','C']
    assert a['clusters'][0]['p_value'] is None
    assert a['clusters'][0]['density_sigma'] is None
    assert a['clusters'][0]['known_family'] is None
    assert len(a['object_assessments'])==4


def test_missing_angles_not_zero_and_no_clipping():
    objects=[neo('A'),neo('B'),neo('missing',node=None),neo('far',a=10)]
    r=NetworkAnalysisSession(PatternAnalysisConfig(group_min_members=2)).run(objects)
    assert r['clusters'][0]['members']==['A','B']
    assert r['modules']['clustering']['excluded'][0]['designation']=='missing'


def test_scalar_nongrav_not_correlation_or_no_signal():
    n=neo(); n.nongrav=NonGravitationalParameters(a1=1e-12,a3=2e-12)
    r=NetworkAnalysisSession(PatternAnalysisConfig(correlation=True)).run([n])
    assert r['correlation_matrix'] is None
    assert r['modules']['correlation']['status']=='unavailable_time_series_required'
    ng=r['modules']['correlation']['inventory'][0]['evidence']
    assert ng==n.nongrav.evidence_report()


def test_native_tdb_gaps_not_dense_synthetic_observations():
    n=neo()
    n.close_approaches=[CloseApproach(designation='A',epoch_jd_tdb=2450000.+k*365.,
         time_scale='TDB',body='Earth') for k in range(4)]
    r=NetworkAnalysisSession().run([n])
    gaps=r['modules']['harmonics']['event_inventory'][0]
    assert gaps['gaps_days']==[365.,365.,365.]
    assert gaps['epochs_jd_tdb']==[2450000.,2450365.,2450730.,2451095.]
    assert gaps['evidence_basis']=='computed_encounter'
    assert r['harmonic_signals']==[]
    assert r['modules']['harmonics']['status']=='descriptive_only_no_calibrated_periodicity'


def test_epoch_body_timescale_and_duplicate_partition():
    n=neo()
    n.close_approaches=[CloseApproach(designation='A',epoch_jd_tdb=2450000.,time_scale='TDB',body=b)
         for b in ('Earth','Earth','Venus')]
    n.close_approaches.append(CloseApproach(designation='A',epoch_jd_tdb=2450001.,time_scale='UTC',body='Earth'))
    r=NetworkAnalysisSession().run([n])['modules']['harmonics']['event_inventory'][0]
    assert r['epochs_jd_tdb']==[2450000.]
    assert r['excluded_count']==3 and r['duplicate_epoch_count']==1


def test_no_hidden_fetch_or_environment_selected_algorithm():
    class Forbidden:
        def __getattr__(self,name):raise AssertionError('No acquisition in scientific session')
    r=NetworkAnalysisSession(fetcher=Forbidden()).run([neo()])
    assert r['modules']['harmonics']['acquisition_status']=='not_requested_by_session'
    assert r['modules']['clustering']['algorithm']=='scaled_circular_connected_components_v1'


@pytest.mark.parametrize('kwargs',[
 {'max_objects':0},{'max_objects':True},{'historical_years':101},
 {'group_min_members':1},{'group_radius':float('nan')},{'group_radius':0},
 {'clustering':1},{'combination_method':'fisher'},{'min_sigma_filter':1.0},
])
def test_invalid_or_deprecated_config_rejected(kwargs):
    with pytest.raises(ValueError):PatternAnalysisConfig(**kwargs)


def test_duplicates_preserved_but_not_population_replication():
    n=neo()
    r=NetworkAnalysisSession(PatternAnalysisConfig(group_min_members=2)).run([n,copy.deepcopy(n)])
    assert len(r['object_assessments'])==2 and r['clusters']==[]
    assert r['modules']['clustering']['duplicate_designations']==['A']


def test_conflicting_duplicate_is_not_silently_selected():
    with pytest.raises(ValueError,match='Conflicting'):NetworkAnalysisSession().run([neo(),neo(a=2)])


def test_cap_explicit_exclusion_and_requested_unavailable():
    r=NetworkAnalysisSession(PatternAnalysisConfig(max_objects=1,rendezvous=True)).run([neo(),neo('B')])
    assert r['designations_analyzed']==1
    assert r['excluded_by_cap']==[{'input_index':1,'designation':'B'}]
    assert r['modules']['rendezvous']['status']=='unavailable_not_implemented'


def test_invalid_object_cannot_be_reported_success():
    with pytest.raises(TypeError):NetworkAnalysisSession().run([{}])
    n=neo();n.orbital_elements.ra_of_ascending_node=float('nan')
    with pytest.raises(ValueError):NetworkAnalysisSession().run([n])


def test_disabled_modules_explicit_and_empty_not_routine():
    r=NetworkAnalysisSession(PatternAnalysisConfig(clustering=False,harmonics=False)).run([])
    assert r['network_tier']=='UNAVAILABLE'
    assert r['modules']['clustering']['status']=='disabled'
    assert r['designations_analyzed']==0 and r['object_assessments']==[]


@pytest.mark.parametrize('module,cls,args',[
    ('clustering','OrbitalElementClusterer',([],)),
    ('harmonics','SynodicHarmonicAnalyzer',([],)),
    ('correlation','NonGravCorrelator',([],[])),
    ('network_sigma','NetworkSigmaCombiner',({'clustering':1e-20},)),
])
def test_unvalidated_legacy_entrypoint_controlled_deprecation(module,cls,args):
    import importlib
    obj=getattr(importlib.import_module('aneos_core.pattern_analysis.'+module),cls)()
    method=obj.combine if module=='network_sigma' else obj.run
    with pytest.raises(NotImplementedError,match='historical reproduction'):
        method(*args)


def test_rehashed_manufactured_report_not_trusted():
    from aneos_core.pattern_analysis.session import verify_network_report, _hash
    r=NetworkAnalysisSession().run([neo()])
    r['network_sigma']=5.0
    r['report_id']=_hash({k:v for k,v in r.items() if k!='report_id'})
    with pytest.raises(ValueError):verify_network_report(r)


def test_policy_and_snapshot_are_bound_to_report_and_replay():
    from aneos_core.analysis.screening import ScreeningPolicy
    from aneos_core.pattern_analysis.session import verify_network_report
    n=neo();r=NetworkAnalysisSession(policy=ScreeningPolicy(candidate_threshold=.1)).run([n])
    assert r['object_assessments'][0]['assessment']['candidate']
    assert verify_network_report(r)==r
    for field,value in [('screening_policy',{'version':'follow-up-ranking-v1','candidate_threshold':99}),
                        ('configuration',{'clustering':False})]:
        bad=copy.deepcopy(r);bad[field]=value
        with pytest.raises(ValueError):verify_network_report(bad)


def test_metadata_native_units_covariance_and_a3_preserved():
    n=neo();n.evidence_metadata={'synthetic_source':{'native_units':'au/d^2','epoch':'2450000.5 TDB'}}
    n.nongrav=NonGravitationalParameters(a3=1e-12)
    r=NetworkAnalysisSession().run([n])
    assert r['input_snapshots'][0]==n.to_dict()
    assert r['object_assessments'][0]['assessment']['non_gravitational_evidence']==n.nongrav.evidence_report()


def test_immutable_policy_and_config():
    from dataclasses import FrozenInstanceError
    cfg=PatternAnalysisConfig()
    with pytest.raises(FrozenInstanceError):cfg.group_radius=2


def test_mismatched_encounter_identity_excluded():
    n=neo();n.close_approaches=[CloseApproach(designation='other',epoch_jd_tdb=2450000.,time_scale='TDB',body='Earth')]
    r=NetworkAnalysisSession().run([n])['modules']['harmonics']['event_inventory'][0]
    assert not r['epochs_jd_tdb'] and r['excluded_indices']==[0]


def test_markdown_retains_full_json_and_refuses_invalid_report():
    from aneos_core.pattern_analysis.session import network_report_to_markdown,verify_network_report
    r=NetworkAnalysisSession().run([neo()])
    md=network_report_to_markdown(r)
    payload=json.loads(md.split('```json\n',1)[1].rsplit('\n```',1)[0])
    assert verify_network_report(payload)==r
    assert 'not calibrated' in md
    bad=copy.deepcopy(r);bad['combined_p_value']=0
    with pytest.raises(ValueError):network_report_to_markdown(bad)


def test_empty_and_disabled_reports_are_deterministic_replayable():
    from aneos_core.pattern_analysis.session import verify_network_report
    report=NetworkAnalysisSession(PatternAnalysisConfig(clustering=False,harmonics=False)).run([])
    assert verify_network_report(report)==report
    assert report==NetworkAnalysisSession(PatternAnalysisConfig(clustering=False,harmonics=False)).run([])


def test_angular_zero_not_fallback_to_conflicting_alias():
    n=neo();n.orbital_elements.ascending_node=20.
    r=NetworkAnalysisSession(PatternAnalysisConfig(group_min_members=2)).run([n,neo('B')])
    assert r['clusters']==[]
    assert r['modules']['clustering']['excluded'][0]['reason']=='conflicting_angular_aliases'


@pytest.mark.parametrize('primary,alias', [('ra_of_ascending_node','ascending_node'),
                                        ('arg_of_periapsis','argument_of_perihelion')])
def test_model_roundtrip_preserves_zero_and_exposes_alias_conflict(primary,alias):
    elements=OrbitalElements.from_dict({primary:0.0,alias:20.0})
    assert getattr(elements,primary)==0.0
    assert getattr(elements,alias)==20.0
    again=OrbitalElements.from_dict(elements.to_dict())
    assert again.to_dict()==elements.to_dict()


def test_menu_displays_verified_positive_not_sigma(monkeypatch):
    from aneos_menu_v2 import ANEOSMenuV2
    from unittest.mock import Mock
    n=neo();n.orbital_elements.inclination=180.
    report=NetworkAnalysisSession().run([n])
    menu=ANEOSMenuV2()
    panel=Mock();monkeypatch.setattr(menu,'display_panel',panel)
    menu._display_pattern_result(report)
    text=panel.call_args[0][0]
    assert 'Follow-up candidates (ranking, not origin): 1' in text
    assert 'not calibrated' in text and 'σ' not in text


def test_menu_rejects_tampered_report(monkeypatch):
    from aneos_menu_v2 import ANEOSMenuV2
    from unittest.mock import Mock
    report=NetworkAnalysisSession().run([neo()]);report['network_sigma']=5
    menu=ANEOSMenuV2();err=Mock();panel=Mock()
    monkeypatch.setattr(menu,'show_error',err);monkeypatch.setattr(menu,'display_panel',panel)
    menu._display_pattern_result(report)
    err.assert_called_once();panel.assert_not_called()


def test_extreme_features_excluded_not_clipped_into_group():
    r=NetworkAnalysisSession().run([neo('extreme',a=1e308)])
    assert r['modules']['clustering']['excluded'][0]['reason']=='unsupported_numerical_range'
    assert r['object_assessments'][0]['assessment']['classification']=='INCONCLUSIVE'


def test_menu_real_session_preserves_order_and_reports_missing(monkeypatch,tmp_path):
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_core.data.fetcher import DataFetcher
    from unittest.mock import Mock
    source=tmp_path/'requests.txt';source.write_text('1998 KY26\n433\n1998 KY26\n')
    n=neo('1998 KY26')
    monkeypatch.setattr(DataFetcher,'fetch_multiple',lambda self,ds:{'1998 KY26':n,'433':None})
    menu=ANEOSMenuV2();info=Mock();display=Mock()
    monkeypatch.setattr(menu,'_ask',lambda *a,**kw:str(source))
    monkeypatch.setattr(menu,'show_info',info);monkeypatch.setattr(menu,'_display_pattern_result',display)
    menu._population_pattern_analysis()
    r=display.call_args[0][0]
    assert len(r['object_assessments'])==2
    assert r['input_snapshots'][0]==r['input_snapshots'][1]
    assert any('Unavailable requested inputs (1)' in str(c) and '433' in str(c) for c in info.call_args_list)
