"""Deterministic object explanations/presentation over the existing screening service.

Binding/replay is not independent source verification, scientific calibration,
complete enrichment or physical-origin fitting. Original source data is retained.
"""
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
import math

from aneos_version import APPLICATION_VERSION
from aneos_core.data.models import NEOData, NonGravitationalParameters
from .screening import ScreeningAnalysisService, ScreeningPolicy

SCHEMA='object-evidence-review-v1'
# Fixed field-presence profile, declared before looking at any object. This is
# neither an all-science denominator nor an inventory of every model attribute.
FIELDS=(
 ('semi_major_axis','Semi-major axis','orbital_elements','semi_major_axis','AU'),
 ('eccentricity','Eccentricity','orbital_elements','eccentricity','dimensionless'),
 ('inclination','Inclination','orbital_elements','inclination','degrees'),
 ('ascending_node','Ascending node','orbital_elements','ra_of_ascending_node','degrees'),
 ('periapsis_argument','Periapsis argument','orbital_elements','arg_of_periapsis','degrees'),
 ('mean_anomaly','Mean anomaly','orbital_elements','mean_anomaly','degrees'),
 ('orbit_epoch','Orbit epoch','orbital_elements','epoch','model timestamp; native solution retained'),
 ('diameter','Diameter','physical_properties','diameter_km','km'),
 ('absolute_magnitude','Absolute magnitude H','physical_properties','absolute_magnitude_h','mag'),
 ('albedo','Albedo','physical_properties','albedo','dimensionless'),
 ('spectral_type','Spectral label','physical_properties','spectral_type','catalogue label'),
 ('rotation_period','Rotation period','physical_properties','rotation_period_hours','h'),
 ('first_observation','First observation','root','first_observation','catalogue timestamp; source precision retained'),
 ('last_observation','Last observation','root','last_observation','catalogue timestamp; source precision retained'),
 ('fetched_at','Retrieved at','root','fetched_at','retrieval timestamp'),
)


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _copy(value):return json.loads(_json(value))


def _hash(value):return sha256(_json(value).encode()).hexdigest()


def review_hash(review):return _hash({k:v for k,v in review.items() if k!='review_id'})


def _time_status(value):
    if value is None:return 'not_available'
    try:
        parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
        return 'offset_recorded_not_time_scale_conversion' if parsed.tzinfo is not None else 'timezone_unspecified'
    except (TypeError,ValueError,AttributeError):return 'invalid_timestamp'


def _inventory(snapshot):
    if not isinstance(snapshot,dict) or not isinstance(snapshot.get('designation'),str):
        raise ValueError('Invalid object snapshot')
    meta=snapshot.get('evidence_metadata') or {}
    if not isinstance(meta,dict):raise ValueError('Invalid source metadata')
    sources=snapshot.get('sources_used',[])
    if not isinstance(sources,list) or any(not isinstance(x,str) for x in sources):
        raise ValueError('Invalid source inventory')
    native_record=(snapshot.get('orbital_elements') or {}).get('native_epoch')
    from aneos_core.data.epoch_evidence import NativeEpoch
    native_epoch=NativeEpoch.from_dict(native_record) if native_record is not None else None
    rows=[]
    for key,label,section,field,units in FIELDS:
        record=snapshot if section=='root' else snapshot.get(section) or {}
        value=record.get(field)
        time_status=_time_status(value) if key in ('orbit_epoch','first_observation','last_observation','fetched_at') else None
        status='not_available' if value is None else 'invalid' if time_status=='invalid_timestamp' else 'present'
        if time_status is None and key!='spectral_type' and value is not None and (type(value) not in (int,float) or not math.isfinite(value)):
            status='invalid'
        path=f'/source_snapshot/{field}' if section=='root' else f'/source_snapshot/{section}/{field}'
        lineage=(meta.get('field_sources') or {}).get('orbital_elements' if section=='orbital_elements' else f'{section}.{field}')
        row={'key':key,'label':label,'value':value,'units':units,'status':status,'references':[path],
             'source_assertion':lineage,'basis':'catalogue_input_not_independently_verified'}
        if time_status:row['time_status']=time_status
        if key=='orbit_epoch' and native_epoch is not None:
            row.update(value=native_epoch.native_value,units=native_epoch.format+' '+native_epoch.time_scale,
                status='present',time_status='native_scale_not_converted_to_UTC',
                references=['/source_snapshot/orbital_elements/native_epoch'],source_assertion=native_epoch.source)
        if key=='diameter' and status=='present':row['display_conversion']={'value':value*1000,'units':'m','factor':1000}
        if key=='rotation_period' and status=='present':row['display_conversion']={'value':value*60,'units':'min','factor':60}
        rows.append(row)
    present=sum(r['status']=='present' for r in rows)
    minimum=[r for r in rows if r['key'] in ('semi_major_axis','eccentricity','inclination')]
    ng=snapshot.get('nongrav')
    ng_report=NonGravitationalParameters.from_dict(ng).evidence_report() if ng else None
    raw_physical=[];identities=[];normalized_physical=[]
    solutions=meta.get('source_solutions') or {}
    if not isinstance(solutions,dict):raise ValueError('Invalid source solution inventory')
    # Older single-source records may place the raw measurements at top level.
    registrations=list(solutions.items()) if solutions else [('unresolved_source',meta)]
    for name,record in registrations:
        if not isinstance(record,dict):raise ValueError('Invalid source record')
        root=f'/source_snapshot/evidence_metadata/source_solutions/{name}' if solutions else '/source_snapshot/evidence_metadata'
        for i,measurement in enumerate(record.get('physical_measurements') or []):
            raw_physical.append({'source':name,'record':_copy(measurement),'reference':root+f'/physical_measurements/{i}',
                                 'status':'raw_not_normalized_or_reconciled'})
        if record.get('physical_evidence') is not None:
            from aneos_core.data.physical_evidence import verify_physical_evidence
            normalized_physical.append({'source':name,'evidence':verify_physical_evidence(record['physical_evidence']),
                'reference':root+'/physical_evidence'})
        solution=record.get('solution') or {}
        if solution:
            identities.append({'source':name,'solution':_copy(solution),'reference':root+'/solution',
                               'status':'catalogue_assertion_identity_compatibility_not_established'})
    encounters=[]
    for i,ca in enumerate(snapshot.get('close_approaches') or []):
        encounters.append({'designation':ca.get('designation'),'distance_au':ca.get('dist'),
            'epoch_jd_tdb':ca.get('epoch_jd_tdb'),'time_scale':ca.get('time_scale'),
            'calendar_literal':ca.get('native_calendar') or ca.get('cd'),'body':ca.get('body'),
            'basis':'computed_encounter','reference':f'/source_snapshot/close_approaches/{i}'})
    inventory={'schema_version':'object-field-inventory-v1','designation':snapshot['designation'],
        'source_snapshot':_copy(snapshot),'source_id':_hash(snapshot),'measurements':rows,
        'coverage':{
            'profile_id':'basic-catalogue-presence-15-v2' if native_epoch else 'basic-catalogue-presence-15-v1',
            'basic_catalogue':{'present':present,'required':len(FIELDS),'fraction_present':present/len(FIELDS)},
            'screening_orbit':{'present':sum(r['status']=='present' for r in minimum),'required':3},
            'overall_data_completeness':None,'eligible_source_coverage':None,
            'meaning':'Field presence in the declared profile only; not full data completeness, source coverage, accuracy or scientific readiness.'},
        'missing_fields':[{'key':r['key'],'status':r['status']} for r in rows if r['status']!='present'],
        'sources_recorded':list(dict.fromkeys(sources)),'source_differences':_copy(meta.get('conflicts') or {}),
        'catalogue_solutions':identities,'raw_physical_assertions':raw_physical,
        'non_gravitational':ng_report,'encounters':encounters,
        'query_coverage':_copy({k:meta[k] for k in ('close_approach_query','historical_query_evidence') if k in meta}),
        'typed_physical_uncertainty_fields':_copy({k:v for k,v in (snapshot.get('physical_properties') or {}).items() if 'uncertainty' in k}),
        'uncertainty_note':'Scalar-model uncertainty fields and raw assertions are retained; unit/measurement reconciliation is not inferred.'}
    if native_epoch is not None:
        inventory['orbital_epoch_evidence']=native_epoch.to_dict()
        inventory['epoch_comparison']={'covariance_epoch':None,'same_julian_instant':None,
            'interpretation':'Epoch equality alone does not establish frame/solution/covariance compatibility; no propagation or time-scale conversion is performed.'}
        if ng_report and ng_report.get('covariance'):
            value=ng_report['covariance']['epoch_jd_tdb']
            covariance_epoch=NativeEpoch(str(value),'JD','TDB','SBDB','covariance')
            inventory['epoch_comparison'].update(covariance_epoch=covariance_epoch.to_dict(),
                same_julian_instant=native_epoch.same_julian_instant(covariance_epoch))
    if normalized_physical:
        inventory['physical_measurement_evidence']=normalized_physical
        inventory['physical_selection']=_copy(meta.get('physical_selection',{}))
    return inventory


def build_data_inventory(neo):
    if not isinstance(neo,NEOData):raise TypeError('Expected typed NEOData')
    return _inventory(_copy(neo.to_dict()))


def _assemble(payload,source_snapshot,policy,*,recorded_application_version=APPLICATION_VERSION):
    if type(policy) is not ScreeningPolicy:raise TypeError('Expected ScreeningPolicy')
    if not isinstance(recorded_application_version,str) or not recorded_application_version or len(recorded_application_version)>100:
        raise ValueError('Invalid recorded application identity')
    payload=_copy(payload)
    service=ScreeningAnalysisService(policy)
    assessment=service.replay(payload['evidence_snapshot'],payload['evidence_id'])
    if assessment.to_payload()!=payload:raise ValueError('Assessment does not replay exactly')
    inventory=None
    if source_snapshot is not None:
        inventory=_inventory(source_snapshot)
        neo=NEOData.from_dict(_copy(source_snapshot))
        actual=service.analyze_neo(neo,payload['evidence_snapshot']['additional'])
        if actual.to_payload()!=payload:raise ValueError('Source snapshot does not match assessment')
    trace=assessment.ranking_trace();score=assessment.screening_score;threshold=policy.candidate_threshold
    if score is None:
        decision={'status':'inconclusive','text':'Screening abstained: required orbital inputs are missing or outside this prototype domain. Missing evidence is not a negative scientific finding.'}
    elif assessment.candidate:
        decision={'status':'follow_up_ranking','text':f'Uncalibrated rank {score!r} meets policy threshold {threshold!r} ranking units. This selects follow-up; it is not a physical-origin finding.'}
    else:
        decision={'status':'not_flagged_by_policy','text':f'Uncalibrated rank {score!r} is below policy threshold {threshold!r} ranking units. This does not establish a natural origin or a perfectly normal object.'}
    insights=[]
    def insight(key,text,basis,refs):insights.append({'key':key,'text':text,'basis':basis,'references':refs})
    insight('reference','The rank uses fixed global prototype reference parameters, not a size/survey-conditioned population or calibrated null. Its proxy channels are not independent observations.',
            'heuristic_proxy',['/ranking_trace/reference_parameters','/assessment/analysis_metadata'])
    if trace['manual_inclination_enhancement']:
        insight('inclination','A hand-set inclination enhancement was applied to the orbital proxy. This is an algorithm assumption, not origin evidence.',
                'heuristic_proxy',['/ranking_trace/manual_inclination_enhancement','/assessment/evidence_snapshot/orbital/i'])
    physical=payload['evidence_snapshot']['physical'];reference=trace['reference_parameters']['physical_properties']
    if physical.get('diameter') is not None:
        diameter=physical['diameter'];centre=reference['diameter']['mean']
        direction='smaller than' if diameter<centre else 'larger than' if diameter>centre else 'equal to'
        insight('diameter',f'The supplied diameter {diameter!r} m is {direction} the legacy reference centre {centre!r} m. The physical proxy compares diameter in log space; this prototype comparison is not origin evidence.',
                'heuristic_proxy',['/assessment/evidence_snapshot/physical/diameter','/ranking_trace/reference_parameters/physical_properties/diameter'])
    if 'absolute_magnitude' in physical or 'diameter' in physical or 'albedo' in physical:
        insight('dependencies','Diameter, absolute magnitude and albedo have physical dependencies. Their availability is not three independent pieces of origin evidence; recorded albedo is not independently scored by the current physical proxy.',
                'heuristic_proxy',['/assessment/evidence_snapshot/physical','/assessment/evidence_sources'])
    if trace['terms'] and score is not None:
        index=max(range(len(trace['terms'])),key=lambda i:trace['terms'][i]['weighted_square'])
        if trace['terms'][index]['weighted_square']>0:
            insight('largest_term',f"The largest squared-ranking term is {trace['terms'][index]['channel']}. This is a decomposition of the heuristic, not the strongest physical evidence or a probability.",
                    'heuristic_proxy',[f'/ranking_trace/terms/{index}'])
    if inventory:
        if inventory['non_gravitational']:
            insight('nongrav','Non-gravitational coefficients/uncertainty/covariance are an inventory of orbit-model evidence; no force-model attribution, propulsion detection or temporal correlation is established by this report.',
                    'fitted_parameter_inventory',['/inventory/non_gravitational'])
        if inventory['source_differences']:
            insight('differences','Source alternatives are retained. Establish compatible epochs, frames, definitions and uncertainty before calling them contradictions or averaging them.',
                    'input_inventory',['/inventory/source_differences'])
        insight('coverage','The displayed profile counts recorded fields, not complete knowledge or verified source coverage. Missing physical uncertainties, observations and unresolved sources still limit scientific use.',
                'input_inventory',['/inventory/coverage','/inventory/missing_fields'])
    review={'schema_version':SCHEMA,'application_version':recorded_application_version,
        'scope':'full_object_snapshot' if inventory else 'assessment_only',
        'designation':assessment.designation,'screening_policy':asdict(policy),'assessment':payload,
        'inventory':inventory,'ranking_trace':trace,'decision':decision,'insights':insights,
        'scientific_status':'not_independently_validated',
        'follow_up':['Verify source identity, dated measurements, required uncertainty and compatible solutions.',
            'Use fitted natural/human-technology/artifact comparisons and withheld observations before physical-origin inference.',
            'Independent population calibration and search-wide error control are required before significance claims.'],
        'limitations':['Binding/replay verifies consistency, not source authenticity or measurement accuracy.',
            'No natural/artificial origin, calibrated probability or five-sigma result is supplied.',
            'Complete local enrichment and scientific operation readiness remain separate requirements.']}
    review['review_id']=review_hash(review)
    return _copy(review)


def build_assessment_review(payload,*,neo=None,policy=None):
    if neo is not None and not isinstance(neo,NEOData):raise TypeError('Expected typed NEOData')
    return _assemble(payload,_copy(neo.to_dict()) if neo else None,policy or ScreeningPolicy())


def build_object_review(neo,*,policy=None,additional_data=None):
    if not isinstance(neo,NEOData):raise TypeError('Expected typed NEOData')
    policy=policy or ScreeningPolicy();source=_copy(neo.to_dict())
    payload=ScreeningAnalysisService(policy).analyze_neo(neo,additional_data).to_payload()
    return _assemble(payload,source,policy)


def verify_object_review(review):
    if not isinstance(review,dict) or review.get('schema_version')!=SCHEMA:
        raise ValueError('Unsupported object review')
    try:
        inventory=review.get('inventory')
        source=inventory['source_snapshot'] if inventory is not None else None
        actual=_assemble(review['assessment'],source,ScreeningPolicy(**review['screening_policy']),
                         recorded_application_version=review['application_version'])
        if _json(actual)!=_json(review):raise ValueError('Object review does not reproduce')
    except (TypeError,KeyError,ValueError,OverflowError) as exc:
        raise ValueError('Object review does not reproduce') from exc
    return actual


def literal(value):
    return ''.join(f'\\x{ord(c):02x}' if ord(c)<32 or ord(c)==127 else c for c in str(value))


def number(value,*,exact=False):
    if value is None:return 'not available'
    if type(value) in (int,float):
        if not math.isfinite(value):return 'invalid non-finite value'
        return repr(value) if exact else format(value,'.9g')
    return literal(value)


def _measurement_value(row):
    value=number(row['value'])
    if row.get('display_conversion'):
        c=row['display_conversion'];value+=f" (~{number(c['value'])} {c['units']})"
    if row.get('time_status') and row['value'] is not None:value+=f"; {row['time_status'].replace('_',' ')}"
    return value


def _sections(review):
    a=review['assessment'];trace=review['ranking_trace'];inventory=review['inventory']
    sections=[('Screening classification',a['classification']),('Interpretation',review['decision']['text']),
        ('Scientific status','not independently validated; significance and origin probability unavailable'),
        ('Evidence binding',f"Review ID: {review['review_id']}\nEvidence ID: {a['evidence_id']}\nPolicy ID: {a['policy_id']}")]
    if inventory:
        basic=inventory['coverage']['basic_catalogue'];minimum=inventory['coverage']['screening_orbit']
        sections.append(('Data coverage',f"Basic catalogue field presence: {basic['present']}/{basic['required']} (declared 15-field profile only).\n"
            f"Screening orbital fields present: {minimum['present']}/3; domain eligibility is separate.\n"
            'Overall data completeness: not established. Eligible source coverage: not established.'))
    else:sections.append(('Data coverage','Full object snapshot unavailable; assessment-only review. No full-object completeness claim.'))
    if inventory and inventory.get('orbital_epoch_evidence'):
        sections.append(('Epochs and compatibility',_epoch_inventory_text(inventory)))
    return sections


def render_object_review(review,*,console=None,verbose=True):
    review=verify_object_review(review)
    if console is None:
        print(object_review_markdown(review,verbose=verbose));return
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    items=[]
    for heading,text in _sections(review):
        items.extend([Text(heading,style='bold cyan'),Text(text),Text('')])
    if review['inventory']:
        table=Table(title='Recorded catalogue data',expand=True,show_lines=False)
        for title in ('Field','Recorded value / display conversion','Units / status'):table.add_column(title,overflow='fold')
        for row in review['inventory']['measurements']:
            table.add_row(Text(row['label']),Text(_measurement_value(row)),Text(row['units']+'; '+row['status']))
        items.extend([table,Text('')])
    if verbose:
        trace=Table(title='Squared-ranking decomposition — not probabilities',expand=True)
        for title in ('Proxy channel','Effect / weight','Weighted square / share'):trace.add_column(title,overflow='fold')
        for term in review['ranking_trace']['terms']:
            share=term['share_of_squared_rank'];bar='' if share is None else ' '+('━'*round(share*20))
            share_text='not combined' if share is None else f'{100*share:.3g}% of squared rank'
            trace.add_row(Text(term['channel']),Text(f"{number(term['effect'])} / {number(term['weight'])}"),
                          Text(f"{number(term['weighted_square'])}; {share_text}{bar}"))
        items.extend([trace if review['ranking_trace']['terms'] else Text('Ranking decomposition unavailable: no supported terms.'),Text('')])
    items.append(Text('Evidence-grounded explanation',style='bold cyan'))
    for insight in review['insights']:items.extend([Text(insight['text']),Text('')])
    if review['inventory'] and review['inventory'].get('physical_measurement_evidence'):
        items.extend([Text('Atomic physical assertions',style='bold cyan'),Text(_physical_inventory_text(review['inventory']))])
    if review['inventory'] and review['inventory']['non_gravitational']:
        ng=Table(title='Non-gravitational coefficient inventory — not thrust',expand=True)
        for label in ('Axis','Native coefficient ± uncertainty','Units / estimation'):ng.add_column(label,overflow='fold')
        for name,component in review['inventory']['non_gravitational']['components'].items():
            native=component.get('native') or component
            ng.add_row(Text(name),Text(f"{number(native.get('value'))} ± {number(native.get('sigma'))}"),
                Text(literal(f"{native.get('units','not available')}; {native.get('kind') or component['status']}")))
        items.extend([ng,Text('Covariance/solution labels and native epochs are preserved in the review JSON; do not assume compatible epochs.')])
    items.extend([Text('Follow-up and limits',style='bold cyan')])
    for text in review['follow_up']+review['limitations']:items.append(Text('• '+text))
    console.print(Panel(Group(*items),title=Text(f"Object review — {literal(review['designation'])}"),border_style='blue',padding=(1,1)))
    if review['inventory']:
        summary=(review['inventory']['source_snapshot'].get('evidence_metadata') or {}).get('acquisition')
        if summary:
            from .acquisition_view import render_acquisition_summary
            render_acquisition_summary(summary,console=console)


def _cell(value):
    return literal(value).replace('|','\\|')


def object_review_markdown(review,*,verbose=True):
    review=verify_object_review(review)
    lines=[f"# Object review: {_cell(review['designation'])}",'']
    for title,text in _sections(review):lines.extend(['## '+title,text,''])
    if review['inventory']:
        lines+=['## Recorded data','| Field | Value | Units / status |','| --- | --- | --- |']
        for row in review['inventory']['measurements']:
            lines.append('| '+' | '.join(_cell(v) for v in (row['label'],_measurement_value(row),row['units']+'; '+row['status']))+' |')
    if review['inventory'] and review['inventory'].get('physical_measurement_evidence'):
        lines+=['','## Atomic physical assertions',_physical_inventory_text(review['inventory'])]
    lines+=['','## Explanation']
    for item in review['insights']:lines.extend([item['text'],'References: '+', '.join(item['references']),''])
    lines+=['## Follow-up and limits']+['- '+s for s in review['follow_up']+review['limitations']]
    if verbose:lines+=['','## Complete unrounded review','```json',json.dumps(review,indent=2,allow_nan=False),'```']
    if review['inventory']:
        summary=(review['inventory']['source_snapshot'].get('evidence_metadata') or {}).get('acquisition')
        if summary:
            from .acquisition_view import acquisition_markdown
            lines.extend(['',acquisition_markdown(summary)])
    return '\n'.join(lines)+'\n'


def _epoch_inventory_text(inventory):
    epoch=inventory.get('orbital_epoch_evidence')
    if not epoch:return ''
    lines=[f"Orbital element epoch: {literal(epoch['native_value'])} {epoch['format']} {epoch['time_scale']} ({literal(epoch['source'])})."]
    if epoch['time_scale']=='UNKNOWN':lines.append('Time scale not provided; no UTC or TDB inference is justified.')
    comparison=inventory['epoch_comparison'];cov=comparison['covariance_epoch']
    if cov:
        lines.append(f"Covariance epoch: {literal(cov['native_value'])} {cov['format']} {cov['time_scale']}.")
        equal=comparison['same_julian_instant']
        lines.append('Epoch values differ; covariance is not propagated to the display epoch.' if equal is False else
            'Epoch values match; this alone does not establish solution/frame compatibility.' if equal is True else
            'Epoch compatibility not established across these representations/time scales.')
    else:lines.append('Covariance epoch: not available in this inventory.')
    lines.append('Native times retained; not propagated, converted to UTC or treated as observation dates.')
    return '\n'.join(lines)


def _physical_inventory_text(inventory):
    lines=[]
    for registration in inventory.get('physical_measurement_evidence',[]):
        evidence=registration['evidence']
        for index,record in enumerate(evidence['records']):
            native=record['native'];n=record['normalized']
            text=f"{literal(native['name'])}: "
            if n:
                text+=f"{number(n['value'])} ± {number(n['sigma'])} {literal(n['units'])}; {n['uncertainty_kind']}"
            else:text+='not normalized; '+', '.join(record['issues'])
            text+=f"; ref: {literal(native.get('ref') or 'not available')}"
            duplicate=evidence['duplicate_records'].get(str(index))
            if duplicate is not None:text+=f'; duplicate of record {duplicate}, not additional evidence'
            lines.append(text)
        for field,choice in evidence['selection'].items():
            if choice['status']!='selected':lines.append(f"{field}: {choice['status']} — no scalar selection from this source.")
    if lines:lines.append('Catalogue assertions, not independently verified measurements. Physical measurement epoch: not available; retrieval/container orbit dates are not measurement dates.')
    return '\n'.join(lines)


def _nongrav_inventory_text(inventory):
    ng=inventory['non_gravitational']
    if not ng:return 'Non-gravitational evidence: not available.'
    lines=['Non-gravitational coefficient inventory — not thrust or origin evidence']
    for name,component in ng['components'].items():
        native=component.get('native') or component
        lines.append(f"{name}: {number(native.get('value'))} ± {number(native.get('sigma'))} {literal(native.get('units','not available'))}; {literal(component['status'])}")
    comparison=inventory.get('epoch_comparison')
    applies=comparison['same_julian_instant'] if comparison else ng['covariance_applies_to_display_epoch']
    if applies is False:
        lines.append('Covariance epoch differs from display epoch; do not combine without propagation.')
    elif applies is None:
        lines.append('Covariance/display epoch compatibility: not established.')
    else:lines.append('Covariance and display epochs match; coordinate ordering/units still require checking.')
    return '\n'.join(lines)


def render_data_inventory(inventory,*,console=None):
    # Verify inventory itself rather than trust edited display values/counts.
    actual=_inventory(inventory['source_snapshot'])
    if _json(actual)!=_json(inventory):raise ValueError('Object inventory does not reproduce')
    basic=actual['coverage']['basic_catalogue'];title=f"Input data — {actual['designation']}"
    if console is None:
        print(literal(title))
        print(f"Basic catalogue field presence {basic['present']}/{basic['required']}; not total completeness or scientific readiness.")
        for row in actual['measurements']:print(f"{row['label']}: {_measurement_value(row)} {row['units']} [{row['status']}]")
        if actual.get('orbital_epoch_evidence'):print(_epoch_inventory_text(actual))
        if actual.get('physical_measurement_evidence'):print(_physical_inventory_text(actual))
        print(_nongrav_inventory_text(actual))
        print('Overall completeness and source coverage: not established.')
        summary=(actual['source_snapshot'].get('evidence_metadata') or {}).get('acquisition')
        if summary:
            from .acquisition_view import render_acquisition_summary
            render_acquisition_summary(summary)
        return
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    table=Table(expand=True)
    for label in ('Field','Recorded value','Units / status'):table.add_column(label,overflow='fold')
    for row in actual['measurements']:table.add_row(Text(row['label']),Text(_measurement_value(row)),Text(row['units']+'; '+row['status']))
    console.print(Panel(Group(Text(f"Basic catalogue field presence {basic['present']}/{basic['required']}; not total completeness or readiness."),
        table,Text(_epoch_inventory_text(actual)),Text(_physical_inventory_text(actual)),Text(_nongrav_inventory_text(actual)),Text('Overall completeness and source coverage: not established. Native solutions/uncertainties remain in the JSON.')),
        title=Text(literal(title)),border_style='blue'))
    summary=(actual['source_snapshot'].get('evidence_metadata') or {}).get('acquisition')
    if summary:
        from .acquisition_view import render_acquisition_summary
        render_acquisition_summary(summary,console=console)
