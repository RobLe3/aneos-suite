"""Frozen orbital-scenario experiments through the deployed canonical policy.

No origin classifier, observed control labels, calibrated null or provider I/O.
The legacy Monte Carlo facades share these mechanics; see ADR-098.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, deque
import hashlib
import json
import math
import multiprocessing
from numbers import Integral

import numpy as np
from scipy import stats

from aneos_core.analysis.screening import ScreeningAnalysisService
from aneos_core.utils.reproducibility import software_metadata

SCHEMA='candidate-policy-experiment-v1'
GENERATOR_SCHEMA='synthetic-orbit-generator-v2'
MAX_SAMPLES=1_000_000
MAX_INPUT_BYTES=256*1024*1024
MAX_REPORT_BYTES=1024*1024*1024
SOURCE_FILES=['aneos_version.py','aneos_core/analysis/screening.py',
    'aneos_core/detection/validated_sigma5_artificial_neo_detector.py',
    'aneos_core/validation/policy_evaluation.py',
    'aneos_core/validation/monte_carlo_false_positive_validation.py',
    'aneos_core/validation/large_scale_monte_carlo.py']
RECIPES={
    'clipped-orbits-v2':{'a_lognormal':[1.25,.4],'a_clip':[.8,2.],
        'e_beta':[2.,4.],'e_multiplier':.7,'e_clip':[.001,.7],'i_rayleigh':8.,'i_clip':[0.,35.]},
    'mixture-orbits-v2':{'component_probabilities':[.02,.08,.6,.3],
        'a_uniform_components':[[.7,1.],[.8,1.]],'a_lognormal_components':[[1.5,.4],[1.8,.3]],
        'a_clip_components':[[1.,4.],[1.2,3.5]],'e_beta_components':[[1.2,2.],[1.5,1.5],[1.3,1.2],[1.4,1.8]],
        'e_multipliers':[.6,.8,.95,.9],'i_rayleigh_components':[8.,10.,12.,15.],'i_clip':[0.,45.]}}
LIMITATIONS=[
    'Synthetic/supplied orbital scenarios, not independently verified natural or human objects.',
    'Candidate means follow-up ranking selection, not artificial origin or a Gaussian significance.',
    'Physical measurements are absent; results describe orbital-only screening under the frozen prototype policy.',
    'No real-world false-positive/negative rate, sensitivity, FDR, origin probability or five-sigma validation is estimated.',
    'Clipped/adjusted toy distributions are not empirically qualified NEO populations; no survey-selection model is fitted.',
    'Hashes and replay consistency do not authenticate initial inputs or independently certify software.']


def _integer(value,name,low,high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or not low<=value<=high:
        raise ValueError(f'{name} must be an integer within {low}..{high}')
    return int(value)


def canonical(value,limit=MAX_REPORT_BYTES):
    def check(item,depth=0):
        if depth>32:raise ValueError('JSON nesting budget exceeded')
        if isinstance(item,dict):
            if any(type(key) is not str for key in item):raise ValueError('String JSON keys required')
            for child in item.values():check(child,depth+1)
        elif isinstance(item,(list,tuple)):
            for child in item:check(child,depth+1)
        elif type(item) not in (str,int,float,bool,type(None)):
            raise ValueError('Finite plain JSON values required')
    try:
        check(value)
        text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
        if len(text.encode())>limit:raise ValueError('Experiment payload budget exceeded')
        return text
    except (TypeError,OverflowError,RecursionError) as exc:
        raise ValueError('Invalid experiment JSON') from exc


def _clone(value,limit=MAX_REPORT_BYTES):return json.loads(canonical(value,limit))
def _hash(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


class SyntheticPopulation(list):
    """List-compatible draw with replay metadata, not an authenticity certificate."""
    def __init__(self,records,metadata):
        super().__init__(records);self.generation_metadata=_clone(metadata)


class OrbitGenerator:
    def __init__(self,seed=None,profile='clipped-orbits-v2'):
        if seed is not None:seed=_integer(seed,'seed',0,2**128-1)
        if profile not in RECIPES:raise ValueError('Unknown toy generator profile')
        self.seed=seed;self.profile=profile
        self.rng=np.random.Generator(np.random.PCG64(seed))

    def generate(self,size):
        size=_integer(size,'sample size',0,MAX_SAMPLES)
        start=_clone(self.rng.bit_generator.state)
        recipe=RECIPES[self.profile];rows=[]
        for index in range(size):
            component=None
            if self.profile=='clipped-orbits-v2':
                a=float(np.clip(self.rng.lognormal(np.log(recipe['a_lognormal'][0]),recipe['a_lognormal'][1]),*recipe['a_clip']))
                e=float(np.clip(self.rng.beta(*recipe['e_beta'])*recipe['e_multiplier'],*recipe['e_clip']))
                inc=float(np.clip(self.rng.rayleigh(recipe['i_rayleigh']),*recipe['i_clip']))
            else:
                component=int(self.rng.choice(4,p=recipe['component_probabilities']))
                if component<2:a=float(self.rng.uniform(*recipe['a_uniform_components'][component]))
                else:
                    mean,spread=recipe['a_lognormal_components'][component-2]
                    a=float(np.clip(self.rng.lognormal(np.log(mean),spread),*recipe['a_clip_components'][component-2]))
                e=float(self.rng.beta(*recipe['e_beta_components'][component])*recipe['e_multipliers'][component])
                inc=float(np.clip(self.rng.rayleigh(recipe['i_rayleigh_components'][component]),*recipe['i_clip']))
            # Explicit toy selection transformation, not a fitted physical model.
            if a*(1-e)>=1.3:e=1-1.25/a
            row={'synthetic_id':f'SYNTH_NEO_{index:08d}','a':a,'e':e,'i':inc,
                'omega':float(self.rng.uniform(0,360)),'w':float(self.rng.uniform(0,360)),
                'M':float(self.rng.uniform(0,360)),'perihelion':a*(1-e),'aphelion':a*(1+e)}
            if component is not None:row['population_component']=component
            rows.append(row)
        metadata={'schema_version':GENERATOR_SCHEMA,'profile':self.profile,'recipe':_clone(recipe),
            'seed_requested':self.seed,'rng':'PCG64','numpy_version':np.__version__,
            'state_before':start,'state_after':_clone(self.rng.bit_generator.state),'count':size,
            'sampling_design':'iid_draws_from_declared_toy_recipe',
            'postprocessing':'If q>=1.3 AU, set e=1-1.25/a; retain full binary64 values, no decimal rounding.',
            'units':{'a':'au','e':'dimensionless','i':'degree'},'input_sha256':_hash(rows),
            'evidence_basis':'synthetic_not_observations','orbital_class_labels':'not_assigned'}
        return SyntheticPopulation(rows,metadata)


def _generator_context(records,metadata):
    if metadata is None:return None
    try:
        metadata=_clone(metadata)
        if metadata['schema_version']!=GENERATOR_SCHEMA or metadata['rng']!='PCG64' or metadata['numpy_version']!=np.__version__:
            raise ValueError('Generator/version unavailable for replay')
        generator=OrbitGenerator(metadata['seed_requested'],metadata['profile'])
        generator.rng.bit_generator.state=metadata['state_before']
        replay=generator.generate(len(records))
        if replay!=records or replay.generation_metadata!=metadata:
            raise ValueError('Frozen generator metadata/inputs do not reproduce')
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Synthetic generator binding cannot be verified') from exc
    return metadata


def binomial_interval(successes,total,confidence=.95):
    total=_integer(total,'trial count',0,MAX_SAMPLES)
    successes=_integer(successes,'success count',0,total)
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<confidence<1:
        raise ValueError('Confidence must be strictly within (0,1)')
    if total==0:return None
    interval=stats.binomtest(successes,total).proportion_ci(confidence_level=confidence,method='exact')
    return [float(interval.low),float(interval.high)]


def _evaluate_chunk(task):
    start,records=task
    service=ScreeningAnalysisService();policy_id=service.describe_policy()['policy_id'];outcomes=[]
    allowed={'a','e','i','omega','Omega','w','M','perihelion','aphelion','synthetic_id','population_component'}
    for offset,row in enumerate(records):
        index=start+offset+1
        result={'input_index':index,'designation':f'scenario-{index}','status':'invalid_input',
                'candidate':None,'screening_score':None,'evidence_id':None,'policy_id':policy_id,'error_code':None}
        try:
            if not isinstance(row,dict) or set(row)-allowed:raise ValueError('Expected explicit orbital scenario')
            name=row.get('synthetic_id',f'scenario-{index}')
            if not isinstance(name,str) or not name or len(name)>128 or any(ord(c)<32 or ord(c)==127 for c in name):
                raise ValueError('Invalid scenario identifier')
            result['designation']=name
            assessment=service.analyze_raw(name,{k:row[k] for k in ('a','e','i') if k in row},
                provenance={'evidence_basis':'supplied_or_synthetic_orbital_scenario_not_observation'})
            if assessment.policy_id!=policy_id:raise RuntimeError('Policy changed inside experiment')
            result.update(status='abstained' if assessment.screening_score is None else 'assessed',
                candidate=assessment.candidate if assessment.screening_score is not None else None,
                screening_score=assessment.screening_score,evidence_id=assessment.evidence_id)
        except (ValueError,TypeError,KeyError):result['error_code']='INVALID_INPUT'
        except Exception:
            result['status']='screening_error';result['error_code']='SCREENING_ERROR'
        outcomes.append(result)
    return outcomes


def _outcomes(records,workers,chunk_size):
    tasks=((i,records[i:i+chunk_size]) for i in range(0,len(records),chunk_size))
    if workers==1:
        return [row for task in tasks for row in _evaluate_chunk(task)]
    output=[]
    # Bounded submission, ordered aggregation, spawn avoids inherited RNG/source state.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=deque()
        for _ in range(2*workers):
            task=next(tasks,None)
            if task is None:break
            pending.append(pool.submit(_evaluate_chunk,task))
        while pending:
            output.extend(pending.popleft().result())  # Whole-worker failure propagates, never all-False.
            task=next(tasks,None)
            if task is not None:pending.append(pool.submit(_evaluate_chunk,task))
    return output


def evaluate_population(records,*,workers=1,chunk_size=1000):
    """Measure the fixed candidate policy on explicitly supplied orbital scenarios."""
    metadata=getattr(records,'generation_metadata',None)
    if not isinstance(records,list):raise ValueError('A finite list of scenario records is required')
    _integer(len(records),'sample size',0,MAX_SAMPLES)
    workers=_integer(workers,'process budget',1,8);chunk_size=_integer(chunk_size,'chunk budget',1,10000)
    frozen=_clone(records,MAX_INPUT_BYTES)
    context=_generator_context(frozen,metadata)
    return _assemble(frozen,_outcomes(frozen,workers,chunk_size),context)


def _assemble(inputs,outcomes,context,software=None):
    counts={'total':len(inputs),'assessed':0,'candidates':0,'not_flagged':0,'abstained':0,'invalid_input':0,'screening_error':0}
    for item in outcomes:
        counts[item['status']]+=1
        if item['status']=='assessed':counts['candidates' if item['candidate'] else 'not_flagged']+=1
    if len(outcomes)!=len(inputs) or [o['input_index'] for o in outcomes]!=list(range(1,len(inputs)+1)):
        raise ValueError('Experiment lost/reordered a requested scenario')
    ids=[o['designation'] for o in outcomes]
    duplicate_ids=len(ids)-len(set(ids))
    feature_hashes=[_hash({k:row.get(k) for k in ('a','e','i')}) for row in inputs if isinstance(row,dict)]
    duplicate_features=len(feature_hashes)-len(set(feature_hashes))
    complete=counts['assessed']==counts['total']
    interval=binomial_interval(counts['candidates'],counts['total']) if context is not None and complete and not duplicate_ids else None
    policy=ScreeningAnalysisService().describe_policy()
    if any(o['policy_id']!=policy['policy_id'] for o in outcomes):raise ValueError('Experiment mixed policies')
    report={'schema_version':SCHEMA,'input_schema':'orbital-scenario-rows-v1','inputs':inputs,
        'input_sha256':_hash(inputs),'outcomes':outcomes,'outcomes_sha256':_hash(outcomes),
        'policy':policy,'generator_context':context,'counts':counts,
        'status':'empty' if not inputs else 'complete' if complete else 'incomplete',
        'candidate_fraction_assessed':counts['candidates']/counts['assessed'] if counts['assessed'] else None,
        'candidate_fraction_denominator':'assessed_scenarios_not_all_requested_rows',
        'unresolved_count':counts['total']-counts['assessed'],
        'candidate_interval_95':interval,
        'interval_status':'conditional_iid_toy_model_not_real_world_error' if interval else 'unavailable_no_complete_verified_iid_draw',
        'duplicate_identity_count':duplicate_ids,'duplicate_scored_feature_count':duplicate_features,
        'duplicate_note':'Matching scored features are not independent evidence; IID toy draws may match by chance. This is not a train/holdout leakage audit.',
        'unscored_fields':sorted({k for row in inputs if isinstance(row,dict) for k in row if k not in ('a','e','i','synthetic_id')}),
        'legacy_inference':dict.fromkeys(['false_positives','empirical_false_positive_rate','theoretical_false_positive_rate',
            'false_discovery_rate','sigma_level_equivalent','sensitivity','false_negative_rate','artificial_probability']),
        'five_sigma_claim_permitted':False,'scientific_validation_status':'not_independently_validated',
        'limitations':LIMITATIONS.copy(),'software':software if software is not None else software_metadata(SOURCE_FILES)}
    report['report_id']=_hash(report)
    return _clone(report)


def verify_experiment(report):
    """Replay supplied frozen scenarios under the current canonical policy, no I/O."""
    try:
        report=_clone(report)
        if report['schema_version']!=SCHEMA:raise ValueError('Unknown experiment schema')
        inputs=_clone(report['inputs'],MAX_INPUT_BYTES)
        _integer(len(inputs),'sample size',0,MAX_SAMPLES)
        context=_generator_context(inputs,report['generator_context'])
        expected=_assemble(inputs,_outcomes(inputs,1,1000),context,software=report['software'])
        if expected!=report:raise ValueError('Experiment does not reproduce')
    except (KeyError,ValueError,TypeError,AttributeError) as exc:
        raise ValueError('Candidate-policy experiment cannot be verified') from exc
    return report
