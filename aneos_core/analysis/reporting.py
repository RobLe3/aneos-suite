"""Reproducible presentation of current screening, not scientific confirmation."""
from aneos_version import APPLICATION_VERSION, BUILD_CHANNEL
import copy
import json
from hashlib import sha256
from .screening import ScreeningAnalysisService


def _software_metadata():
    """Read-only run metadata; file hashes are not a dependency lock or signature."""
    import sys
    import subprocess
    from importlib.metadata import version, PackageNotFoundError
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    commit, dirty = None, None
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
            capture_output=True, text=True, timeout=2, check=True)
        status = subprocess.run(['git', '--no-optional-locks', 'status', '--porcelain'], cwd=root,
            capture_output=True, text=True, timeout=2, check=True)
        commit, dirty = revision.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass  # A packaged/non-Git installation is explicitly unknown, never clean.
    packages = {}
    for name in ('numpy', 'scipy', 'pydantic', 'SQLAlchemy', 'requests', 'astroquery'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    modules = ['aneos_version.py', 'aneos_core/analysis/pipeline.py', 'aneos_core/analysis/screening.py',
        'aneos_core/analysis/reporting.py', 'aneos_core/analysis/object_review.py',
        'aneos_core/data/physical_evidence.py', 'aneos_core/data/epoch_evidence.py', 'simple_neo_analyzer.py',
        'aneos_core/analysis/advanced_aneos_analyzer.py', 'aneos_core/analysis/impact_evidence.py',
        'aneos_core/interfaces/unified_analysis.py', 'aneos_core/data/models.py',
        'aneos_core/data/orbit_evidence.py', 'aneos_core/data/fetcher.py',
        'aneos_core/data/sources/sbdb.py', 'aneos_core/data/sources/horizons.py',
        'aneos_core/data/sources/mpc.py', 'aneos_core/data/sources/neodys.py',
        'aneos_core/data/acquisition.py', 'aneos_core/data/chronology.py', 'aneos_core/analysis/chronology_view.py', 'aneos_core/data/neodys_evidence.py', 'aneos_core/data/reconciliation.py',
        'aneos_core/data/enrichment.py', 'aneos_core/data/cache.py',
        'aneos_api/enrichment_store.py', 'aneos_api/acquisition_repository.py',
        'aneos_core/detection/validated_sigma5_artificial_neo_detector.py']
    hashes = {name:sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else None
        for name in modules}
    return {'application_version':APPLICATION_VERSION, 'build_channel':BUILD_CHANNEL,
        'repository_commit':commit, 'working_tree_dirty':dirty,
        'python_version':sys.version.split()[0], 'package_versions':packages,
        'float_mantissa_bits':sys.float_info.mant_dig, 'float_epsilon':sys.float_info.epsilon,
        'source_hashes_at_report_time':hashes,
        'limitations':'Source-file hashes describe files at report time, not loaded bytecode, authentication or a complete environment lock. Dirty code must be preserved separately to reproduce it.'}


def build_screening_report(result, additional_evidence=None, impact_assessment=None):
    payload = getattr(result, 'screening_assessment', None)
    if payload is None:
        analysis = getattr(result, 'analysis', {}) or {}
        payload = analysis.get('assessment') if isinstance(analysis, dict) else None
    if isinstance(result, dict):
        payload = result.get('screening_assessment', result if result.get('schema_version') == 'screening-assessment-v1' else payload)
    errors = list(getattr(result, 'errors', []) or [])
    status = getattr(result, 'status', 'unavailable')
    assessment = None
    if isinstance(payload, dict):
        try:
            replay = ScreeningAnalysisService().replay(payload['evidence_snapshot'], payload['evidence_id'])
            if replay.policy_id != payload['policy_id']:
                raise ValueError('Unsupported screening policy')
            assessment = replay.to_payload()  # Do not trust edited legacy inference fields.
            status = 'completed' if replay.screening_score is not None else 'inconclusive'
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f'Assessment cannot be replayed: {exc}')
            status = 'unavailable'
    elif not errors:
        errors.append('No replayable canonical assessment supplied; legacy claims are not scientific evidence')
    neo = getattr(result, 'neo_data', None)
    report = {'schema_version':'scientific-screening-report-v1',
        'designation': assessment['designation'] if assessment else getattr(result, 'designation', 'Unknown'),
        'status':status, 'errors':errors, 'screening_assessment':assessment,
        'classification':assessment['classification'] if assessment else 'INCONCLUSIVE',
        'is_artificial':None, 'artificial_probability':None, 'confidence_level':None,
        'confidence':None, 'sigma_statistical_level':None,
        'risk_assessment':'not_assessed', 'threat_level':'not_assessed',
        'analysis_method':'canonical_candidate_screening', 'detection_method':'canonical_candidate_screening',
        'validation_status':'unavailable_uncalibrated',
        'impact_assessment':copy.deepcopy(impact_assessment),
        'neo_data':neo.to_dict() if neo is not None else None,
        'analysis_metadata':copy.deepcopy(getattr(result, 'analysis_metadata', {}) or {}),
        'data_quality':copy.deepcopy(getattr(result, 'data_quality', {}) or {}),
        'software':_software_metadata(),
        'scientific_validation_status':'unavailable_uncalibrated',
        'five_sigma_claim_permitted':False,
        'additional_evidence':copy.deepcopy(additional_evidence)}
    if neo is not None:
        from aneos_core.validation.review_evidence import freeze_review_inputs
        _, report['review_input_id'] = freeze_review_inputs(neo)
    else:
        report['review_input_id'] = None
    # Own a JSON-safe snapshot, never stringify model objects or emit NaN/Infinity.
    report = json.loads(json.dumps(report, allow_nan=False))
    report['report_id'] = _report_hash(report)
    verify_screening_report(report)
    return report


def _report_hash(report):
    body = {key:value for key,value in report.items() if key != 'report_id'}
    return sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_screening_report(report):
    if report.get('schema_version') != 'scientific-screening-report-v1' or report.get('report_id') != _report_hash(report):
        raise ValueError('Report fingerprint/schema mismatch')
    for key in ('is_artificial', 'artificial_probability', 'confidence_level', 'confidence', 'sigma_statistical_level'):
        if key in report and report[key] is not None:
            raise ValueError('Unsupported standalone scientific claim in screening report')
    assessment = report.get('screening_assessment')
    if assessment is not None:
        service = ScreeningAnalysisService()
        replay = service.replay(assessment['evidence_snapshot'], assessment['evidence_id'])
        if replay.policy_id != assessment['policy_id'] or replay.to_payload() != assessment:
            raise ValueError('Report assessment cannot be replayed under the current policy')
    if report.get('neo_data') is not None:
        from aneos_core.data.models import NEOData
        from aneos_core.validation.review_evidence import freeze_review_inputs
        neo = NEOData.from_dict(report['neo_data'])
        _, review_id = freeze_review_inputs(neo)
        if review_id != report.get('review_input_id'):
            raise ValueError('Report review-input fingerprint mismatch')
        if assessment is not None and service.analyze_neo(neo).evidence_id != assessment['evidence_id']:
            raise ValueError('Report source data differs from screening evidence')
    return True


def _cell(value):
    if value is None:
        return 'not available'
    text = str(value).replace('|', r'\|').replace('\n', ' ').replace('\r', ' ')
    return ''.join(f'\\x{ord(c):02x}' if ord(c) < 32 or ord(c) == 127 else c for c in text)


def render_screening_markdown(report):
    assessment = report.get('screening_assessment') or {}
    rank = assessment.get('screening_score')
    lines = [f"# NEO screening: {_cell(report['designation'])}", '',
        f"Execution status: {_cell(report['status'])}",
        f"Screening classification: {_cell(assessment.get('classification', 'INCONCLUSIVE'))}",
        f"Uncalibrated screening rank: {_cell(rank)}", '',
        'Statistical significance: unavailable (no calibrated null/search-wide procedure)',
        'Artificial-origin probability: Not calibrated',
        'False-discovery estimate: Not reliably estimated',
        'Origin: inconclusive; human origin has not been excluded.', '',
        'This rank selects follow-up candidates. It does not identify an origin.', '',
        f"Evidence ID: {_cell(assessment.get('evidence_id'))}",
        f"Policy ID: {_cell(assessment.get('policy_id'))}",
        f"Review input ID: {_cell(report.get('review_input_id'))}",
        f"Report ID: {_cell(report.get('report_id'))}",
        f"Application version: {_cell(report.get('software', {}).get('application_version'))}",
        f"Build channel: {_cell(report.get('software', {}).get('build_channel'))}",
        f"Repository commit: {_cell(report.get('software', {}).get('repository_commit'))}",
        f"Working tree dirty: {_cell(report.get('software', {}).get('working_tree_dirty'))}", '',
        '## Non-gravitational coefficients', '',
        '| Component | Native value | Uncertainty | Units | Estimation status | Evidence basis/status |',
        '| --- | --- | --- | --- | --- | --- |']
    evidence = assessment.get('non_gravitational_evidence') or {}
    for name in ('A1','A2','A3'):
        component = evidence.get('components', {}).get(name, {})
        native = component.get('native', component)
        lines.append('| ' + ' | '.join(_cell(v) for v in [name,native.get('value'),
            native.get('sigma'),native.get('units'),native.get('kind'),
            f"{component.get('evidence_basis', 'unknown')}: {component.get('status', 'not_available')}"]) + ' |')
    raw = report.get('neo_data') or {}
    orbit = raw.get('orbital_elements') or {}
    lines += ['', '## Source inventory', '',
        f"Sources: {_cell(', '.join(raw.get('sources_used', [])) or None)}",
        f"Retrieved at: {_cell(raw.get('fetched_at'))}",
        f"First/last observation: {_cell(raw.get('first_observation'))} / {_cell(raw.get('last_observation'))}",
        f"Displayed orbit epoch: {_cell(orbit.get('epoch'))}; provider-native epoch/solution metadata remains in the JSON source record.", '', 'These are force-model coefficients, not direct propulsion evidence.',
        f"Covariance: {_cell(evidence.get('covariance_status'))}; preserve its native labels and epoch.", '',
        '## Evidence basis', '', '| Contribution | Basis | Dependency group | Available |',
        '| --- | --- | --- | --- |']
    for contribution in assessment.get('evidence_sources', []):
        lines.append('| ' + ' | '.join(_cell(contribution.get(k)) for k in
            ['type','evidence_basis','dependency_group','analyzed']) + ' |')
    lines += ['', '## Limitations', '']
    lines += ['- ' + _cell(v) for v in assessment.get('limitations', [])]
    lines += ['- Physical hypothesis fitting and independent scientific validation are not installed.']
    lines += ['- ' + _cell(e) for e in report.get('errors', [])]
    if report.get('impact_assessment'):
        impact = report['impact_assessment']
        lines += ['', '## Impact assessment', '',
            f"Status: {_cell(impact.get('status'))}",
            'Collision probability: unavailable; no fitted impact model was run.',
            'Unavailable is not zero or negligible risk. Origin screening is not an impact assessment.']
    if report.get('additional_evidence'):
        lines += ['', '## Additional evidence', '',
            'Computed ephemerides are not observations or independent origin evidence.', '',
            '```json', json.dumps(report['additional_evidence'], indent=2, allow_nan=False), '```']
    return '\n'.join(lines) + '\n'
