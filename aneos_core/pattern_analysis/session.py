"""BC11 frozen-input population description and canonical screening (ADR-077).

Population resemblance and computed encounter gaps are exploratory descriptors,
not a calibrated null model, physical association or evidence of artificial origin.
Acquisition, execution and storage belong outside this deterministic boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import copy
import json
import math
from typing import Optional

from aneos_core.data.models import NEOData
from aneos_core.analysis.screening import ScreeningAnalysisService, ScreeningPolicy

SCHEMA = 'network-evidence-report-v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class PatternAnalysisConfig:
    clustering: bool = True
    harmonics: bool = True
    correlation: bool = False
    rendezvous: bool = False
    historical_years: int = 30
    max_objects: int = 500
    # Old names remain accepted only as explicit non-operational defaults.
    min_sigma_filter: Optional[float] = None
    combination_method: Optional[str] = None
    group_radius: float = 1.0
    group_min_members: int = 3

    def __post_init__(self):
        for name in ('clustering', 'harmonics', 'correlation', 'rendezvous'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        for name, low, high in (('historical_years', 1, 100), ('max_objects', 1, 500),
                                ('group_min_members', 2, 500)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        if (type(self.group_radius) not in (int, float) or not math.isfinite(self.group_radius)
                or not 0 < self.group_radius <= 10):
            raise ValueError('group_radius must be finite in (0, 10]')
        if self.min_sigma_filter is not None or self.combination_method is not None:
            raise ValueError('min_sigma_filter/combination_method are deprecated: '
                             'ranking is not sigma and dependent selected p-values are not calibrated')


def _vector(neo):
    oe = neo.orbital_elements
    if oe is None:
        return None, 'missing_orbital_elements'
    fields = ('semi_major_axis', 'eccentricity', 'inclination',
              'ra_of_ascending_node', 'arg_of_periapsis')
    values = [getattr(oe, f) for f in fields]
    if any(x is None for x in values):
        return None, 'missing_' + '_'.join(f for f, x in zip(fields, values) if x is None)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError('Invalid orbital grouping features')
    # No clipping, imputation, inferred family label or background distribution.
    a, e, i, node, peri = values
    if a <= 0 or not 0 <= e < 1 or not 0 <= i <= 180:
        return None, 'unsupported_non_elliptical_or_invalid_elements'
    for primary, alias in (('ra_of_ascending_node','ascending_node'),
                           ('arg_of_periapsis','argument_of_perihelion')):
        other = getattr(oe, alias)
        if other is not None and other != getattr(oe, primary):
            return None, 'conflicting_angular_aliases'
    angular_scale = math.radians(10)
    vector = [a / .1, e / .05, i / 5,
            math.sin(math.radians(node % 360)) / angular_scale,
            math.cos(math.radians(node % 360)) / angular_scale,
            math.sin(math.radians(peri % 360)) / angular_scale,
            math.cos(math.radians(peri % 360)) / angular_scale]
    if not all(math.isfinite(x) for x in vector):
        return None, 'unsupported_numerical_range'
    return vector, None


def _groups(objects, config):
    metadata = {
        'status': 'descriptive_only', 'evidence_basis': 'heuristic_proxy',
        'algorithm': 'scaled_circular_connected_components_v1',
        'scales': {'semi_major_axis_au': .1, 'eccentricity': .05,
                   'inclination_degrees': 5, 'angular_chord_radians': math.radians(10)},
        'radius': config.group_radius, 'min_members': config.group_min_members,
        'excluded': [], 'duplicate_designations': [],
        'limitations': [
            'Engineering scales for exploration, not fitted population or uncertainty scales.',
            'Connected components permit chaining; members need not all be mutually close.',
            'Catalogue elements are not propagated to a common epoch; frame, epoch and covariance compatibility require physical follow-up.',
            'Missing elements are excluded. No survey-selection correction or artificial-origin inference.',
            'Angular degeneracies near circular/equatorial orbits are not resolved by this engineering metric.',
        ],
    }
    unique = {}
    for neo in objects:
        if neo.designation in unique:
            metadata['duplicate_designations'].append(neo.designation)
        else:
            unique[neo.designation] = neo
    points = []
    for name, neo in sorted(unique.items()):
        vector, reason = _vector(neo)
        if reason:
            metadata['excluded'].append({'designation': name, 'reason': reason})
        else:
            points.append((name, vector))
    # At most 500 inputs: explicit deterministic O(n^2), no optional backend switch.
    neighbours = [set() for _ in points]
    for i, (_, x) in enumerate(points):
        for j in range(i):
            y = points[j][1]
            if math.dist(x, y) <= config.group_radius:
                neighbours[i].add(j)
                neighbours[j].add(i)
    groups, visited = [], set()
    for i in range(len(points)):
        if i in visited:
            continue
        pending, component = [i], set()
        while pending:
            k = pending.pop()
            if k in component:
                continue
            component.add(k)
            pending.extend(neighbours[k] - component)
        visited.update(component)
        members = sorted(points[k][0] for k in component)
        if len(members) >= config.group_min_members:
            groups.append({'cluster_id': 'group_' + _hash(members)[:16],
                           'members': members, 'n_members': len(members),
                           'centroid': {}, 'density_sigma': None, 'p_value': None,
                           'known_family': None, 'evidence_basis': 'heuristic_proxy',
                           'interpretation': 'descriptive_catalogue_neighbourhood_not_physical_association'})
    metadata['objects_with_complete_features'] = len(points)
    return groups, metadata


def _events(neo):
    epochs, excluded, duplicates = set(), [], 0
    for index, ca in enumerate(neo.close_approaches):
        epoch = ca.epoch_jd_tdb
        if (ca.designation != neo.designation or ca.body != 'Earth' or ca.time_scale != 'TDB'
                or type(epoch) not in (int, float) or not math.isfinite(epoch) or epoch <= 0):
            excluded.append(index)
            continue
        if epoch in epochs:
            duplicates += 1
        epochs.add(epoch)
    ordered = sorted(epochs)
    return {'designation': neo.designation, 'evidence_basis': 'computed_encounter',
            'body': 'Earth', 'time_scale': 'TDB', 'epochs_jd_tdb': ordered,
            'gaps_days': [b - a for a, b in zip(ordered, ordered[1:])],
            'excluded_count': len(excluded) + duplicates, 'excluded_indices': excluded,
            'duplicate_epoch_count': duplicates,
            'coverage_status': 'not_established_by_event_list',
            'periodicity_significance': None}


class NetworkAnalysisSession:
    """Pure report over frozen NEOData; no hidden fetch or legacy score selection.

    `fetcher` is retained for call compatibility but is never invoked. Historical
    acquisition must be explicit before this boundary and preserved in inputs.
    `harmonics=True` requests computed-event gap inventory, not a spectral claim.
    """
    def __init__(self, config=None, fetcher=None, policy=None):
        self.config = config or PatternAnalysisConfig()
        if not isinstance(self.config, PatternAnalysisConfig):
            raise TypeError('config must be PatternAnalysisConfig')
        self.policy = policy or ScreeningPolicy()
        if not isinstance(self.policy, ScreeningPolicy):
            raise TypeError('policy must be ScreeningPolicy')
        self.fetcher = fetcher

    def run(self, neo_objects):
        if not isinstance(neo_objects, list) or any(not isinstance(n, NEOData) for n in neo_objects):
            raise TypeError('neo_objects must be a list of NEOData')
        if len(neo_objects) > 10000:
            raise ValueError('Too many requested inputs; split the request explicitly')
        # JSON freeze retains full source/solution/uncertainty lineage and native nulls.
        snapshots = json.loads(_json([n.to_dict() for n in neo_objects]))
        frozen = [NEOData.from_dict(n) for n in snapshots]
        seen = {}
        for n, snapshot in zip(frozen, snapshots):
            if not isinstance(n.designation, str) or not n.designation.strip():
                raise ValueError('Each input requires a nonempty designation')
            if n.designation in seen and seen[n.designation] != snapshot:
                raise ValueError('Conflicting duplicate designation: ' + n.designation)
            seen[n.designation] = snapshot
        objects = frozen[:self.config.max_objects]
        screening = ScreeningAnalysisService(policy=self.policy)
        assessments = [{'input_index': index, 'assessment': screening.analyze_neo(n).to_payload()}
                       for index, n in enumerate(objects)]
        modules = {name: {'status': 'disabled', 'requested': False}
                   for name in ('clustering', 'harmonics', 'correlation', 'rendezvous')}
        groups = []
        if self.config.clustering:
            groups, modules['clustering'] = _groups(objects, self.config)
            modules['clustering']['requested'] = True
        skipped = 0
        if self.config.harmonics:
            inventory = [_events(n) for n in objects]
            skipped = sum(len(e['epochs_jd_tdb']) < 2 for e in inventory)
            modules['harmonics'] = {
                'requested': True, 'status': 'descriptive_only_no_calibrated_periodicity',
                'acquisition_status': 'not_requested_by_session',
                'historical_years_requested': self.config.historical_years,
                'historical_window_applied': False, 'event_inventory': inventory,
                'limitations': ['Computed encounters are not observed acceleration samples.',
                    'No zero-filled daily observations, spectral selection or Gaussian null is inferred.',
                    'Acquisition coverage/window and detection completeness require separate evidence.']}
        if self.config.correlation:
            modules['correlation'] = {
                'requested': True, 'status': 'unavailable_time_series_required',
                'inventory': [{'designation': n.designation,
                               'evidence': n.nongrav.evidence_report() if n.nongrav else None}
                              for n in objects],
                'reason': 'One fitted scalar per component per object is not a paired time series; within-orbit covariance is not cross-object temporal correlation.'}
        if self.config.rendezvous:
            modules['rendezvous'] = {'requested': True, 'status': 'unavailable_not_implemented',
                                     'reason': 'In-session physical propagation is not implemented (ADR-045).'}
        report = {
            'schema_version': SCHEMA, 'engine_version': 'network-descriptors-v1', 'input_snapshots': snapshots,
            'configuration': asdict(self.config), 'screening_policy': asdict(self.policy),
            'input_id': _hash(snapshots),
            'configuration_id': _hash({'config': asdict(self.config), 'policy': asdict(self.policy)}),
            'designations_requested': len(frozen), 'designations_analyzed': len(objects),
            'excluded_by_cap': [{'input_index': i, 'designation': n.designation}
                                for i, n in enumerate(frozen) if i >= len(objects)],
            'object_assessments': assessments, 'modules': modules, 'clusters': groups,
            'harmonic_signals': [], 'correlation_matrix': None,
            'network_sigma': None, 'combined_p_value': None,
            'network_tier': 'UNAVAILABLE', 'sub_module_p_values': {}, 'method': None,
            'analysis_metadata': {
                'five_sigma_claim_permitted': False,
                'significance_status': 'unavailable_uncalibrated_dependent_population_procedure',
                'objects_skipped_insufficient_harmonics': skipped,
                'n_clusters_evaluated': len(groups),
                'limitations': ['Statistical unusualness, origin and human-origin exclusion remain separate.',
                    'Duplicate identities do not increase grouping evidence; ordered screening results retain requested duplicates.',
                    'Input hashes/replay detect changes, not source authenticity or scientific calibration.']},
        }
        report['report_id'] = _hash(report)
        return copy.deepcopy(report)


def verify_network_report(report):
    """Recompute the complete versioned report; no hash-only trust of results."""
    if not isinstance(report, dict) or report.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported network report')
    try:
        actual = NetworkAnalysisSession(PatternAnalysisConfig(**report['configuration']),
            policy=ScreeningPolicy(**report['screening_policy'])).run(
                [NEOData.from_dict(n) for n in report['input_snapshots']])
        if _json(actual) != _json(report):
            raise ValueError('Network report does not reproduce')
    except (TypeError, KeyError, ValueError, OverflowError) as exc:
        raise ValueError('Network report does not reproduce') from exc
    return actual


def network_report_to_markdown(report):
    """Human-readable verified summary plus complete replay input/output as JSON."""
    valid = verify_network_report(report)
    lines = ['# Population evidence report', '',
             f"Report ID: `{valid['report_id']}`", '',
             f"Inputs analysed: {valid['designations_analyzed']} of {valid['designations_requested']}.",
             f"Descriptive groups: {len(valid['clusters'])}.",
             'Statistical significance: not calibrated. Origin: unresolved.',
             'Grouping does not establish common physical origin or trajectory convergence.', '',
             '## Requested module outcomes', '']
    for name, module in valid['modules'].items():
        lines.append(f"- {name}: {module['status']}")
    lines += ['', '## Complete versioned evidence and results', '',
              'Hashes/replay are change checks, not source authentication or calibration.', '',
              '```json', json.dumps(valid, indent=2, sort_keys=True, allow_nan=False), '```', '']
    return '\n'.join(lines)
