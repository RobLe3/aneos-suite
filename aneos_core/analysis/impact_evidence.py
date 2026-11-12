"""Actual impact-input inventory, not a collision-probability calculator.

Existing research calculators remain separate. A catalogue orbit/encounter or
origin rank alone does not supply a fitted impact probability or a safe verdict.
"""
import math


def describe_impact_prerequisites(neo=None):
    covariance = neo.nongrav.covariance if neo is not None and neo.nongrav is not None else None
    arc = None
    if neo is not None and neo.first_observation is not None and neo.last_observation is not None:
        try:
            days = (neo.last_observation - neo.first_observation).total_seconds() / 86400
            if math.isfinite(days) and days >= 0:
                arc = days
        except (TypeError, ValueError):
            pass
    return {
        'status':'unavailable_observational_model_not_installed',
        'collision_probability':None, 'collision_probability_per_year':None,
        'moon_collision_probability':None, 'risk_level':'not_assessed',
        'probability_uncertainty':None, 'calculation_confidence':None, 'methodology':None,
        'input_inventory':{
            'orbital_elements_available':bool(neo is not None and neo.orbital_elements is not None),
            'observation_arc_days':arc,
            'catalogue_covariance_available':covariance is not None,
            'covariance_epoch_jd_tdb':covariance.epoch_jd_tdb if covariance is not None else None,
            'computed_encounter_count':len(neo.close_approaches) if neo is not None else 0,
            'encounter_count_semantics':'supplied_computed_predictions_not_observed_residuals',
        },
        'missing_prerequisites':['Compatible observation fit and residual uncertainty model',
            'Covariance propagation and encounter/impact model over a specified time horizon',
            'Independent validation or scoped authoritative impact solution'],
        'interpretation':'Unavailable is neither zero probability nor negligible risk. Origin screening does not establish impact probability.',
    }


import threading

_sentry_service_lock = threading.Lock()
_sentry_service_instance = None


def get_sentry_impact_service():
    """One process-local source service, including concurrent first requests."""
    global _sentry_service_instance
    with _sentry_service_lock:
        if _sentry_service_instance is None:
            from ..data.sources.sentry import SentryImpactService
            _sentry_service_instance = SentryImpactService()
        return _sentry_service_instance


def render_sentry_markdown(report):
    """Render only replay-verified reports; no local Torino/risk estimate."""
    from ..data.sources.sentry import verify_sentry_report
    if not verify_sentry_report(report):
        raise ValueError('Invalid or modified Sentry report')
    source = report['source']
    probability = report['collision_probability']
    lines = [f"# Sentry impact evidence — {report['designation']}",
        f"Status: {report['status']}",
        'Basis: external JPL model output, not a local aNEOS fit.',
        f"Retrieved (UTC): {source['retrieved_at']}",
        f"Source API: {source['api_version']}; method: {source['method'] or 'not available'}",
        f"Tabulated Earth-impact probability: {probability if probability is not None else 'not available'}",
        'Probability interval / aNEOS confidence: not supplied.',
        'Lunar probability / aNEOS risk classification: not assessed.']
    if report['time_scope']:
        scope = report['time_scope']
        lines.append(f"Listed-event span (UTC): {scope['first_listed_event_utc']} to {scope['last_listed_event_utc']}; {scope['event_count']} virtual impactors.")
    if source['computation_time']:
        lines.append(f"Source computation: {source['computation_time']['native']} [Pacific Local; not converted to UTC].")
    if source['removed_at_utc']:
        lines.append(f"Source removal time (UTC): {source['removed_at_utc']}")
    lines += ['','## Limits'] + [f'- {item}' for item in report['limitations']]
    lines += ['', f"Source: {source['documentation']}",
        f"Raw response SHA-256: {report['source_snapshot']['sha256']}",
        f"Report SHA-256: {report['report_sha256']}"]
    return '\n'.join(lines) + '\n'
