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
