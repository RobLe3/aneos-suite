#!/usr/bin/env python3
"""Offline historical replay, separate from current scientific analysis.

Only executes the SHA-256-verified repository fixture from the recorded revision.
Never acquires data or modifies the baseline. --output refuses existing files.
"""
from pathlib import Path
import argparse
import hashlib
import types
import json
import logging
import platform
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]


def reproduce(root=ROOT):
    fixture = root / 'tests/fixtures/ky26_baseline.json'
    baseline = json.loads(fixture.read_text())
    source = root / baseline['reproduction']['frozen_source']
    source_text = ''.join(json.loads(source.read_text())['source_lines'])
    actual = hashlib.sha256(source_text.encode('utf-8')).hexdigest()
    if actual != baseline['reproduction']['source_sha256']:
        raise ValueError('Frozen source fingerprint mismatch')
    name = '_aneos_ky26_legacy_' + actual
    module = types.ModuleType(name)
    module.__file__ = str(source)
    sys.modules[name] = module
    exec(compile(source_text, str(source), 'exec'), module.__dict__)
    logger = logging.getLogger(name)
    previous_disabled = logger.disabled
    logger.disabled = True  # Historical toy-validation messages are not new detections.
    try:
        detector = module.ValidatedSigma5ArtificialNEODetector()
        values = baseline['rounded_input']
        result = detector.analyze_neo_validated(values['orbital'], values['physical'])
    finally:
        logger.disabled = previous_disabled
    import numpy as np
    output = {'sigma_confidence': float(result.sigma_confidence),
              'bayesian_probability': float(result.bayesian_probability),
              'combined_p_value': float(result.combined_p_value),
              'is_artificial': bool(result.is_artificial),
              'evidence': [{'type': e.evidence_type.value, 'effect_size': float(e.effect_size),
                            'p_value': float(e.p_value), 'quality': float(e.quality_score)}
                           for e in result.evidence_sources]}
    return {'schema_version': 'ky26-legacy-replay-v1',
            'code_revision': baseline['reproduction']['code_revision'],
            'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'detector_sha256': actual, 'input': baseline['rounded_input'], 'result': output,
            'environment': {'python': platform.python_version(), 'numpy': version('numpy'),
                            'scipy': version('scipy'), 'platform': platform.system(),
                            'machine': platform.machine(), 'float64_bits': np.finfo(np.float64).bits,
                            'numpy_error_settings': np.geterr()},
            'interpretation': 'Historical heuristic replay only; not calibrated discovery significance or artificial-origin probability.',
            'exact_historical_reproduction': False,
            'limitation': 'Full-precision historical inputs and original provider provenance are missing.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(reproduce(), sort_keys=True, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
