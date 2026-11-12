"""Typed orbit-fit evidence; no origin inference or independent-force claim.

SBDB cov=mat uses native au/degree/day coordinates at its OWN solution epoch.
The display-element epoch may differ. Labels, epochs and model constants travel
with the matrix; this module never silently propagates or decorrelates it.
"""
from dataclasses import asdict, dataclass, field
import copy
import math
from typing import Any, Dict, List, Optional

import numpy as np

AU_DAY2_TO_M_S2 = 149597870700.0 / 86400.0**2


def finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(number):
        raise ValueError(f'{name} must be a finite number')
    return number


@dataclass
class OrbitModelParameter:
    name: str
    value: float
    sigma: Optional[float]
    units: Optional[str]
    kind: str
    n: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    raw_value: Any = None
    raw_sigma: Any = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Model parameter name is required')
        self.value = finite_number(self.value, self.name)
        if self.sigma is not None:
            self.sigma = finite_number(self.sigma, f'{self.name}.sigma')
            if self.sigma < 0:
                raise ValueError('Parameter sigma must be nonnegative or unavailable')
        if self.raw_value is not None and finite_number(self.raw_value, 'raw parameter value') != self.value:
            raise ValueError('Native parameter value differs from preserved raw literal')
        if self.raw_sigma is not None and finite_number(self.raw_sigma, 'raw parameter sigma') != self.sigma:
            raise ValueError('Native parameter sigma differs from preserved raw literal')
        if self.units is not None and not isinstance(self.units, str):
            raise ValueError('Parameter units must be a string or unavailable')
        if self.kind not in {'SET', 'EST', 'CON'}:
            raise ValueError(f'Unsupported parameter estimation kind: {self.kind}')
        if self.n is not None and (isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0):
            raise ValueError('Model parameter index must be a nonnegative integer')

    @classmethod
    def from_provider(cls, data):
        return cls(name=data['name'], value=data.get('value'), sigma=data.get('sigma'),
                   units=data.get('units'), kind=data.get('kind'), n=data.get('n'),
                   title=data.get('title'), description=data.get('desc'),
                   raw_value=data.get('value'), raw_sigma=data.get('sigma'))

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return copy.deepcopy(asdict(self))

    def in_si(self):
        if self.name not in {'A1', 'A2', 'A3'} or self.units not in {'au/d^2', 'AU/day^2'}:
            raise ValueError(f'Unsupported acceleration coefficient units: {self.units}')
        return {'name': self.name, 'value': self.value * AU_DAY2_TO_M_S2,
                'sigma': self.sigma * AU_DAY2_TO_M_S2 if self.sigma is not None else None,
                'units': 'm/s^2', 'kind': self.kind,
                'evidence_basis': 'fitted_parameter' if self.kind == 'EST' else
                                  'considered_parameter' if self.kind == 'CON' else 'fixed_model_parameter',
                'interpretation': 'Force-model coefficient, not a direct thrust measurement'}


@dataclass
class OrbitalCovariance:
    labels: List[str]
    data: List[List[float]]
    epoch_jd_tdb: Any
    elements: List[Dict[str, Any]] = field(default_factory=list)
    representation: str = 'mat'
    coordinate_units: str = 'au,deg,d; model-parameter units retained separately'
    raw_data: Optional[List[List[Any]]] = None

    def __post_init__(self):
        if self.representation != 'mat':
            raise ValueError('Only explicitly requested cov=mat is supported')
        if not self.labels or any(not isinstance(label, str) or not label for label in self.labels):
            raise ValueError('Covariance labels are required')
        canonical = [{'om': 'node', 'w': 'peri'}.get(label, label) for label in self.labels]
        if len(set(canonical)) != len(canonical):
            raise ValueError('Covariance labels must be unique')
        finite_number(self.epoch_jd_tdb, 'covariance epoch')
        n = len(self.labels)
        if len(self.data) != n or any(not isinstance(row, list) or len(row) != n for row in self.data):
            raise ValueError('Covariance dimensions must match labels')
        native = copy.deepcopy(self.data)
        matrix = np.array([[finite_number(v, 'covariance entry') for v in row] for row in self.data])
        diagonal = np.diag(matrix)
        if np.any(diagonal < 0):
            raise ValueError('Covariance variances must be nonnegative')
        scale = np.sqrt(diagonal)
        # Normalize units before symmetry/PSD checks: a universal absolute
        # tolerance would silently accept invalid 1e-24 acceleration variances.
        zero = scale == 0
        if np.any(matrix[zero, :] != 0) or np.any(matrix[:, zero] != 0):
            raise ValueError('Zero variance requires a zero covariance row/column')
        safe_scale = np.where(zero, 1.0, scale)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            corr = matrix / np.outer(safe_scale, safe_scale)
        if not np.isfinite(corr).all():
            raise ValueError('Covariance must have finite normalized correlations')
        if not np.allclose(corr, corr.T, atol=1e-12, rtol=1e-10):
            raise ValueError('Covariance must be symmetric in normalized units')
        if np.linalg.eigvalsh((corr + corr.T) / 2).min() < -1e-10:
            raise ValueError('Covariance must be positive semidefinite')
        self.data = matrix.tolist()
        if self.raw_data is not None:
            try:
                raw_matrix = np.array([[finite_number(v, 'raw covariance entry') for v in row] for row in self.raw_data])
            except (TypeError, ValueError) as exc:
                raise ValueError('Invalid raw covariance representation') from exc
            if not np.array_equal(raw_matrix, matrix):
                raise ValueError('Covariance differs from preserved native raw matrix')
        self.raw_data = native if self.raw_data is None else copy.deepcopy(self.raw_data)
        self.labels = list(self.labels)
        self.elements = copy.deepcopy(self.elements)

    @classmethod
    def from_provider(cls, data):
        return cls(labels=data['labels'], data=data['data'], epoch_jd_tdb=data['epoch'],
                   elements=data.get('elements', []))

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return copy.deepcopy(asdict(self))

    def correlation(self, left, right):
        i, j = self.labels.index(left), self.labels.index(right)
        denominator = math.sqrt(self.data[i][i]) * math.sqrt(self.data[j][j])
        return self.data[i][j] / denominator if denominator else None

    def select(self, labels):
        if len(set(labels)) != len(labels):
            raise ValueError('Requested covariance labels must be unique')
        indices = [self.labels.index(label) for label in labels]
        return {'labels': list(labels), 'data': [[self.data[i][j] for j in indices] for i in indices],
                'epoch_jd_tdb': self.epoch_jd_tdb, 'time_scale': 'TDB'}
