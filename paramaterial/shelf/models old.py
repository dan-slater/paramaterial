"""
Module containing 1D constitutive models for fitting to stress-strain curves.
"""
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.optimize as op
from scipy.optimize import OptimizeResult

from paramaterial.plug import DataItem, DataSet


@dataclass
class Model(ABC):
    name: str = None
    func: Callable = None
    param_names: List[str] = None
    bounds: List[Tuple[float, float]] = None
    constraints: Dict = None
    strain_vec: np.ndarray = None
    stress_vec: np.ndarray = None
    opt_res: OptimizeResult = None

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, x_lin: np.ndarray) -> np.ndarray:
        pass


@dataclass
class IsoReturnMapModel(Model):
    def fit(self):
        print(f'{".": <10}Fitting "{self.name}".')
        if self.constraints is not None:
            self.opt_res = op.differential_evolution(
                lambda params: np.linalg.norm(
                    self.stress_vec - self.func(self.strain_vec, *params, vec='stress'))/np.sqrt(
                    len(self.stress_vec)),
                bounds=self.bounds,
                constraints=op.LinearConstraint(**self.constraints)
            )
        else:
            self.opt_res = op.differential_evolution(
                lambda params: np.linalg.norm(
                    self.stress_vec - self.func(self.strain_vec, *params, vec='stress'))/np.sqrt(
                    len(self.stress_vec)),
                bounds=self.bounds
            )
        return self

    def predict(self, x_lin: np.ndarray) -> np.ndarray:
        return self.func(x_lin, *self.opt_res.x, vec='stress')

    def predict_plastic_strain(self, x_lin: np.ndarray):
        return self.func(x_lin, *self.opt_res.x, vec='plastic strain')

    def predict_accumulated_plastic_strain(self, x_lin: np.ndarray):
        return self.func(x_lin, *self.opt_res.x, vec='accumulated plastic strain')


def iso_return_map(yield_stress_func: Callable, vec: str = 'stress'):
    @wraps(yield_stress_func)
    def wrapper(
            x: np.ndarray,
            E: float,
            s_y: float,
            *mat_params
    ):
        y = np.zeros(x.shape)  # predicted stress
        x_p = np.zeros(x.shape)  # plastic strain
        aps = np.zeros(x.shape)  # accumulated plastic strain
        y_yield: callable = yield_stress_func(E, s_y, *mat_params)  # yield stress

        for i in range(len(x) - 1):
            y_trial = E*(x[i + 1] - x_p[i])
            f_trial = np.abs(y_trial) - y_yield(aps[i])
            if f_trial <= 0:
                y[i + 1] = y_trial
                x_p[i + 1] = x_p[i]
                aps[i + 1] = aps[i]
            else:
                d_aps = op.root(
                    lambda d: f_trial - d*E - y_yield(aps[i] + d) + y_yield(aps[i]),
                    aps[i]
                ).x[0]
                y[i + 1] = y_trial*(1 - d_aps*E/np.abs(y_trial))
                x_p[i + 1] = x_p[i] + np.sign(y_trial)*d_aps
                aps[i + 1] = aps[i] + d_aps

        if vec == 'stress':
            return y
        elif vec == 'plastic strain':
            return x_p
        elif vec == 'accumulated plastic strain':
            return aps
        else:
            return None

    return wrapper


@iso_return_map
def perfect(E, s_y):
    """Perfect plasticity yield function, no hardening."""
    return lambda a: s_y


@iso_return_map
def linear(E, s_y, K):
    """Linear isotropic hardening yield function."""
    return lambda a: s_y + K*a


@iso_return_map
def quadratic(E, s_y, Q):
    """Quadratic isotropic hardening yield function."""
    return lambda a: s_y + E*(a - Q*a**2)


@iso_return_map
def voce(E, s_y, s_u, d):
    """Exponential isotropic hardening yield function."""
    return lambda a: s_y + (s_u - s_y)*(1 - np.exp(-d*a))


@iso_return_map
def ramberg(E, s_y, C, n):
    """Ramberg-Osgood isotropic hardening yield function."""
    return lambda a: s_y + C*(a**n)


def sample(dataitem: DataItem, sample_size: int, delete_neg_strain: bool = True):
    dataitem.info['nr of points sampled'] = sample_size
    df = dataitem.data

    x_data = df['Strain'].values
    y_data = df['Stress(MPa)'].values

    if delete_neg_strain:
        for i, x_val in enumerate(x_data):
            if x_val < 0:
                x_data = np.delete(x_data, [i])
                y_data = np.delete(y_data, [i])

    sampling_stride = int(len(x_data)/sample_size)
    if sampling_stride < 1:
        sampling_stride = 1

    x_data = x_data[::sampling_stride]
    y_data = y_data[::sampling_stride]

    return x_data, y_data
