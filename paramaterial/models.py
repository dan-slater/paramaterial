"""
Module containing 1D constitutive models for fitting to stress-strain curves.
"""
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import List, Tuple, Dict

import numpy as np
import scipy.optimize as op
from scipy.optimize import OptimizeResult
from sklearn.metrics import mean_squared_error


@dataclass
class Model(ABC):
    name: str = None
    func: Callable = None
    param_names: List[str] = None
    bounds: List[Tuple[float, float]] = None
    constraints: Dict = None
    x_data: np.ndarray = None
    y_data: np.ndarray = None
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
        if self.constraints is not None:
            self.opt_res = op.differential_evolution(
                lambda params: np.linalg.norm(self.y_data - self.func(self.x_data, *params, vec='stress')) / np.sqrt(
                    len(self.y_data)),
                bounds=self.bounds,
                constraints=op.LinearConstraint(**self.constraints)
            )
        else:
            # a = 0
            # b = self.x_data
            # c = self.y_data
            self.opt_res = op.differential_evolution(
                lambda params: np.linalg.norm(self.y_data - self.func(self.x_data, *params, vec='stress')) / np.sqrt(
                    len(self.y_data)),
                bounds=self.bounds
            )

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
            *mat_params,
            vec: str = 'stress'
    ):
        y = np.zeros(x.shape)  # predicted stress
        x_p = np.zeros(x.shape)  # plastic strain
        aps = np.zeros(x.shape)  # accumulated plastic strain
        y_yield: callable = yield_stress_func(E, s_y, *mat_params)  # yield stress

        for i in range(len(x) - 1):
            y_trial = E * (x[i + 1] - x_p[i])
            f_trial = np.abs(y_trial) - y_yield(aps[i])
            if f_trial <= 0:
                y[i + 1] = y_trial
                x_p[i + 1] = x_p[i]
                aps[i + 1] = aps[i]
            else:
                d_aps = op.root(
                    lambda d: f_trial - d * E - y_yield(aps[i] + d) + y_yield(aps[i]),
                    aps[i]
                ).x[0]
                y[i + 1] = y_trial * (1 - d_aps * E / np.abs(y_trial))
                x_p[i + 1] = x_p[i] + np.sign(y_trial) * d_aps
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
    return lambda a: s_y + K * a


@iso_return_map
def quadratic(E, s_y, Q):
    """Quadratic isotropic hardening yield function."""
    return lambda a: s_y + E * (a - Q * a ** 2)


@iso_return_map
def voce(E, s_y, s_u, d):
    """Exponential isotropic hardening yield function."""
    return lambda a: s_y + (s_u - s_y) * (1 - np.exp(-d * a))


@iso_return_map
def ramberg(E, s_y, C, n):
    """Ramberg-Osgood isotropic hardening yield function."""
    return lambda a: s_y + C * (a ** n)

# todo: figure out why -ve a being passed in
