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
class ModelItem:
    model_id: str
    info: pd.Series
    params: pd.Series
    result: OptimizeResult
    data: pd.DataFrame

    @staticmethod
    def from_model(
            model_func: Callable[[pd.Series, pd.Series]],
            info: pd.Series,
            params: pd.Series,
            result: OptimizeResult
    ) -> 'ModelItem':
        model_id = 'model_' + str(info[0])
        return ModelItem(model_id, info, params, result, model_func(info, params))


class ModelSet:
    def __init__(
            self,
            model_func: Callable[[np.ndarray, *List[float]], np.ndarray],
            param_names: List[str],
            bounds: List[Tuple[float, float]]|None = None,
            initial_guess: np.ndarray|None = None,
            constraints: Dict[str, Any]|None = None,
            scipy_func: str = 'minimize',
            scipy_kwargs: Dict[str, Any]|None = None
    ):
        self.model_func = model_func
        self.param_names = param_names
        self.bounds = bounds
        self.initial_guess = initial_guess
        self.constraints = constraints
        self.scipy_func = scipy_func
        self.scipy_kwargs = scipy_kwargs
        self.fitted_ds: DataSet|None = None
        self.x_key: str|None = None
        self.y_key: str|None = None
        self.model_map: map|None = None

    @property
    def params_table(self) -> pd.DataFrame:
        return pd.DataFrame([mi.params for mi in self.model_map])

    @property
    def results_table(self) -> pd.DataFrame:
        return pd.DataFrame([mi.result for mi in self.model_map])

    @property
    def model_items(self) -> List[ModelItem]:
        return [mi for mi in self.model_map]

    def objective_function(self, params: List[float], di: DataItem) -> float:
        di_error = np.linalg.norm(
            (di.data[self.y_key].values - self.model_func(di.data[self.x_key], *params))
            /np.sqrt(len(di.data[self.y_key]))
        )
        return di_error

    def predict(self) -> DataSet:
        predict_ds = DataSet()
        predict_ds.data_map = self.model_map
        return predict_ds

    def fit(self, ds: DataSet, x_key: str, y_key: str, scipy_func: str = 'minimize') -> None:
        self.fitted_ds = ds
        self.x_key = x_key
        self.y_key = y_key
        self.model_map = map(self.fit_item, ds.data_items)

    def fit_item(self, di: DataItem) -> ModelItem:
        if self.scipy_func == 'minimize':
            result = op.minimize(
                self.objective_function,
                self.initial_guess,
                args=(di,),
                bounds=self.bounds,
                constraints=self.constraints,
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'differential_evolution':
            result = op.differential_evolution(
                self.objective_function,
                self.bounds,
                args=(di,),
                constraints=op.LinearConstraint(**self.constraints),
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'basinhopping':
            result = op.basinhopping(
                self.objective_function,
                self.initial_guess,
                minimizer_kwargs=dict(
                    args=(di,),
                    bounds=self.bounds,
                    constraints= self.constraints,
                    **self.scipy_kwargs
                )
            )
        elif self.scipy_func == 'dual_annealing':
            result = op.dual_annealing(
                self.objective_function,
                self.bounds,
                args=(di,),
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'shgo':
            result = op.shgo(
                self.objective_function,
                self.bounds,
                args=(di,),
                constraints=self.constraints,
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'brute':
            result = op.brute(
                self.objective_function,
                self.bounds,
                args=(di,),
                **self.scipy_kwargs
            )
        else:
            raise ValueError(f'Invalid scipy_func: {self.scipy_func}')
        params = pd.Series(result.x, index=self.param_names)
        return ModelItem.from_model(self.model_func, di.info, params, result)


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
