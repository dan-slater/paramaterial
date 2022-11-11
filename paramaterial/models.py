"""
Module containing 1D constitutive models for fitting to stress-strain curves.
"""
import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.optimize as op
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult

from paramaterial.plug import DataItem, DataSet


@dataclass
class ModelItem:
    model_id: str
    info: pd.Series
    params: List[float]
    result: OptimizeResult
    data: pd.DataFrame

    @staticmethod
    def from_model(
            model_func: Callable[[np.ndarray, List[float]], np.ndarray],
            x_min: float,
            x_max: float,
            info: pd.Series,
            param_names: List[str],
            params: List[float],
            result: OptimizeResult,
            resolution: int,
    ) -> 'ModelItem':
        model_id = 'model_' + str(info[0])
        x_model = np.linspace(x_min, x_max, resolution)
        y_model = model_func(x_model, params)
        data = pd.DataFrame({'x': x_model, 'y': y_model})
        info['test id'] = model_id
        info['param_names'] = param_names
        info['params'] = params
        return ModelItem(model_id, info, params, result, data)

    def write_data_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.model_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self

class ModelSet:
    def __init__(
            self,
            model_func: Callable[[np.ndarray, List[float]], np.ndarray],
            param_names: List[str],
            bounds: List[Tuple[float, float]] | None = None,
            initial_guess: np.ndarray | None = None,
            scipy_func: str = 'minimize',
            scipy_kwargs: Dict[str, Any] | None = None,
    ):
        self.model_func = model_func
        self.param_names = param_names
        self.bounds = bounds
        self.initial_guess = initial_guess if initial_guess is not None else [0.0]*len(param_names)
        self.scipy_func = scipy_func
        self.scipy_kwargs = scipy_kwargs if scipy_kwargs is not None else {}
        self.resolution: int | None = None
        self.fitted_ds: DataSet | None = None
        self.x_key: str | None = None
        self.y_key: str | None = None
        self.x_min: float | None = None
        self.x_max: float | None = None
        self.model_map: map | None = None

    @property
    def params_table(self) -> pd.DataFrame:
        return pd.DataFrame([mi.params for mi in self.model_map])

    @property
    def results_table(self) -> pd.DataFrame:
        return pd.DataFrame([mi.result for mi in self.model_map])

    @property
    def model_items(self) -> List[ModelItem]:
        return [mi for mi in copy.deepcopy(self.model_map)]

    def objective_function(self, params: List[float], di: DataItem) -> float:
        data = di.data[di.data[self.x_key] > 0]

        x_data = data[self.x_key].values
        y_data = data[self.y_key].values

        sampling_stride = int(len(x_data)/self.resolution)
        if sampling_stride < 1:
            sampling_stride = 1
        x_data = x_data[::sampling_stride]
        y_data = y_data[::sampling_stride]
        y_model = self.model_func(x_data, params)
        return np.linalg.norm((y_data - y_model)/np.sqrt(len(y_data)))**2

    def predict(self) -> DataSet:
        predict_ds = DataSet()
        predict_ds.data_map = self.model_map
        return predict_ds

    def fit(self, ds: DataSet, x_key: str, y_key: str, resolution: int = 30) -> None:
        self.fitted_ds = ds
        self.x_key = x_key
        self.y_key = y_key
        self.x_min = 0
        self.x_max = max([di.data[x_key].max() for di in ds])
        self.resolution = resolution
        self.model_map = map(self.fit_item, ds.data_items)

    def fit_item(self, di: DataItem) -> ModelItem:
        if self.scipy_func == 'minimize':
            result = op.minimize(
                self.objective_function,
                self.initial_guess,
                args=(di,),
                bounds=self.bounds,
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'differential_evolution':
            result = op.differential_evolution(
                self.objective_function,
                self.bounds,
                args=(di,),
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'basinhopping':
            result = op.basinhopping(
                self.objective_function,
                self.initial_guess,
                minimizer_kwargs=dict(
                    args=(di,),
                    bounds=self.bounds,
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
        return ModelItem.from_model(self.model_func, self.x_min, self.x_max, di.info, self.param_names, result.x, result, self.resolution)


def iso_return_map(yield_stress_func: Callable, vec: str = 'stress'):
    @wraps(yield_stress_func)
    def wrapper(
            x: np.ndarray,
            mat_params
    ):
        y = np.zeros(x.shape)  # predicted stress
        x_p = np.zeros(x.shape)  # plastic strain
        aps = np.zeros(x.shape)  # accumulated plastic strain
        y_yield: callable = yield_stress_func(mat_params)  # yield stress
        E = mat_params[0]  # elastic modulus

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
def perfect(mat_params):
    """Perfect plasticity yield function, no hardening."""
    E, s_y = mat_params
    return lambda a: s_y


@iso_return_map
def linear(mat_params):
    """Linear isotropic hardening yield function."""
    E, s_y, K = mat_params
    return lambda a: s_y + K*a


@iso_return_map
def quadratic(mat_params):
    """Quadratic isotropic hardening yield function."""
    E, s_y, Q = mat_params
    return lambda a: s_y + E*(a - Q*a ** 2)


@iso_return_map
def voce(mat_params):
    """Exponential isotropic hardening yield function."""
    E, s_y, s_u, d = mat_params
    return lambda a: s_y + (s_u - s_y)*(1 - np.exp(-d*a))


@iso_return_map
def ramberg(mat_params):
    """Ramberg-Osgood isotropic hardening yield function."""
    E, s_y, C, n = mat_params
    return lambda a: s_y + C*(np.sign(a) * (np.abs(a)) ** n)


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
