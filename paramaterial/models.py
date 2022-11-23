"""
Module containing 1D constitutive models for fitting to stress-strain curves.
"""
import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
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
    model_func: Callable[[np.ndarray, List[float]], np.ndarray]
    x_min: float
    x_max: float
    resolution: int = 50

    @staticmethod
    def from_results_dict(results_dict: Dict[str, Any]):
        model_id = results_dict['model_id']
        info = pd.Series(results_dict['info'])
        params = results_dict['params']
        param_names = results_dict['param_names']
        variables = results_dict['variables']
        variable_names = results_dict['variable_names']
        info = pd.concat([info,
                          pd.Series(variables, index=variable_names, dtype=float),
                          pd.Series(params, index=param_names, dtype=float)])
        info['model_id'] = model_id
        model_func = results_dict['model_func']
        info['model_name'] = model_func.__name__
        x_min = results_dict['x_min']
        x_max = results_dict['x_max']
        info['x_min'] = x_min
        info['x_max'] = x_max
        input_params = np.hstack([variables, params]) if variable_names is not None else params
        return ModelItem(model_id, info, input_params, model_func, x_min, x_max)

    def read_row_from_params_table(self, params_table: pd.DataFrame, model_id_key: str):
        self.params = params_table.loc[params_table[model_id_key] == self.model_id].squeeze()
        self.params.name = None
        return self

    @property
    def data(self) -> pd.DataFrame:
        x = np.linspace(self.x_min, self.x_max, self.resolution)
        y = self.model_func(x, self.params)
        return pd.DataFrame({'x': x, 'y': y})

    @property
    def test_id(self) -> str:
        return self.info['test id']

    def write_data_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.model_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self


class ModelSet:
    def __init__(
            self,
            model_func: Callable[[np.ndarray, List[float]], np.ndarray],
            param_names: List[str],
            var_names: List[str]|None = None,
            bounds: List[Tuple[float, float]]|None = None,
            initial_guess: np.ndarray|None = None,
            scipy_func: str = 'minimize',
            scipy_kwargs: Dict[str, Any]|None = None,
    ):
        self.model_func = model_func
        self.params_table = pd.DataFrame(columns=['model id'] + param_names)
        self.results_dict_list = []
        self.param_names = param_names
        self.var_names = var_names
        self.bounds = bounds
        self.initial_guess = initial_guess if initial_guess is not None else [0.0]*len(param_names)
        self.scipy_func = scipy_func
        self.scipy_kwargs = scipy_kwargs if scipy_kwargs is not None else {}
        self.sample_size: int|None = None
        self.fitted_ds: DataSet|None = None
        self.x_key: str|None = None
        self.y_key: str|None = None
        self.model_map: map|None = None

    @property
    def model_items(self) -> List[ModelItem]:
        return [mi for mi in copy.deepcopy(self.model_map)]

    @staticmethod
    def from_info_table(info_table: pd.DataFrame,
                        model_func: Callable[[np.ndarray, List[float]], np.ndarray],
                        param_names: List[str],
                        model_id_key: str = 'model_id') -> 'ModelSet':
        model_ids = info_table[model_id_key].values
        info_rows = [info_table.drop(columns=param_names).iloc[i] for i in range(len(info_table))]
        params_lists = [info_table[param_names].iloc[i].values for i in range(len(info_table))]
        model_funcs = [model_func for _ in range(len(info_table))]
        x_mins = [info_table['x_min'].iloc[i] for i in range(len(info_table))]
        x_maxs = [info_table['x_max'].iloc[i] for i in range(len(info_table))]
        resolutions = [info_table['resolution'].iloc[i] for i in range(len(info_table))]
        ms = ModelSet(model_func, param_names)
        ms.model_map = map(ModelItem, model_ids, info_rows, params_lists, model_funcs, x_mins, x_maxs, resolutions)
        return ms

    def objective_function(self, params: List[float], di: DataItem) -> float:
        data = di.data[di.data[self.x_key] > 0]
        if self.var_names is not None:
            variables = [di.info[var_name] for var_name in self.var_names]
            params = np.hstack([variables, params])
        x_data = data[self.x_key].values
        y_data = data[self.y_key].values
        sampling_stride = int(len(x_data)/self.sample_size)
        if sampling_stride < 1:
            sampling_stride = 1
        x_data = x_data[::sampling_stride]
        y_data = y_data[::sampling_stride]
        y_model = self.model_func(x_data, params)
        # return max((y_data - y_model)/np.sqrt(len(y_data)))
        return np.linalg.norm((y_data - y_model)/np.sqrt(len(y_data)))**2

    def predict(self, resolution: int = 50) -> DataSet:
        predict_ds = DataSet()

        def update_resolution(mi: ModelItem):
            mi.resolution = resolution
            mi.info['resolution'] = resolution
            return mi

        self.model_map = map(lambda mi: update_resolution(mi), self.model_map)
        predict_ds.data_map = copy.deepcopy(self.model_map)
        return predict_ds

    def fit_to(self, ds: DataSet, x_key: str, y_key: str, sample_size: int = 50) -> None:
        self.fitted_ds = ds
        self.x_key = x_key
        self.y_key = y_key
        self.sample_size = sample_size
        for _ in tqdm(map(self.fit_item, ds.data_items), unit='fits', leave=False):
            pass
        self.model_map = map(ModelItem.from_results_dict, self.results_dict_list)

    def fit_item(self, di: DataItem) -> None:
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
            raise ValueError(
                f'Invalid scipy_func: {self.scipy_func}\nMust be one of:'
                f' minimize, differential_evolution, basinhopping, dual_annealing, shgo, brute')
        model_id = 'model_' + str(di.info[0])
        results_dict = {
            'model_id': model_id,
            'info': di.info,
            'params': result.x,
            'param_names': self.param_names,
            'variables': [di.info[var_name] for var_name in self.var_names] if self.var_names is not None else None,
            'variable_names': self.var_names,
            'model_func': self.model_func,
            'x_min': di.data[self.x_key].min(),
            'x_max': di.data[self.x_key].max(),
        }
        self.results_dict_list.append(results_dict)
        self.params_table = pd.concat([self.params_table,
                                       pd.DataFrame([[model_id] + list(result.x)],
                                                    columns=['model id'] + self.param_names)])


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

        if not np.isclose(x[0], 0):
            y_trial_0 = E*(x[1])
            f_trial_0 = np.abs(y_trial_0) - y_yield(0)
            if f_trial_0 <= 0:
                y[0] = E*x[0]
            else:
                d_aps = op.root(lambda d: f_trial_0 - d*E - y_yield(d) + y_yield(0), 0).x[0]
                y[0] = y_trial_0*(1 - d_aps*E/np.abs(y_trial_0))

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
    return lambda a: s_y + E*(a - Q*a**2)


@iso_return_map
def voce(mat_params):
    """Exponential isotropic hardening yield function."""
    E, s_y, s_u, d = mat_params
    return lambda a: s_y + (s_u - s_y)*(1 - np.exp(-d*a))


@iso_return_map
def ramberg(mat_params):
    """Ramberg-Osgood isotropic hardening yield function."""
    E, s_y, C, n = mat_params
    return lambda a: s_y + C*(np.sign(a)*(np.abs(a))**n)


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
