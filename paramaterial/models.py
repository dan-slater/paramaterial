"""
Module containing 1D constitutive models for fitting to stress-strain curves.
"""
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import scipy.optimize as op
from scipy.optimize import OptimizeResult
# from sklearn.metrics import mean_squared_error

from paramaterial.plug import DataItem, DataSet


class Model:
    def __init__(self,
                 objective_function: Callable[[List[float], DataItem], float],
                 initial_guess: List[float],
                 bounds: List[Tuple[float, float]] | None = None,
                 dataitem_func: Callable[[pd.Series, List[float]], DataItem] | None = None,
                 opt_result: op.OptimizeResult | None = None):
        self.objective_function = objective_function
        self.initial_guess = np.array(initial_guess)
        self.bounds = bounds
        self.dataitem_func = dataitem_func
        self.opt_result = opt_result
        self.fitted_ds: DataSet | None = None

    def fit_item(self, di: DataItem) -> DataItem:
        def di_obj_func(params) -> float:
            return self.objective_function(params, di)

        self.opt_result = op.minimize(di_obj_func, self.initial_guess, bounds=self.bounds)
        test_id = di.test_id

        return self.predict_item(di.info)

    def fit_to(self, ds: DataSet, **kwargs) -> None:
        def ds_obj_func(params) -> float:
            return sum([self.objective_function(params, di) for di in ds])

        self.opt_result = op.minimize(ds_obj_func, self.initial_guess, bounds=self.bounds, **kwargs)
        self.fitted_ds = ds

    def predict_item(self, info: pd.Series) -> DataItem:
        return self.dataitem_func(info, self.opt_result.x)

    def predict_set(self, info_table: Optional[pd.DataFrame]) -> DataSet:
        new_set = DataSet('model', 'model')
        # dataset.info_table = info_table
        #     dataset.data_map = map(self.predict_item, [info for _, info in info_table.iterrows()])
        #     return dataset
        # elif dataset is not None:
        #     dataset.data_map = map(self.predict_item, [info for _, info in info_table.iterrows()])
        # else:
        #     new_set.info_table = self.fitted_ds.info_table
        #     new_set.data_map = map(self.predict_item, [info for _, info in self.fitted_ds.info_table.iterrows()])




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

    sampling_stride = int(len(x_data) / sample_size)
    if sampling_stride < 1:
        sampling_stride = 1

    x_data = x_data[::sampling_stride]
    y_data = y_data[::sampling_stride]

    return x_data, y_data

# todo: sample data with minimum in given area (i.e both x and y tolerance)
# todo: variable sampling runtime paramaterial
# todo: full elastic range with plastic sampling only
# todo: equidistant strain sampling
# todo: if strain decreasing, omit point


@dataclass
class Model2:
    name: str
    func: callable
    params: List[str]
    bounds: List[Tuple[float, float]]
    init_guess: List[float]
    units: Dict[str, str]
    latex: Dict[str, str]
    description: str
    data: DataSet = None
    fit: OptimizeResult = None
    fit_params: Dict[str, Any] = None
    fit_errors: Dict[str, Any] = None
    fit_r2: float = None
    fit_rmse: float = None

    def __post_init__(self):
        self.fit_params = {param: None for param in self.params}
        self.fit_errors = {param: None for param in self.params}

    def fit(self, data: DataSet, obj_func: Callable[[DataItem], float], **kwargs):
        self.data = data
        self.fit = op.curve_fit(
            self.func,
            self.data.x,
            self.data.y,
            p0=self.init_guess,
            bounds=self.bounds
        )
        self.fit_params = {param: value for param, value in zip(self.params, self.fit[0])}
        self.fit_errors = {param: value for param, value in zip(self.params, np.sqrt(np.diag(self.fit[1])))}
        # self.fit_r2 = 1 - mean_squared_error(self.data.y, self.func(self.data.x, *self.fit_params))/np.var(self.data.y)
        # self.fit_rmse = mean_squared_error(self.data.y, self.func(self.data.x, *self.fit_params))


@dataclass
class ModelItem:
    info: pd.Series = None
    params: pd.Series = None
    model_func: Callable[[pd.Series], pd.DataFrame] = None

    def load_model_func(self, model_func: Callable[[pd.Series], pd.DataFrame]):
        self.model_func = model_func
        return self

    #
    # def objective_function(self, params: pd.Series, data: pd.DataFrame, x: str, y: str) -> float:
    #     residual = np.linalg.norm(data[y] - self.model_func(data[x], *params))/np.sqrt(
    #         len(self.stress_vec)),
    #     bounds = self.bounds,)
    #     constraints = op.LinearConstraint(**self.constraints))
    #
    #     def read_params(self, params_table: pd.DataFrame, model_id_key: str = 'model id'):
    #         self.params = params_table.loc[params_table[model_id_key] == self.model_id].squeeze()
    #         return self
    #
    #     def generate_data(self) -> pd.DataFrame:
    #         return self.model_func(self.params)

    @dataclass
    class ModelSet:
        """A class that fits a model to a dataset and stores the results."""
        params_path: str = None
        model_map: map = None
        params_table: pd.DataFrame = None
        dataset: DataSet = None
        var_keys: List[str] = None
        model_func: Callable[[np.ndarray, ...], np.ndarray] = None
        x: str = None
        y: str = None

        def __init__(self, params_path: str):
            self.params_path = params_path
            # self.model_map = map(lambda obj: ModelItem.read_params(obj, self.params_table), self.model_map)
            self.params_table = pd.read_excel(self.params_path)

        def objective_function(self, params) -> float:
            residual = 0.
            for di in self.dataset:
                var_dict = {var_key: di.info[var_key] for var_key in self.var_keys}
                residual += np.linalg.norm(
                    di.data[self.y].values - self.model_func(di.data[self.x], *params, **var_dict)) \
                            /np.sqrt(len(di[self.y]))
            return residual

        def fit_model(self, dataset: DataSet, x: str, y: str, var_keys: List[str],
                      model_func: Callable[[np.ndarray], np.ndarray],
                      bounds: List[Tuple[float, float]], constraints: Dict[str, Any],
                      fitting_method: str = 'minimize'
                      ) -> OptimizeResult:
            """Fit a model to a dataset and return the fit results."""
            self.dataset = dataset
            self.model_func = model_func
            self.x = x
            self.y = y
            self.var_keys = var_keys
            if fitting_method == 'minimize':
                fit_result = op.minimize(self.objective_function, bounds=bounds,
                                         constraints=op.LinearConstraint(**constraints))
            else:
                raise NotImplementedError
            return fit_result

        # def fit_models(self, dataset: DataSet, model_id_key: str = 'model id'):
        #     model_ids = self.params_table[model_id_key]
        #     self.model_map = map(lambda model_id: ModelItem(model_id), model_ids)
        #     self.model_map = map(lambda obj: ModelItem.read_params(obj, self.params_table), self.model_map)
        #     self.model_map = map(lambda obj: ModelItem.load_model_func(obj, self.get_model_func(obj.model_id)), self.model_map)
        #     self.model_map = map(lambda obj: ModelItem.generate_data(obj), self.model_map)
        #     return self

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
                *mat_params,
                vec: str = 'stress'
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
        return lambda a: s_y + E*(a - Q*a ** 2)

    @iso_return_map
    def voce(E, s_y, s_u, d):
        """Exponential isotropic hardening yield function."""
        return lambda a: s_y + (s_u - s_y)*(1 - np.exp(-d*a))

    @iso_return_map
    def ramberg(E, s_y, C, n):
        """Ramberg-Osgood isotropic hardening yield function."""
        return lambda a: s_y + C*(a ** n)

    # todo: figure out why -ve a being passed in
