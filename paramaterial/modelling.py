from typing import Callable, List, Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import optimize as op

from paramaterial.plug import DataSet, DataItem


def _call_scipy_method(scipy_method: str,
                       initial_guess: Tuple[float, ...],
                       bounds: List[Tuple[float, float]],
                       objective_function: Callable[[Tuple[float, ...], Union[DataItem, pd.DataFrame]], float],
                       storage_object: Union[DataItem, pd.DataFrame],
                       **scipy_method_kwargs):
    if scipy_method == 'minimize':
        result = op.minimize(objective_function, np.array(initial_guess), args=(storage_object,),
                             bounds=bounds,
                             **scipy_method_kwargs)
    elif scipy_method == 'differential_evolution':
        result = op.differential_evolution(objective_function, bounds, args=(storage_object,),
                                           **scipy_method_kwargs)
    elif scipy_method == 'basinhopping':
        result = op.basinhopping(objective_function, initial_guess,
                                 minimizer_kwargs=dict(args=(storage_object,), bounds=bounds,
                                                       **scipy_method_kwargs))
    elif scipy_method == 'dual_annealing':
        result = op.dual_annealing(objective_function, bounds, args=(storage_object,),
                                   **scipy_method_kwargs)
    elif scipy_method == 'shgo':
        result = op.shgo(objective_function, bounds, args=(storage_object,), **scipy_method_kwargs)
    elif scipy_method == 'brute':
        result = op.brute(objective_function, bounds, args=(storage_object,), **scipy_method_kwargs)
    else:
        raise ValueError(f'Invalid scipy_method: {scipy_method}\nMust be one of:'
                         f' minimize, differential_evolution, basinhopping, dual_annealing, shgo, brute')
    return result


def _error_norm(y_data: np.ndarray, y_model: np.ndarray) -> float:
    return np.linalg.norm((y_data - y_model) / np.sqrt(len(y_data))) ** 2


class ModelTable:
    """Class that fits a model between info_table columns."""

    def __init__(self,
                 model_func: Callable[[np.ndarray, Tuple[float]], np.ndarray],
                 x_key: str,
                 y_key: str,
                 param_names: List[str],
                 variable_names: List[str] = None,
                 bounds: List[Tuple[float, float]] = None,
                 initial_guess: Tuple[float] = None,
                 scipy_func: str = 'minimize',
                 scipy_method_kwargs: Dict[str, Any] = None
                 ):
        self.model_func = model_func
        self.x_key = x_key
        self.y_key = y_key
        self.variable_names = variable_names

        self.param_names = param_names
        self.bounds = bounds
        self.initial_guess = initial_guess if initial_guess else [0.0] * len(param_names)

        self.fitting_table: pd.DataFrame = pd.DataFrame(
            columns=variable_names + param_names + ['error'])

    def _objective_function(self, params: Tuple[float, ...], info_table: pd.DataFrame) -> float:
        params = pd.Series(params, index=self.param_names)
        info_table = info_table.assign(**params.to_dict())

        variables_and_params_keys = self.variable_names + self.param_names if self.variable_names else self.param_names
        y_model = info_table.apply(lambda row: self.model_func(row[self.x_key], row[variables_and_params_keys]), axis=1)
        y_info = info_table[self.y_key].values

        info_table[f'{self.y_key}_{self.model_func.__name__}'] = y_model
        info_table[f'error_{self.y_key}_{self.model_func.__name__}'] = (y_info - y_model) / np.sqrt(len(y_info))

        return _error_norm(y_info, y_model.values)

    def _fit_table(self, info_table: pd.DataFrame):
        pass

    def fit_table(self, info_table: pd.DataFrame):
        pass


class ModelSet:
    """Class that acts as model DataSet."""

    def __init__(self,
                 model_func: Callable[[np.ndarray, Tuple[float]], np.ndarray],
                 var_names: List[str],
                 param_names: List[str],
                 bounds: List[Tuple[float, float]] = None,
                 initial_guess: Tuple[float] = None,
                 sample_range: Tuple[float, float] = (None, None),
                 sample_size: int = 50,
                 model_id_key: str = 'model_id',
                 scipy_func: str = 'minimize',
                 ):
        self.model_func = model_func
        self.variable_names = var_names  # Updated name
        self.param_names = param_names
        self.bounds = bounds
        self.initial_guess = initial_guess if initial_guess else [0.0] * len(param_names)
        self.sample_range = sample_range
        self.sample_size = sample_size
        self.model_id_key = model_id_key
        self.scipy_func = scipy_func
        # self.fitting_table: pd.DataFrame = pd.DataFrame(
        #     columns=[model_id_key] + ['var_' + var_name for var_name in var_names] +
        #             ['param_' + param_name for param_name in param_names] + ['error'])
        self.fitting_table: pd.DataFrame = pd.DataFrame(
            columns=[model_id_key] + ['var_' + var_name for var_name in var_names] +
                    [param_name for param_name in param_names] + ['error'])

    def fit_to(self, ds: DataSet, x_key: str, y_key: str, sample_range: Tuple[float, float] = (None, None),
               sample_size: int = 50, **scipy_method_kwargs):
        # Set the keys
        self.x_col = x_key
        self.y_col = y_key

        # Call the existing fit method
        self.fit_items(ds, sample_range, sample_size, self.scipy_func, **scipy_method_kwargs)

    def predict(self, x_range: Optional[Tuple[float, float, float]] = None,
                xmin: Optional[float] = None,
                xmax: Optional[float] = None,
                info_table: Optional[pd.DataFrame] = None,
                model_id_key: str = 'model_id'):

        # If x_range is not provided, check for xmin and xmax
        if x_range is None:
            if xmin is None or xmax is None:
                x_range = (0, 0.01, 0.0001)  # Example default value
            else:
                x_range = (xmin, xmax, 0.0001)  # Example step value, adjust as needed

        return self.predict_ds(x_range, info_table, model_id_key)

    def _sample_data(self, di: DataItem) -> Tuple[np.ndarray, np.ndarray]:
        sample_range = self.sample_range
        sample_size = self.sample_size
        x_data = di.data[self.x_col].values
        y_data = di.data[self.y_col].values
        if sample_range[0] is not None and sample_range[1] is not None:
            mask = (x_data > sample_range[0]) & (x_data < sample_range[1])
            x_data, y_data = x_data[mask], y_data[mask]
        sampling_stride = max(int(len(x_data) / sample_size), 1)
        x_data, y_data = x_data[::sampling_stride], y_data[::sampling_stride]
        return x_data, y_data

    def _objective_function(self, params: Tuple[float, ...], di: DataItem) -> float:
        x_data, y_data = self._sample_data(di)
        if self.variable_names is not None:
            variables = tuple(di.info[var_name] for var_name in self.variable_names)
        else:
            variables = ()
        variables_and_params = variables + tuple(params)
        y_model = self.model_func(x_data, variables_and_params)
        return _error_norm(y_data, y_model)

    def _fit_item(self, di: DataItem, scipy_method: str, **scipy_method_kwargs) -> op.OptimizeResult:
        return _call_scipy_method(scipy_method=scipy_method, initial_guess=self.initial_guess, bounds=self.bounds,
                                  objective_function=self._objective_function, storage_object=di, **scipy_method_kwargs)

    def fit_items(self, ds: DataSet, sample_range: Tuple[float, float] = (None, None), sample_size: int = 50,
                  scipy_method: str = 'minimize', **scipy_method_kwargs):
        self.sample_range = sample_range
        self.sample_size = sample_size
        # fit each DataItem and add a row to fitting_table for each
        fitting_dfs = []
        pad = int(np.log10(len(ds))) + 1
        for i, di in enumerate(ds):
            model_id = f'{self.model_id_key}_{i+1:0{pad}}'
            # run optimisation
            fitting_result = self._fit_item(di, scipy_method, **scipy_method_kwargs)
            # extract results
            params = fitting_result.x
            error = fitting_result.fun
            variables = di.info[self.variable_names]
            # add 'var_' prefix to variable names
            variables.index = 'var_' + variables.index
            # add 'param_' prefix to param names
            # params = pd.Series(params, index='param_' + pd.Series(self.param_names))
            params = pd.Series(params, index=pd.Series(self.param_names))
            # add row to fitting_table
            # concatenate data
            data = np.hstack([model_id, variables, params, error, di.info.to_list()]).reshape(1, -1)
            # define columns
            columns = self.fitting_table.columns.tolist() + di.info.index.to_list()
            # create DataFrame and append
            fitting_dfs.append(pd.DataFrame(data, columns=columns))
        # concatenate fitting_dfs into fitting_table
        self.fitting_table = pd.concat(fitting_dfs)

    def predict_ds(self, x_range: Tuple[float, float, float], info_table: Optional[pd.DataFrame] = None,
                   model_id_key: str = 'model_id'):
        if info_table is None:
            info_table = self.fitting_table
        # generate DataItems for each row of info_table
        model_items = []
        for model_id in info_table[model_id_key].to_list():
            di_info = info_table.loc[info_table[model_id_key] == model_id, :].squeeze()
            # extract variables and optimised params from info_table
            variables_keys = self.variable_names if self.variable_names else []
            variables_keys = ['var_' + var_key for var_key in variables_keys]
            # params_keys = ['param_' + param_key for param_key in self.param_names]
            params_keys = [param_key for param_key in self.param_names]
            variables_and_params = di_info[variables_keys + params_keys].to_list()
            # generate model data and create DataItem
            x_model = np.arange(*x_range)
            y_model = self.model_func(x_model, variables_and_params)
            data = pd.DataFrame({self.x_col: x_model, self.y_col: y_model})
            model_items.append(DataItem(model_id, data=data, info=di_info))
        # create DataSet from model_items and return
        ds = DataSet(test_id_key=model_id_key)
        ds.data_items = model_items
        ds.info_table = info_table
        return ds
#
# class ModelSet:
#     """Class that acts as model DataSet."""
#
#     def __init__(self,
#                  model_func: Callable[[np.ndarray, Tuple[float]], np.ndarray],
#                  x_col: str,
#                  y_col: str,
#                  param_names: List[str],
#                  variable_names: List[str] = None,
#                  bounds: List[Tuple[float, float]] = None,
#                  initial_guess: Tuple[float] = None,
#                  sample_range: Tuple[float, float] = (None, None),
#                  sample_size: int = 50,
#                  model_id_key: str = 'model_id'
#                  ):
#         self.model_func = model_func
#         self.x_col = x_col
#         self.y_col = y_col
#         self.variable_names = variable_names
#
#         self.param_names = param_names
#         self.bounds = bounds
#         self.initial_guess = initial_guess if initial_guess else [0.0] * len(param_names)
#         self.sample_range = sample_range
#         self.sample_size = sample_size
#
#         self.model_id_key = model_id_key
#         self.fitting_table: pd.DataFrame = pd.DataFrame(
#             columns=[model_id_key] + ['var_' + var_name for var_name in variable_names] +
#                     ['param_' + param_name for param_name in param_names] + ['error'])
#
#
