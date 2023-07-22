from typing import Callable, List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import optimize as op

from paramaterial.plug import DataSet


class ModelSet:
    """Class that acts as model DataSet."""
    SCIPY_METHODS = {
        'minimize': op.minimize,
        'differential_evolution': op.differential_evolution,
        'basinhopping': op.basinhopping,
        'dual_annealing': op.dual_annealing,
        'shgo': op.shgo,
        'brute': op.brute,
    }

    def __init__(self,
                 model_func: Callable[[np.ndarray, Tuple[float]], np.ndarray],
                 param_names: List[str],
                 variable_names: List[str] = None,
                 bounds: List[Tuple[float]] = None,
                 initial_guess: Tuple[float] = None,
                 scipy_method_name: str = 'minimize',
                 scipy_method_kwargs: Dict[str, Any] = None
                 ):
        self.model_func = model_func
        self.param_names = param_names
        self.variable_names = variable_names
        self.bounds = bounds
        self.initial_guess = initial_guess
        self.scipy_method_name = scipy_method_name
        self.scipy_method_kwargs = scipy_method_kwargs

        if self.scipy_method_name not in self.SCIPY_METHODS:
            raise ValueError(f'Invalid scipy_method_name: {self.scipy_method_name}\nMust be one of:'
                             f' {list(self.SCIPY_METHODS.keys())}')

        self.scipy_method = self.SCIPY_METHODS[self.scipy_method_name]

    def _error_norm(self, x_data: np.ndarray, y_data: np.ndarray, params: Tuple[float]) -> float:
        y_model = self.model_func(x_data, params)
        return np.linalg.norm((y_data - y_model)/np.sqrt(len(y_data)))**2

    def _objective_function(self, params: Tuple[float], variables: Tuple[float]) -> float:
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

    def _fit_item(self, di: DataItem) -> None:
        if self.scipy_func == 'minimize':
            result = op.minimize(self._objective_function, self.initial_guess, args=(di,), bounds=self.bounds,
                                 **self.scipy_kwargs)
        elif self.scipy_func == 'differential_evolution':
            result = op.differential_evolution(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'basinhopping':
            result = op.basinhopping(self._objective_function, self.initial_guess,
                                     minimizer_kwargs=dict(args=(di,), bounds=self.bounds, **self.scipy_kwargs))
        elif self.scipy_func == 'dual_annealing':
            result = op.dual_annealing(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'shgo':
            result = op.shgo(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'brute':
            result = op.brute(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        else:
            raise ValueError(f'Invalid scipy_func: {self.scipy_func}\nMust be one of:'
                             f' minimize, differential_evolution, basinhopping, dual_annealing, shgo, brute')
        model_id = 'model_' + str(di.info[0])
        results_dict = {'model_id': model_id, 'info': di.info, 'params': result.x, 'param_names': self.param_names,
                        'error': result.fun,
                        'variables': [di.info[var_name] for var_name in
                                      self.var_names] if self.var_names is not None else None,
                        'variable_names': self.var_names, 'model_func': self.model_func,
                        'x_key': self.x_key, 'y_key': self.y_key,
                        'x_min': di.data[self.x_key].min(),
                        'x_max': di.data[self.x_key].max(), }
        self.results_dict_list.append(results_dict)
        self.params_table = pd.concat(
            [self.params_table, pd.DataFrame([[model_id] + list(result.x) + [result.fun]],
                                             columns=['model_id'] + self.param_names + ['fitting error'])])


    def fit_to(self, ds: DataSet, x_col: str, y_col: str,
               sample_range: Tuple[float] = None, sample_size: int = 50) -> None:
        """Fit the model to the DataSet.

        Args:
            ds: DataSet to fit the model to.
            x_col: Key of the x values in the DataSet.
            y_col: Key of the y values in the DataSet.
            sample_range: Range of x values to sample from the DataSet.

        Returns: None
        """

        total_error = 0
        params = self.initial_guess

        for di in ds:

            x_data = di.data[x_col].values
            y_data = di.data[y_col].values
            if sample_range is not None:
                x_data = x_data[(x_data > sample_range[0]) & (x_data < sample_range[1])]
                y_data = y_data[(x_data > sample_range[0]) & (x_data < sample_range[1])]
            sampling_stride = int(len(x_data)/sample_size)
            if sampling_stride < 1:
                sampling_stride = 1
            x_data = x_data[::sampling_stride]
            y_data = y_data[::sampling_stride]

            vars = [di.info[var_name] for var_name in self.variable_names] if self.variable_names is not None else []
            vars_and_params = tuple(vars) + tuple(params)

            total_error += self._error_norm(x_data, y_data, vars_and_params)

    # ,
    #

    # @staticmethod
    # def from_info_table(info_table: pd.DataFrame, model_func: Callable[[np.ndarray, List[float]], np.ndarray],
    #                     param_names: List[str], model_id_key: str = 'model_id') -> 'ModelSet':
    #     """Create a ModelSet from an info table."""
    #     model_ids = info_table[model_id_key].values
    #     info_rows = [info_table.drop(columns=param_names).iloc[i] for i in range(len(info_table))]
    #     params_lists = [info_table[param_names].iloc[i].values for i in range(len(info_table))]
    #     model_funcs = [model_func for _ in range(len(info_table))]
    #     x
    #     x_mins = [info_table['x_min'].iloc[i] for i in range(len(info_table))]
    #     x_maxs = [info_table['x_max'].iloc[i] for i in range(len(info_table))]
    #     resolutions = [info_table['resolution'].iloc[i] for i in range(len(info_table))]
    #     ms = ModelSet(model_func, param_names)
    #     ms.model_items = list(
    #         map(ModelItem, model_ids, info_rows, params_lists, model_funcs, x_keys, y_keys, x_mins, x_maxs,
    #             resolutions))
    #     return ms

    # def fit_to(self, ds: DataSet, x_key: str, y_key: str, sample_size: int = 50) -> None:
    #     """Fit the model to the DataSet.
    #
    #     Args:
    #         ds: DataSet to fit the model to.
    #         x_key: Key of the x values in the DataSet.
    #         y_key: Key of the y values in the DataSet.
    #         sample_size: Number of samples to draw from the x-y data in the DataSet.
    #
    #     Returns: None
    #     """
    #     self.fitted_ds = ds
    #     self.x_key = x_key
    #     self.y_key = y_key
    #     self.sample_size = sample_size
    #     for _ in tqdm(map(self._fit_item, ds.data_items), unit='fits', leave=False):
    #         pass
    #     self.model_items = list(map(ModelItem.from_results_dict, self.results_dict_list))
    #
    # def predict(self, resolution: int = 50, xmin=None, xmax=None, info_table=None) -> DataSet:
    #     """Return a ds with generated data with optimised model parameters added to the info table.
    #     If an info table is provided, data items will be generated to match the rows of the info table, using the
    #     var_names and param_names and model_func.
    #
    #     Args:
    #         resolution: Number of points to generate between the x_min and x_max.
    #
    #     Returns: DataSet with generated data.
    #     """
    #     predict_ds = DataSet()
    #
    #     if info_table is not None:
    #         predict_ds.info_table = info_table
    #         for i, row in info_table.iterrows():
    #             # make a data item for each row in the info table
    #             x_data = np.linspace(row['x_min'], row['x_max'], resolution)
    #             y_data = self.model_func(x_data, row[self.param_names].values)
    #             data = {self.x_key: x_data, self.y_key: y_data}
    #             info = row
    #             test_id = row['test_id']
    #             di = DataItem()
    #
    #
    #     # predict_ds.test_id_key = 'model_id'
    #
    #     def update_resolution(mi: ModelItem):
    #         mi.resolution = resolution
    #         mi.info['resolution'] = resolution
    #         mi.info['x_min'] = xmin if xmin is not None else mi.info['x_min']
    #         mi.info['x_max'] = xmax if xmax is not None else mi.info['x_max']
    #         return mi
    #
    #     self.model_items = list(map(lambda mi: update_resolution(mi), self.model_items))
    #     predict_ds.data_items = copy.deepcopy(self.model_items)
    #     for di in predict_ds.data_items:
    #         di.info['test_id'] = di.info['model_id']
    #     predict_ds.info_table = pd.DataFrame([di.info for di in predict_ds.data_items])
    #
    #     return predict_ds

    def _objective_function(self, params: List[float], di: DataItem) -> float:
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

    def _fit_item(self, di: DataItem) -> None:
        if self.scipy_func == 'minimize':
            result = op.minimize(self._objective_function, self.initial_guess, args=(di,), bounds=self.bounds,
                                 **self.scipy_kwargs)
        elif self.scipy_func == 'differential_evolution':
            result = op.differential_evolution(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'basinhopping':
            result = op.basinhopping(self._objective_function, self.initial_guess,
                                     minimizer_kwargs=dict(args=(di,), bounds=self.bounds, **self.scipy_kwargs))
        elif self.scipy_func == 'dual_annealing':
            result = op.dual_annealing(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'shgo':
            result = op.shgo(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'brute':
            result = op.brute(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        else:
            raise ValueError(f'Invalid scipy_func: {self.scipy_func}\nMust be one of:'
                             f' minimize, differential_evolution, basinhopping, dual_annealing, shgo, brute')
        model_id = 'model_' + str(di.info[0])
        results_dict = {'model_id': model_id, 'info': di.info, 'params': result.x, 'param_names': self.param_names,
                        'error': result.fun,
                        'variables': [di.info[var_name] for var_name in
                                      self.var_names] if self.var_names is not None else None,
                        'variable_names': self.var_names, 'model_func': self.model_func,
                        'x_key': self.x_key, 'y_key': self.y_key,
                        'x_min': di.data[self.x_key].min(),
                        'x_max': di.data[self.x_key].max(), }
        self.results_dict_list.append(results_dict)
        self.params_table = pd.concat(
            [self.params_table, pd.DataFrame([[model_id] + list(result.x) + [result.fun]],
                                             columns=['model_id'] + self.param_names + ['fitting error'])])

    @property
    def fitting_results(self) -> pd.DataFrame:
        """Return a DataFrame with the results of the fitting."""
        # get the fitted parameters and make a table with them and include only the relevant info and the fitting error
        return self.params_table.merge(self.fitted_ds.info_table, on='model_id')
