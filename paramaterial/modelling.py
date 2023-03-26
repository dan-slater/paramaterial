"""
Module for modelling materials test data.
"""
import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import pandas as pd
import scipy.optimize as op
from tqdm import tqdm

from paramaterial.plug import DataItem, DataSet


def make_representative_data(ds: DataSet, info_path: str, data_dir: str, repres_col: str, group_by_keys: List[str],
                             interp_by: str, interp_res: int = 200,
                             interp_range: Union[str, Tuple[float, float]] = 'outer',
                             group_info_cols: Optional[List[str]] = None):
    """Make representative curves of the ds and save them to a directory.

    Args:
        ds: The ds to make representative curves from.
        data_dir: The directory to save the representative curves to.
        info_path: The path to the info file.
        repres_col: The column to group by.
        group_by_keys: The columns to group by.
        interp_by: The column to interpolate by.
        interp_res: The resolution of the interpolation.
        min_interp_val: The minimum value of the interpolation.
        interp_end: The end of the interpolation.
        group_info_cols: The columns to group by.

    Returns:
        None
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    value_lists = [ds.info_table[col].unique() for col in group_by_keys]

    # make a dataset filter for each representative curve
    subset_filters = []
    for i in range(len(value_lists[0])):
        subset_filters.append({group_by_keys[0]: value_lists[0][i]})
    for i in range(1, len(group_by_keys)):
        new_filters = []
        for fltr in subset_filters:
            for value in value_lists[i]:
                new_filter = fltr.copy()
                new_filter[group_by_keys[i]] = value
                new_filters.append(new_filter)
        subset_filters = new_filters

    # make list of repres_ids and initialise info table for the representative data
    repres_ids = [f'repres_id_{i + 1:0>4}' for i in range(len(subset_filters))]
    repr_info_table = pd.DataFrame(columns=['repres_id'] + group_by_keys)

    # make representative curves and take means of info table columns
    for repres_id, subset_filter in zip(repres_ids, subset_filters):
        # get representative subset
        repres_subset = ds.subset(subset_filter)
        if repres_subset.info_table.empty:
            continue
        # add row to repr_info_table
        repr_info_table = pd.concat(
            [repr_info_table,
             pd.DataFrame({'repres_id': [repres_id], **subset_filter, 'nr averaged': [len(repres_subset)]})])

        # add means of group info columns to repr_info_table
        if group_info_cols is not None:
            for col in group_info_cols:
                df_col = repres_subset.info_table[col]
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, '' + col] = df_col.mean()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'std_' + col] = df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'upstd_' + col] = df_col.mean() + df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'downstd_' + col] = df_col.mean() - df_col.std()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'max_' + col] = df_col.max()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'min_' + col] = df_col.min()

        # find minimum of maximum interp_by vals in subset
        if interp_range == 'outer':
            min_interp_val = min([min(dataitem.data[interp_by]) for dataitem in repres_subset])
            max_interp_val = max([max(dataitem.data[interp_by]) for dataitem in repres_subset])
        elif interp_range == 'inner':
            min_interp_val = max([min(dataitem.data[interp_by]) for dataitem in repres_subset])
            max_interp_val = min([max(dataitem.data[interp_by]) for dataitem in repres_subset])
        elif type(interp_range) == tuple:
            min_interp_val = interp_range[0]
            max_interp_val = interp_range[1]
        else:
            raise ValueError(f'interp_range must be "outer", "inner" or a tuple, not {interp_range}')

        # make monotonically increasing vector to interpolate by
        interp_vec = np.linspace(min_interp_val, max_interp_val, interp_res)

        # make interpolated data for averaging, staring at origin
        interp_data = pd.DataFrame(data={interp_by: interp_vec})

        for n, dataitem in enumerate(repres_subset):
            # drop columns and rows outside interp range
            data = dataitem.data[[interp_by, repres_col]].reset_index(drop=True)
            data = data[(data[interp_by] <= max_interp_val)&(data[interp_by] >= min_interp_val)]
            # interpolate the repr_by column and add to interp_data
            # add 0 to start of data to ensure interpolation starts at origin
            interp_data[f'interp_{repres_col}_{n}'] = np.interp(interp_vec, data[interp_by].tolist(),
                                                                data[repres_col].tolist())

        # make representative data from stats of interpolated data
        interp_data = interp_data.drop(columns=[interp_by])
        repr_data = pd.DataFrame({f'{interp_by}': interp_vec})
        repr_data[f'{repres_col}'] = interp_data.mean(axis=1)
        repr_data[f'std_{repres_col}'] = interp_data.std(axis=1)
        repr_data[f'up_std_{repres_col}'] = repr_data[f'{repres_col}'] + repr_data[f'std_{repres_col}']
        repr_data[f'down_std_{repres_col}'] = repr_data[f'{repres_col}'] - repr_data[f'std_{repres_col}']
        repr_data[f'up_2std_{repres_col}'] = repr_data[f'{repres_col}'] + 2*repr_data[f'std_{repres_col}']
        repr_data[f'down_2std_{repres_col}'] = repr_data[f'{repres_col}'] - 2*repr_data[f'std_{repres_col}']
        repr_data[f'up_3std_{repres_col}'] = repr_data[f'{repres_col}'] + 3*repr_data[f'std_{repres_col}']
        repr_data[f'down_3std_{repres_col}'] = repr_data[f'{repres_col}'] - 3*repr_data[f'std_{repres_col}']
        repr_data[f'min_{repres_col}'] = interp_data.min(axis=1)
        repr_data[f'max_{repres_col}'] = interp_data.max(axis=1)
        repr_data[f'q1_{repres_col}'] = interp_data.quantile(0.25, axis=1)
        repr_data[f'q3_{repres_col}'] = interp_data.quantile(0.75, axis=1)

        # write the representative data and info
        repr_data.to_csv(os.path.join(data_dir, f'{repres_id}.csv'), index=False)
        repr_info_table.to_excel(info_path, index=False)


def make_representative_info(ds: DataSet, repr_by_cols: List[str], group_info_cols: List[str] = None):
    """Make a table of representative info for each group in a DataSet.

    Args:
        ds: DataSet to make representative info for.
        info_path: Path to save representative info table to.
        repr_by_cols: Columns to group by and make representative info for.
        group_info_cols: Columns to include in representative info table.
    """
    subset_filters = []
    value_lists = [ds.info_table[col].unique() for col in repr_by_cols]
    for i in range(len(value_lists[0])):
        subset_filters.append({repr_by_cols[0]: [value_lists[0][i]]})
    for i in range(1, len(repr_by_cols)):
        new_filters = []
        for fltr in subset_filters:
            for value in value_lists[i]:
                new_filter = fltr.copy()
                new_filter[repr_by_cols[i]] = [value]
                new_filters.append(new_filter)
        subset_filters = new_filters

    # make list of repres_ids and initialise info table for the representative data
    repres_ids = [f'repres_id_{i + 1:0>4}' for i in range(len(subset_filters))]
    repr_info_table = pd.DataFrame(columns=['repres_id'] + repr_by_cols)

    for fltr, repres_id in zip(subset_filters, repres_ids):
        # get representative subset
        repr_subset = ds.subset(fltr)
        if repr_subset.info_table.empty:
            continue
        # add row to repr_info_table
        repr_info_table = pd.concat(
            [repr_info_table, pd.DataFrame({'repres_id': [repres_id], **fltr, 'nr averaged': [len(repr_subset)]})])

        # add means of group info columns to repr_info_table
        if group_info_cols is not None:
            for col in group_info_cols:
                df_col = repr_subset.info_table[col]
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, '' + col] = df_col.mean()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'std_' + col] = df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'upstd_' + col] = df_col.mean() + df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'downstd_' + col] = df_col.mean() - df_col.std()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'max_' + col] = df_col.max()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'min_' + col] = df_col.min()

    return repr_info_table


@dataclass
class ModelItem:
    """Class that acts as a model DataItem."""
    model_id: str
    info: pd.Series
    params: List[float]
    model_func: Callable[[np.ndarray, List[float]], np.ndarray]
    x_key: str
    y_key: str
    x_min: float
    x_max: float
    resolution: int = 50

    @property
    def data(self) -> pd.DataFrame:
        """Generate the model data and return as a DataFrame."""
        x = np.linspace(self.x_min, self.x_max, self.resolution)
        y = self.model_func(x, self.params)
        # return pd.DataFrame({'Strain': x, 'Stress(MPa)': y})
        return pd.DataFrame({self.x_key: x, self.y_key: y})

    @property
    def test_id(self) -> str:
        """Return the test_id of the corresponding DataItem."""
        return self.info['test_id']

    @staticmethod
    def from_results_dict(results_dict: Dict[str, Any]):
        model_id = results_dict['model_id']
        info = pd.Series(results_dict['info'])
        params = results_dict['params']
        param_names = results_dict['param_names']
        error = results_dict['error']
        variables = results_dict['variables']
        variable_names = results_dict['variable_names']
        var_vals = pd.Series(variables, index=variable_names, dtype=float)
        param_vals = pd.Series(params, index=param_names, dtype=float)
        param_vals['error'] = error
        info = pd.concat([info, var_vals, param_vals])
        info['model_id'] = model_id
        model_func = results_dict['model_func']
        info['model_name'] = model_func.__name__
        x_key = results_dict['x_key']
        y_key = results_dict['y_key']
        x_min = results_dict['x_min']
        x_max = results_dict['x_max']
        info['x_min'] = x_min
        info['x_max'] = x_max
        input_params = np.hstack([variables, params]) if variable_names is not None else params
        # drop duplicate columns from info
        info = info.loc[~info.index.duplicated(keep='first')]
        return ModelItem(model_id, info, input_params, model_func, x_key, y_key, x_min, x_max)

    def read_info_from(self, info_table: pd.DataFrame, test_id_key: str):
        self.info = info_table.loc[info_table[test_id_key] == self.test_id].squeeze()
        self.info.name = None
        return self

    def read_row_from_params_table(self, params_table: pd.DataFrame, model_id_key: str):
        self.params = params_table.loc[params_table[model_id_key] == self.model_id].squeeze()
        self.params.name = None
        return self


class ModelSet:
    """Class that acts as model DataSet."""

    def __init__(self, model_func: Callable[[np.ndarray, List[float]], np.ndarray], param_names: List[str],
                 var_names: Optional[List[str]] = None, bounds: Optional[List[Tuple[float, float]]] = None,
                 initial_guess: Optional[np.ndarray] = None, scipy_func: str = 'minimize',
                 scipy_kwargs: Optional[Dict[str, Any]] = None, ):
        self.model_func = model_func
        self.params_table = pd.DataFrame(columns=['model_id'] + param_names)
        self.results_dict_list = []
        self.param_names = param_names
        self.var_names = var_names
        self.bounds = bounds
        self.initial_guess = initial_guess if initial_guess is not None else [0.0]*len(param_names)
        self.scipy_func = scipy_func
        self.scipy_kwargs = scipy_kwargs if scipy_kwargs is not None else {}
        self.sample_size: Optional[int] = None
        self.fitted_ds: Optional[DataSet] = None
        self.x_key: Optional[str] = None
        self.y_key: Optional[str] = None
        self.model_items: Optional[List] = None

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

    def fit_to(self, ds: DataSet, x_key: str, y_key: str, sample_size: int = 50) -> None:
        """Fit the model to the DataSet.

        Args:
            ds: DataSet to fit the model to.
            x_key: Key of the x values in the DataSet.
            y_key: Key of the y values in the DataSet.
            sample_size: Number of samples to draw from the x-y data in the DataSet.

        Returns: None
        """
        self.fitted_ds = ds
        self.x_key = x_key
        self.y_key = y_key
        self.sample_size = sample_size
        for _ in tqdm(map(self._fit_item, ds.data_items), unit='fits', leave=False):
            pass
        self.model_items = list(map(ModelItem.from_results_dict, self.results_dict_list))

    def predict(self, resolution: int = 50, xmin=None, xmax=None) -> DataSet:
        """Return a ds with generated data with optimised model parameters added to the info table.

        Args:
            resolution: Number of points to generate between the x_min and x_max.

        Returns: DataSet with generated data.
        """
        predict_ds = DataSet()

        # predict_ds.test_id_key = 'model_id'

        def update_resolution(mi: ModelItem):
            mi.resolution = resolution
            mi.info['resolution'] = resolution
            mi.info['x_min'] = xmin if xmin is not None else mi.info['x_min']
            mi.info['x_max'] = xmax if xmax is not None else mi.info['x_max']
            return mi

        self.model_items = list(map(lambda mi: update_resolution(mi), self.model_items))
        predict_ds.data_items = copy.deepcopy(self.model_items)
        for di in predict_ds.data_items:
            di.info['test_id'] = di.info['model_id']
        predict_ds.info_table = pd.DataFrame([di.info for di in predict_ds.data_items])

        return predict_ds

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
