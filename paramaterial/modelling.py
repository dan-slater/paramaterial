"""
Module for modelling materials test data.
"""
import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.optimize as op
from tqdm import tqdm

from paramaterial.plug import DataItem, DataSet





def make_representative_data(
        ds: DataSet, data_dir: str, info_path: str,
        repr_col: str, repr_by_cols: List[str],
        interp_by: str, interp_res: int = 200, min_interp_val: float = 0., interp_end: str = 'max_all',
        group_info_cols: List[str]|None = None
):
    """Make representative curves of the ds and save them to a directory.

    Args:
        ds: The ds to make representative curves from.
        data_dir: The directory to save the representative curves to.
        info_path: The path to the info file.
        repr_col: The column to group by.
        repr_by_cols: The columns to group by.
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

    # make list of repr_ids and initialise info table for the representative data
    repr_ids = [f'repr_id_{i + 1:0>4}' for i in range(len(subset_filters))]
    repr_info_table = pd.DataFrame(columns=['repr id'] + repr_by_cols)

    # make representative curves and take means of info table columns
    for repr_id, subset_filter in zip(repr_ids, subset_filters):
        # get representative subset
        repr_subset = ds.subset(subset_filter)
        if repr_subset.info_table.empty:
            continue
        # add row to repr_info_table
        repr_info_table = pd.concat(
            [repr_info_table, pd.DataFrame({'repr id': [repr_id], **subset_filter, 'nr averaged': [len(repr_subset)]})])

        # add means of group info columns to repr_info_table
        if group_info_cols is not None:
            for col in group_info_cols:
                repr_info_table.loc[repr_info_table['repr id'] == repr_id, 'mean_' + col] \
                    = repr_subset.info_table[col].mean()

        # find minimum of maximum interp_by vals in subset
        if interp_end == 'max_all':
            max_interp_val = max([max(dataitem.data[interp_by]) for dataitem in repr_subset])
        elif interp_end == 'min_of_maxes':
            max_interp_val = min([max(dataitem.data[interp_by]) for dataitem in repr_subset])
        else:
            raise ValueError(f'interp_end must be "max_all" or "min_of_maxes", not {interp_end}')
        # make monotonically increasing vector to interpolate by
        interp_vec = np.linspace(min_interp_val, max_interp_val, interp_res)

        # make interpolated data for averaging, staring at origin
        interp_data = pd.DataFrame(data={interp_by: interp_vec})

        for n, dataitem in enumerate(repr_subset):
            # drop columns and rows outside interp range
            data = dataitem.data[[interp_by, repr_col]].reset_index(drop=True)
            data = data[(data[interp_by] <= max_interp_val)&(data[interp_by] >= min_interp_val)]
            # interpolate the repr_by column and add to interp_data
            # add 0 to start of data to ensure interpolation starts at origin
            interp_data[f'interp_{repr_col}_{n}'] = np.interp(interp_vec, [0] + data[interp_by].tolist(),
                                                              [0] + data[repr_col].tolist())

        # make representative data from stats of interpolated data
        interp_data = interp_data.drop(columns=[interp_by])
        repr_data = pd.DataFrame({f'interp_{interp_by}': interp_vec})
        repr_data[f'mean_{repr_col}'] = interp_data.mean(axis=1)
        repr_data[f'std_{repr_col}'] = interp_data.std(axis=1)
        repr_data[f'up_std_{repr_col}'] = repr_data[f'mean_{repr_col}'] + repr_data[f'std_{repr_col}']
        repr_data[f'down_std_{repr_col}'] = repr_data[f'mean_{repr_col}'] - repr_data[f'std_{repr_col}']
        repr_data[f'up_2std_{repr_col}'] = repr_data[f'mean_{repr_col}'] + 2*repr_data[f'std_{repr_col}']
        repr_data[f'down_2std_{repr_col}'] = repr_data[f'mean_{repr_col}'] - 2*repr_data[f'std_{repr_col}']
        repr_data[f'up_3std_{repr_col}'] = repr_data[f'mean_{repr_col}'] + 3*repr_data[f'std_{repr_col}']
        repr_data[f'down_3std_{repr_col}'] = repr_data[f'mean_{repr_col}'] - 3*repr_data[f'std_{repr_col}']
        repr_data[f'min_{repr_col}'] = interp_data.min(axis=1)
        repr_data[f'max_{repr_col}'] = interp_data.max(axis=1)
        repr_data[f'q1_{repr_col}'] = interp_data.quantile(0.25, axis=1)
        repr_data[f'q3_{repr_col}'] = interp_data.quantile(0.75, axis=1)

        # make representative info from subset info

        # write the representative data and info
        repr_data.to_csv(os.path.join(data_dir, f'{repr_id}.csv'), index=False)
        repr_info_table.to_excel(info_path, index=False)


@dataclass
class ModelItem:
    """Class that acts as a model DataItem."""
    model_id: str
    info: pd.Series
    params: List[float]
    model_func: Callable[[np.ndarray, List[float]], np.ndarray]
    x_min: float
    x_max: float
    resolution: int = 50

    @property
    def data(self) -> pd.DataFrame:
        """Generate the model data and return as a DataFrame."""
        x = np.linspace(self.x_min, self.x_max, self.resolution)
        y = self.model_func(x, self.params)
        return pd.DataFrame({'x': x, 'y': y})

    @property
    def test_id(self) -> str:
        """Return the test_id of the corresponding DataItem."""
        return self.info['test_id']

    def write_data_to_csv(self, output_dir: str):
        """Write the generated data to a csv file."""
        output_path = output_dir + '/' + self.model_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self

    @staticmethod
    def _from_results_dict(results_dict: Dict[str, Any]):
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

    def _read_row_from_params_table(self, params_table: pd.DataFrame, model_id_key: str):
        self.params = params_table.loc[params_table[model_id_key] == self.model_id].squeeze()
        self.params.name = None
        return self


class ModelSet:
    """Class that acts as model DataSet."""

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
        """Return a list of ModelItem objects."""
        return [mi for mi in copy.deepcopy(self.model_map)]

    @staticmethod
    def from_info_table(info_table: pd.DataFrame,
                        model_func: Callable[[np.ndarray, List[float]], np.ndarray],
                        param_names: List[str],
                        model_id_key: str = 'model_id') -> 'ModelSet':
        """Create a ModelSet from an info table."""
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
        self.model_map = map(ModelItem._from_results_dict, self.results_dict_list)

    def predict(self, resolution: int = 50) -> DataSet:
        """Return a ds with generated data with optimised model parameters added to the info table.

        Args:
            resolution: Number of points to generate between the x_min and x_max.

        Returns: DataSet with generated data.
        """
        predict_ds = DataSet()

        def update_resolution(mi: ModelItem):
            mi.resolution = resolution
            mi.info['resolution'] = resolution
            return mi

        self.model_map = map(lambda mi: update_resolution(mi), self.model_map)
        predict_ds.data_map = copy.deepcopy(self.model_map)
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
            result = op.minimize(
                self._objective_function,
                self.initial_guess,
                args=(di,),
                bounds=self.bounds,
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'differential_evolution':
            result = op.differential_evolution(
                self._objective_function,
                self.bounds,
                args=(di,),
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'basinhopping':
            result = op.basinhopping(
                self._objective_function,
                self.initial_guess,
                minimizer_kwargs=dict(
                    args=(di,),
                    bounds=self.bounds,
                    **self.scipy_kwargs
                )
            )
        elif self.scipy_func == 'dual_annealing':
            result = op.dual_annealing(
                self._objective_function,
                self.bounds,
                args=(di,),
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'shgo':
            result = op.shgo(
                self._objective_function,
                self.bounds,
                args=(di,),
                **self.scipy_kwargs
            )
        elif self.scipy_func == 'brute':
            result = op.brute(
                self._objective_function,
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
