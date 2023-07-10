""" In charge of handling data and executing I/O. """
import copy
import os
from dataclasses import dataclass
from typing import Dict, Callable, List, Any, Union, Optional

import pandas as pd
from tqdm import tqdm


@dataclass
class DataItem:
    """A class for handling a single data item.
    Args:
        test_id: The test id.
        info: A pandas Series containing the info for the test.
        data: A pandas DataFrame containing the data for the test.
     """
    test_id: str
    info: pd.Series
    data: pd.DataFrame

    def read_info_from(self, info_table: pd.DataFrame, test_id_key: str):
        self.info = info_table.loc[info_table[test_id_key] == self.test_id].squeeze()
        self.info.name = None
        return self


class DataSet:
    """A class for handling data. The data is stored in a list of DataItems. The info table is stored in a pandas DataFrame.
    Args:
        info_path: The path to the info table file.
        data_dir: The directory containing the data files.
        test_id_key: The column name in the info table that contains the test ids.
    """
    def __init__(self, info_path: Optional[str] = None, data_dir: Optional[str] = None, test_id_key: str = 'test_id'):
        # store initialization parameters
        self.info_path = info_path
        self.data_dir = data_dir
        self.test_id_key = test_id_key

        # if empty inputs, create empty ds
        if info_path is None and data_dir is None:
            self._info_table = pd.DataFrame()
            self.data_items = []
            return

        # read the info table
        if self.info_path.endswith('.xlsx'):
            self._info_table = pd.read_excel(self.info_path)
        elif self.info_path.endswith('.csv'):
            self._info_table = pd.read_csv(self.info_path)
        else:
            raise ValueError(f'Info table must be a csv or xlsx file path. Got {self.info_path}')

        # read the data files to a list of data-items
        test_ids = self._info_table[test_id_key].tolist()
        file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in test_ids]
        self.data_items = [DataItem(test_id, pd.Series(dtype=object), pd.read_csv(file_path)) for test_id, file_path in
                           zip(test_ids, file_paths)]
        self.data_items = list(map(lambda di: di.read_info_from(self._info_table, self.test_id_key), self.data_items))

        # run checks
        assert len(list(self.data_items)) == len(self._info_table)
        assert all([di.test_id == test_id for di, test_id in zip(self.data_items, self._info_table[self.test_id_key])])
        assert all(
            [di.info.equals(self._info_table.loc[self._info_table[self.test_id_key] == di.test_id].squeeze()) for di in
             self.data_items])

    @property
    def info_table(self) -> pd.DataFrame:
        """Return the info table."""
        return self._info_table

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        self._info_table = info_table
        self._update_data_items()

    def _update_data_items(self):
        # update the list of dataitems
        new_test_ids = self._info_table[self.test_id_key].tolist()
        old_test_ids = [di.test_id for di in self.data_items]
        self.data_items = [self.data_items[old_test_ids.index(new_test_id)] for new_test_id in new_test_ids]
        # update the info in the data items
        self.data_items = list(map(lambda di: di.read_info_from(self._info_table, self.test_id_key),
                                   self.data_items))

    def apply(self, func: Callable[[DataItem, ...], DataItem], **kwargs) -> 'DataSet':
        """Apply a function to every dataitem in a copy of the ds and return the copy."""
        self._update_data_items()

        def wrapped_func(di: DataItem):
            di = func(di, **kwargs)
            di.data.reset_index(drop=True, inplace=True)
            assert self.test_id_key in di.info.index
            return di

        new_ds = self.copy()
        new_ds.data_items = list(map(wrapped_func, new_ds.data_items))
        new_ds.info_table = pd.DataFrame([di.info for di in new_ds.data_items])
        return new_ds

    def write_output(self, out_info_path: str, out_data_dir: str) -> None:
        """Execute the processing operations and write the output of the ds to a directory.
        Args:
            out_data_dir: The directory to write the data to.
            out_info_path: The path to write the info table to.
        """
        # make the output directory if it doesn't exist
        self._update_data_items()
        if not os.path.exists(out_data_dir):
            os.makedirs(out_data_dir)
        # write the info table
        if out_info_path.endswith('.xlsx'):
            self._info_table.to_excel(out_info_path, index=False)
        elif out_info_path.endswith('.csv'):
            self._info_table.to_csv(out_info_path, index=False)
        else:
            raise ValueError(f'Info table must be a csv or xlsx file. Got {out_info_path}')
        # write the data files
        for di in self.data_items:
            output_path = out_data_dir + '/' + di.test_id + '.csv'
            di.data.to_csv(output_path, index=False)

    def sort_by(self, column: Union[str, List[str]], ascending: bool = True) -> 'DataSet':
        """Sort a copy of the ds by a column in the info table and return the copy."""
        self._update_data_items()
        new_ds = self.copy()
        new_ds.info_table = new_ds.info_table.sort_values(by=column, ascending=ascending).reset_index(drop=True)
        return new_ds

    def __iter__(self):
        """Iterate over the ds."""
        self._update_data_items()
        for dataitem in tqdm(self.copy().data_items, unit='DataItems', leave=False):
            yield dataitem

    def __getitem__(self, specifier: Union[int, str, slice, Dict[str, List[Any]]]) -> Union['DataSet', DataItem]:
        """Get a subset of the ds using a dictionary of column names and lists of values or using normal list
        indexing. """
        self._update_data_items()
        if isinstance(specifier, int):
            return self.data_items[specifier]
        elif isinstance(specifier, str):
            return self.data_items[self._info_table[self.test_id_key].tolist().index(specifier)]
        elif isinstance(specifier, slice):
            new_ds = self.copy()
            new_ds.info_table = new_ds.info_table.iloc[specifier]
            return new_ds
        else:
            raise ValueError(f'Invalid ds[specifier] specifier type: {type(specifier)}')

    def subset(self, filter_dict: Dict[str, Union[str, List[Any]]]) -> 'DataSet':
        self._update_data_items()
        new_ds = self.copy()
        for key, value in filter_dict.items():
            if key not in new_ds.info_table.columns:
                raise ValueError(f'Invalid filter key: {key}')
            if not isinstance(value, list):
                filter_dict[key] = [value]
        query_string = ' and '.join([f"`{key}` in {str(values)}" for key, values in filter_dict.items()])
        try:
            new_ds.info_table = self._info_table.query(query_string)
        except Exception as e:
            print(f'Error applying query "{query_string}" to info table: {e}')
        return new_ds

    def copy(self) -> 'DataSet':
        """Return a copy of the ds."""
        self._update_data_items()
        return copy.deepcopy(self)

    def __repr__(self):
        self._update_data_items()
        repr_string = f'DataSet with {len(self._info_table)} DataItems containing\n'
        repr_string += f'\tinfo: columns -> {", ".join(self._info_table.columns)}\n'
        repr_string += f'\tdata: len = {len(self.data_items[0].data)}, columns -> {", ".join(self.data_items[0].data.columns)}\n'
        return repr_string

    def __len__(self):
        """Get the number of dataitems in the ds."""
        self._update_data_items()
        if len(self._info_table) != len(self.data_items):
            raise ValueError('Length of info table and datamap are different.')
        return len(self._info_table)

    def __hash__(self):
        self._update_data_items()
        return hash(tuple(map(hash, self.data_items)))



class ModelSet:
    """Class that acts as model DataSet.

    Args:
        model_func: The model function to be used for fitting.
        param_names: The names of the parameters of the model function.
        var_names: The names of the variables of the model function.
        bounds: The bounds for the parameters of the model function.
        initial_guess: The initial guess for the parameters of the model function.
        scipy_func: The scipy function to be used for fitting.
        scipy_kwargs: The kwargs for the scipy function.
    """

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
        self.predicted_ds: Optional[DataSet] = None

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

    def predict(self, resolution: int = 50, xmin=None, xmax=None, info_table=None) -> DataSet:
        """Return a ds with generated data with optimised model parameters added to the info table.
        If an info table is provided, data items will be generated to match the rows of the info table, using the
        var_names and param_names and model_func.

        Args:
            resolution: Number of points to generate between the x_min and x_max.

        Returns: DataSet with generated data.
        """
        predict_ds = DataSet()

        if info_table is not None:
            predict_ds.info_table = info_table
            for i, row in info_table.iterrows():
                # make a data item for each row in the info table
                x_data = np.linspace(row['x_min'], row['x_max'], resolution)
                y_data = self.model_func(x_data, row[self.param_names].values)
                data = {self.x_key: x_data, self.y_key: y_data}
                info = row
                test_id = row['test_id']
                di = DataItem()


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
