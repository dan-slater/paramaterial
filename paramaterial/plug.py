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
        data: A pandas DataFrame containing the data for the test.
        _info_table: Reference to info_table from parent DataSet. Should not be set manually.

    Attributes:
        info: A Series containing the corresponding row from the info_table of the parent DataSet.
        Modifying the info of a DataItem will modify the info_table of the parent DataSet and vice versa.
    """
    test_id: str
    data: pd.DataFrame
    _info_table: pd.DataFrame  # Reference to info_table from parent DataSet

    @property
    def info(self) -> pd.Series:
        """Return the corresponding info for this DataItem from the info_table."""
        return self._info_table.loc[self._info_table['test_id'] == self.test_id]

    @info.setter
    def info(self, new_info: pd.Series):
        """Update the corresponding info for this DataItem in the info_table."""
        self._info_table.loc[self._info_table['test_id'] == self.test_id] = new_info


class DataSet:
    """A class for handling data.
    Args:
        info_path: The path to the info table file.
        data_dir: The directory containing the data files.
        test_id_key: The column name in the info table that contains the test ids.
    """

    def __init__(self, info_path: Optional[str] = None, data_dir: Optional[str] = None, test_id_key: str = 'test_id'):
        self.info_path = info_path
        self.data_dir = data_dir
        self.test_id_key = test_id_key

        if info_path is None and data_dir is None:
            self._info_table = pd.DataFrame()
            self._data_items = []
            return

        if info_path is None or data_dir is None:
            raise ValueError('Both info_path and data_dir must be specified, or neither.')

        self._info_table = self._read_info_table()
        self._data_items = self._load_data_items()
        self._check_data_info_match()

    def _read_info_table(self):
        if self.info_path.endswith('.xlsx'):
            self._info_table = pd.read_excel(self.info_path)
        elif self.info_path.endswith('.csv'):
            self._info_table = pd.read_csv(self.info_path)
        elif self.info_path.endswith('.ods'):
            self._info_table = pd.read_excel(self.info_path, engine='odf')
        else:
            raise ValueError(
                f'Info_path must end with ".csv", ".xlsx" or ".ods". Not ".{self.info_path.split(".")[-1]}"')

    def _read_info_row(self, test_id: str):
        return self._info_table.loc[self._info_table[self.test_id_key] == test_id].squeeze()

    def _load_data_items(self) -> List[DataItem]:
        try:
            test_ids = self._info_table[self.test_id_key].tolist()
        except KeyError:
            raise KeyError(f'Could not find test_id column "{self.test_id_key}" in info table.')

        file_extensions = set([file.split('.')[-1] for file in os.listdir(self.data_dir)])
        if len(file_extensions) > 1:
            raise ValueError(f'Data files have multiple extensions: {file_extensions}.'
                             f'Must have only one extension: csv, xlsx or ods.')
        if file_extensions.pop() not in ['csv', 'xlsx', 'ods']:
            raise ValueError(f'Data files must have extension csv, xlsx or ods. Got {file_extensions}.')

        file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in test_ids]

        return [DataItem(test_id, self._read_info_row(test_id, self.test_id_key), pd.read_csv(file_path)) for
                test_id, file_path in zip(test_ids, file_paths)]

    def _check_data_info_match(self):
        if len(self._info_table) != len(self._data_items):
            raise ValueError('Lengths of info_table and data_items are different.')
        if any([di.test_id != self._info_table[self.test_id_key][i] for i, di in enumerate(self._data_items)]):
            raise ValueError('test_id\'s in info_table and data_items are different.')
        if any([not di.info.equals(self._info_table.loc[self._info_table[self.test_id_key] == di.test_id].squeeze()) for
                di in self._data_items]):
            raise ValueError('At least one DataItem.info different to corresponding info_table row.')

    @property
    def info_table(self) -> pd.DataFrame:

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        self._info_table = info_table
        self._update_data_items()

    def _read_info_row(self, test_id: str, test_id_key: str):
        return self._info_table.loc[self._info_table[test_id_key] == test_id].squeeze()

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


#####

@dataclass
class DataItem:
    test_id: str
    info: pd.Series
    data: pd.DataFrame


class DataSet:
    def __init__(self, info_path: Optional[str] = None, data_dir: Optional[str] = None):
        self.info_path = info_path
        self.data_dir = data_dir

        if info_path is None and data_dir is None:
            self._info_table = pd.DataFrame()
            self.data_items = []
            return

        if info_path is None or data_dir is None:
            raise ValueError('Both info_path and data_dir must be specified, or neither.')

        if self.info_path.endswith('.xlsx'):
            self._info_table = pd.read_excel(self.info_path)
        elif self.info_path.endswith('.csv'):
            self._info_table = pd.read_csv(self.info_path)
        elif self.info_path.endswith('.ods'):
            self._info_table = pd.read_excel(self.info_path, engine='odf')
        else:
            raise ValueError(
                f'Info_path must end with ".csv", ".xlsx" or ".ods". Not ".{self.info_path.split(".")[-1]}"')

        self.update_data_items()

    @property
    def info_table(self):
        return self._info_table

    @info_table.setter
    def info_table(self, info_table):
        self._info_table = info_table
        self.update_data_items()

    def update_data_items(self, test_id_key: str = 'test_id'):
        test_ids = self._info_table[test_id_key].tolist()
        if self.data_dir is not None:
            file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in test_ids]
            self.data_items = [DataItem(test_id, self.read_info_from(test_id, test_id_key), pd.read_csv(file_path)) for
                               test_id, file_path in zip(test_ids, file_paths)]
        else:
            self.data_items = [DataItem(test_id, self.read_info_from(test_id, test_id_key), pd.DataFrame()) for test_id
                               in test_ids]

    def read_info_from(self, test_id: str, test_id_key: str):
        return self._info_table.loc[self._info_table[test_id_key] == test_id].squeeze()

    def subset(self, filter_dict: Dict[str, Union[str, List[Any]]]) -> 'DataSet':
        new_ds = self.copy()
        for key, value in filter_dict.items():
            if key not in new_ds._info_table.columns:
                raise ValueError(f'Invalid filter key: {key}')
            if isinstance(value, str):
                value = [value]
            if not all(elem in new_ds._info_table[key].values for elem in value):
                raise ValueError(f'Invalid values for filter key: {key}')
            new_ds._info_table = new_ds._info_table[new_ds._info_table[key].isin(value)]
        return new_ds

    # Other methods...
