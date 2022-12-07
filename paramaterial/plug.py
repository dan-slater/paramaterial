""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
from dataclasses import dataclass
from typing import Dict, Callable, List, Any, Union

import pandas as pd
from tqdm import tqdm


@dataclass
class DataItem:
    test_id: str
    info: pd.Series
    data: pd.DataFrame

    @staticmethod
    def read_data_from_csv(file_path: str):
        test_id = os.path.split(file_path)[1].split('.')[0]
        data = pd.read_csv(file_path)
        return DataItem(test_id, pd.Series(dtype=object), data)

    def read_info_from_table(self, info_table: pd.DataFrame, test_id_key: str):
        self.info = info_table.loc[info_table[test_id_key] == self.test_id].squeeze()
        self.info.name = None
        return self

    def write_data_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.test_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self


class DataSet:
    def __init__(self, data_dir: str, info_path: str, test_id_key: str = 'test id'):
        """Initialize the ds.
        Args:
            data_dir: The directory containing the data.
            info_path: The path to the info table.
        """
        self.data_dir = data_dir
        self.info_path = info_path
        self.test_id_key = test_id_key
        if self.info_path.endswith('.xlsx'):
            self._info_table = pd.read_excel(self.info_path)
        elif self.info_path.endswith('.csv'):
            self._info_table = pd.read_csv(self.info_path)
        else:
            raise ValueError(f'Info table must be a csv or xlsx file path. Got {self.info_path}')
        self.file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in self._info_table[self.test_id_key]]
        self.data_map = self.data_map = map(DataItem.read_data_from_csv, self.file_paths)
        self.data_map = map(lambda di: di.read_info_from_table(self._info_table, self.test_id_key), self.data_map)
        self.applied_funcs = []

    def update_data_map(self) -> None:
        self.file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in self._info_table[self.test_id_key]]
        self.data_map = map(DataItem.read_data_from_csv, self.file_paths)
        self.data_map = map(lambda di: di.read_info_from_table(self.info_table, self.test_id_key), self.data_map)
        # map the applied functions to the new data map
        for func in self.applied_funcs:
            self.data_map = map(func, self.data_map)

    @property
    def info_table(self) -> pd.DataFrame:
        """Assemble the info table from the dataitems."""
        return self._info_table

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        """Distribute the info table in the dataitems."""
        self._info_table = info_table
        self.update_data_map()

    @property
    def data_items(self) -> List[DataItem]:
        return list(copy.deepcopy(self.data_map))

    def apply(self, func: Callable[[DataItem, ...], DataItem], **kwargs) -> 'DataSet':
        """Apply a function to every dataitem in a copy of the ds and return the copy."""

        def wrapped_func(di: DataItem):
            try:
                di = func(di, **kwargs)
                di.data.reset_index(drop=True, inplace=True)
                assert self.test_id_key in di.info.index
                return di
            except Exception as e:
                print(f'Error applying {func.__name__} to {di.test_id}: {e}')
                return di

        new_ds = self.copy()
        new_ds.applied_funcs.append(wrapped_func)
        new_ds.update_data_map()
        return new_ds

    def write_output(self, data_dir: str, info_path: str) -> None:
        """Execute the processing operations and write the output of the ds to a directory.
        Args:
            data_dir: The directory to write the data to.
            info_path: The path to write the info table to.
        """
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for dataitem in self:
            dataitem.write_data_to_csv(data_dir)

        if info_path.endswith('.xlsx'):
            self.info_table.to_excel(info_path, index=False)
        elif info_path.endswith('.csv'):
            self.info_table.to_csv(info_path, index=False)
        else:
            raise ValueError(f'Info table must be a csv or xlsx file. Got {info_path}')

    def sort_by(self, column: str | List[str], ascending: bool = True) -> 'DataSet':
        """Sort a copy of the ds by a column in the info table and return the copy."""
        new_ds = self.copy()
        new_ds.info_table = new_ds.info_table.sort_values(by=column, ascending=ascending)
        new_ds.update_data_map()
        return new_ds

    def __iter__(self):
        """Iterate over the ds."""
        for dataitem in tqdm(self.copy().data_map, unit='DataItems', leave=False):
            yield dataitem

    def __getitem__(self, specifier: Union[int, str, slice, Dict[str, List[Any]]]) -> Union['DataSet', DataItem]:
        """Get a subset of the ds using a dictionary of column names and lists of values or using normal list
        indexing. """
        if isinstance(specifier, int):
            return self.data_items[specifier]
        elif isinstance(specifier, str):
            return self.data_items[self.info_table[self.test_id_key].tolist().index(specifier)]
        elif isinstance(specifier, slice):
            new_ds = self.copy()
            new_ds.info_table = new_ds.info_table.iloc[specifier]
            new_ds.update_data_map()
            return new_ds
        else:
            raise ValueError(f'Invalid ds[specifier] specifier type: {type(specifier)}')

    def subset(self, filter_dict: Dict[str, List[Any]]) -> 'DataSet':
        new_ds = self.copy()
        for key, value in filter_dict.items():
            if key not in self.info_table.columns:
                raise ValueError(f'Invalid filter key: {key}')
            if not isinstance(value, list):
                filter_dict[key] = [value]
        query_string = ' and '.join([f'{key} in {str(values)}' for key, values in filter_dict.items()])
        try:
            new_ds.info_table = self.info_table.query(query_string)
        except Exception as e:
            print(f'Error applying query "{query_string}" to info table: {e}')
        return new_ds

    def copy(self) -> 'DataSet':
        """Return a copy of the ds."""
        return copy.deepcopy(self)

    def __repr__(self):
        repr_string = f'DataSet with {len(self.info_table)} DataItems containing\n'
        repr_string += f'\tinfo: columns -> {", ".join(self.info_table.columns)}\n'
        repr_string += f'\tdata: len = {len(self.data_items[0].data)}, columns -> {", ".join(self.data_items[0].data.columns)}\n'
        return repr_string

    def __len__(self):
        """Get the number of dataitems in the ds."""
        if len(self.info_table) != len(self.data_items):
            raise ValueError('Length of info table and datamap are different.')
        return len(self.info_table)

    def __hash__(self):
        return hash(tuple(map(hash, self.data_items)))
