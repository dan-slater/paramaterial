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

    def read_info_from(self, info_table: pd.DataFrame, test_id_key: str):
        self.info = info_table.loc[info_table[test_id_key] == self.test_id].squeeze()
        self.info.name = None
        return self


class DataSet:
    def __init__(self, info_path: str, data_dir: str, test_id_key: str = 'test_id'):
        """Initialize the ds.
        Args:
            info_path: The path to the info table file.
            data_dir: The directory containing the data files.
            test_id_key: The column name in the info table that contains the test ids.
        """

        # store initialization parameters
        self.info_path = info_path
        self.data_dir = data_dir
        self.test_id_key = test_id_key

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
        return self._info_table

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        # update the list of dataitems
        new_test_ids = info_table[self.test_id_key].tolist()
        old_test_ids = [di.test_id for di in self.data_items]
        self.data_items = [self.data_items[old_test_ids.index(new_test_id)] for new_test_id in new_test_ids]
        # set the internal copy of the info table
        self._info_table = info_table
        # update the info in the data items
        self.data_items = list(map(lambda di: di.read_info_from(self._info_table, self.test_id_key),
                                   self.data_items))

    def apply(self, func: Callable[[DataItem, ...], DataItem], **kwargs) -> 'DataSet':
        """Apply a function to every dataitem in a copy of the ds and return the copy."""

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

    def sort_by(self, column: str | List[str], ascending: bool = True) -> 'DataSet':
        """Sort a copy of the ds by a column in the info table and return the copy."""
        new_ds = self.copy()
        new_ds.info_table = new_ds.info_table.sort_values(by=column, ascending=ascending).reset_index(drop=True)
        return new_ds

    def __iter__(self):
        """Iterate over the ds."""
        for dataitem in tqdm(self.copy().data_items, unit='DataItems', leave=False):
            yield dataitem

    def __getitem__(self, specifier: Union[int, str, slice, Dict[str, List[Any]]]) -> Union['DataSet', DataItem]:
        """Get a subset of the ds using a dictionary of column names and lists of values or using normal list
        indexing. """
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

    def subset(self, filter_dict: Dict[str, List[Any]]) -> 'DataSet':
        new_ds = self.copy()
        for key, value in filter_dict.items():
            if key not in new_ds.info_table.columns:
                raise ValueError(f'Invalid filter key: {key}')
            if not isinstance(value, list):
                filter_dict[key] = [value]
        query_string = ' and '.join([f"'{key}' in {str(values)}" for key, values in filter_dict.items()])
        try:
            new_ds.info_table = self._info_table.query(query_string)
        except Exception as e:
            print(f'Error applying query "{query_string}" to info table: {e}')
        return new_ds

    def copy(self) -> 'DataSet':
        """Return a copy of the ds."""
        return copy.deepcopy(self)

    def __repr__(self):
        repr_string = f'DataSet with {len(self._info_table)} DataItems containing\n'
        repr_string += f'\tinfo: columns -> {", ".join(self._info_table.columns)}\n'
        repr_string += f'\tdata: len = {len(self.data_items[0].data)}, columns -> {", ".join(self.data_items[0].data.columns)}\n'
        return repr_string

    def __len__(self):
        """Get the number of dataitems in the ds."""
        if len(self._info_table) != len(self.data_items):
            raise ValueError('Length of info table and datamap are different.')
        return len(self._info_table)

    def __hash__(self):
        return hash(tuple(map(hash, self.data_items)))
