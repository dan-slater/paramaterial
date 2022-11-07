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
        return self

    def write_data_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.test_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self


class DataSet:
    def __init__(self, data_dir: str | None = None, info_path: str | None = None, test_id_key: str = 'test id'):
        """Initialize the dataset.
        Args:
            data_dir: The directory containing the data.
            info_path: The path to the info table.
        """
        self.data_dir = data_dir
        self.info_path = info_path
        self.test_id_key = test_id_key
        self.file_paths: List[str] | None = None
        self.data_map: map | None = None
        if data_dir is not None and info_path is not None:
            self.load()

    def load(self) -> None:
        if self.info_path.endswith('.xlsx'):
            _info_table = pd.read_excel(self.info_path)
        elif self.info_path.endswith('.csv'):
            _info_table = pd.read_csv(self.info_path)
        else:
            raise ValueError(f'Info table must be a csv or xlsx file path. Got {self.info_path}')

        self.file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in _info_table[self.test_id_key]]
        self.data_map = map(DataItem.read_data_from_csv, self.file_paths)
        self.data_map = map(lambda di: di.read_info_from_table(_info_table, self.test_id_key), self.data_map)

    @property
    def info_table(self) -> pd.DataFrame:
        """Assemble the info table from the dataitems."""
        _info_table = pd.DataFrame()
        for dataitem in copy.deepcopy(self.data_map):
            _info_table = pd.concat([_info_table, dataitem.info.to_frame().T], ignore_index=True)
        return _info_table

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        """Distribute the info table in the dataitems."""
        self.data_map = map(lambda di: DataItem.read_info_from_table(di, info_table, self.test_id_key), self.data_map)

    @property
    def dataitems(self) -> List[DataItem]:
        return list(copy.deepcopy(self.data_map))

    def apply(self, func: Callable[[DataItem, ...], DataItem], **kwargs) -> 'DataSet':
        """Apply a function to every dataitem in a copy of the dataset and return the copy."""

        # def wrapped_func(di: DataItem):
        #     try:
        #         di = func(di)
        #         di.data.reset_index(drop=True, inplace=True)
        #         assert type(di.data) == pd.DataFrame
        #         assert type(di.info) == pd.Series
        #         assert self.test_id_key in di.info.index
        #         return di
        #     except Exception as e:
        #         print(f'Error applying {func.__name__} to {di.test_id}: {e}')
        #         return di

        def wrapped_func(di: DataItem):
                di = func(di, **kwargs)
                di.data.reset_index(drop=True, inplace=True)
                assert type(di.data) == pd.DataFrame
                assert type(di.info) == pd.Series
                assert self.test_id_key in di.info.index
                return di

        new_set = self.copy()
        new_set.data_map = map(wrapped_func, new_set.data_map)

        return new_set

    def write_output(self, data_dir: str, info_path: str) -> None:
        """Execute the processing operations and write the output of the dataset to a directory.
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
        """Sort a copy of the dataset by a column in the info table and return the copy."""
        new_set = DataSet()
        new_set.data_dir = self.data_dir
        new_set.info_path = self.info_path
        new_set.test_id_key = self.test_id_key
        _info_table = self.info_table.sort_values(by=column, ascending=ascending)
        new_set.file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in _info_table[self.test_id_key]]
        new_set.data_map = map(DataItem.read_data_from_csv, new_set.file_paths)
        new_set.data_map = map(lambda di: di.read_info_from_table(_info_table, self.test_id_key), new_set.data_map)
        new_set.info_table = _info_table
        return new_set

    def __iter__(self):
        """Iterate over the dataset."""
        for dataitem in tqdm(self.copy().data_map, unit='DataItems', leave=False):
            yield dataitem

    def __getitem__(self, specifier: Union[int, str, slice, Dict[str, List[Any]]]) -> Union['DataSet', DataItem]:
        """Get a subset of the dataset using a dictionary of column names and lists of values or using normal list
        indexing. """
        if isinstance(specifier, int):
            return self.dataitems[specifier]

        elif isinstance(specifier, str):
            return self.dataitems[self.info_table[self.test_id_key].tolist().index(specifier)]

        elif isinstance(specifier, slice):
            new_set = DataSet()
            new_set.data_dir = self.data_dir
            new_set.info_path = self.info_path
            new_set.test_id_key = self.test_id_key
            new_set.file_paths = self.file_paths[specifier]
            new_set.data_map = map(DataItem.read_data_from_csv, new_set.file_paths)
            new_set.data_map = map(lambda di: di.read_info_from_table(self.info_table, self.test_id_key), new_set.data_map)
            return new_set

        elif isinstance(specifier, dict):
            new_dataset = self.copy()
            new_dataset.data_map = filter(lambda di: all(
                [di.info[key] in values for key, values in specifier.items()]), new_dataset.data_map)
            return new_dataset
            # new_set = self.copy()
            # _info_table = new_set.info_table
            # for info_col, vals in specifier.items():
            #     if not isinstance(vals, list):
            #         vals = [vals]
            #     _info_table = _info_table.loc[_info_table[info_col].isin(vals)]
            # test_ids = _info_table[self.test_id_key].tolist()
            # new_set.data_map = filter(lambda di: di.info[self.test_id_key] in test_ids, new_set.data_map)
            # new_set.file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in test_ids]
            # return new_set
        else:
            raise ValueError(f'Invalid dataset[specifier] specifier type: {type(specifier)}')

    def get_subset(self, subset_filter: Dict[str, List[Any]]) -> 'DataSet':
        subset = copy.deepcopy(self)
        info_table = subset.info_table
        for col_name, vals in subset_filter.items():
            info_table = info_table.loc[info_table[col_name].isin(vals)]
        subset.file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in info_table[self.test_id_key]]
        subset.datamap = map(lambda path: DataItem.read_data_from_csv(path), subset.file_paths)
        subset.datamap = map(lambda di: di.read_info_from_table(info_table, self.test_id_key), subset.datamap)
        subset.info_table = info_table
        return subset

    def get_subset_2(self, subset_filter: Dict[str, List[Any]]) -> 'DataSet':
        new_dataset = self.copy()
        new_dataset.data_map = filter(lambda di: all(
            [di.info[key] in values for key, values in subset_filter.items()]), new_dataset.data_map)
        return new_dataset

    def copy(self) -> 'DataSet':
        """Return a copy of the dataset."""
        return copy.deepcopy(self)

    def __repr__(self):
        repr_string = f'DataSet with {len(self.info_table)} DataItems.\n'
        repr_string += f'Columns in info table: {", ".join(self.info_table.columns)}\n'
        repr_string += f'Columns in data: {", ".join(self.dataitems[0].data.columns)}'
        return repr_string

    def __len__(self):
        """Get the number of dataitems in the dataset."""
        if len(self.info_table) != len(self.dataitems):
            raise ValueError('Length of info table and datamap are different.')
        return len(self.info_table)

    def __hash__(self):
        return hash(tuple(map(hash, self.dataitems)))
