""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable, List, Any, Union

import pandas as pd
from tqdm import tqdm


@dataclass
class DataItem:
    test_id: str = None
    data: pd.DataFrame = None
    info: pd.Series = None

    @staticmethod
    def read_data(file_path: str):
        test_id = os.path.split(file_path)[1].split('.')[0]
        data = pd.read_csv(file_path)
        return DataItem(test_id, data)

    def read_info_row(self, info_table: pd.DataFrame, test_id_key: str = 'test id'):
        self.info = info_table.loc[info_table[test_id_key] == self.test_id].squeeze() # todo: move this to dataset class
        return self

    def write_data_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.test_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        repr_string = f'DataItem with test id {self.test_id}.\n'
        repr_string += f'Info: {self.info.to_dict()}\n'
        repr_string += f'Data: {self.data.head(2)}'
        return repr_string


class DataSet:
    def __init__(self, data_dir: str, info_path: str, test_id_key: str = 'test id', load: bool = True):
        """Initialize the dataset.
        Args:
            data_dir: The directory containing the data.
            info_path: The path to the info table.
        """
        self.data_dir = data_dir
        self.info_path = info_path
        self.test_id_key = test_id_key
        self.info_table = pd.read_excel(self.info_path)
        file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in self.info_table[test_id_key]]
        self.data_map = map(lambda path: DataItem.read_data(path), file_paths)
        self.data_map = map(lambda di: DataItem.read_info_row(di, self.info_table, test_id_key), self.data_map)

    def __iter__(self):
        """Iterate over the dataset."""
        for dataitem in tqdm(copy.deepcopy(self.data_map), unit='DataItems', leave=False):
            if dataitem.test_id in self.info_table[self.test_id_key].values:
                yield dataitem

    def __getitem__(self, item: Union[Dict[str, List[Any]], int, slice]) -> Union['DataSet', DataItem]:
        """Get a subset of the dataset using a dictionary of column names and lists of values or using normal list
        indexing. """
        if isinstance(item, int):
            test_id = self.info_table.iloc[item]['test id']
            data = list(copy.deepcopy(self.data_map))[item].data
            info = self.info_table.loc[self.info_table['test id'] == test_id].squeeze()
            return DataItem(test_id, data, info)
        elif isinstance(item, slice):
            subset = copy.deepcopy(self)
            subset.data_map = list(self.data_map)[item]
            subset.info_table = subset.info_table.iloc[item]
            return subset
        elif isinstance(item, dict):
            new_ds = self.copy()
            new_ds.data_map = filter(lambda dataitem: all(
                [dataitem.info[column] in values for column, values in item.items()]), new_ds.data_map)
            for col_name, vals in item.items():
                if not isinstance(vals, list):
                    vals = [vals]
                new_ds.info_table = new_ds.info_table.loc[new_ds.info_table[col_name].isin(vals)]
            return new_ds
        else:
            raise ValueError(f'Invalid argument type: {type(item)}')

    def apply_function(self, func: Callable[[DataItem], DataItem], update_info: bool = True) -> 'DataSet':
        """Apply a processing function to the dataset."""

        def wrapped_func(di: DataItem):
            di = func(di)
            di.data.reset_index(drop=True, inplace=True)
            return di

        new_set = self.copy()
        new_set.data_map = map(wrapped_func, new_set.data_map)
        if update_info:
            new_info_table = pd.DataFrame()
            for i, dataitem in enumerate(new_set):
                new_info_table = pd.concat([new_info_table, dataitem.info.to_frame().T], ignore_index=True)
            new_set.info_table = new_info_table
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
        self.info_table.to_excel(info_path, index=False)
        out_info_table = pd.DataFrame()
        for i, dataitem in enumerate(copy.deepcopy(self.data_map)):
            dataitem.write_data_to_csv(data_dir)
            out_info_table = pd.concat([out_info_table, dataitem.info.to_frame().T], ignore_index=True)
            out_info_table.to_excel(info_path, index=False)

    def sort_by(self, column: str | List[str], ascending: bool = True) -> 'DataSet':
        """Sort the dataset by a column in the info table."""
        new_dataset = self.copy()
        new_dataset.info_table.sort_values(by=column, inplace=True, ascending=ascending)
        # also sort the data map by the test ids in the info table
        new_dataset.data_map = sorted(new_dataset.data_map, key=lambda x: x.test_id)
        return new_dataset

    def copy(self) -> 'DataSet':
        """Return a copy of the dataset."""
        return copy.deepcopy(self)

    def __repr__(self):
        repr_string = f'DataSet with {len(self.info_table)} DataItems.\n'
        repr_string += f'Columns in info table: {", ".join(self.info_table.columns)}\n'
        repr_string += f'Columns in data: {", ".join(self[0].data.columns)}'
        return repr_string

    def __eq__(self, other):
        """Check if the datamaps and info tables of the datasets are equal."""
        return self.data_map == other.data_map and self.info_table.equals(other.info_table)

    def __len__(self):
        """Get the number of dataitems in the dataset."""
        if len(self.info_table) != len(list(copy.deepcopy(self.data_map))):
            raise ValueError('Length of info table and datamap are different.')
        return len(self.info_table)

