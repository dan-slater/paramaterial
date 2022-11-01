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
    data: pd.DataFrame
    info: pd.Series = None

    @staticmethod
    def read_data_from_csv(file_path: str):
        test_id = os.path.split(file_path)[1].split('.')[0]
        data = pd.read_csv(file_path)
        return DataItem(test_id, data)

    def set_info(self, info_table: pd.DataFrame, test_id_key: str):
        self.info = info_table.loc[info_table[test_id_key] == self.test_id].squeeze()
        return self

    def write_data_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.test_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self


class DataSet:
    def __init__(self, data_dir: str, info_path: str, test_id_key: str = 'test id'):
        """Initialize the dataset.
        Args:
            data_dir: The directory containing the data.
            info_path: The path to the info table.
        """
        self.data_dir: str = data_dir
        self.info_path: str = info_path
        self.test_id_key: str = test_id_key

        _info_table = pd.read_excel(self.info_path)
        file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in _info_table[test_id_key]]

        self.dataitems = map(DataItem.read_data_from_csv, file_paths)
        self.dataitems = map(lambda di: di.set_info(_info_table, test_id_key), self.dataitems)

    @property
    def info_table(self) -> pd.DataFrame:
        info_table = pd.DataFrame()
        for dataitem in copy.deepcopy(self.dataitems):
            info_table = pd.concat([info_table, dataitem.info.to_frame().T], ignore_index=True)
        return info_table

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        self.dataitems = map(lambda di: DataItem.set_info(di, info_table, self.test_id_key), self.dataitems)

    def __iter__(self):
        """Iterate over the dataset."""
        for dataitem in tqdm(copy.deepcopy(self.dataitems), unit='DataItems', leave=False):
            yield dataitem

    def __getitem__(self, item: Union[Dict[str, List[Any]], int, slice]) -> Union['DataSet', DataItem]:
        """Get a subset of the dataset using a dictionary of column names and lists of values or using normal list
        indexing. """
        if isinstance(item, int):
            test_id = self.info_table.iloc[item]['test id']
            data = list(copy.deepcopy(self.dataitems))[item].data
            info = self.info_table.loc[self.info_table['test id'] == test_id].squeeze()
            return DataItem(test_id, data, info)
        elif isinstance(item, slice):
            subset = copy.deepcopy(self)
            subset.dataitems = list(self.dataitems)[item]
            subset.info_table = subset.info_table.iloc[item]
            return subset
        elif isinstance(item, dict):
            new_ds = self.copy()
            new_ds.dataitems = filter(lambda dataitem: all(
                [dataitem.info[column] in values for column, values in item.items()]), new_ds.dataitems)
            new_info_table = pd.DataFrame(columns=self.info_table.columns)
            for col_name, vals in item.items():
                if not isinstance(vals, list):
                    vals = [vals]
                new_info_table = pd.concat([new_info_table,
                                            self.info_table.loc[self.info_table[col_name].isin(vals)]],
                                           ignore_index=True)
            new_ds.info_table = new_info_table
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
        new_set.dataitems = map(wrapped_func, new_set.dataitems)
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
        for i, dataitem in enumerate(copy.deepcopy(self.dataitems)):
            dataitem.write_data_to_csv(data_dir)
            out_info_table = pd.concat([out_info_table, dataitem.info.to_frame().T], ignore_index=True)
            out_info_table.to_excel(info_path, index=False)

    def sort_by(self, column: str | List[str], ascending: bool = True) -> 'DataSet':
        """Sort the dataset by a column in the info table."""
        new_dataset = self.copy()
        new_dataset.info_table.sort_values(by=column, inplace=True, ascending=ascending)
        # also sort the data map by the test ids in the info table
        new_dataset.dataitems = sorted(new_dataset.dataitems, key=lambda x: x.test_id)
        return new_dataset

    def copy(self) -> 'DataSet':
        """Return a copy of the dataset."""
        return copy.deepcopy(self)

    def __repr__(self):
        repr_string = f'DataSet with {len(self.info_table)} DataItems.\n'
        repr_string += f'Columns in info table: {", ".join(self.info_table.columns)}\n'
        repr_string += f'Columns in data: {", ".join(list(self.dataitems)[0].data.columns)}'
        return repr_string

    def __len__(self):
        """Get the number of dataitems in the dataset."""
        if len(self.info_table) != len(list(copy.deepcopy(self.dataitems))):
            raise ValueError('Length of info table and datamap are different.')
        return len(self.info_table)


