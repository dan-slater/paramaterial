""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
from dataclasses import dataclass
from typing import Dict, Callable, List, Any, Union

import pandas as pd
from tqdm import tqdm


@dataclass
class DataItem:
    test_id: str = None
    data: pd.DataFrame = None
    info: pd.Series = None

    def __len__(self):
        return len(self.data)

    @staticmethod
    def read_from_csv(file_path: str):
        test_id = os.path.split(file_path)[1].split('.')[0]
        data = pd.read_csv(file_path)
        return DataItem(test_id, data)

    def get_row_from_info_table(self, info_table: pd.DataFrame, test_id_key: str = 'test id'):
        self.info = info_table.loc[info_table[test_id_key] == self.test_id].squeeze()
        return self

    def write_data_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.test_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self


@dataclass
class DataSet:
    data_dir: str = None
    info_path: str = None
    datamap: map = None
    info_table: pd.DataFrame = None

    def __init__(self, data_dir: str, info_path: str, test_id_key: str = 'test id'):
        """Initialize the dataset.
        Args:
            data_dir: The directory containing the data.
            info_path: The path to the info table.
        """
        self.test_id_key = test_id_key
        self.data_dir = data_dir
        self.info_path = info_path
        self.info_table = pd.read_excel(self.info_path)
        file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in self.info_table[test_id_key]]
        self.datamap = map(lambda path: DataItem.read_from_csv(path), file_paths)
        self.datamap = map(lambda obj: DataItem.get_row_from_info_table(obj, self.info_table, test_id_key=test_id_key),
                           self.datamap)

    def __iter__(self):
        """Iterate over the dataset."""
        for dataitem in tqdm(copy.deepcopy(self.datamap), desc='Iterating over DataItems in DataSet'):
            if dataitem.test_id in self.info_table[self.test_id_key].values:
                yield dataitem

    # get subset using a subset filter dictionary
    def get_subset(self, subset_filter: Dict[str, List[Any]]) -> 'DataSet':
        """Get a subset of the dataset.
        Args:
            subset_filter: A dictionary of column names and values to filter by.
        Returns:
            A new dataset object with the subset.
        """
        subset = copy.deepcopy(self)
        info_table = subset.info_table
        for col_name, vals in subset_filter.items():
            info_table = info_table.loc[info_table[col_name].isin(vals)]
        subset.datamap = map(lambda path: DataItem.read_from_csv(path),
                             [self.data_dir + f'/{test_id}.csv' for test_id in info_table[self.test_id_key]])
        subset.datamap = map(
            lambda obj: DataItem.get_row_from_info_table(obj, info_table, test_id_key=self.test_id_key),
            subset.datamap)
        subset.info_table = info_table
        return subset

    def apply_function(self, func: Callable[[DataItem], DataItem]) -> 'DataSet':
        """Apply a processing function to the dataset."""

        def wrapped_func(dataitem: DataItem):
            try:
                dataitem = func(dataitem)
                dataitem.data.reset_index(drop=True, inplace=True)
                return dataitem
            except Exception as e:
                print(f'Error applying "{func.__name__}": {e}')
                return dataitem

        new_dataset = self.copy()
        new_dataset.datamap = map(wrapped_func, new_dataset.datamap)
        return new_dataset

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
        for i, dataitem in enumerate(copy.deepcopy(self.datamap)):
            dataitem.write_data_to_csv(data_dir)
            out_info_table = pd.concat([out_info_table, dataitem.info.to_frame().T], ignore_index=True)
            out_info_table.to_excel(info_path, index=False)

    def copy(self) -> 'DataSet':
        """Return a copy of the dataset."""
        return copy.deepcopy(self)

    def __eq__(self, other):
        """Check if the datamaps and info tables of the datasets are equal."""
        return self.datamap == other.datamap and self.info_table.equals(other.info_table)

    def __len__(self):
        """Get the number of dataitems in the dataset."""
        if len(self.info_table) != len(list(copy.deepcopy(self.datamap))):
            raise ValueError('Length of info table and datamap are different.')
        return len(self.info_table)

    def __getitem__(self, item: Union[Dict[str, List[Any]], int, slice]) -> Union['DataSet', DataItem]:
        """Get a subset of the dataset using a dictionary of column names and lists of values or using normal list
        indexing. """
        if isinstance(item, int):
            # get the test id of the ith row in the info table
            test_id = self.info_table.iloc[item]['test id']
            data = list(copy.deepcopy(self.datamap))[item].data
            info = self.info_table.loc[self.info_table['test id'] == test_id].squeeze()
            return DataItem(test_id, data, info)
        elif isinstance(item, slice):
            subset = copy.deepcopy(self)
            subset.datamap = list(self.datamap)[item]
            subset.info_table = subset.info_table.iloc[item]
            return subset
        elif isinstance(item, dict):
            subset = copy.deepcopy(self)
            info_table = subset.info_table
            for col_name, vals in item.items():
                if not isinstance(vals, list):
                    vals = [vals]
                info_table = info_table.loc[info_table[col_name].isin(vals)]
            subset.datamap = map(lambda path: DataItem.read_from_csv(path),
                                 [self.data_dir + f'/{test_id}.csv' for test_id in info_table[self.test_id_key]])
            subset.datamap = map(
                lambda obj: DataItem.get_row_from_info_table(obj, info_table, test_id_key=self.test_id_key),
                subset.datamap)
            subset.info_table = info_table
            return subset
        else:
            raise ValueError(f'Invalid argument type: {type(item)}')
