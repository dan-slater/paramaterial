""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Callable, List, Any, Union

import pandas as pd
from tqdm import tqdm


@dataclass
class DataItem:
    test_id: str = None
    data: pd.DataFrame = None
    info: pd.Series = None

    @staticmethod
    def read_from_csv(file_path: str):
        test_id = os.path.split(file_path)[1].split('.')[0]
        data = pd.read_csv(file_path)
        return DataItem(test_id, data)

    def get_row_from_info_table(self, info_table: pd.DataFrame):
        self.info = info_table.loc[info_table['test id'] == self.test_id].squeeze()
        return self

    def write_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.test_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self

    def __contains__(self, other: 'DataItem'):
        if self.test_id != other.test_id:
            return False
        if other.data is not None and other.data not in self.data:
            return False
        if other.info is not None and other.info not in self.info:
            return False
        return True


@dataclass
class DataSet:
    data_dir: str = None
    info_path: str = None

    def __init__(self, data_dir: str, info_path: str):
        self.data_dir = data_dir
        self.info_path = info_path
        self.info_table = pd.read_excel(self.info_path)
        if 'test id' not in self.info_table.columns:
            raise ValueError('No column called "test id" found in info table.')
        file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in self.info_table['test id']]
        self.datamap = map(lambda path: DataItem.read_from_csv(path), file_paths)
        self.datamap = map(lambda obj: DataItem.get_row_from_info_table(obj, self.info_table), self.datamap)

    def __iter__(self):
        """Iterate over the dataset."""
        for dataitem in copy.deepcopy(self.datamap):
            if dataitem.test_id in self.info_table['test id'].values:
                yield dataitem

    def __len__(self):
        """Get the number of dataitems in the dataset."""
        if len(self.info_table) != len(list(copy.deepcopy(self.datamap))):
            raise ValueError('Length of info table and datamap are different.')
        return len(self.info_table)

    def __contains__(self, item: DataItem):
        """Check if a dataitem is in the dataset."""
        return item.test_id in self.info_table['test id'].values

    def __getitem__(self, item: Union[Dict[str, List[Any]], int, slice]) -> Union['DataSet', DataItem]:
        """Get a subset of the dataset using a dictionary of column names and lists of values or using normal list
        indexing. """
        if isinstance(item, int):
            # get the test id of the ith row in the info table
            test_id = self.info_table.iloc[item]['test id']
            data = pd.read_csv(self.data_dir + '/' + test_id + '.csv')
            info = self.info_table.loc[self.info_table['test id'] == test_id].squeeze()
            return DataItem(test_id, data, info)
        elif isinstance(item, slice):
            subset = copy.deepcopy(self)
            subset.datamap = list(self.datamap)[item]
            subset.info_table = subset.info_table.iloc[item]
            return subset
        elif isinstance(item, dict):
            subset = copy.deepcopy(self)
            info = self.info_table
            for col_name, vals in item.items():
                if col_name not in self.info_table.columns:
                    raise ValueError(f'Column {col_name} not found in info table.')
                if not all([val in self.info_table[col_name].values for val in vals]):
                    raise ValueError(f'Values not found in "{col_name}" column:\n'
                                     f'\t{[val for val in vals if val not in self.info_table[col_name].values]}.')
                if not isinstance(vals, list):
                    vals = [vals]
                info = info.loc[info[col_name].isin(vals)]
            subset.info_table = info
            return subset
        else:
            raise ValueError(f'Invalid argument type: {type(item)}')

    def __add__(self, other: 'DataSet') -> 'DataSet':
        """Combine two datasets into one."""
        new_dataset = copy.deepcopy(self)
        new_dataset.datamap = list(self.datamap) + list(other.datamap)
        # check for duplicate test ids, if any, rename the duplicate test ids in the other dataset
        test_ids = self.info_table['test id'].values
        for test_id in other.info_table['test id'].values:
            if test_id in test_ids:
                new_test_id = test_id + '_2'
                other.info_table.loc[other.info_table['test id'] == test_id, 'test id'] = new_test_id
                other.datamap = map(lambda obj: DataItem(new_test_id, obj.data, obj.info) if obj.test_id == test_id
                                    else obj, other.datamap)
        new_dataset.info_table = pd.concat([self.info_table, other.info_table])
        return new_dataset

    def __sub__(self, other: 'DataSet') -> 'DataSet':
        """Remove a subset of the dataset."""
        new_dataset = copy.deepcopy(self)
        new_dataset.datamap = list(filter(lambda x: x not in other, self.datamap))
        new_dataset.info_table = self.info_table.loc[~self.info_table['test id'].isin(other.info_table['test id'])]
        return new_dataset

    def __eq__(self, other: 'DataSet') -> bool:
        """Check if two datasets are equal."""
        if self.info_table.equals(other.info_table) and self.datamap == other.datamap:
            return True
        return False

    def __ne__(self, other: 'DataSet') -> bool:
        """Check if two datasets are not equal."""
        return not self.__eq__(other)

    def copy(self) -> 'DataSet':
        """Return a copy of the dataset."""
        return copy.deepcopy(self)

    def apply(self, func: Callable[[DataItem], DataItem]) -> 'DataSet':
        """Apply a processing operation to the dataset.
        Args:
            func: The processing operation (function) to apply to the dataset.
            *args: Arguments to pass to the processing function.
            **kwargs: Keyword arguments to pass to the processing function.
        Returns:
            A new dataset with the processing operation applied.
        """
        new_dataset = copy.deepcopy(self)
        new_dataset.datamap = map(lambda dataitem: func(dataitem), new_dataset.datamap)
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
            dataitem.write_to_csv(data_dir)
        self.info_table.to_excel(info_path, index=False)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        loading_bar = tqdm(range(len(self.info_table)))
        out_info_table = pd.DataFrame()
        for i, dataitem in enumerate(copy.deepcopy(self.datamap)):
            loading_bar.update()
            dataitem.write_to_csv(data_dir)
            out_info_table = pd.concat([out_info_table, dataitem.info.to_frame().T], ignore_index=True)
            out_info_table.to_excel(info_path, index=False)
        loading_bar.close()
