""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
import shutil
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Callable, Optional, List, Tuple, Any

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tqdm import tqdm

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from paramaterial.processing import processing_function
from paramaterial.plotting.dataset_plot import dataset_plot, dataset_subplots

IO_Paths = namedtuple('IO_Paths', ['input_data', 'input_info', 'output_data', 'output_info'])


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

    def __enter__(self):
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
        return len(self.info_table)

    def __contains__(self, item: DataItem):
        """Check if a dataitem is in the dataset."""
        return item.test_id in self.info_table['test id'].values

    def __getitem__(self, index) -> 'DataSet':
        """Get a subset of the dataset using pandas filtering syntax."""
        subset = copy.deepcopy(self)
        subset.info_table = self.info_table.loc[index]
        return subset

    def get_subset(self, subset_keys: Dict[str, List[Any]]) -> 'DataSet':
        """Get a subset of the dataset.

        Args:
            subset_keys: A dictionary of column names and lists of values to use for the subset.
        """
        subset = copy.deepcopy(self)
        info = self.info_table
        for col_name, vals in subset_keys.items():
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

    def add_proc_op(self, func: Callable[[DataItem, ...], DataItem], *args, **kwargs) -> None:
        """Add a processing operation to the dataset.

        Args:
            func: The function to apply to the dataset.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        self.datamap = map(lambda dataitem: func(dataitem, *args, **kwargs), self.datamap)

    def write_output(self, data_dir: str, info_path: str) -> None:
        """Exectue the processing operations and write the output of the dataset to a directory.

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

