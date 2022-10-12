""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
import shutil
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Callable, Optional, List, Tuple

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tqdm import tqdm

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

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


@dataclass
class DataSet:
    info_table: pd.DataFrame = None
    datamap: map = None

    def __init__(self, data_dir: str, info_path: str):
        self.info_table = pd.read_excel(info_path)
        if 'test id' not in self.info_table.columns:
            raise ValueError('No column called "test id" found in info table.')
        file_paths = [data_dir + f'/{test_id}.csv' for test_id in self.info_table['test id']]
        self.datamap = map(lambda path: DataItem.read_from_csv(path), file_paths)
        self.datamap = map(lambda obj: DataItem.get_row_from_info_table(obj, self.info_table), self.datamap)

    def __iter__(self):
        for dataitem in copy.deepcopy(self.datamap):
            if dataitem.test_id in self.info_table['test id'].values:
                yield dataitem

    def __len__(self):
        return len(self.info_table)

    def plot(self,
             x: str,
             y: str,
             colorby: Optional[str] = None,
             styleby: Optional[str] = None,
             markerby: Optional[str] = None,
             widthby: Optional[str] = None,
             cbar: bool = False,
             cbar_label: Optional[str] = None,
             **kwargs) -> plt.Axes:
        """ Plot the data in the dataset.

        Args:
            x: The column name of the x-axis data.
            y: The column name of the y-axis data.
            colorby: The info column to use for coloring.
            styleby: The info column to use for line style.
            markerby: The info column to use for marker style.
            widthby: The info column to use for line width.
            cbar: Whether to add a colorbar.
            cbar_label: The label for the colorbar.
         """
        return dataset_plot(self, x, y,
                            colorby, styleby, markerby, widthby,
                            cbar, cbar_label,
                            **kwargs)

    def subplots(
            self,
            x: str,
            y: str,
            nrows: int,
            ncols: int,
            cols_by: str,
            rows_by: str,
            col_keys: List[str],
            row_keys: List[str],
            figsize: Tuple[float, float] = (6.4, 4.8),
            row_titles: Optional[List[str]] = None,
            col_titles: Optional[List[str]] = None,
            plot_titles: Optional[List[str]] = None,
            **kwargs
    ) -> tuple[Figure, Axes]:
        """Plot a subplot of the dataset.
        Args:
            nrows: The number of rows in the subplot.
            ncols: The number of columns in the subplot.
            col_keys: The info column keys to use for the columns.
            row_keys: The info column keys to use for the rows.
            figsize: The figure size.
            row_titles: The titles for the rows.
            col_titles: The titles for the columns.
            plot_titles: The titles for the plots.
            **kwargs: Keyword arguments to pass to the dataset_plot() function.
        Returns: The figure and axes.
        """
        return dataset_subplots(self, x=x, y=y, nrows=nrows, ncols=ncols, cols_by=cols_by, rows_by=rows_by,
                                col_keys=col_keys, row_keys=row_keys, figsize=figsize, row_titles=row_titles,
                                col_titles=col_titles, plot_titles=plot_titles, **kwargs)

    def get_subset(self, subset_keys: Dict) -> 'DataSet':
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

    def add_proc_op(self, func: Callable[[DataItem, ...], DataItem], func_cfg: Optional[Dict] = None):
        if func_cfg is not None:
            self.datamap = map(lambda dataitem: func(dataitem, func_cfg), self.datamap)
        else:
            self.datamap = map(lambda dataitem: func(dataitem), self.datamap)

    def output(self, data_dir: str, info_path: str) -> None:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        loading_bar = tqdm(range(len(self.info_table)))
        out_info_table = pd.DataFrame()
        for i, dataitem in enumerate(copy.deepcopy(self.datamap)):
            loading_bar.update()
            dataitem.write_to_csv(data_dir)
            out_info_table = pd.concat([out_info_table, dataitem.info.to_frame().T], ignore_index=True)
            out_info_table.to_excel(info_path, index=False)


if __name__ == '__main__':
    dataset = DataSet('../examples/aakash study/data/01 raw data', '../examples/aakash study/info/01 raw info.xlsx')
    dataset.subplots(
        x='Strain',
        y='Stress_MPa',
        ylabel='Stress (MPa)',
        nrows=2,
        ncols=2,
        cols_by='test type',
        rows_by='material',
        col_keys=['P', 'T'],
        row_keys=['G', 'H']
    )
    plt.show()
