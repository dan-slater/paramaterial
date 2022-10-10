""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
import shutil
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Callable, Optional

import matplotlib

from tqdm import tqdm

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from paramaterial.plotting.dataset_plot import dataset_plot

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


def empty_folder_at(dir_path: str) -> None:
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)


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
        return copy.deepcopy(self.datamap)

    def __len__(self):
        return len(self.info_table)

    def plot(self, ax: plt.Axes, colourby: Optional[str] = None, **df_plot_kwargs):
        # raise error if ax in df_plot_kwargs
        if 'ax' in df_plot_kwargs:
            raise ValueError('Cannot specify "ax" in df_plot_kwargs.')
        dataset_plot(self, ax, colourby, **df_plot_kwargs)

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
    dataset = DataSet('../examples/vos ringing study/data/01 prepared data',
                      '../examples/vos ringing study/info/01 prepared info.xlsx')
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    dataset.plot(ax, x='Strain', y='Stress (MPa)', ylabel='Stress (MPa)', legend=False)
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    def neg_stress(dataitem):
        dataitem.data['Stress (MPa)'] = -dataitem.data['Stress (MPa)']
        dataitem.data['Strain'] = -dataitem.data['Strain']
        return dataitem
    dataset.add_proc_op(neg_stress)
    dataset.output('../examples/vos ringing study/data/02 processed data',
                   '../examples/vos ringing study/info/02 processed info.xlsx')
    dataset = DataSet('../examples/vos ringing study/data/02 processed data',
                      '../examples/vos ringing study/info/02 processed info.xlsx')
    dataset.plot(ax, x='Strain', y='Stress (MPa)', ylabel='Stress (MPa)', legend=False, colourby='rate')
    plt.show()
