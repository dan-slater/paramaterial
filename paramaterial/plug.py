""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import os
import shutil
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict, Callable

import pandas as pd
import copy

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
        self.info = info_table.loc[self.test_id].rename(self.test_id)
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

    def __init__(self, data_dir: str, info_path: str):
        self.datamap: map = None
        self.info_table: pd.DataFrame = None
        self.load_data(data_dir, info_path)

    def __iter__(self):
        return copy.deepcopy(self.datamap)

    def load_data(self, data_dir: str, info_path: str) -> None:
        self.info_table = pd.read_excel(info_path, index_col='test id')
        file_paths = [data_dir + f'/{test_id}.csv' for test_id in self.info_table.index]
        self.datamap = map(lambda path: DataItem.read_from_csv(path), file_paths)
        self.datamap = map(lambda obj: DataItem.get_row_from_info_table(obj, self.info_table), self.datamap)

    def get_subset(self, subset_cfg: Dict) -> 'DataSet':
        subset = copy.deepcopy(self)
        info = self.info_table
        for col_name, vals in subset_cfg.items():
            if col_name not in self.info_table.columns:
                raise ValueError(f'Column {col_name} not found in info table.')
            if not all([val in self.info_table[col_name].values for val in vals]):
                raise ValueError(f'Value not found in column {col_name}.')
            if not isinstance(vals, list):
                vals = [vals]
            if col_name == 'test id':
                info = info.loc[vals]
            elif len(vals) > 0:
                info = info.loc[info[col_name].isin(vals)]
        subset.info_table = info
        return subset

    def add_proc_op(self, func: Callable[[DataItem, Dict], DataItem], func_cfg: Dict):
        self.datamap = map(lambda dataitem: func(dataitem, func_cfg), self.datamap)

    def output(self, data_dir: str, info_path: str) -> None:
        tot = len(self.info_table)
        out_info_table = pd.DataFrame()
        for i, dataitem in enumerate(copy.deepcopy(self.datamap)):
            print(f'{dataitem.test_id} [{i}/{tot}]:')
            dataitem.write_to_csv(data_dir)
            info_row = pd.concat([pd.Series({'test id': dataitem.test_id}), dataitem.info])
            out_info_table = out_info_table.append(info_row, ignore_index=True)
            out_info_table.to_excel(info_path, index=False)
