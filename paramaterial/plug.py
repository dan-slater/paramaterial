""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import os
import shutil
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd


IO_Paths = namedtuple('IO_Paths', ['input_data', 'input_info', 'output_data', 'output_info'])


@dataclass
class DataSet:
    datamap: map = None
    info_table: pd.DataFrame = None

    def load(self, input_data_path: str, input_info_path: str, subset_config: Dict = None) -> None:
        df = pd.read_excel(input_info_path, index_col='test id')
        if subset_config is not None:
            for col_name, vals in subset_config.items():
                if len(vals) > 0:
                    df = df.loc[df[col_name].isin(vals)]
        self.info_table = df
        file_paths = [input_data_path + f'/{test_id}.csv' for test_id in self.info_table.index]
        self.datamap = map(lambda path: DataItem.read_from_csv(path), file_paths)
        self.datamap = map(lambda obj: DataItem.get_row_from_info_table(obj, self.info_table), self.datamap)

    def execute_mapping_and_write_output(self, output_data_path: str, output_info_path: str) -> None:
        tot = len(self.info_table)
        out_info_table = pd.DataFrame()
        for i, dataitem in enumerate(self.datamap):
            print(f'{dataitem.test_id} [{i}/{tot}]:')
            dataitem.write_to_csv(output_data_path)
            info_row = pd.concat([pd.Series({'test id': dataitem.test_id}), dataitem.info_row])
            out_info_table = out_info_table.append(info_row, ignore_index=True)
            out_info_table.to_excel(output_info_path, index=False)


@dataclass
class DataItem:
    test_id: str = None
    data: pd.DataFrame = None
    info_row: pd.Series = None

    @staticmethod
    def read_from_csv(file_path: str):
        test_id = os.path.split(file_path)[1].split('.')[0]
        data = pd.read_csv(file_path)
        return DataItem(test_id, data)

    def get_row_from_info_table(self, info_table: pd.DataFrame):
        self.info_row = info_table.loc[self.test_id].rename(self.test_id)
        return self

    def write_to_csv(self, output_dir: str):
        output_path = output_dir + '/' + self.test_id + '.csv'
        self.data.to_csv(output_path, index=False)
        return self


def empty_folder_at(dir_path: str) -> None:
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)
