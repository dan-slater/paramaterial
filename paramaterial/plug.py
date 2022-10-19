""" In charge of handling data and executing I/O. [danslater, 1march2022] """
import copy
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Callable, List, Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

@dataclass
class DataItem:
    test_id: str = None
    data: pd.DataFrame = None
    info: pd.Series = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        """Return a string with the test id, the info Series and the data DataFrame."""
        # make a more informative string
        return f'DataItem test_id:{self.test_id}\nDataItem info:\n{self.info}\nDataItem data:\n{self.data}'

    @staticmethod
    def read_from_csv(file_path: str):
        test_id = os.path.split(file_path)[1].split('.')[0]
        data = pd.read_csv(file_path)
        return DataItem(test_id, data)

    def get_row_from_info_table(self, info_table: pd.DataFrame):
        self.info = info_table.loc[info_table['test id'] == self.test_id].squeeze()
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

    def __init__(self, data_dir: str, info_path: str):
        """Initialize the dataset.
        Args:
            data_dir: The directory containing the data.
            info_path: The path to the info table.
        """
        self.data_dir = data_dir
        self.info_path = info_path
        self.info_table = pd.read_excel(self.info_path)
        if 'test id' not in self.info_table.columns:
            raise ValueError('No column called "test id" found in info table.')
        file_paths = [self.data_dir + f'/{test_id}.csv' for test_id in self.info_table['test id']]
        self.datamap = map(lambda path: DataItem.read_from_csv(path), file_paths)
        self.datamap = map(lambda obj: DataItem.get_row_from_info_table(obj, self.info_table), self.datamap)

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

    def make_representative_curves(self, data_dir: str, info_path: str, filter_keys: List[str], interp_by: str, interp_res: int = 200):
        """Make representative curves of the dataset and save them to a directory.
        Args:

        """
        subset_filters = []
        for col in filter_keys:
            values = self.info_table[col].unique()
            for value in values:
                subset_filters.append({col: [value]})
        repr_ids = [f'repr_id_{i:0>4}' for i in range(len(subset_filters))]
        for repr_id, subset_filter in zip(repr_ids, subset_filters):
            repr_df = pd.DataFrame()
            repr_subset = self[subset_filter]
            for di in repr_subset:
                interp_vec = np.linspace(di.data[interp_by].min(), di.data[interp_by].max(), interp_res)
                interp_df = pd.DataFrame({f'interp_vec_{interp_by}': interp_vec})
                for col in di.data.columns:
                    if col != interp_by:
                        interp_col = np.interp(interp_vec, di.data[interp_by], di.data[col])
                        interp_df[f'interp_{col}'] = interp_col
                repr_df = pd.concat([repr_df, interp_df])
            repr_df = repr_df.groupby(f'interp_vec_{interp_by}').mean().reset_index()
            repr_df.to_csv(data_dir + '/' + repr_id + '.csv', index=False)

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
        loading_bar = tqdm(range(len(self.info_table)))
        out_info_table = pd.DataFrame()
        for i, dataitem in enumerate(copy.deepcopy(self.datamap)):
            loading_bar.update()
            dataitem.write_data_to_csv(data_dir)
            out_info_table = pd.concat([out_info_table, dataitem.info.to_frame().T], ignore_index=True)
            out_info_table.to_excel(info_path, index=False)
        loading_bar.close()

    def copy(self) -> 'DataSet':
        """Return a copy of the dataset."""
        return copy.deepcopy(self)

    def __iter__(self):
        """Iterate over the dataset."""
        for dataitem in tqdm(copy.deepcopy(self.datamap)):
            if dataitem.test_id in self.info_table['test id'].values:
                yield dataitem

    def __eq__(self, other):
        """Check if the datamaps and info tables of the datasets are equal."""
        return self.datamap == other.datamap and self.info_table.equals(other.info_table)

    def __len__(self):
        """Get the number of dataitems in the dataset."""
        if len(self.info_table) != len(list(copy.deepcopy(self.datamap))):
            raise ValueError('Length of info table and datamap are different.')
        return len(self.info_table)

    def __repr__(self):
        """Get a string with the info table head and column headers of the first dataitem."""
        return f'DataSet:\n' \
               f'info_table:\n' \
               f'rows={len(self.info_table)}\n' \
               f'head=\n{self.info_table.head().to_string()}\n' \
               f'First {self[0]}'

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
