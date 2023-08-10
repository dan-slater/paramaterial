""" In charge of handling data and executing I/O. """
import copy
import os
from dataclasses import dataclass, field
from typing import Dict, Callable, List, Any, Union, Optional

import pandas as pd
from tqdm import tqdm


class UnsupportedExtensionError(Exception):
    """Exception raised when an unsupported file extension is encountered."""

    def __init__(self, extension, message="Unsupported file extension."):
        self.extension = extension
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.extension} -> {self.message}'


def _read_file(path: str) -> pd.DataFrame:
    """Read a file into a pandas DataFrame.
    Args:
        path: The path to the file to be read.
    Returns:
        A pandas DataFrame containing the data from the file.
    """
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    elif path.endswith('.ods'):
        return pd.read_excel(path, engine='odf')
    else:
        raise UnsupportedExtensionError(path.split('.')[-1], "Read file extension not supported.")


def _write_file(df: pd.DataFrame, path: str) -> None:
    """Write a pandas DataFrame to a file.
    Args:
        df: The pandas DataFrame to be written.
        path: The path to the file to be written.
    """
    if path.endswith('.csv'):
        df.to_csv(path, index=False)
    elif path.endswith('.xlsx'):
        df.to_excel(path, index=False)
    elif path.endswith('.ods'):
        df.to_excel(path, index=False, engine='odf')
    else:
        raise UnsupportedExtensionError(path.split('.')[-1], "Write file extension not supported.")


@dataclass
class DataItem:
    test_id: str
    data: pd.DataFrame
    info: pd.Series


class DataSet:
    """A class for handling data.
    Args:
        info_path: The path to the info table file.
        data_dir: The directory containing the data files.
        test_id_key: The column name in the info table that contains the test ids.
    """

    def __init__(self, info_path: Optional[str] = None, data_dir: Optional[str] = None, test_id_key: str = 'test_id'):
        self.info_path = info_path
        self.data_dir = data_dir
        self.test_id_key = test_id_key

        if info_path is None and data_dir is None:
            self.data_items = []
        elif info_path is None or data_dir is None:
            raise ValueError('Both info_path and data_dir must be specified, or neither.')
        else:
            self.data_items: List[DataItem] = self._load_data_items()

    def _load_data_items(self) -> List[DataItem]:
        file_paths = [self.data_dir + f'/{file}' for file in os.listdir(self.data_dir)]
        info_table = _read_file(self.info_path)

        try:
            test_ids = info_table[self.test_id_key].tolist()
        except KeyError:
            raise KeyError(f'Could not find test_id column "{self.test_id_key}" in info_table.')

        info_rows = [info_table.loc[info_table[self.test_id_key] == test_id].squeeze() for test_id in test_ids]
        return [DataItem(t_id, _read_file(f_path), info) for t_id, f_path, info in zip(test_ids, file_paths, info_rows)]

    @property
    def info_table(self) -> pd.DataFrame:
        info_table = pd.DataFrame([di.info for di in self.data_items])
        info_table.index = range(len(info_table))
        return info_table

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        data_item_dict = {data_item.test_id: data_item for data_item in self.data_items}
        test_ids = info_table[self.test_id_key].tolist()

        new_data_items = []
        for test_id in test_ids:
            data_item = data_item_dict.get(test_id)
            if data_item:
                data_item.info = info_table.loc[info_table[self.test_id_key] == test_id].squeeze()
                new_data_items.append(data_item)

        self.data_items = new_data_items

    def write_output(self, info_path: str, data_dir: str) -> None:
        _write_file(self.info_table, info_path)
        for di in self.data_items:
            _write_file(di.data, f'{data_dir}/{di.test_id}.csv')

    def __iter__(self):
        for test_id in tqdm(self.info_table[self.test_id_key].tolist(), unit='DataItems', leave=False):
            data_item = next((di for di in self.data_items if di.test_id == test_id), None)
            if data_item is None:
                raise ValueError(f"No DataItem found with test_id={test_id}.")
            yield data_item

    def apply(self, func: Callable[[DataItem, Dict], DataItem], **kwargs) -> 'DataSet':
        new_ds = self.copy()
        new_ds.data_items = [func(di, **kwargs) for di in self.data_items]
        return new_ds

    def copy(self) -> 'DataSet':
        new_ds = DataSet(test_id_key=self.test_id_key)
        new_ds.data_items = [DataItem(di.test_id, di.data.copy(), di.info.copy()) for di in self.data_items]
        return new_ds

    def sort_by(self, column: Union[str, List[str]], ascending: bool = True) -> 'DataSet':
        """Sort the info table of the current DataSet in place by a column or list of columns."""
        if isinstance(column, str):
            column = [column]
        self.info_table = self.info_table.sort_values(by=column, ascending=ascending)
        return self

    def __getitem__(self, specifier: Union[int, str, slice]) -> Union[List[DataItem], DataItem]:
        if isinstance(specifier, int):
            return self.data_items[specifier]
        elif isinstance(specifier, str):
            return self.data_items[self.info_table[self.test_id_key].tolist().index(specifier)]
        elif isinstance(specifier, slice):
            return self.data_items[specifier]
        else:
            raise ValueError(
                f'Invalid ds[<specifier>] specifier type: {type(specifier)}. Must be int, str (test_id), or slice.')

    def subset(self, filter_dict: Dict[str, Union[str, List[Any]]]) -> 'DataSet':
        new_ds = self.copy()
        for key, value in filter_dict.items():
            if key not in new_ds.info_table.columns:
                raise ValueError(f'Invalid filter key: {key}. Must be one of {new_ds.info_table.columns}.')
            if not isinstance(value, list):
                filter_dict[key] = [value]
        query_string = ' and '.join([f"`{key}` in {str(values)}" for key, values in filter_dict.items()])
        try:
            new_info_table = new_ds.info_table.query(query_string)
            new_ds.info_table = new_info_table
        except Exception as e:
            print(f'Error applying query "{query_string}" to info_table: {e}')
        return new_ds

    def __repr__(self):
        return f"DataSet({len(self.data_items)} DataItems)"

    def __len__(self):
        return len(self.data_items)
