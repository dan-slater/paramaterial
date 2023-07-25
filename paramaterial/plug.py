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


# class SyncedInfoSeries:
#     def __init__(self, info_table: pd.DataFrame, test_id: str):
#         self.info_table = info_table
#         self.test_id = test_id
#
#     def __getitem__(self, key):
#         # Return a single value instead of a Series
#         return self.info_table.loc[self.info_table['test_id'] == self.test_id, key].values[0]
#
#     def __setitem__(self, key, value):
#         self.info_table.loc[self.info_table['test_id'] == self.test_id, key] = value
#
#
# @dataclass
# class DataItem:
#     test_id: str
#     data: pd.DataFrame
#     info_table: pd.DataFrame  # Reference to DataSet.info_table from parent
#     _info: SyncedInfoSeries = field(init=False)
#
#     def __post_init__(self):
#         self._info = SyncedInfoSeries(self.info_table, self.test_id)
#
#     @property
#     def info(self):
#         return self._info

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
            # self._info_table = pd.DataFrame()
            self.data_items = []
            return

        if info_path is None or data_dir is None:
            raise ValueError('Both info_path and data_dir must be specified, or neither.')

        # self._info_table = _read_file(info_path)
        self.data_items: List[DataItem] = self._load_data_items()

    def _load_data_items(self) -> List[DataItem]:
        file_paths = [self.data_dir + f'/{file}' for file in os.listdir(self.data_dir)]
        _info_table = _read_file(self.info_path)

        try:
            test_ids = _info_table[self.test_id_key].tolist()
        except KeyError:
            raise KeyError(f'Could not find test_id column "{self.test_id_key}" in info_table.')

        info_rows = [_info_table.loc[_info_table[self.test_id_key] == test_id].squeeze() for test_id in test_ids]

        return [DataItem(t_id, _read_file(f_path), info) for t_id, f_path, info in zip(test_ids, file_paths, info_rows)]

    @property
    def info_table(self) -> pd.DataFrame:
        a = [data_item.info for data_item in self.data_items]
        b = pd.DataFrame([data_item.info for data_item in self.data_items])
        return pd.DataFrame([data_item.info for data_item in self.data_items])

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        for data_item in self.data_items:
            data_item.info = info_table.loc[info_table[self.test_id_key] == data_item.test_id].squeeze()


    # @property
    # def info_table(self) -> pd.DataFrame:
    #     return self._info_table
    #
    # @info_table.setter
    # def info_table(self, info_table: pd.DataFrame):
    #     for data_item in self.data_items:
    #         data_item.info_table = info_table
    #     self._info_table = info_table

    # @property
    # def data_items(self) -> List[DataItem]:
    #     return self._data_items
    #
    # @data_items.setter
    # def data_items(self, data_items: List[DataItem]):
    #     if not all([data_item.info_table is self._info_table for data_item in data_items]):
    #         new_info_table = pd.concat([data_item.info_table for data_item in data_items], axis=1)
    #         new_info_table = new_info_table.loc[:, ~new_info_table.columns.duplicated()]
    #         self.info_table = new_info_table
    #     self._data_items = data_items

    def write_output(self, info_path: str, data_dir: str) -> None:
        """Execute the processing operations and write the output of the ds to a directory.
        Args:
            info_path: The path to write the info table to.
            data_dir: The directory to write the data files into.
        """
        _write_file(self.info_table, info_path)
        for data_item in self.data_items:
            _write_file(data_item.data, f'{data_dir}/{data_item.test_id}.csv')

    def __iter__(self):
        for test_id in tqdm(self.info_table[self.test_id_key].tolist(), unit='DataItems', leave=False):
            data_item = next((item for item in self.data_items if item.test_id == test_id), None)
            if data_item is None:
                raise ValueError(f"No DataItem found with test_id={test_id}.")
            yield data_item

    def apply(self, func: Callable[[DataItem, Dict], DataItem], **kwargs) -> 'DataSet':
        new_data_items = [func(data_item, **kwargs) for data_item in self.data_items]
        self.data_items = new_data_items
        return self

    def copy(self) -> 'DataSet':
        copied_dataset = DataSet(test_id_key=self.test_id_key)
        copied_dataset.data_items = [DataItem(di.test_id, di.data.copy(), di.info.copy())
                                     for di in self.data_items]
        return copied_dataset

    def sort_by(self, column: Union[str, List[str]], ascending: bool = True) -> 'DataSet':
        """Sort the info table of the current DataSet in place by a column or list of columns."""
        if isinstance(column, str):
            column = [column]
        new_info_table = self.info_table.sort_values(by=column, ascending=ascending)
        self.info_table = new_info_table
        return self

    def __getitem__(self, specifier: Union[int, str, slice, Dict[str, List[Any]]]) -> Union[List[DataItem], DataItem]:
        """Get a subset of the ds using a dictionary of column names and lists of values or using normal list
        indexing. """
        sorted_test_ids = self.info_table[self.test_id_key].tolist()
        sorted_data_items = [self.data_items[sorted_test_ids.index(test_id)] for test_id in sorted_test_ids]
        if isinstance(specifier, int):
            return sorted_data_items[specifier]
        elif isinstance(specifier, str):
            return self.data_items[self.info_table[self.test_id_key].tolist().index(specifier)]
        elif isinstance(specifier, slice):
            return sorted_data_items[specifier]
        else:
            raise ValueError(f'Invalid ds[<specifier>] specifier type: {type(specifier)}')

    def subset(self, filter_dict: Dict[str, Union[str, List[Any]]]) -> 'DataSet':
        """ Subset the DataSet based on a provided filtering dictionary.
        Args:
            filter_dict: A dictionary where the keys are column names from the info_table and values are lists of
            acceptable
                         values for the corresponding column.
        Returns:
            A new DataSet instance where the info_table and the data_items are filtered based on the provided
            filter_dict.
        """

        new_ds = self.copy()

        for key, value in filter_dict.items():
            if key not in new_ds.info_table.columns:
                raise ValueError(f'Invalid filter key: {key}')

            if not isinstance(value, list):
                filter_dict[key] = [value]

        query_string = ' and '.join([f"`{key}` in {str(values)}" for key, values in filter_dict.items()])

        try:
            new_info_table = new_ds.info_table.query(query_string)
        except Exception as e:
            print(f'Error applying query "{query_string}" to info_table: {e}')

        new_ds.info_table = new_info_table

        return new_ds

    def __repr__(self):
        return f"DataSet({len(self.info_table)} DataItems)"

    def __len__(self):
        return len(self.info_table)
