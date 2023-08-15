"""
This module is responsible for handling data and executing I/O within the Paramaterial library.

The central components of the plug module are the DataSet and DataItem classes:
    - `DataItem`: A data structure that encapsulates a single test's information and data. A `DataItem` object holds:
        - `test_id`: A string identifier for the test.
        - `data`: A pandas DataFrame containing the test data.
        - `info`: A pandas Series containing the metadata associated with the test.
    - `DataSet`: A container class that manages a collection of `DataItem` objects. It provides various methods for
    managing and manipulating the data, including reading from files, writing to files, filtering, sorting, and applying
    custom functions.

Key Interactions between DataSet and DataItem:
    - Loading Data: A `DataSet` is initialized by providing paths to metadata and data files. It reads the files and
    constructs a collection of `DataItem` objects.
    - Accessing Data: You can access individual `DataItem` objects within a `DataSet` using index-based or test_id-based
    access through the `__getitem__` method.
    - Applying Functions: You can use the `apply` method in the `DataSet` class to apply a custom function to each
    `DataItem`. This enables complex data transformations and analyses.
    - Iterating: The `DataSet` class supports iteration over its `DataItem` objects, allowing you to loop through the
    data items using a standard `for` loop.
    - Writing Data: The `write_output` method allows you to save the information in the `DataSet` back to files,
    preserving changes made to the `DataItem` objects.

Examples:
    >>> # Load a DataSet from files
    >>> ds = DataSet('info.xlsx', 'data')
    >>> # Access a specific DataItem by test_id
    >>> di = ds['T01']
    >>> # Apply a custom function to all DataItems
    >>> def custom_function(di: DataItem) -> DataItem:
    ...     di.data['Stress_MPa'] *= 2
    ...     di.info['max_stress'] = di.data['Stress_MPa'].max()
    ...     return di
    >>> ds_modified = ds.apply(custom_function)
    >>> # Save the DataSet to new files
    >>> ds_modified.write_output('new_info.xlsx', 'new_data')
"""

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Callable, List, Any, Union, Optional

import pandas as pd
from tqdm import tqdm


class UnsupportedExtensionError(Exception):
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
    """A storage class for data and metadata related to a single test.

    Attributes:
        test_id: The unique identifier for the test.
        data: A pandas DataFrame containing the data related to the test.
        info: A pandas Series containing metadata related to the test.
    """
    test_id: str
    data: pd.DataFrame
    info: pd.Series


class DataSet:
    """A class for handling data, loading from files, and performing various operations.

    The DataSet class provides functionality for loading data from specified files, manipulating the data,
    and writing output. It contains a collection of DataItem objects, each representing a single test.

    Args:
        info_path: The path to the info table file containing metadata.
        data_dir: The directory containing the data files.
        test_id_key: The column name in the info table that contains the test IDs.

    Examples:
        >>> ds = DataSet(info_path='info/01_prepared_info.xlsx', data_dir='data/01_prepared_data')
        >>> len(ds)
        10

    Raises:
        ValueError: If only one of info_path and data_dir is specified.
        FileNotFoundError: If a file is not found for a given test_id.
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
        info_table = _read_file(self.info_path)

        try:
            test_ids = info_table[self.test_id_key].tolist()
        except KeyError:
            raise KeyError(f'Could not find test_id column "{self.test_id_key}" in info_table.')

        data_items = []
        for test_id in test_ids:
            file_path = os.path.join(self.data_dir, f'{test_id}.csv')
            if os.path.exists(file_path):
                info = info_table.loc[info_table[self.test_id_key] == test_id].iloc[0]
                data = _read_file(file_path)
                data_items.append(DataItem(test_id, data, info))
            else:
                raise FileNotFoundError(f"File not found for test_id={test_id}: {file_path}")

        return data_items

    @property
    def info_table(self) -> pd.DataFrame:
        info_table = pd.DataFrame([di.info for di in self.data_items])
        info_table.index = range(len(info_table))
        info_table = info_table.apply(pd.to_numeric, errors='ignore')
        return info_table

    @info_table.setter
    def info_table(self, info_table: pd.DataFrame):
        # Attempt to convert all columns to numeric, and ignore errors for non-numeric columns
        info_table = info_table.apply(pd.to_numeric, errors='ignore')

        data_item_dict = {data_item.test_id: data_item for data_item in self.data_items}
        test_ids = info_table[self.test_id_key].tolist()

        new_data_items = []
        for test_id in test_ids:
            data_item = data_item_dict.get(test_id)
            if data_item:
                # Get the corresponding row from the info_table
                row = info_table.loc[info_table[self.test_id_key] == test_id].iloc[0]
                # Update the data_item's info attribute
                data_item.info = row
                new_data_items.append(data_item)

        self.data_items = new_data_items

    def write_output(self, info_path: str, data_dir: str) -> None:
        """Write the DataSet to files.

        Args:
            info_path: The path to the info table file to be written.
            data_dir: The directory to write the data files to.

        Examples:
            >>> ds.write_output(info_path='info/processed_info.xlsx', data_dir='data/processed_data')

        Raises:
            FileNotFoundError: If the data_dir does not exist.
        """
        # Create the data directory if it doesn't exist
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        _write_file(self.info_table, info_path)
        for di in self.data_items:
            _write_file(di.data, f'{data_dir}/{di.test_id}.csv')

    def __iter__(self):
        """Iterate over the DataItems in the DataSet.

        Yields:
            DataItem: The next DataItem in the DataSet.

        Examples:
            >>> for di in ds:
            ...     print(di.test_id)
        """
        for test_id in tqdm(self.info_table[self.test_id_key].tolist(), unit='DataItems', leave=False):
            data_item = next((di for di in self.data_items if di.test_id == test_id), None)
            if data_item is None:
                raise ValueError(f"No DataItem found with test_id={test_id}.")
            yield data_item

    def apply(self, func: Callable[[DataItem, Dict], DataItem], **kwargs) -> 'DataSet':
        """Apply a function to each DataItem in the DataSet and return a new DataSet with the results.

        Args:
            func: The function to apply to each DataItem. It must take a DataItem and optional keyword arguments and
            return a DataItem.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A new DataSet containing the DataItems after applying the function.

        Examples:
            >>> def double_stress(di: DataItem) -> DataItem:
            ...     di.data['Stress_MPa'] *= 2
            ...     return di
            >>> ds_doubled = ds.apply(double_stress)
        """
        new_ds = self.copy()
        new_ds.data_items = [func(di, **kwargs) for di in copy.deepcopy(self.data_items)]
        return new_ds

    def copy(self) -> 'DataSet':
        """Create a copy of the DataSet.

        Returns:
            A copy of the DataSet.

        Examples:
            >>> ds_copy = ds.copy()
        """
        new_ds = DataSet(test_id_key=self.test_id_key)
        new_ds.data_items = [DataItem(di.test_id, di.data.copy(), di.info.copy()) for di in self.data_items]
        return new_ds

    def sort_by(self, column: Union[str, List[str]], ascending: bool = True) -> 'DataSet':
        """Sort a copy of the DataSet by a column in the info table and return the copy.

        Args:
            column: Column or list of columns to sort by.
            ascending: Whether to sort in ascending order. (Default: True)

        Returns:
            A new DataSet sorted by the specified column(s).

        Examples:
            >>> ds = DataSet(info_path='info/prepared_info.xlsx', data_dir='data/prepared_data')
            >>> ds_sorted = ds.sort_by('temperature')
        """
        new_ds = self.copy()

        if isinstance(column, str):
            column = [column]

        # Ensure that the specified columns are numeric before sorting
        for col in column:
            if new_ds.info_table[col].dtype == 'object':
                new_ds.info_table[col] = pd.to_numeric(new_ds.info_table[col], errors='ignore')

        new_ds.info_table = new_ds.info_table.sort_values(by=column, ascending=ascending).reset_index(drop=True)
        return new_ds

    def __getitem__(self, specifier: Union[int, str, slice]) -> Union[List[DataItem], DataItem]:
        """Get a DataItem or a list of DataItems by index, test_id, or slice.

        Args:
            specifier: An int for index-based access, a str for test_id-based access, or a slice for slicing.

        Returns:
            A DataItem or a list of DataItems depending on the specifier.

        Examples:
            >>> di = ds[0]         # Get the first DataItem
            >>> di = ds['T01']     # Get the DataItem with test_id='T01'
            >>> dis = ds[0:5]      # Get the first five DataItems

        Raises:
            ValueError: If an invalid specifier type is provided.
        """
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
        """Create a subset of the DataSet based on specified filters.

        Args:
            filter_dict: A dictionary containing column names as keys and values or list of values to filter by.

        Returns:
            A new DataSet containing only the filtered DataItems.

        Examples:
            >>> ds_tensile = ds.subset({'test_type': ['T']})

        Raises:
            ValueError: If an invalid filter key is provided.
        """
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
