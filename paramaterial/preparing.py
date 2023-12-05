"""Functions to be used for preparing the experimental data for batch processing."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from paramaterial.plug import DataSet

import os
import pandas as pd

# todo: figure out docstring convention to work with both pycharm and mkdocs
# todo: setting up plotting how to guide
# todo: experimental matrix multi plot easy.. use subplot wrapper?
# todo: generalise matrix plot, data plot, info plot? then keep subplot wrapper as a super for all these

def format_data_file_contents(raw_data_dir: str, prepared_data_dir: str, header_rows=None,
                              delimiter: str = ',', header_rename_dict: Dict[str, str] = None):
    """
    Format CSV files in a directory by combining column headers into one row and changing delimeter to a comma.

    Args:
    - directory_path: Path to the directory containing the original data files.
    - formatted_directory_path: Path to the directory where formatted files will be saved.
    - rename_dict: Dictionary mapping original column names to new column names.
    - header_rows: List of rows to use as headers (0-indexed). Rows above the highest-indexed row will be
    dropped. The original headers will be combined into a single row in the formatted files.
    - delimiter: Delimiter used in the CSV file. This will be changed to a comma in the formatted files.

    Example usage:
    >>> format_data_file_contents('data/00 raw data', 'data/01 prepared data', header_rename_dict={'Total Strain':
    'Strain', 'True Stress': 'Stress'}, header_rows=[0, 1], delimiter='\t')
    """
    # Default header rows to the first row
    if header_rows is None:
        header_rows = [0]

    # Check if the directory exists
    if not os.path.exists(raw_data_dir):
        raise ValueError(f'Directory not found: {raw_data_dir}')

    # Iterate through each file in the directory
    for file in os.listdir(raw_data_dir):

        if file.endswith('.csv'):
            # Read the CSV file with specified headers
            file_path = os.path.join(raw_data_dir, file)
            df = pd.read_csv(file_path, header=header_rows, delimiter=delimiter)

            # Combine multi-level columns and rename as specified
            new_headers = [' '.join(map(str, col)) if isinstance(col, tuple) else str(col)
                           for col in df.columns]
            df.columns = [header_rename_dict.get(col, col) for col in new_headers]

            # Save the formatted DataFrame to the new directory
            new_file_path = os.path.join(prepared_data_dir, file)
            df.to_csv(new_file_path, index=False)


def rename_data_files(data_dir: str, rename_table: Union[pd.DataFrame, str],
                      old_filename_col='old_filename', test_id_col='test_id'):
    """
    Rename data files in a directory based on test ID mappings provided in an info table.

    Args:
    - directory_path (str): Path to the directory containing the data files to be renamed.
    - info_table (pd.DataFrame): DataFrame containing the metadata for the tests.
    - old_filename_col (str): Column name in the info table for the old filenames.
    - test_id_col (str): Column name in the info table for the test IDs.

    Example usage:
    >>> rename_data_files(data_dir='data/01 prepared data', rename_table='info/00 rename list.xlsx',
    old_filename_col='old filename', test_id_col='test id')
    """
    # Read the info table if it is a string
    if type(rename_table) == str and rename_table.endswith('.xlsx'):
        rename_table = pd.read_excel(rename_table)
    elif type(rename_table) == str and rename_table.endswith('.csv'):
        rename_table = pd.read_csv(rename_table)
    else:
        assert isinstance(rename_table, pd.DataFrame)

    # Check rename table
    if old_filename_col not in rename_table.columns:
        raise ValueError(f'There is no "{old_filename_col}" column in the rename table.')
    if test_id_col not in rename_table.columns:
        raise ValueError(f'There is no "{test_id_col}" column in the rename table.')
    if rename_table[test_id_col].duplicated().any():
        raise ValueError(f'There are duplicate test_ids.')
    if rename_table[old_filename_col].duplicated().any():
        raise ValueError(f'There are duplicate old_filenames.')

    # Iterate over the rows in the info table
    for _, row in rename_table.iterrows():
        old_filename = row[old_filename_col] + '.csv'
        new_filename = f'{row[test_id_col]}.csv'
        old_filepath = os.path.join(data_dir, old_filename)
        new_filepath = os.path.join(data_dir, new_filename)

        # Rename the file if old file exists, and replace new file if it already exists
        try:
            if os.path.exists(old_filepath):
                os.rename(old_filepath, new_filepath)
                print(f'Renamed {old_filepath} to {new_filepath}.')
            else:
                print(f"File not found: {old_filepath}")
        except FileExistsError:
            os.remove(new_filepath)
            os.rename(old_filepath, new_filepath)
            print(f'Renamed {old_filepath} to {new_filepath} and replaced {new_filepath}.')


def check_formatting(data_dir: str, info_path: str, skip_headers: List[str] = None):
    """Performs various checks on the formatting of the data files and info table.
    Will print out formatting checks and warnings.

    Args:
        data_dir: Path to folder containing data files.
        info_path: Path to info table spreadsheet.
        skip_headers: Headers that can be different across data files. Other headers must be consistent.
    """
    # check column headers
    file_list = os.listdir(data_dir)
    first_file = pd.read_csv(f'{data_dir}/{file_list[0]}')
    first_columns = first_file.columns

    if skip_headers is not None:
        for exception_header in skip_headers:
            if exception_header in first_columns:
                first_columns = first_columns.drop(exception_header)

    mismatch = False
    for file in file_list[1:]:
        df = pd.read_csv(f'{data_dir}/{file}')
        df_columns = df.columns

        if skip_headers is not None:
            for exception_header in skip_headers:
                if exception_header in df_columns:
                    df_columns = df_columns.drop(exception_header)

        if not df_columns.equals(first_columns):
            print(f'Column headers in {file} don\'t match column headers of first file.'
                  f'{file} headers:\n\t{list(df.columns)}')
            mismatch = True

    if not mismatch and skip_headers is not None:
        print(f'Headers, except for {skip_headers}, are the same in all files as in the first file.')
        print('Consistent headers are: ' + ', '.join([col for col in first_columns if col not in skip_headers]))

    elif not mismatch and skip_headers is None:
        print(f'Headers are the same in all files as in the first file.')
        print(f'Consistent headers are: ' + ', '.join([col for col in first_columns]))

    # check for duplicate files
    hashes = [hash(open(f'{data_dir}/{file}', 'rb').read()) for file in os.listdir(data_dir)]
    if len(hashes) != len(set(hashes)):
        duplicates = [file for file, filehash in zip(os.listdir(data_dir), hashes) if hashes.count(filehash) > 1]
        raise ValueError(f'Duplicate files found in {data_dir} by hashing contents.\n'
                         'The duplicates are:' + '\n\t'.join(duplicates))
    else:
        print(f'No duplicate files (by hashing contents) found in "{data_dir}".')

    # check that test IDs match across data files and info table
    # todo



def experimental_matrix(info_table: pd.DataFrame, index: Union[str, List[str]], columns: Union[str, List[str]],
                        as_heatmap: bool = False, title: str = None, xlabel: str = None,
                        ylabel: str = None, tick_params: Dict = None, **kwargs) -> Union[pd.DataFrame, plt.Axes]:
    """Make an experimental matrix showing the distribution of test across metadata categories.

    Args:
        info_table: DataFrame containing the metadata for the tests.
        index: Column(s) of the info_table to use as the index of the matrix.
        columns: Column(s) of the info_table to use as the columns of the matrix.
        as_heatmap: If True, return a heatmap of the matrix. If False, return the matrix as a DataFrame.
        title: Title of the heatmap.
        xlabel: Label for the x-axis of the heatmap.
        ylabel: Label for the y-axis of the heatmap.
        tick_params: Parameters to pass to the ax.tick_params method of matplotlib.
        **kwargs: Additional keyword arguments to pass to the heatmap function.

    Returns:
        If as_heatmap is False, returns a DataFrame of the experimental matrix.
        If as_heatmap is True, returns a heatmap of the experimental matrix.
    """
    # make sure index and columns are lists
    if isinstance(index, str):
        index = [index]
    if isinstance(columns, str):
        columns = [columns]

    # make the experimental matrix
    exp_matrix = info_table.groupby(index + columns).size().unstack(columns).fillna(0).astype(int)

    # return the matrix as a DataFrame if as_heatmap is False
    if not as_heatmap:
        return exp_matrix

    # update the default kwargs
    default_kwargs = dict(linewidths=2, cbar=False, annot=True, square=True, cmap='Blues')
    default_kwargs.update(kwargs)

    ax = sns.heatmap(exp_matrix, **default_kwargs)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if tick_params:
        ax.tick_params(**tick_params)
    return ax
