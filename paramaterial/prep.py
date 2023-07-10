"""Functions to be used for preparing the experimental data for batch processing."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from paramaterial.plug import DataSet


def check_formatting(ds: DataSet):
    """Check that the formatting of the data is correct. This includes checking that the column headers are the same in all
    files, that the column headers don't contain spaces, and that there are no duplicate files. A ValueError will be raised
    if any of these conditions are not met.

    Args:
        ds: DataSet object containing the data to be checked.
    """
    check_column_headers(ds.data_dir)
    check_for_duplicate_files(ds.data_dir)


def check_column_headers(data_dir: str, exception_headers: List[str] = None):
    """Check that all files in a directory have the same column headers and that column headers don't contain spaces.
    A ValueError will be raised if the column headers don't match or if a column header contains a space.

    Args:
        data_dir: Path to the directory containing the files to be checked.
        exception_headers: List of column headers that are allowed to be different between files.
    """
    file_list = os.listdir(data_dir)
    first_file = pd.read_csv(f'{data_dir}/{file_list[0]}')
    first_columns = first_file.columns
    if exception_headers is not None:
        for exception_header in exception_headers:
            if exception_header in first_columns:
                first_columns = first_columns.drop(exception_header)
    print("Checking column headers...")
    for column_header in first_file.columns:
        if len(column_header.split(' ')) > 1:
            raise ValueError(f'Column header "{column_header}" contains a space.')
    print(f'First file headers:\n\t{list(first_file.columns)}')
    for file in file_list[1:]:
        df = pd.read_csv(f'{data_dir}/{file}')
        df_columns = df.columns
        if exception_headers is not None:
            for exception_header in exception_headers:
                if exception_header in df_columns:
                    df_columns = df_columns.drop(exception_header)
        if not df_columns.equals(first_columns):
            raise ValueError(f'Column headers in {file} don\'t match column headers of first file.'
                             f'{file} headers:\n\t{list(df.columns)}')
    print(f'Headers in all files are the same as in the first file, except for {exception_headers}.')


def check_for_duplicate_files(data_dir: str):
    """Check that there are no duplicate files in a directory by hashing the contents of the files.
    A ValueError will be raised if there are duplicate files.

    Args:
        data_dir: Path to the directory containing the files to be checked.
    """
    print('Checking for duplicate files...')
    hashes = [hash(open(f'{data_dir}/{file}', 'rb').read()) for file in os.listdir(data_dir)]
    if len(hashes) != len(set(hashes)):
        duplicates = [file for file, filehash in zip(os.listdir(data_dir), hashes) if hashes.count(filehash) > 1]
        raise ValueError(f'There are duplicate files in {data_dir}.\n'
                         'The duplicates are:' + '\n\t'.join(duplicates))
    else:
        print(f'No duplicate files found in "{data_dir}".')


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


def convert_gleeble_output_files_to_csv(directory_path: str):
    """Convert all files in a directory from Gleeble output format to csv format."""
    for file in os.listdir(directory_path):
        if not file.endswith('.csv'):
            df = pd.read_csv(f'{directory_path}/{file}', header=[0, 1], delimiter='\t')
            df.columns = \
                [col[0] if str(col[1]).startswith('Unnamed') else ' '.join(col).strip() for col in df.columns]
            df.to_csv(f'{directory_path}/{file[:-4]}.csv', index=False)


def copy_data_and_rename_by_test_id(data_in: str, data_out: str, info_table: pd.DataFrame, test_id_col='test_id'):
    """Rename files in data directory by test_id in info table and copy to new directory. The info_table must have a column
    named 'old_filename' containing the original filenames and a column named 'test_id'. The new filenames will be the
    test_ids with the extension '.csv'.

    Args:
        data_in: Path to the directory containing the data to be copied.
        data_out: Path to the directory where the data will be copied.
        info_table: DataFrame containing the metadata for the tests.
        test_id_col: Column in the info table containing the test_ids.

    Returns:
        None
    """
    # make data directory if it doesn't exist
    if not os.path.exists(data_out):
        os.mkdir(data_out)

    # check info table
    if 'old_filename' not in info_table.columns:
        raise ValueError(f'There is no "old_filename" column in the info table.')
    if test_id_col not in info_table.columns:
        raise ValueError(f'There is no "{test_id_col}" column in the info table.')
    if info_table[test_id_col].duplicated().any():
        raise ValueError(f'There are duplicate test_ids.')
    if info_table['old_filename'].duplicated().any():
        raise ValueError(f'There are duplicate old_filenames.')

    for filename, test_id in zip(info_table['old_filename'], info_table[test_id_col]):
        # check that file exists
        if not os.path.exists(f'{data_in}/{filename}'):
            raise FileNotFoundError(f'File {filename} does not exist in {data_in}.')
        # copy and rename file
        shutil.copy(f'{data_in}/{filename}', f'{data_out}/{test_id}.csv')

    print(f'Copied {len(info_table)} files in {data_in} to {data_out}.')
