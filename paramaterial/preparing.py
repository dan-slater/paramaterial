"""Functions to be used for preparing the experimental data for batch processing."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from paramaterial.plug import DataSet


def copy_data_and_rename_by_test_id(data_in: str, data_out: str, info_table: pd.DataFrame, test_id_col='test_id'):
    """Rename files in data directory by test_id in info table."""
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


def check_column_headers(data_dir: str, exception_headers: List[str] = None):
    """Check that all files in a directory have the same column headers and that column headers don't contain spaces."""
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
    """Check that there are no duplicate files in the data directory by hashing the contents."""
    print('Checking for duplicate files...')
    hashes = [hash(open(f'{data_dir}/{file}', 'rb').read()) for file in os.listdir(data_dir)]
    if len(hashes) != len(set(hashes)):
        duplicates = [file for file, filehash in zip(os.listdir(data_dir), hashes) if hashes.count(filehash) > 1]
        raise ValueError(f'There are duplicate files in {data_dir}.\n'
                         'The duplicates are:' + '\n\t'.join(duplicates))
    else:
        print(f'No duplicate files found in "{data_dir}".')


def make_experimental_matrix(info_table: pd.DataFrame, index: Union[str, List[str]], columns: Union[str, List[str]],
                             as_heatmap: bool = False, title: str = None, xlabel: str = None,
                             ylabel: str = None, tick_params: Dict = None, **kwargs) -> pd.DataFrame|plt.Axes:
    if isinstance(index, str):
        index = [index]
    if isinstance(columns, str):
        columns = [columns]
    exp_matrix = info_table.groupby(index + columns).size().unstack(columns).fillna(0).astype(int)
    if not as_heatmap:
        return exp_matrix
    else:
        ax = sns.heatmap(exp_matrix, **kwargs)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if tick_params:
            ax.tick_params(**tick_params)
        return ax


def convert_files_in_directory_to_csv(directory_path: str):
    for file in os.listdir(directory_path):
        if not file.endswith('.csv'):
            df = pd.read_csv(f'{directory_path}/{file}', header=[0, 1], delimiter='\t')
            df.columns = \
                [col[0] if str(col[1]).startswith('Unnamed') else ' '.join(col).strip() for col in df.columns]
            df.to_csv(f'{directory_path}/{file[:-4]}.csv', index=False)


def extract_info(in_dir, info_path):
    info_df = pd.DataFrame(columns=['filename', 'test type', 'temperature', 'material'])
    for filename in os.listdir(in_dir):
        info_row = pd.Series()
        info_row['filename'] = filename
        name_list = filename.split('_')
        if name_list[0] == 'P':
            info_row['test type'] = 'PST'
        else:
            info_row['test type'] = 'UT'
        info_row['temperature'] = float(name_list[1])
        info_row['material'] = 'AA6061-T651_' + name_list[2]
        info_df = info_df.append(info_row, ignore_index=True)
    info_df.to_excel(info_path, index=False)
