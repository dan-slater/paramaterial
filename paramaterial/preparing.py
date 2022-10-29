import os
import shutil
from typing import List, Dict, Union

import pandas as pd

from paramaterial.plug import DataSet
# from paramaterial.screening import make_screening_pdf


def make_info_table(data_dir: str, columns: List[str]) -> pd.DataFrame:
    """Make a table of information about the tests in the directory."""
    columns = ['test id', 'old filename'] + columns
    info_df = pd.DataFrame(columns=columns)
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            info_row = pd.Series(dtype=str)
            info_row['old filename'] = filename
            info_df = pd.concat([info_df, info_row.to_frame().T], ignore_index=True)
    return info_df


def copy_data_and_info(old_data_dir: str, new_data_dir: str, old_info_path: str, new_info_path: str) -> None:
    """Copy data and info from old directory and path to new directory and path."""

    # checks
    if not os.path.exists(old_data_dir):
        raise FileNotFoundError(f'Old data directory {old_data_dir} does not exist.')
    if not os.path.exists(old_info_path):
        raise FileNotFoundError(f'Old info file {old_info_path} does not exist.')
    if not old_info_path.endswith('.xlsx'):
        raise ValueError(f'Old info file {old_info_path} is not an excel file.')
    if not new_info_path.endswith('.xlsx'):
        raise ValueError(f'New info file {new_info_path} is not an excel file.')

    # copy data
    if not os.path.exists(new_data_dir):
        os.mkdir(new_data_dir)
    for file in os.listdir(old_data_dir):
        shutil.copy(f'{old_data_dir}/{file}', f'{new_data_dir}/{file}')

    # copy info
    shutil.copy(old_info_path, new_info_path)

    print(f'Copied {len(os.listdir(old_data_dir))} files from {old_data_dir} to {new_data_dir}.')
    print(f'Copied info table from {old_info_path} to {new_info_path}.')


def rename_by_test_id(data_dir, info_path, test_id_col='test id'):
    """Rename files in data directory by test id in info table."""
    # make data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # read info table
    info_df = pd.read_excel(info_path)

    # check info table
    if 'old filename' not in info_df.columns:
        raise ValueError(f'There is no "old filename" column in {info_path}. Please add it.'
                         f'\nExisting columns are: {list(info_df.columns)}')
    if test_id_col not in info_df.columns:
        raise ValueError(f'There is no "test id" column in {info_path}.')
    if info_df[test_id_col].duplicated().any():
        raise ValueError(f'There are duplicate test ids {info_path}.')
    if info_df['old filename'].duplicated().any():
        raise ValueError(f'There are duplicate old filenames in {info_path}.')

    # rename files if they exist and if new name is not already taken
    for filename, test_id in zip(info_df['old filename'], info_df[test_id_col]):
        if os.path.exists(f'{data_dir}/{filename}'):
            if os.path.exists(f'{data_dir}/{test_id}.csv'):
                raise ValueError(f'File {test_id}.csv already exists in {data_dir}.')
            os.rename(f'{data_dir}/{filename}', f'{data_dir}/{test_id}.csv')
            print(f'Renamed {filename} to {test_id}.csv.')
        else:
            print(f'File {filename}.csv does not exist in {data_dir}.')

    print(f'Renamed {len(info_df)} files in {data_dir}.')


def check_column_headers(data_dir: str):
    file_list = os.listdir(data_dir)
    first_file = pd.read_csv(f'{data_dir}/{file_list[0]}')
    print("Checking column headers...")
    print(f'First file headers:\n\t{list(first_file.columns)}')
    for file in file_list[1:]:
        df = pd.read_csv(f'{data_dir}/{file}')
        try:
            assert set(first_file.columns) == set(df.columns)
        except AssertionError:
            raise ValueError(f'Column headers in {file} don\'t match column headers of first file.'
                             f'{file} headers:\n\t{list(df.columns)}')
    print(f'Headers in all files are the same as in the first file.')


def make_experimental_matrix(dataset: DataSet, index: Union[str, List[str]], columns: Union[str, List[str]]):
    if isinstance(index, str):
        index = [index]
    if isinstance(columns, str):
        columns = [columns]
    return dataset.info_table.groupby(index + columns).size().unstack(columns).fillna(0).astype(int)


def copy_and_rename_by_test_id(old_dir: str, new_dir: str, info_path: str):
    info_df = pd.read_excel(info_path)
    for old_name, new_name in zip(info_df['old filename'], info_df['test id']):
        os.rename(f'{old_dir}/{old_name}.csv', f'{new_dir}/{new_name}.csv')


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
