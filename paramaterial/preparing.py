import os
import shutil
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd

from paramaterial.plug import DataSet
# from paramaterial.screening import make_screening_pdf


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


def copy_data_and_rename_by_test_id(data_in: Path, data_out: Path, info_table: pd.DataFrame, test_id_col='test id'):
    """Rename files in data directory by test id in info table."""
    # make data directory if it doesn't exist
    if not os.path.exists(data_out):
        os.mkdir(data_out)

    # check info table
    if 'old filename' not in info_table.columns:
        raise ValueError(f'There is no "old filename" column in the info table.')
    if test_id_col not in info_table.columns:
        raise ValueError(f'There is no "{test_id_col}" column in the info table.')
    if info_table[test_id_col].duplicated().any():
        raise ValueError(f'There are duplicate test ids.')
    if info_table['old filename'].duplicated().any():
        raise ValueError(f'There are duplicate old filenames.')

    for filename, test_id in zip(info_table['old filename'], info_table[test_id_col]):
        # check that file exists
        if not os.path.exists(f'{data_in}/{filename}'):
            raise FileNotFoundError(f'File {filename} does not exist in {data_in}.')
        # copy and rename file
        shutil.copy(f'{data_in}/{filename}', f'{data_out}/{test_id}.csv')

    print(f'Copied {len(info_table)} files in {data_in} to {data_out}.')


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


# compare csv files
def check_for_duplicate_files(data_dir: str):
    hashes = [hash(open(f'{data_dir}/{file}', 'rb').read()) for file in os.listdir(data_dir)]
    if len(hashes) != len(set(hashes)):
        duplicates = [file for file, filehash in zip(os.listdir(data_dir), hashes) if hashes.count(filehash) > 1]
        raise ValueError(f'There are duplicate files in {data_dir}.\n'
                         'The duplicates are:' + '\n\t'.join(duplicates))
    else:
        print(f'No duplicate files found in "{data_dir}".')



def make_experimental_matrix(info_table: pd.DataFrame, index: Union[str, List[str]], columns: Union[str, List[str]]):
    if isinstance(index, str):
        index = [index]
    if isinstance(columns, str):
        columns = [columns]
    return info_table.groupby(index + columns).size().unstack(columns).fillna(0).astype(int)


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
