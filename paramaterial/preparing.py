import os
from typing import List, Dict

import numpy as np
import pandas as pd

from paramaterial.plug import DataSet
from paramaterial.screening import make_screening_pdf


def make_info_table(data_dir: str, output_info_path: str):
    """Make a table of information about the tests in the directory."""
    info_df = pd.DataFrame(columns=['test id', 'old filename', 'test type', 'material', 'temperature', 'rate'])
    for filename in os.listdir(data_dir):
        info_row = pd.Series(dtype=str)
        info_row['old filename'] = filename
        info_df = pd.concat([info_df, info_row.to_frame().T], ignore_index=True)
    info_df.to_excel(output_info_path, index=False)
    return info_df


def screen_data(data_dir: str, pdf_path: str, df_plt_kwargs: Dict,
                check_headers: bool = True, screening_pdf: bool = True):
    # check if column headers are the same, if not throw error
    if check_headers:
        check_column_headers(data_dir)
    # make screening pdf for data
    if screening_pdf:
        make_screening_pdf(data_dir, pdf_path, df_plt_kwargs)


def rename_by_test_id(data_dir, info_path):
    info_df = pd.read_excel(info_path)
    if 'old filename' not in info_df.columns:
        raise ValueError(f'There is no "old filename" column in {info_path}. Please add it.'
                         f'\nExisting columns are: {list(info_df.columns)}')
    if 'test id' not in info_df.columns:
        raise ValueError(f'There is no "test id" column in {info_path}.')
    if info_df['test id'].duplicated().any():
        raise ValueError(f'There are duplicate test ids {info_path}.')
    if info_df['old filename'].duplicated().any():
        raise ValueError(f'There are duplicate old filenames in {info_path}.')
    for filename, test_id in zip(info_df['old filename'], info_df['test id']):
        os.rename(f'{data_dir}/{filename}', f'{data_dir}/{test_id}.csv')
    print(f'Renamed {len(info_df)} files in {data_dir}')


def check_column_headers(data_dir: str):
    file_list = os.listdir(data_dir)
    first_file = pd.read_csv(f'{data_dir}/{file_list[0]}')
    print("Checking column headers...")
    print(f'First file headers:\n\t{list(first_file.columns)}')
    for file in file_list[1:]:
        df = pd.read_csv(f'{data_dir}/{file}')
        if not np.all(first_file.columns == df.columns):
            raise ValueError('Column headers are not the same in all files.')
    print(f'Headers in all files are the same as in the first file.')


def make_preparing_screening_pdf(data_dir: str, pdf_path: str, df_plt_kwargs: Dict):
    print(os.getcwd())
    make_screening_pdf(data_dir, pdf_path, df_plt_kwargs)



if __name__ == '__main__':
    make_preparing_screening_pdf('../examples/baron study/data/01 raw data',
                '../examples/baron study/info/01 raw screening.pdf',
                df_plt_kwargs={'x': 'Jaw(mm)', 'y': 'Force(kN)'})

def make_prepared_data():
    ...


def copy_and_rename_by_test_id(old_dir: str, new_dir: str, info_path: str):
    info_df = pd.read_excel(info_path)
    for old_name, new_name in zip(info_df['old filename'], info_df['test id']):
        os.rename(f'{old_dir}/{old_name}.csv', f'{new_dir}/{new_name}.csv')


def convert_files_in_directory_to_csv(directory_path: str):
    # loop through files in directory
    # if a file is not a .csv file read it into a pandas dataframe and combine the first two rows as header if second column not unnamed
    # remove the suffix from the original name and save as a .csv file

    for file in os.listdir(directory_path):
        if not file.endswith('.csv'):
            df = pd.read_csv(f'{directory_path}/{file}', header=[0, 1], delimiter='\t')
            df.columns = \
                [col[0] if str(col[1]).startswith('Unnamed') else ' '.join(col).strip() for col in df.columns]
            df.to_csv(f'{directory_path}/{file[:-4]}.csv', index=False)


def print_file_names_in_directory(directory_path: str):
    for file in os.listdir(directory_path):
        print(file)


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

# def prepare_data(cfg):
#     in_dir = cfg['io'][0]
#     in_info_path = cfg['io'][1]
#     out_dir = cfg['io'][2]
#     out_info_path = cfg['io'][3]
#
#     in_info_df = pd.read_excel(in_info_path)
#     out_info_df = pd.DataFrame()
#
#     for filename in os.listdir(in_dir):
#         in_data_df = pd.read_csv(f'{in_dir}/{filename}').dropna()
#         info_row = in_info_df.loc[in_info_df['filename'] == filename].squeeze()
#         test_id = info_row['test id']
#
#         eng_strain = in_data_df['Strain']
#         eng_stress = in_data_df['Stress_MPa']
#         true_strain = np.log(1 + eng_strain)
#         true_stress = eng_stress * (1 + eng_strain)
#
#         out_data_df = pd.DataFrame()
#         out_data_df['eng strain'] = eng_strain
#         out_data_df['eng stress'] = eng_stress
#         out_data_df['Strain'] = true_strain
#         out_data_df['Stress(MPa)'] = true_stress
#
#         out_data_df.to_csv(f'{out_dir}/{test_id}.csv', index=False)
#         out_info_df = out_info_df.append(info_row)
#
#     out_info_df.to_excel(out_info_path, index=False)
