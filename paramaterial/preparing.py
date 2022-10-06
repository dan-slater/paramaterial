import os
from typing import List

import numpy as np
import pandas as pd

from paramaterial.plug import DataSet
from screening import make_screening_pdf

def make_info_table(in_dir: str, info_path: str):
    """Make a table of information about the tests in the directory."""
    info_df = pd.DataFrame(columns=['test id', 'filename', 'test type', 'material', 'temperature', 'rate'])
    for filename in os.listdir(in_dir):
        info_row = pd.Series(dtype=str)
        info_row['filename'] = filename
        # info_df = info_df.append(info_row, ignore_index=True)
        info_df = pd.concat([info_df, info_row.to_frame().T], ignore_index=True)
    info_df.to_excel(info_path, index=False)
    return info_df


if __name__ == '__main__':
    make_info_table(r'../examples/baron study/data/00 backup data', r'../examples/baron study/info/00 backup info.xlsx')


def screen_raw_data(data_dir: str, info_path: str, screening_x: str, screening_y: str, check_headers: bool = True,
                    screening_pdf: bool = True):
    # check if column headers are the same, if not throw error
    if check_headers:
        check_column_headers(data_dir)
    # make screening pdf for data
    if make_screening_pdf:
        dataset = DataSet()
        dataset.load_data(data_dir, info_path)
        make_screening_pdf(dataset, screening_x, screening_y)

def check_column_headers(data_dir: str):
    file_list = os.listdir(data_dir)
    first_file = pd.read_csv(f'{data_dir}/{file_list[0]}')
    print("Checking column headers...")
    print(f'All headers should be:\n\t{first_file.columns}')
    for file in file_list[1:]:
        df = pd.read_csv(f'{data_dir}/{file}')
        if not np.all(first_file.columns == df.columns):
            raise ValueError('Column headers are not the same in all files.')


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
