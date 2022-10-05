import os

import numpy as np
import yaml
import pandas as pd

IN_DIR = r'../data/01 raw data'
IN_INFO_PATH = r'../info/01 raw info.xlsx'
OUT_DIR = r'../data/02 prepared data'
OUT_INFO_PATH = r'../info/02 prepared info.xlsx'
CFG = {'io': [IN_DIR, IN_INFO_PATH, OUT_DIR, OUT_INFO_PATH]}



def convert_files_in_directory_to_csv(directory_path: str):
    # loop through files in directory
    # if a file is not a .csv file read it into a pandas dataframe and combine the first two rows as header if second column not unnamed
    # remove the suffix from the original name and save as a .csv file

    for file in os.listdir(directory_path):
        if not file.endswith('.csv'):
            df = pd.read_csv(f'{directory_path}/{file}', header=[0,1], delimiter='\t')
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


def prepare_data(cfg):
    in_dir = cfg['io'][0]
    in_info_path = cfg['io'][1]
    out_dir = cfg['io'][2]
    out_info_path = cfg['io'][3]

    in_info_df = pd.read_excel(in_info_path)
    out_info_df = pd.DataFrame()

    for filename in os.listdir(in_dir):
        in_data_df = pd.read_csv(f'{in_dir}/{filename}').dropna()
        info_row = in_info_df.loc[in_info_df['filename'] == filename].squeeze()
        test_id = info_row['test id']

        eng_strain = in_data_df['Strain']
        eng_stress = in_data_df['Stress_MPa']
        true_strain = np.log(1 + eng_strain)
        true_stress = eng_stress * (1 + eng_strain)

        out_data_df = pd.DataFrame()
        out_data_df['eng strain'] = eng_strain
        out_data_df['eng stress'] = eng_stress
        out_data_df['Strain'] = true_strain
        out_data_df['Stress(MPa)'] = true_stress

        out_data_df.to_csv(f'{out_dir}/{test_id}.csv', index=False)
        out_info_df = out_info_df.append(info_row)

    out_info_df.to_excel(out_info_path, index=False)


if __name__ == '__main__':
    # extract_info(IN_DIR, INFO_PATH)
    # prepare_data(CFG)
    convert_files_in_directory_to_csv(r"C:\Users\DS\paramaterial\examples\uniaxial case study\data\00 backup data")
