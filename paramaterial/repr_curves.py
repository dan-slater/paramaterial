from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from paramaterial.plug import DataSet, DataItem


def main():
    # data_path = '../examples/data/04 corrected data'
    # info_path = '../examples/info/02 screened info.xlsx'
    # out_data_path = '../examples/data/11 representative data'
    # out_info_path = '../examples/info/11 representative info.xlsx'
    data_path = '../examples/hot axisymmetric compression example/data/12 fitted data ramberg'
    info_path = '../examples/hot axisymmetric compression example/info/02 screened info.xlsx'
    out_data_path = '../examples/hot axisymmetric compression example/data/13 representative data ramberg'
    out_info_path = '../examples/hot axisymmetric compression example/info/13 representative info ramberg.xlsx'
    subset_filters = create_filter_permutations(info_path=info_path)
    make_representative_data(data_path, info_path, subset_filters, out_data_path, out_info_path)






# todo: find a way to make this more general
def create_filter_permutations(info_path):
    filters = []
    info = pd.read_excel(info_path)
    references = info['reference'].unique()
    for ref in references:
        ref_info = info.loc[info['reference'] == ref]
        test_types = ref_info['test type'].unique()
        for test_type in test_types:
            test_type_info = ref_info.loc[ref_info['test type'] == test_type]
            materials = test_type_info['material'].unique()
            for mat in materials:
                mat_info = test_type_info.loc[test_type_info['material'] == mat]
                rates = mat_info['rate'].unique()
                for rate in rates:
                    temp_info = mat_info.loc[mat_info['rate'] == rate]
                    temps = temp_info['temperature'].unique()
                    for temp in temps:
                        filters.append({'reference': [ref], 'test type': [test_type], 'material': [mat], 'rate': [rate],
                                        'temperature': [temp]})
    return filters


def make_dataset_permutations(dataset: DataSet, columns: List[str]):
    filters = []
    for column in columns:
        values = dataset.info_table[column].unique()
        for value in values:
            filters.append({column: [value]})
    return filters


def make_representative_data(data_path, info_path, subset_filters, out_data_path, out_info_path, strain_res: int = 500):
    # make info dataframe for output
    out_info_df = pd.DataFrame(columns=['test id', 'reference', 'test type', 'material', 'rate', 'temperature'])
    # make a dataframe for each subset filter
    for i, fltr in enumerate(subset_filters):
        # setup data storage objects
        repr_df = pd.DataFrame()
        dataset = DataSet() # todo: update to use data_path and info_path
        dataitem: DataItem  # type hint for dataitem objects returned by iterating through dataset
        dataset.load_data(data_path, info_path, fltr)
        # find max_strain and make monotonically increasing strain vector
        # max_strain = max([dataitem.data['Strain'].max() for dataitem in dataset])
        # max_strain = max([dataitem.data['model strain'].max() for dataitem in dataset])
        max_strain = .95
        strain_vec = np.linspace(0, max_strain, strain_res)
        repr_df['strain'] = strain_vec
        # add stress for each dataitem to interp dataframe
        interp_df = pd.DataFrame()
        for _, dataitem in enumerate(dataset):
            # ensure data starts at zero
            dataitem.data = pd.concat([pd.Series(0, index=dataitem.data.columns), dataitem.data.loc[:]]).reset_index(
                drop=True)
            interp_df[f'stress{_}'] = np.interp(
                # strain_vec, dataitem.data['Strain'], dataitem.data['Stress(MPa)'])
                strain_vec, dataitem.data['model strain'], dataitem.data['ramberg stress'])
        # add min, mean and max to repres df and smooth
        repr_df['min'] = interp_df.min(axis=1)
        repr_df['mean'] = interp_df.mean(axis=1)
        repr_df['max'] = interp_df.max(axis=1)
        repr_df['upperstd'] = interp_df.std(axis=1)
        repr_df['lowerstd'] = interp_df.std(axis=1)
        repr_df = repr_df.fillna(0)
        # save repr data
        repr_id = f'reprID_{i:0>4}'
        repr_df.to_csv(f'{out_data_path}/{repr_id}.csv', index=False)
        info_row = pd.Series(fltr).apply(lambda l: l[0])
        info_row['test id'] = f'{repr_id}'
        out_info_df = out_info_df.append(info_row, ignore_index=True)
        out_info_df.to_excel(out_info_path, index=False)


if __name__ == '__main__':
    main()
