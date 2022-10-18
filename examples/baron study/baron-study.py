"""Module for examnple study of baron data."""
from matplotlib import pyplot as plt
import seaborn as sns

import paramaterial as pam
from paramaterial.plug import DataSet, DataItem


def main():
    """Main function."""

    raw_dataset = DataSet(data_dir='data/01 prepared data', info_path='info/01 prepared info.xlsx')
    dataitem = raw_dataset[{'rate': [1], 'material': ['H560']}][0]
    # make pairplot from dataitem.data, color by temperature
    print(dataitem.data[::100])
    sns.pairplot(dataitem.data[::1000])
    plt.show()


# remove the trailing data in the dataitem.data after the maximimum Force
def remove_trailing_data(dataitem: DataItem):
    max_force = dataitem.data['Force'].max()
    max_force_index = dataitem.data[dataitem.data['Force'] == max_force].index[0]
    dataitem.data = dataitem.data[:max_force_index]
    return dataitem


def dataset_plot(dataset: DataSet):
    pam.plotting.dataset_plot(dataset, x='Strain', y='Stress(MPa)', ylabel='Stress (MPa)', cbar_by='temperature',
                              cbar_label=r'Temperature ($^{\circ}$C)', style_by='material', alpha=0.8,
                              xlim=(-0.2, 1.5), grid=True, style_by_label='Material', width_by='rate', width_by_label='Rate (/s)')


def dataset_subplot(dataset: DataSet):
    pam.plotting.dataset_subplots(dataset, x='Strain', y='Stress(MPa)', ylabel='Stress (MPa)',
                                  nrows=3, ncols=4,
                                  rows_by='material', cols_by='rate',
                                  row_keys=[['AC'],['H560'],['H580']], col_keys=[[1],[10],[50],[100]],
                                  col_titles=['1 s$^{-1}$', '10 s$^{-1}$', '50 s$^{-1}$', '100 s$^{-1}$'],
                                  cbar_by='temperature', cbar_label='Temperature (C)', grid=True)
    plt.show()

if __name__ == '__main__':
    main()
