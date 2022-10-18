"""Module for examnple study of baron data."""
from matplotlib import pyplot as plt

import paramaterial as pam
from paramaterial.plug import DataSet


def main():
    """Main function."""
    dataset = DataSet(data_dir='data/01 prepared data', info_path='info/01 prepared info.xlsx')
    pam.plotting.dataset_plot(dataset, x='Strain', y='Stress(MPa)', ylabel='Stress (MPa)', cbar_by='temperature',
                              style_by='rate', alpha=1,
                              xlim=(-0.2, 2), grid=True)
    pam.plotting.dataset_subplots(dataset, x='Strain', y='Stress(MPa)', ylabel='Stress (MPa)',
                                  nrows=3, ncols=4,
                                  rows_by='material', cols_by='rate',
                                  row_keys=[['AC'],['H560'],['H580']], col_keys=[[1],[10],[50],[100]],
                                  col_titles=['1 s$^{-1}$', '10 s$^{-1}$', '50 s$^{-1}$', '100 s$^{-1}$'],
                                  cbar_by='temperature', cbar_label='Temperature (C)', grid=True)
    plt.show()

if __name__ == '__main__':
    main()
