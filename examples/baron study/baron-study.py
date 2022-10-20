"""Module for examnple study of baron data."""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import paramaterial as pam
from paramaterial.plug import DataSet, DataItem


def main():
    """Main function."""

    dataset = DataSet('data/02 processed data', 'info/02 processed info.xlsx')

    def drop_cols(dataitem: DataItem):
        dataitem.data = dataitem.data.drop(columns=[
            'Time(sec)', 'Force(kN)', 'Jaw(mm)', 'TC1(C)', 'time diff', 'Pressure(MPa)'
        ])
        return dataitem

    dataset = dataset.apply_function(drop_cols)
    dataset.make_representative_curves(
        'data/03 repr data', 'info/03 repr info.xlsx',
        ['material', 'rate', 'temperature'],
        interp_res=100, interp_by='Strain')


if __name__ == '__main__':
    main()
