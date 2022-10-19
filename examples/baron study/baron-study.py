"""Module for examnple study of baron data."""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import paramaterial as pam
from paramaterial.plug import DataSet, DataItem


def main():
    """Main function."""

    dataset = DataSet('data/02 processed data', 'info/02 processed info.xlsx')
    dataset.make_representative_curves('data/03 repr data', 'info/03 repr info.xlsx',
                                       ['material', 'rate', 'temperature'],
                                       interp_res=100,
                                       interp_by='Strain')


if __name__ == '__main__':
    main()
