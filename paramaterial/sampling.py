"""For sampling data to pass into fitting. [danslater, 7apr22]"""
import numpy as np

from paramaterial.plug import DataItem


def sample(dataitem: DataItem, sample_size: int, delete_neg_strain: bool = True):
    dataitem.info['nr of points sampled'] = sample_size
    df = dataitem.data

    x_data = df['Strain'].values
    y_data = df['Stress(MPa)'].values

    if delete_neg_strain:
        for i, x_val in enumerate(x_data):
            if x_val < 0:
                x_data = np.delete(x_data, [i])
                y_data = np.delete(y_data, [i])

    sampling_stride = int(len(x_data) / sample_size)
    if sampling_stride < 1:
        sampling_stride = 1

    x_data = x_data[::sampling_stride]
    y_data = y_data[::sampling_stride]

    return x_data, y_data

# todo: sample data with minimum in given area (i.e both x and y tolerance)
# todo: variable sampling runtime paramaterial
# todo: full elastic range with plastic sampling only
# todo: equidistant strain sampling
# todo: if strain decreasing, omit point
