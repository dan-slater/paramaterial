"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

import paramaterial as pam
from paramaterial import DataSet, DataItem

dataset = DataSet('data/02 processed data', 'info/02 processed info.xlsx')


def determine_upper_proportional_limit(di: DataItem, preload: float = 0.0, max_strain: float = 0.02,
                                       strain_key: str = 'Strain', stress_key: str = 'Stress',
                                       preload_key: str = 'Stress') -> DataItem:
    # trim data before preload
    df = di.data[di.data[preload_key] >= preload]
    # trim data after max_strain
    df = df[df[strain_key] <= max_strain]
    # get strain and stress vectors
    x_vec = df[strain_key].values
    y_vec = df[stress_key].values
    # loop through all points and find the one that minimizes the relative deviation from the slope
    min_S_mrel = np.inf
    UPL = None
    for i in range(3, len(x_vec)):
        # fit a line to the first i points
        x = x_vec[:i]
        y = y_vec[:i]
        n = len(x)
        slope = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/(n*np.sum(np.square(x)) - np.square(np.sum(x)))
        intercept = (np.sum(y) - slope*np.sum(x))/n
        # empirical covariance
        S_xy = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/(n - 1)
        # x standard deviation
        S_x = np.sqrt((n*np.sum(np.square(x)) - np.square(np.sum(x)))/(n - 1))
        # y standard deviation
        S_y = np.sqrt((n*np.sum(np.square(y)) - np.square(np.sum(y)))/(n - 1))
        # correlation coefficient
        r = S_xy/(S_x*S_y)
        # slope standard deviation
        S_m = np.sqrt((1 - r**2)/(n - 2))*S_y/S_x
        # relative deviation of slope
        S_mrel = S_m/slope
        # check for minimum relative deviation of slope and store UPL
        if i == 2:
            min_S_mrel = S_mrel
            UPL = (x[-1], y[-1])
        elif S_mrel < min_S_mrel:
            min_S_mrel = S_mrel
            UPL = (x[-1], y[-1])
    # store UPL in dataitem info
    di.info['UPL'] = UPL
    return di


def determine_lower_proportional_limit(di: DataItem, strain_key: str = 'Strain', stress_key: str = 'Stress') -> DataItem:
    # trim data after UPL
    df = di.data[di.data[strain_key] <= di.info['UPL'][0]]
    # get strain and stress vectors
    x_vec = df[strain_key].values
    y_vec = df[stress_key].values
    # loop through points from UPL backwards and find the one that minimizes the relative deviation from the slope
    min_S_mrel = np.inf
    LPL = None
    for i in range(len(x_vec) - 1, 2, -1):
        # fit a line to the last i points
        x = x_vec[i:]
        y = y_vec[i:]
        n = len(x)
        a = (n*np.sum(x*y) - np.sum(x)*np.sum(y))
        b = (n*np.sum(np.square(x)) - np.square(np.sum(x)))
        print(b)
        slope = np.divide((n*np.sum(x*y) - np.sum(x)*np.sum(y)),(n*np.sum(np.square(x)) - np.square(np.sum(x))))
        intercept = (np.sum(y) - slope*np.sum(x))/n
        # empirical covariance
        S_xy = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/(n - 1)
        # x standard deviation
        S_x = np.sqrt((n*np.sum(np.square(x)) - np.square(np.sum(x)))/(n - 1))
        # y standard deviation
        S_y = np.sqrt((n*np.sum(np.square(y)) - np.square(np.sum(y)))/(n - 1))
        # correlation coefficient
        r = S_xy/(S_x*S_y)
        # slope standard deviation
        S_m = np.sqrt((1 - r**2)/(n - 2))*S_y/S_x
        # relative deviation of slope
        S_mrel = S_m/slope
        # check for minimum relative deviation of slope and store LPL
        if i == len(x_vec) - 1:
            min_S_mrel = S_mrel
            LPL = (x[0], y[0])
        elif S_mrel > min_S_mrel:
            min_S_mrel = S_mrel
            LPL = (x[0], y[0])
    # store LPL in dataitem info
    di.info['LPL'] = LPL
    return di


styler = pam.plotting.Styler(color_by='temperature', cmap='plasma', color_by_label='(°C)', plot_kwargs={'grid': True, 'alpha': 0.2})
styler.style_to(dataset)


def ds_plot(ds: DataSet, **kwargs):
    return pam.plotting.dataset_plot(ds, styler=styler, **kwargs)


ax = ds_plot(dataset, x='Strain', y='Stress_MPa')


def plot_UPL(di):
    temp = di.info['temperature']
    ax.plot(di.info['UPL'][0], di.info['UPL'][1], 'x', color=styler.color_dict[temp], alpha=0.9, markersize=10,
            mfc='none', zorder=temp+500)
    return di


def plot_LPL(di):
    temp = di.info['temperature']
    ax.plot(di.info['LPL'][0], di.info['LPL'][1], 'o', color=styler.color_dict[temp], alpha=0.9, markersize=10,
            mfc='none', zorder=temp+500)
    return di

dataset = dataset.apply(determine_upper_proportional_limit, preload=40., preload_key='Stress_MPa', stress_key='Stress_MPa')
dataset = dataset.apply(determine_lower_proportional_limit, stress_key='Stress_MPa')

dataset = dataset.apply(plot_UPL).apply(plot_LPL)

list(dataset)
plt.show()




