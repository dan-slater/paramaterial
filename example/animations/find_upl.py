"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""

import matplotlib.pyplot as plt
import numpy as np

import paramaterial as pam
from paramaterial import DataSet, DataItem

dataset = DataSet('../data/02 processed data', 'info/02 processed info.xlsx')


def trim_at_max_force(di: DataItem):
    """Quote: The maximum force is determined by the maximum force recorded during the test. The data is then
    trimmed to this point."""
    di.info['Max_Force_idx'] = di.data['Force(kN)'].idxmax()
    di.data = di.data[:di.info['Max_Force_idx']]
    return di


dataset = dataset.apply(trim_at_max_force)

styler = pam.plotting.Styler(color_by='temperature', cmap='plasma', color_by_label='(°C)', plot_kwargs={'grid': True})
styler.style_to(dataset)


def ds_plot(ds: DataSet, **kwargs):
    return pam.plotting.dataset_plot(ds, styler=styler, **kwargs)


preload = 30  # MPa


def trim_at_preload(di: DataItem):
    """Quote: The data is then trimmed to the preload point."""
    di.info['preload_stress'] = preload
    di.info['preload_strain'] = np.interp(preload, di.data['Stress_MPa'], di.data['Strain'])
    di.data = di.data[di.data['Stress_MPa'] >= preload]
    return di


def trim_at_strain(di: DataItem):
    """Quote: The data is then trimmed to the preload point."""
    di.data = di.data[di.data['Strain'] <= 0.01]
    return di


dataset = dataset.apply(trim_at_preload).apply(trim_at_strain)

di = dataset[0]

# make animation of linear fits


x_data = di.data['Strain'].values
y_data = di.data['Stress_MPa'].values
plt.plot(x_data, y_data, 'o', color='b', mfc='none', alpha=0.5, label='data')

plt.ylim(0, 400)

for i in range(2, len(x_data)):
    x = x_data[:i]
    y = y_data[:i]
    n = len(x)
    slope = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/(n*np.sum(np.square(x)) - np.square(np.sum(x)))
    intercept = (np.sum(y) - slope*np.sum(x))/n
    S_xy = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/(n - 1)
    S_x = np.sqrt((n*np.sum(np.square(x)) - np.square(np.sum(x)))/(n - 1))
    S_y = np.sqrt((n*np.sum(np.square(y)) - np.square(np.sum(y)))/(n - 1))
    r = S_xy/(S_x*S_y)
    S_m = np.sqrt((1 - r ** 2)/(n - 2))*S_y/S_x
    S_b = S_m*np.sqrt(((n - 1)/n)*S_x ** 2 + np.mean(x) ** 2)
    S_mrel = S_m/slope
    if i == 2:
        min_S_mrel = S_mrel
        min_S_mrel_slope = slope
        min_S_mrel_intercept = intercept
        UPL = (x[-1], y[-1])
        l_min, = plt.plot(x[[0, -1]], y[[0, -1]], color='k', label='linear fit min variance', lw=1, ls='--', zorder=10)
    elif S_mrel < min_S_mrel:
        l_min.remove()
        min_S_mrel = S_mrel
        min_S_mrel_slope = slope
        min_S_mrel_intercept = intercept
        UPL = (x[-1], y[-1])
        l_min, = plt.plot(x[[0, -1]], y[[0, -1]], color='k', label='linear fit min variance', lw=1, ls='--', zorder=10)

    l_data, = plt.plot(x, y, color='r', alpha=0.9, label='data', lw=0, marker='o', mfc='none')
    l_fit, = plt.plot(x, slope*x + intercept, color='k', alpha=0.5, label='linear fit', lw=1)
    l_ups, = plt.plot(UPL[0], UPL[1], 's', color='k', label='UPL', lw=0, mfc='none', ms=10, zorder=10)
    plt.legend()
    plt.pause(0.1)
    l_data.remove()
    l_fit.remove()
    l_ups.remove()

plt.show()
