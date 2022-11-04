"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""

import matplotlib.pyplot as plt
import numpy as np

import paramaterial as pam
from paramaterial import DataSet, DataItem

dataset = DataSet('data/02 processed data', 'info/02 processed info.xlsx')


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


dataset = dataset.apply(trim_at_preload)

ax = ds_plot(dataset, x='Strain', y='Stress_MPa')
ax.axhline(preload, color='k', linestyle='--')
plt.show()


# data is already trimmed to preload
# use curve_fit
# fit linear curve between first point and next point
# fit linear curve between first point and next two points
# fit linear curve between first point and next three points etc.
# find the point that minimizes the residuals of the fitted linear curve
def determine_upl(di):
    """Quote: Starting from a preload (which was chosen to be [133.5 kN] to minimize errors in the strain measurements
    caused by factors such as initial grip alignment), the upper proportional limit (UPL) is determined by linear
    regression as the point that minimizes the residuals of the slope fit between that point and the preload."""
    x = di.data['Strain'].values
    y = di.data['Stress_MPa'].values
    # fit line between first point and every other point
    slopes = []
    for i in range(1, len(x)):
        # fit line between first point and next i points
        x_fit = x[:i]
        y_fit = y[:i]
        p,  = np.polyfit(x_fit, y_fit, 1, full=True)
        slopes.append(p[0])
    # find the point that minimizes the residuals of the fitted linear curve
    slopes = np.array(slopes)
    di.info['upl_idx'] = np.argmin(np.abs(slopes - preload))
    di.info['upl_strain'] = x[di.info['upl_idx']]
    di.info['upl_stress'] = y[di.info['upl_idx']]
    return di


