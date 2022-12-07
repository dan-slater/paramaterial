from typing import Callable

import paramaterial as pam
import numpy as np
from paramaterial import DataItem, DataSet, ModelSet
import matplotlib.pyplot as plt

prepared_ds = DataSet('data/01 prepared data', 'info/01 prepared info.xlsx').sort_by(['temperature', 'lot'])
styler = pam.plotting.Styler(color_by='temperature', cmap='plasma', color_by_label='(°C)',
                             plot_kwargs=dict(grid=True)).style_to(prepared_ds)
subplot_cfg = dict(shape=(3, 3), figsize=(8, 6), hspace=0.2,
                   sharex='all', sharey='all',
                   rows_by='lot', row_vals=[[a] for a in 'ABCDEFGHI'],
                   cols_by='lot', col_vals=[[a] for a in 'ABCDEFGHI'],
                   # if rows_by = cols_by, row_vals/col_vals are assigned to axs.flat
                   plot_titles=[f'Lot {a}' for a in 'ABCDEFGHI'])
stress_strain_labels = dict(x='Strain', y='Stress_MPa', ylabel='Stress (MPa)')


def ds_plot(ds: DataSet, **kwargs):
    return pam.plotting.dataset_plot(ds, styler=styler, **kwargs)


def ds_subplots(ds: DataSet, **kwargs):
    return pam.plotting.dataset_subplots(ds=ds, styler=styler, plot_legend=False, **subplot_cfg, **kwargs)


def subplot_wrapper(ds: DataSet, plot_func: Callable[[DataItem], DataItem], **plot_func_kwargs):
    return pam.plotting.subplot_wrapper(ds=ds, plot_func=plot_func, **subplot_cfg, **plot_func_kwargs)


def trim_small(di):
    di.data = di.data[di.data['Strain'] < 0.01]
    return di


trim_small_ds = DataSet('data/02 trimmed large data', 'info/02 trimmed large info.xlsx').apply(trim_small)
# ds_plot(trim_small_ds, **stress_strain_labels)
# plt.show()

properties_ds = trim_small_ds.apply(pam.find_upl_and_lpl,
                                    preload=36, preload_key='Stress_MPa',  # Stress at which to start searching for UPL
                                    suppress_numpy_warnings=True)
properties_ds.write_output('data/02 properties data', 'info/02 properties info.xlsx')
properties_ds = DataSet('data/02 properties data', 'info/02 properties info.xlsx')


def plot_upl_and_lpl(di, ax):  # to use the subplot wrapper, DataItem and plt.axes arguments are required
    temp = di.info['temperature']
    UPL = (di.info['UPL_0'], di.info['UPL_1'])
    LPL = (di.info['LPL_0'], di.info['LPL_1'])
    color = styler.color_dict[temp]
    ax.axline(UPL, slope=di.info['E'], c=color, ls='--', alpha=0.4, zorder=500 + temp)
    ax.axline(UPL, slope=di.info['E'], lw=0.5, ls=':', c='k', alpha=0.1, zorder=500 + temp)
    ax.plot(*UPL, c='k', mfc=color, marker=4, markersize=6, zorder=1000 + temp)
    ax.plot(*LPL, c='k', mfc=color, marker=5, markersize=6, zorder=1000 + temp)
    return di


def pl_plot(ds):
    _ax = ds_plot(ds, **stress_strain_labels)
    list(ds.apply(plot_upl_and_lpl, ax=_ax))


def pl_plots(ds):
    _axs = ds_subplots(ds, **stress_strain_labels)
    subplot_wrapper(ds, plot_upl_and_lpl, axs=_axs)  # use the subplot wrapper


pl_plots(properties_ds)
plt.show()
