from typing import Callable

import paramaterial as pam
import numpy as np
from paramaterial import DataItem, DataSet, ModelSet
import matplotlib.pyplot as plt

from paramaterial.modelling import ramberg, voce

processed_ds = DataSet('data/02 processed data', 'info/02 processed info.xlsx')

styler = pam.plotting.Styler(
    color_by='temperature', cmap='plasma', color_by_label='(°C)', color_norm=plt.Normalize(20, 320),
    plot_kwargs=dict(grid=True))
styler.style_to(processed_ds.sort_by(['temperature', 'lot']))


def ds_plot(ds: DataSet, **kwargs):
    return pam.plotting.dataset_plot(ds, styler=styler, **kwargs)


subplot_cfg = dict(
    shape=(3, 3), sharex='all', sharey='all', hspace=0.2,
    rows_by='lot', row_vals=[[a] for a in 'ABCDEFGHI'],
    cols_by='lot', col_vals=[[a] for a in 'ABCDEFGHI'],
    plot_titles=[f'Lot {a}' for a in 'ABCDEFGHI']
)


def ds_subplots(ds: DataSet, **kwargs):
    return pam.plotting.dataset_subplots(ds=ds, styler=styler, plot_legend=False, **subplot_cfg, **kwargs)


def subplot_wrapper(ds: DataSet, plot_func: Callable[[DataItem], DataItem], **plot_func_kwargs):
    return pam.plotting.subplot_wrapper(ds=ds, plot_func=plot_func, **subplot_cfg, **plot_func_kwargs)


stress_strain_labels = dict(x='Strain', y='Stress_MPa', ylabel='Stress (MPa)')


def trim_for_fitting(di):
    di.data = di.data[di.data['Strain'] <= 0.01]
    return di


fit_ds = DataSet('data/02 processed data', 'info/02 processed info.xlsx').apply(trim_for_fitting)

# ramberg_ms = ModelSet(ramberg, ['E', 's_y', 'C', 'n'],
#                       bounds=[(35e3, 90e3), (1., 280.), (30, 220.), (0.01, 0.8)],
#                       initial_guess=[50e3, 100., 100., 0.2],
#                       scipy_func='minimize')
# ramberg_ms.fit_to(fit_ds, 'Strain', 'Stress_MPa', sample_size=40)
# ramberg_ms.params_table.to_excel('info/03 ramberg params.xlsx')
# ramberg_ds = ramberg_ms.predict()
#
# ramberg_axs = ds_subplots(fit_ds, **stress_strain_labels, alpha=0.4)
# ds_subplots(ramberg_ds, x='x', y='y', ls='--', alpha=0.9, axs=ramberg_axs)
# plt.savefig('03 ramberg.png', dpi=300)
# plt.close()

# voce_ms = ModelSet(voce, ['E', 's_y', 's_u', 'd'],
#                    bounds=[(35e3, 70e3), (1., 220.), (120., 320.), (1., 150.)],
#                    initial_guess=np.array([50e3, 80., 200., 50.]),
#                    scipy_func='minimize')
# voce_ms.fit_to(fit_ds, 'Strain', 'Stress_MPa', sample_size=40)
# voce_ms.params_table.to_excel('info/03 voce params.xlsx')
# voce_ds = voce_ms.predict()
#
# voce_axs = ds_subplots(fit_ds, **stress_strain_labels, alpha=0.4)
# ds_subplots(voce_ds, x='x', y='y', ls='--', alpha=0.9, axs=voce_axs)
# plt.savefig('03 voce.png', dpi=300)
