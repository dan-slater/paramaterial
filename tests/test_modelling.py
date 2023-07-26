import numpy as np
import pandas as pd
import paramaterial as pam
import matplotlib.pyplot as plt

prepared_ds = pam.DataSet('modelling_test_data/03 temp repres info.xlsx', 'modelling_test_data/repres_data',
                          test_id_key='repres_id')

styler = pam.plotting.Styler(
    color_by='temperature', color_by_label='(°C)', cmap='plasma', color_norm=plt.Normalize(20, 310)
).style_to(prepared_ds)

stress_strain_labels = dict(x='Strain', y='Stress_MPa', ylabel='Stress (MPa)')


def ds_plot(ds, title="", **kwargs):
    return pam.dataset_plot(ds=ds, styler=styler, **stress_strain_labels, title=title, **kwargs)


def ds_subplots(ds, **kwargs):
    axs = pam.dataset_subplots(
        ds=ds, shape=(6, 5), styler=styler, figsize=(10, 12), hspace=0.05, wspace=0.05,
        rows_by='temperature', row_vals=[[20], [100], [150], [200], [250], [300]],
        cols_by='lot', col_vals=[['A'], ['B'], ['C'], ['D'], ['E']],
        col_titles=[f'Lot {lot}' for lot in 'ABCDE'], subplots_adjust=0.02,
        **stress_strain_labels, **kwargs
    )
    for ax in axs.flat:
        ax.set_ylabel('Stress (MPa)')
    return axs


ds_plot(prepared_ds, fill_between=('down_std_Stress_MPa', 'up_std_Stress_MPa'), title='Representative Curves')

temp_repres_ds = prepared_ds

ramberg_ms = pam.ModelSet(
    model_func=pam.models.ramberg,
    x_col='Strain',
    y_col='Stress_MPa',
    variable_names=['E', 'UPL_1'],
    param_names=['H', 'n'],
    bounds=[(0., 1000.), (0.01, 0.8)]
)

ramberg_ms.fit_items(temp_repres_ds)
ramberg_ds = ramberg_ms.predict_ds(x_range=(0, 0.1, 0.001))
ax = ds_plot(temp_repres_ds, alpha=0.5)
ds_plot(ramberg_ds, ls='--', ax=ax, title='Fitted Curves')
plt.show()