"""Module for developing code for hyde-study notebook."""
from matplotlib import pyplot as plt

from paramaterial.plug import DataItem, DataSet
import paramaterial as pam


def main():
    ds = DataSet('data/02 trimmed data', 'info/02 trimmed info.xlsx')

    styler = pam.plotting.Styler(
        linestyle_by='rate', linestyle_by_label=r'Strain-Rate ($\dot{\varepsilon}$)',
        color_by='temperature', color_by_label=r'Temperature ($^{\circ}$C)',
        marker_by='test type', marker_by_label='Test Type', color_norm=plt.Normalize(300, 500)
    ).style_to(ds)

    ds_plot = lambda ds, **kwargs: pam.plotting.dataset_plot(ds, 'Strain', 'Stress(MPa)', styler=styler)

    def ds_subplot(ds: DataSet, **kwargs):
        axxs = pam.plotting.subplot_wrapper(
            ds, ds_plot, auto_legend_on=False, legend=False,
            shape=(2, 3), wspace=0.05, hspace=0.05, figsize=(10, 7),
            rows_by='test type', cols_by='rate',
            row_vals=[['PSC'], ['PSC*']], col_vals=[[10], [30], [100]],
            row_titles=['PSC', 'PSC*'], col_titles=['10 s$^{-1}$', '30 s$^{-1}$', '100 s$^{-1}$'],
            **kwargs
        )
        axxs.flat[0].get_figure().legend(handles=styler.legend_handles(), loc='right', frameon=True)
        return axxs

    axs = ds_subplot(ds)

    pam.plotting.dataset_subplots(ds, x='Strain', y='Force(kN)', axs=axs, shape=(2, 3),
                                  rows_by='test type', cols_by='rate', styler=pam.plotting.Styler(),
                                  row_vals=[['PSC'], ['PSC*']], col_vals=[[10], [30], [100]], )

    handles = styler.legend_handles()
    # add string handle to end
    handles.append(plt.Line2D([0], [0], color='k', linestyle='-', label='String'))

    # append a patch with a colorbar
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import numpy as np
    norm = Normalize(vmin=300, vmax=500)
    sm = ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    handles.append(Rectangle((0, 0), 1, 1, facecolor=sm.to_rgba(300), edgecolor='none'))
    handles.append(Rectangle((0, 0), 1, 1, facecolor=sm.to_rgba(500), edgecolor='none'))
    handles.append(Rectangle((0, 0), 1, 1, facecolor=sm.to_rgba(400), edgecolor='none'))


    plt.show()


if __name__ == '__main__':
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    dict2 = {'a': 4, 'b': 5}
    dict3 = {**dict2, **dict1}
    print(dict3)

# def print_data_item(di):
#     print(di)
#     return di
#
# list(map(print_data_item, ds))