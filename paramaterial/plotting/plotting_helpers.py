from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paramaterial.plug import DataSet

TEXTWIDTH = 6.4  # inches
DATA_CMAP = 'plasma'
PARAMS_CMAP = 'viridis'


def results_grid_plot(
        stage: str,
        name: str,
        grid_kwargs: Dict,
        axis_kwargs: Dict,
        plot_kwargs: Dict,
        legend_kwargs: Dict,
        out_dir: str = 'graphics',
):
    _configure_plt_formatting()
    axs = _setup_grid(**grid_kwargs)
    # plot data curves
    if stage in ['screened', 'trimmed', 'prepared', 'processed', 'fitted', 'representative']:
        _setup_data_axs(axs, **axis_kwargs)
        _add_test_data_curves(stage, axs, **plot_kwargs, **legend_kwargs)
        _add_colorbar(**legend_kwargs)
    # or add scatter plots and regression lines of fitted params
    elif stage in ['ramberg', 'voce']:
        _setup_params_axs(axs, **axis_kwargs)
        _add_params_scatter_fits(axs, **plot_kwargs)
        _add_params_key(**legend_kwargs)
    # save figure
    plt.savefig(f'{out_dir}/{name}.pdf', dpi=90, bbox_inches='tight')


def _configure_plt_formatting():
    plt.style.use('seaborn-dark')
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', size=9)
    plt.rc('axes', titlesize=9, labelsize=9)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=7)
    plt.rc('figure', titlesize=11)


def _setup_data_axs(axs: np.ndarray,
                    x_sep: float,
                    x_max: float,
                    y_sep: float,
                    y_max: float):
    # format ax ticks and limits
    # for ax in axs.flat:
    #     ax.set_xticks(np.arange(0, x_max + 0.5*x_sep, x_sep))
    #     ax.set_xlim(xmin=0 - 0.5*x_sep, xmax=x_max)
    #     ax.set_yticks(np.arange(0, y_max + 0.5*y_sep, y_sep))
    #     ax.set_ylim(ymin=0 - 0.5*y_sep, ymax=y_max)
    #     ax.tick_params(axis="y", direction="in")
    #     ax.tick_params(axis="x", direction="in")
    #     ax.grid()
    axs.set_xticks(np.arange(0, x_max + 0.5*x_sep, x_sep))
    axs.set_xlim(xmin=0 - 0.5*x_sep, xmax=x_max)
    axs.set_yticks(np.arange(0, y_max + 0.5*y_sep, y_sep))
    axs.set_ylim(ymin=0 - 0.5*y_sep, ymax=y_max)
    axs.tick_params(axis="y", direction="in")
    axs.tick_params(axis="x", direction="in")
    axs.grid()
    # add axis labels
    # for ax in axs[:, 0]:
    #     ax.set_ylabel('Stress (MPa)')
    # for ax in axs[-1, :]:
    #     ax.set_xlabel('Strain (mm/mm)')
    axs.set_ylabel('Stress (MPa)')
    axs.set_xlabel('Strain (mm/mm)')



def _add_data_curves(
        axs: np.ndarray,
        dataset_paths: Dict,
        dataset_config: Dict,
        row_vals: List,
        col_vals: List,
        rows_key: str,
        cols_key: str,
        min_val=None,
        max_val=None
):
    # find max and min temperatures
    dataset = DataSet()
    dataset.load_data(**dataset_paths, subset_config=dataset_config)
    info = dataset.info_table
    if min_val is None:
        min_val = info['temperature'].min()
    if max_val is None:
        max_val = info['temperature'].max()
    # add line plots
    for i, row_name in enumerate(row_vals):
        for j, col_name in enumerate(col_vals):
            if type(col_name) is list:
                col_name = col_vals[j][i]
            # ax = axs[i, j]
            ax = axs
            sub_config = dataset_config.copy()
            sub_config.update({rows_key: [row_name], cols_key: [col_name]})
            dataset = DataSet()
            dataset.load_data(dataset_paths['data_dir'], dataset_paths['info_path'], sub_config)
            for dataitem in dataset:
                T = dataitem.info['temperature']
                color = plt.get_cmap(DATA_CMAP)((T - min_val)/(max_val - min_val))
                x, y = dataitem.data[x_data_key].values, dataitem.data[y_data_key].values
                ax.plot(x, y, color=color, lw=linewidth, zorder=int(T))
                # ax.annotate(f'{dataitem.test_id}', (x[-1], y[-1]), fontsize=0.5)
                if stage == 'representative':
                    lower, upper = dataitem.data[y_lower_key], dataitem.data[y_upper_key]
                    ax.fill_between(x, lower, upper, color=color, alpha=0.4, zorder=int(T))


def _add_colorbar(min_val: float, max_val: float, cbar_label: Optional[str] = None):
    """Add a colorbar to the right of the given axis."""
    plt.subplots_adjust(right=0.875)
    cax = plt.axes([0.9, 0.3, 0.014, 0.4])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(DATA_CMAP), norm=plt.Normalize(vmin=min_val, vmax=max_val))
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label)

