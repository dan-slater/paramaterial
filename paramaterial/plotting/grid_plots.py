from typing import Dict, List

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


def _setup_grid(height: float,
                width: float,
                rows: int,
                cols: int,
                row_titles=None,
                col_titles=None,
                subplot_titles=None) -> np.ndarray:
    # setup grid
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.1)
    # axs = gs.subplots(sharex=True, sharey=True)
    axs = gs.subplots(sharex='col', sharey='row')
    # add row titles
    if row_titles is not None:
        for ax, row_title in zip(axs[:, 0], row_titles):
            ax.annotate(row_title, xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)
    # add grid column titles
    if col_titles is not None:
        # for ax, column_title in zip(axs[0, :], col_titles):
        #     ax.set_title(column_title)
        axs.set_title(col_titles[0])
    # add subplot titles
    if subplot_titles is not None:
        for ax, subplot_title in zip(axs.flat, subplot_titles):
            ax.set_title(subplot_title)
    return axs


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


def _setup_params_axs(axs: np.ndarray,
                      x_sep: float,
                      x_max: float,
                      y_seps: List[float],
                      y_maxs: List[float],
                      y_labels: List[str]):
    # format ax ticks and limits
    for i in range(len(y_seps)):
        for ax in axs[i]:
            ax.set_xticks(np.arange(0, x_max + 0.5*x_sep, x_sep))
            ax.set_xlim(xmin=0 - 0.5*x_sep, xmax=x_max)
            ax.set_yticks(np.arange(0, y_maxs[i] + 0.5*y_seps[i], y_seps[i]))
            ax.set_ylim(ymin=0 - 0.5*y_seps[i], ymax=y_maxs[i])
            ax.tick_params(axis="y", direction="in")
            ax.tick_params(axis="x", direction="in")
            ax.grid()
    # add axis labels
    for ax, y_lab in zip(axs[:, 0], y_labels):
        ax.set_ylabel(y_lab)
    for ax in axs[-1, :]:
        ax.set_xlabel(r'Temperature ($^{\circ}$C)')


def _add_test_data_curves(
        stage: str,
        axs: np.ndarray,
        dataset_paths: Dict,
        dataset_config: Dict,
        row_vals: List,
        col_vals: List,
        rows_key: str,
        cols_key: str,
        x_data_key: str = 'Strain',
        y_data_key: str = 'Stress(MPa)',
        linewidth: float = np.sqrt(2),
        y_lower_key=None,
        y_upper_key=None,
        min_T=None,
        max_T=None
):
    # find max and min temperatures
    dataset = DataSet()
    dataset.load(**dataset_paths, subset_config=dataset_config)
    info = dataset.info_table
    if min_T is None:
        min_T = info['temperature'].min()
    if max_T is None:
        max_T = info['temperature'].max()
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
            dataset.load(dataset_paths['data_dir'], dataset_paths['info_path'], sub_config)
            for dataitem in dataset:
                T = dataitem.info['temperature']
                color = plt.get_cmap(DATA_CMAP)((T - min_T)/(max_T - min_T))
                x, y = dataitem.data[x_data_key].values, dataitem.data[y_data_key].values
                ax.plot(x, y, color=color, lw=linewidth, zorder=int(T))
                # ax.annotate(f'{dataitem.test_id}', (x[-1], y[-1]), fontsize=0.5)
                if stage == 'representative':
                    lower, upper = dataitem.data[y_lower_key], dataitem.data[y_upper_key]
                    ax.fill_between(x, lower, upper, color=color, alpha=0.4, zorder=int(T))


def _add_params_scatter_fits(
        axs: np.ndarray,
        dataset_paths: Dict,
        dataset_config: Dict,
        row_data_keys: List,
        cols_key: str,
        col_vals: List,
        lines_key: str,
        line_vals: List,
        line_styles: Dict[str, Dict],
        x_data_key: str = 'T',
        linewidth: float = np.sqrt(2),
):
    # add scatter plots
    for row_i, row_data_key in enumerate(row_data_keys):
        for col_j, col_val in enumerate(col_vals):
            ax = axs[row_i, col_j]
            for line_val in line_vals:
                # load data and info for line
                sub_config = dataset_config.copy()
                if type(col_val) is list:
                    sub_config.update({lines_key: [line_val], cols_key: col_val})
                else:
                    sub_config.update({lines_key: [line_val], cols_key: [col_val]})
                dataset = DataSet()
                dataset.load(**dataset_paths, subset_config=sub_config)
                # plot params and regression line
                for dataitem in dataset:  # (should only be one dataitem)
                    data = dataitem.data
                    ax.plot(data[x_data_key], data[row_data_key], lw=0, alpha=0.5, **line_styles[line_val])
                    ax.plot(data['temp vec'], data[str(row_data_key) + ' reg line'], **line_styles[line_val], ms=0)


def _add_colorbar(min_T: float, max_T: float):
    # add colorbar
    plt.subplots_adjust(right=0.875)
    cax = plt.axes([0.9, 0.3, 0.014, 0.4])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(DATA_CMAP), norm=plt.Normalize(vmin=min_T, vmax=max_T))
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r'Temperature ($^{\circ}$C)')


def _add_params_key():
    # add marker key
    plt.subplots_adjust(right=0.875)
