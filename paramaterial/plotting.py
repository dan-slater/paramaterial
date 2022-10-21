"""Module containing the plotting functions for the dataset class."""
from typing import Optional, Tuple, List, Any, Union, Dict

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from paramaterial.plug import DataItem


def configure_plt_formatting():
    """Configure the matplotlib formatting."""
    import matplotlib as mpl
    plt.style.use('seaborn-dark')
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', size=11)
    plt.rc('axes', titlesize=11, labelsize=11)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('legend', fontsize=9)
    plt.rc('figure', titlesize=13)


configure_plt_formatting()


def dataset_plot(
        dataset,
        x: str,
        y: str,
        ax: Optional[plt.Axes] = None,
        color_by: Optional[str] = None,
        style_by: Optional[str] = None,
        marker_by: Optional[str] = None,
        width_by: Optional[str] = None,
        color_by_label: Optional[str] = None,
        style_by_label: Optional[str] = None,
        marker_by_label: Optional[str] = None,
        width_by_label: Optional[str] = None,
        cbar: bool = False,
        color_norm: Optional[plt.Normalize] = None,
        cbar_label: Optional[str] = None,
        cmap: str = 'plasma',
        width_by_scale: float = 1.0,
        fill_between: Optional[Tuple[str, str]] = None,
        auto_legend_on: bool = False,
        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Make a single plot from the dataframe of every item in the dataset."""
    if ax is None:
        fig, ax = plt.subplots(kwargs.get('figsize', (10, 6)))
    kwargs['ax'] = ax

    # clear legend if ax has one
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    handles = []  # list of legend entries

    # color function
    unique_color_vals = dataset.info_table[color_by].unique() if color_by else None

    colormap = plt.get_cmap(cmap)
    color_norm = plt.Normalize(vmin=unique_color_vals.min(), vmax=unique_color_vals.max())

    def color(val: float) -> str:
        if color_by is not None:
            return colormap(color_norm(val))
        else:
            return ax._get_lines.get_next_color()

    # color legend
    if color_by_label is not None:
        handles.append(mpatches.Patch(label=color_by_label, alpha=0))
    if color_by is not None:
        for color_val in unique_color_vals:
            handles.append(Line2D([], [], color=colormap(color_norm(color_val)), label=color_val,
                                  marker='o', linestyle=''))

    # colorbar
    if cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm).set_array([])
        # sm.set_array([])
        cbar = plt.colorbar(sm, ax=kwargs['ax'], fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label) if cbar_label is not None else None
        cbar.ax.yaxis.set_ticks_position('right').set_label_position('right')
        # cbar.ax.yaxis.set_label_position('right')

    # linestyle function
    unique_style_vals = dataset.info_table[style_by].unique() if style_by else None

    linestyles = ['-', '--', ':', '-.']
    while len(linestyles) < len(unique_style_vals):
        linestyles.extend(linestyles)

    def style(val: float) -> str:
        if style_by is not None:
            return linestyles[unique_style_vals.tolist().index(val)]
        else:
            return '-'

    # linestyle legend
    if style_by_label is not None:
        handles.append(mpatches.Patch(label=style_by_label, alpha=0))
    if style_by is not None:
        for style_val in unique_style_vals:
            handles.append(Line2D([], [], color='black', linestyle=linestyles[unique_style_vals.tolist().index(style_val)],
                                  label=style_val))

    # marker function
    unique_marker_vals = dataset.info_table[marker_by].unique() if marker_by is not None else None
    markers = ['o', 's', 'v', '^', 'd', 'p', 'h', '8', '>', '<', 'x', 'D', 'P', 'H', 'X']
    if marker_by is not None:
        while len(markers) < len(unique_marker_vals):
            markers.extend(markers)

    def marker(val: float) -> Union[str, None]:
        if marker_by is not None:
            return markers[unique_marker_vals.tolist().index(val)]
        else:
            return None

    # marker legend
    if marker_by_label is not None:
        handles.append(mpatches.Patch(label=marker_by_label, alpha=0))
    if marker_by is not None:
        for marker_val in unique_marker_vals:
            handles.append(Line2D([], [], color='black', marker=markers[unique_marker_vals.tolist().index(marker_val)],
                                  label=marker_val, linestyle='None'))

    # width function
    unique_vals = dataset.info_table[width_by].unique()
    width_norm = plt.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())

    def width(val: float) -> float:
        if width_by is not None:
            return width_by_scale*width_norm(val) + 1
        else:
            return 1.0

    # width legend
    if width_by_label is not None:
        handles.append(mpatches.Patch(label=width_by_label, alpha=0))
    if width_by is not None:
        for width_val in unique_vals:
            handles.append(
                Line2D([], [], color='black', linewidth=width_by_scale*width_norm(width_val) + 1,
                       label=width_val, linestyle='-'))

    def curve_formatters(dataitem: DataItem) -> Dict[str, Any]:
        formatters = {
            'color': color(dataitem.info[color_by]) if color_by is not None else None,
            'linestyle': style(dataitem.info[style_by]) if style_by is not None else None,
            'marker': marker(dataitem.info[marker_by]) if marker_by is not None else None,
            'linewidth': width(dataitem.info[width_by]) if width_by is not None else None,
            'zorder': dataitem.info[color_by] if color_by is not None else None
        }
        return {k: v for k, v in formatters.items() if v is not None}

    # plot the curve for each dataitem
    for di in dataset:
        # plot the curve
        ax = di.data.plot(x=x, y=y, legend=auto_legend_on, **curve_formatters(di), **kwargs)
        # fill between curves
        if fill_between is not None:
            ax.fill_between(di.data[x], di.data[fill_between[0]], di.data[fill_between[1]], alpha=0.2,
                            **curve_formatters(di))

    # add the legend
    if len(handles) > 0:
        ax.legend(handles=handles, loc='best')

    return ax


def dataset_subplots(
        dataset,
        x: str,
        y: str,
        shape: Tuple[int, int],
        rows_by: str,
        cols_by: str,
        row_vals: List[List[Any]],
        col_vals: List[List[Any]],
        figsize: Tuple[float, float] = (12, 8),
        sharex: str = 'col',
        sharey: str = 'row',
        wspace: float = 0.1,
        hspace: float = 0.1,
        row_titles: Optional[List[str]] = None,
        col_titles: Optional[List[str]] = None,
        plot_titles: Optional[List[str]] = None,
        color_by: Optional[str] = None,
        cbar: bool = False,
        cbar_label: Optional[str] = None,
        color_norm: Optional[plt.Normalize] = None,
        cmap: str = 'plasma',
        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a dataset as a grid of subplots, split by the 'rows_by' and 'cols_by' columns in the info_table."""

    # setup grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(shape[0], shape[1], wspace=wspace, hspace=hspace)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    if axs.ndim == 1:
        axs = np.array([axs])

    # color normaliser
    if color_norm is None and color_by is not None:
        color_norm = plt.Normalize(vmin=dataset.info_table[color_by].min(), vmax=dataset.info_table[color_by].max())
    elif color_norm is not None and color_by is None:
        raise ValueError('color_norm is set but color_by is not')

    # loop through the grid of axes and plot the subsets
    for row, row_val in enumerate(row_vals):
        for col, col_val in enumerate(col_vals):
            ax = axs[row, col]
            subset = dataset[{cols_by: col_val, rows_by: row_val}]
            dataset_plot(subset, x=x, y=y, ax=ax, color_by=color_by, color_norm=color_norm, **kwargs)

    # add colorbar
    if cbar:
        plt.subplots_adjust(right=0.875)
        cax = plt.axes([0.9, 0.2, 0.014, 0.6])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cmap), cax=cax)

        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')

        if cbar_label is not None:
            cbar.set_label(cbar_label)

    # add row titles
    if row_titles is not None:
        for ax, row_title in zip(axs[:, 0], row_titles):
            ax.annotate(row_title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)

    # add column titles
    if col_titles is not None:
        for ax, column_title in zip(axs[0, :], col_titles):
            ax.set_title(column_title)

    # add subplot titles
    if plot_titles is not None:
        for ax, subplot_title in zip(axs.flat, plot_titles):
            ax.set_title(subplot_title)

    return fig, axs
