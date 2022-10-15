"""Module containing the plotting functions for the dataset class."""
from typing import Optional, Tuple, List

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def configure_plt_formatting():
    """Configure the matplotlib formatting."""
    import matplotlib as mpl
    plt.style.use('seaborn-dark')
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', size=9)
    plt.rc('axes', titlesize=9, labelsize=9)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=7)
    plt.rc('figure', titlesize=11)


configure_plt_formatting()


def dataset_plot(dataset, x: str, y: str, figsize: Tuple[float, float] = (6.4, 4.8),
                 cbar_by: Optional[str] = None, cbar_label: Optional[str] = None, cmap: str = 'plasma',
                 color_by: Optional[str] = None, style_by: Optional[str] = None, marker_by: Optional[str] = None,
                 width_by: Optional[str] = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the curves from every item in the dataset using pandas.DataFrame.plot().
    Args:
        dataset: The dataset to plot.
        x: The column to use for the x-axis.
        y: The column to use for the y-axis.
        figsize: The figure size.
        color_by: The info column to use for coloring.
        cbar_by: The info column to use for the colorbar.
        cbar_label: The label for the colorbar.
        cmap: The colormap to use for the colorbar.
        style_by: The info column to use for line style.
        marker_by: The info column to use for marker style.
        width_by: The info column to use for line width.
        **kwargs: Keyword arguments to pass to the pandas.DataFrame.plot() function.
    Returns:
        The figure and axes.
    """
    if plt.gcf() is None or plt.gca() is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
        ax = plt.gca()

    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'v', '^', 'd', 'p', 'h', '8', '>', '<', 'x', 'D', 'P', 'H', 'X']

    unique_colors = sorted(dataset.info_table[color_by].unique().tolist()) if color_by is not None else [None]
    unique_styles = sorted(dataset.info_table[style_by].unique().tolist()) if style_by is not None else [None]
    unique_markers = sorted(dataset.info_table[marker_by].unique().tolist()) if marker_by is not None else [None]
    unique_widths = sorted(dataset.info_table[width_by].unique().tolist()) if width_by is not None else [None]

    # make the colorbar
    if cbar_by is not None:
        vmin = dataset.info_table[cbar_by].min()
        vmax = dataset.info_table[cbar_by].max()
        cbar_norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cbar_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    for dataitem in dataset:
        # get the curve formatters
        linestyle = '-'
        marker = None
        width = 1
        zorder = 1
        if style_by is not None:
            style_val = dataitem.info[style_by]
            linestyle = linestyles[unique_styles.index(style_val)]
        if marker_by is not None:
            marker_val = dataitem.info[marker_by]
            marker = markers[unique_markers.index(marker_val)]
        if width_by is not None:
            width_val = dataitem.info[width_by]
            width = unique_widths.index(width_val)*0.1 + 1
        if color_by is None:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
        else:
            color_val = dataitem.info[color_by]
            color = plt.get_cmap(cmap)(unique_colors.index(color_val)/len(unique_colors))
        if cbar_by is not None:
            color = plt.get_cmap(cmap)(cbar_norm(dataitem.info[cbar_by]))
            zorder = dataitem.info[cbar_by]

        # plot the curve, catch error from pd.plot() and raise a more informative error
        try:
            dataitem.data.plot(ax=ax, x=x, y=y, linestyle=linestyle, marker=marker, linewidth=width, color=color,
                               zorder=zorder, **kwargs)
        except AttributeError as e:
            raise AttributeError(f"Error when calling pandas.DataFrame.plot(): {e}")

    # add the legend
    handles = []
    if color_by is not None:
        for color_val in unique_colors:
            handles.append(
                mpatches.Patch(color=plt.get_cmap(cmap)(unique_colors.index(color_val)/len(unique_colors)),
                               label=color_val))
    if style_by is not None:
        for style_val in unique_styles:
            handles.append(Line2D([0], [0], color='black', linestyle=linestyles[unique_styles.index(style_val)],
                                  label=style_val))
    if marker_by is not None:
        for marker_val in unique_markers:
            handles.append(Line2D([0], [0], color='black', marker=markers[unique_markers.index(marker_val)],
                                  linestyle='None', label=marker_val))
    if width_by is not None:
        for width_val in unique_widths:
            handles.append(Line2D([0], [0], color='black', linewidth=unique_widths.index(width_val)*0.1 + 1,
                                  label=width_val))
    if len(handles) > 0:
        if cbar_by is not None:
            ax.legend(handles=handles, bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=6)
        else:
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fig.tight_layout()

    return fig, ax


def dataset_subplots(
        dataset,
        x: str,
        y: str,
        nrows: int,
        ncols: int,
        cols_by: str,
        rows_by: str,
        col_keys: List[List[str]],
        row_keys: List[List[str]],
        col_titles: Optional[List[str]] = None,
        row_titles: Optional[List[str]] = None,
        plot_titles: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (6.4, 4.8),
        cbar_by: Optional[str] = None,
        cbar_label: Optional[str] = None,
        cmap: str = 'plasma',
        color_by: Optional[str] = None,
        style_by: Optional[str] = None,
        marker_by: Optional[str] = None,
        width_by: Optional[str] = None,
        sharex: str = 'col',
        sharey: str = 'row',
        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey)

    # plot the data
    for row, row_key in enumerate(row_keys):
        for col, col_key in enumerate(col_keys):
            ax = axes[row, col]
            if plot_titles is not None:
                ax.set_title(plot_titles[row*ncols + col])
            subset = dataset[{rows_by: row_key, cols_by: col_key}]
            if subset is not None:
                fig, ax = dataset_plot(subset, x, y, ax=ax,
                                        cbar=cbar, cbar_label=cbar_label,
                                       color_by=color_by,
                                       style_by=style_by, marker_by=marker_by, width_by=width_by,
                                       **kwargs)

    # add the legend
    _add_legend(dataset, color_by, cbar, cbar_label, style_by, marker_by, width_by, style_legend, color_legend,
                marker_legend, width_legend)

    return fig, axes


def _setup_grid(nrows: int, ncols: int, figsize: Tuple[float, float] = (6.4, 4.8), row_titles=None, col_titles=None,
                subplot_titles=None, wspace: float = 0.3, hspace: float = 0.3, sharex: str = 'col',
                sharey: str = 'row', ) -> np.ndarray:
    # setup grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    # add row titles
    if row_titles is not None:
        for ax, row_title in zip(axs[:, 0], row_titles):
            ax.annotate(row_title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)
    # add grid column titles
    if col_titles is not None:
        for ax, column_title in zip(axs[0, :], col_titles):
            ax.set_title(column_title)
    # add subplot titles
    if subplot_titles is not None:
        for ax, subplot_title in zip(axs.flat, subplot_titles):
            ax.set_title(subplot_title)
    return axs
