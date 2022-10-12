"""Module containing the plotting functions for the dataset class."""
from typing import Optional, Tuple, List

import numpy as np
from matplotlib import pyplot as plt


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


def dataset_plot(
        dataset,  # todo: type hint but avoid circular import
        x: str,
        y: str,
        colorby: Optional[str] = None,
        styleby: Optional[str] = None,
        markerby: Optional[str] = None,
        widthby: Optional[str] = None,
        cbar: bool = False,
        cbar_label: Optional[str] = None,
        **kwargs
) -> plt.Axes:
    """Plot the curves from every item in the dataset using pandas.DataFrame.plot().
    Args:
        dataset: The dataset to plot.
        x: The column to use for the x-axis.
        y: The column to use for the y-axis.
        colorby: The info column to use for coloring.
        styleby: The info column to use for line style.
        markerby: The info column to use for marker style.
        widthby: The info column to use for line width.
        cbar: Whether to add a colorbar.
        cbar_label: The label for the colorbar.
        **kwargs: Keyword arguments to pass to the pandas.DataFrame.plot() function.
    """
    ax = None
    for dataitem in dataset:
        ls, marker, width, color = _get_curve_formatters(styleby, markerby, widthby, colorby, dataset, dataitem)
        ax = dataitem.data.plot(linestyle=ls, marker=marker, linewidth=width, color=color, **kwargs)
    if colorby is not None and cbar:
        _add_colorbar(dataset, colorby, cbar_label)
    return ax


# get a subset of the data for each ax in the subplot based on the col_keys and row_keys
def dataset_subplots(
        dataset,
        x: str,
        y: str,
        nrows: int,
        ncols: int,
        cols_by: str,
        rows_by: str,
        col_keys: List[str],
        row_keys: List[str],
        colorby: Optional[str] = None,
        styleby: Optional[str] = None,
        markerby: Optional[str] = None,
        widthby: Optional[str] = None,
        figsize: Tuple[float, float] = (6.4, 4.8),
        row_titles: Optional[List[str]] = None,
        col_titles: Optional[List[str]] = None,
        plot_titles: Optional[List[str]] = None,
        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the curves from every item in the dataset using pandas.DataFrame.plot().

    Args:
        dataset: The dataset to plot.
        nrows: The number of rows in the subplot.
        ncols: The number of columns in the subplot.
        colsby: The info column to use for the columns.
        rowsby: The info column to use for the rows.
        col_keys: The keys to use for the columns.
        row_keys: The keys to use for the rows.
        figsize: The figure size.
        row_titles: The titles for the rows.
        col_titles: The titles for the columns.
        plot_titles: The titles for the plots.
        **kwargs: Keyword arguments to pass to the pandas.DataFrame.plot() function.

    Returns:
        The figure and axes.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for i, row_key in enumerate(row_keys):
        for j, col_key in enumerate(col_keys):
            ax = axes[i, j]
            for dataitem in dataset:
                if dataitem.info[rows_by] == row_key and dataitem.info[cols_by] == col_key:
                    linestyle, marker, width, color = _get_curve_formatters(styleby, markerby, widthby, colorby,
                                                                            dataset, dataitem)
                    dataitem.data.plot(x=x, y=y, linestyle=linestyle, marker=marker, linewidth=width, color=color,
                                       ax=ax, **kwargs)
            if row_titles is not None:
                ax.set_title(row_titles[i])
            if col_titles is not None:
                ax.set_title(col_titles[j])
            if plot_titles is not None:
                ax.set_title(plot_titles[i*ncols + j])
    return fig, axes


def _get_curve_formatters(styleby: str, markerby: str, widthby: str, colorby: str, dataset, dataitem):
    """Helper function to get the curve formatters for a given dataitem."""
    linestyle = '-'
    marker = None
    width = 1
    if styleby is not None:
        unique_styles = dataset.info_table[styleby].unique()
        style_val = dataitem.info[styleby]
        linestyles = ['-', '--', '-.', ':']
        linestyle = linestyles[unique_styles.tolist().index(style_val)]
    if markerby is not None:
        unique_markers = dataset.info_table[markerby].unique()
        marker_val = dataitem.info[markerby]
        markers = ['o', 's', 'v', '^', 'd', 'p', 'h', '8', '>', '<', 'x', 'D', 'P', 'H', 'X']
        marker = markers[unique_markers.tolist().index(marker_val)]
    if widthby is not None:
        unique_widths = dataset.info_table[widthby].unique()
        width_val = dataitem.info[widthby]
        width = unique_widths.tolist().index(width_val)*0.1 + 1
    if colorby is None:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
    else:
        unique_colors = dataset.info_table[colorby].unique()
        color_val = dataitem.info[colorby]
        color = plt.get_cmap('plasma')(unique_colors.tolist().index(color_val)/len(unique_colors))

    return linestyle, marker, width, color


def _add_colorbar(
        dataset,
        colorby: str,
        min_val=None,
        max_val=None,
        cbar_label: Optional[str] = None):
    """Add a colorbar to the right of the given axis."""
    # find max and min temperatures
    if min_val is None:
        min_val = dataset.info_table[colorby].min()
    if max_val is None:
        max_val = dataset.info_table[colorby].max()
    # add colorbar
    plt.subplots_adjust(right=0.875)
    cax = plt.axes([0.9, 0.3, 0.014, 0.4])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=plt.Normalize(vmin=min_val, vmax=max_val))
    cbar = plt.colorbar(sm, cax=cax)
    if cbar_label is None:
        cbar_label = colorby
    cbar.set_label(cbar_label)


def _setup_grid(
        nrows: int,
        ncols: int,
        figsize: Tuple[float, float] = (6.4, 4.8),
        row_titles=None,
        col_titles=None,
        subplot_titles=None,
        wspace: float = 0.3,
        hspace: float = 0.3,
        sharex: str = 'col',
        sharey: str = 'row',
) -> np.ndarray:
    # setup grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    # add row titles
    if row_titles is not None:
        for ax, row_title in zip(axs[:, 0], row_titles):
            ax.annotate(row_title, xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
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
