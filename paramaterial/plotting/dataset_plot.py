"""Module containing the plotting functions for the dataset class."""
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


def dataset_plot(
        dataset,  # todo: type hint but avoid circular import
        ax: plt.Axes,
        colorby: Optional[str] = None,
        styleby: Optional[str] = None,
        markerby: Optional[str] = None,
        widthby: Optional[str] = None,
        add_cbar: bool = False,
        cbar_label: Optional[str] = None,
        **df_plot_kwargs
):
    """Plot the curves from every item in the dataset using pandas.DataFrame.plot().
    Args:
        dataset: The dataset to plot.
        ax: The axes to plot on.
        colorby: The info column to use for coloring.
        styleby: The info column to use for line style.
        markerby: The info column to use for marker style.
        widthby: The info column to use for line width.
        add_cbar: Whether to add a colorbar.
        cbar_label: The label for the colorbar.
        **df_plot_kwargs: Keyword arguments to pass to the pandas.DataFrame.plot() function.
    """
    _configure_plt_formatting()
    for dataitem in dataset:
        ls, marker, width, color = _get_curve_formatters(styleby, markerby, widthby, colorby, dataset, dataitem)
        dataitem.data.plot(ax=ax, linestyle=ls, marker=marker, linewidth=width, color=color, **df_plot_kwargs)
    if colorby is not None and add_cbar:
        _add_colorbar(dataset, colorby, cbar_label)


def _configure_plt_formatting():
    """Helper function to configure matplotlib formatting."""
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


def _get_curve_formatters(styleby: str, markerby: str, widthby: str, colorby: str, dataset, dataitem):
    """Helper function to get the curve formatters for a given dataitem."""
    linestyle = '-'
    marker = None
    width = 1
    color = None
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
    if colorby is not None:
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


