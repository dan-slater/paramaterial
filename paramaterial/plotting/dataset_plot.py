"""Module containing the plotting functions for the dataset class."""
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt



def dataset_plot(dataset, ax: plt.Axes, colourby: str, **df_plot_kwargs):
    """Plot the dataset using the given parameters.

    Args:
        ax: The axes to plot on.
        colourby: The info column to use for coloring.
        styleby: The info column to use for styling.
        **df_plot_kwargs: Keyword arguments to pass to the pandas.DataFrame.plot() function.
    """
    _configure_plt_formatting()
    _add_colourbar_curves(ax, dataset, colourby, **df_plot_kwargs)


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


def _add_colourbar_curves(
        ax: plt.Axes,
        dataset,
        colourby: str,
        min_val=None,
        max_val=None,
        **df_plot_kwargs
):
    # find max and min temperatures
    if min_val is None:
        min_val = dataset.info_table[colourby].min()
    if max_val is None:
        max_val = dataset.info_table[colourby].max()
    # add line plots
    for dataitem in dataset:
        colour_val = dataitem.info[colourby]
        color = plt.get_cmap('plasma')((colour_val - min_val)/(max_val - min_val))  # todo: add parameter for cmap
        dataitem.data.plot(ax=ax, color=color, **df_plot_kwargs)
    # add colorbar
    _add_colorbar(min_val, max_val, cbar_label=colourby)


def _add_colorbar(min_val: float, max_val: float, cbar_label: Optional[str] = None):
    """Add a colorbar to the right of the given axis."""
    plt.subplots_adjust(right=0.875)
    cax = plt.axes([0.9, 0.3, 0.014, 0.4])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('plasma'), norm=plt.Normalize(vmin=min_val, vmax=max_val))
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label)


def _setup_ax_grid(ax: plt.Axes, x_sep: float, x_max: float, y_sep: float, y_max: float):
    """Helper function for axis limits and grid formatting.

    Args:
        ax: The axis to format.
        x_sep: The x-axis separation.
        x_max: The maximum x-axis value.
        y_sep: The y-axis separation.
        y_max: The maximum y-axis value.
    """
    ax.set_xticks(np.arange(0, x_max + 0.5*x_sep, x_sep))
    ax.set_xlim(xmin=0 - 0.5*x_sep, xmax=x_max)
    ax.set_yticks(np.arange(0, y_max + 0.5*y_sep, y_sep))
    ax.set_ylim(ymin=0 - 0.5*y_sep, ymax=y_max)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.grid()
