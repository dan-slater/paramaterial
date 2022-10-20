"""Module containing the plotting functions for the dataset class."""
from typing import Optional, Tuple, List, Any

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
    plt.rc('font', size=11)
    plt.rc('axes', titlesize=11, labelsize=11)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('legend', fontsize=9)
    plt.rc('figure', titlesize=13)


configure_plt_formatting()


def dataset_plot(dataset, x: str, y: str,
                 cbar_by: Optional[str] = None, cbar_label: Optional[str] = None, cmap: str = 'plasma',
                 color_by: Optional[str] = None, color_by_label: Optional[str] = None,
                 style_by: Optional[str] = None, style_by_label: Optional[str] = None,
                 marker_by: Optional[str] = None, marker_by_label: Optional[str] = None,
                 width_by: Optional[str] = None, width_by_label: Optional[str] = None, width_by_scale: float = 0.2,
                 **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the curves from every item in the dataset using pandas.DataFrame.plot().
    Args:
        dataset: The dataset to plot.
        x: The column to use for the x-axis.
        y: The column to use for the y-axis.
        cbar_by: The info column to use for the colorbar.
        cbar_label: The label for the colorbar.
        cmap: The colormap to use for the colorbar.
        color_by: The info column to use for coloring.
        color_by_label: The label for the color_by column.
        style_by: The info column to use for line style.
        style_by_label: The label for the line style.
        marker_by: The info column to use for marker style.
        marker_by_label: The label for the marker style.
        width_by: The info column to use for line width.
        width_by_label: The label for the line width.
        width_by_scale: The scale factor for the line width.
        **kwargs: Keyword arguments to pass to the pandas.DataFrame.plot() function.
    Returns:
        The plt.Axes object.
    """
    # make ax if not given
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (6, 4)
    if 'ax' not in kwargs:
        fig, ax = plt.subplots(figsize=kwargs['figsize'])

    # setup curve formatters
    linestyles = ['-', '--', ':', '-.']
    markers = ['o', 's', 'v', '^', 'd', 'p', 'h', '8', '>', '<', 'x', 'D', 'P', 'H', 'X']

    unique_colors = sorted(dataset.info_table[color_by].unique().tolist()) if color_by is not None else [None]
    unique_styles = sorted(dataset.info_table[style_by].unique().tolist()) if style_by is not None else [None]
    unique_markers = sorted(dataset.info_table[marker_by].unique().tolist()) if marker_by is not None else [None]
    unique_widths = sorted(dataset.info_table[width_by].unique().tolist()) if width_by is not None else [None]

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
            width = unique_widths.index(width_val)*width_by_scale + 1
        if color_by is None:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
        else:
            color_val = dataitem.info[color_by]
            color = plt.get_cmap(cmap)(unique_colors.index(color_val)/len(unique_colors))
        if cbar_by is not None:
            vmin = dataset.info_table[cbar_by].min()
            vmax = dataset.info_table[cbar_by].max()
            cbar_norm = plt.Normalize(vmin=vmin, vmax=vmax)
            color = plt.get_cmap(cmap)(cbar_norm(dataitem.info[cbar_by]))
            zorder = dataitem.info[cbar_by]

        # plot the curve
        try:
            ax = dataitem.data.plot(ax=ax, x=x, y=y, linestyle=linestyle, marker=marker, linewidth=width, color=color,
                                    zorder=zorder, legend=False, **kwargs)
        except AttributeError as e:
            raise AttributeError(f"Error when calling pandas.DataFrame.plot(): {e}")

    # make the colorbar
    if cbar_by is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cbar_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')

    # add the legend
    handles = []
    if color_by is not None:
        if color_by_label is not None:
            handles.append(mpatches.Patch(label=color_by_label, alpha=0))
        for color_val in unique_colors:
            handles.append(mpatches.Patch(color=plt.get_cmap(cmap)(unique_colors.index(color_val)/len(unique_colors)),
                                          label=color_val))
    if style_by is not None:
        if style_by_label is not None:
            handles.append(mpatches.Patch(label=style_by_label, alpha=0))
        for style_val in unique_styles:
            handles.append(
                Line2D([0], [0], color='black', linestyle=linestyles[unique_styles.index(style_val)], label=style_val))
    if marker_by is not None:
        if marker_by_label is not None:
            handles.append(mpatches.Patch(label=marker_by_label, alpha=0))
        for marker_val in unique_markers:
            handles.append(
                Line2D([0], [0], color='black', marker=markers[unique_markers.index(marker_val)], linestyle='None',
                       label=marker_val))
    if width_by is not None:
        if width_by_label is not None:
            handles.append(mpatches.Patch(label=width_by_label, alpha=0))
        for width_val in unique_widths:
            handles.append(
                Line2D([0], [0], color='black', linewidth=unique_widths.index(width_val)*width_by_scale + 1,
                       label=width_val))

    if len(handles) > 0:
        if cbar_by is not None:
            ax.legend(handles=handles, loc='best')
        else:
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    return ax


def dataset_subplots(dataset, x: str, y: str, nrows: int, ncols: int, cols_by: str, rows_by: str,
                     col_keys: List[List[Any]], row_keys: List[List[Any]], col_titles: Optional[List[str]] = None,
                     row_titles: Optional[List[str]] = None, plot_titles: Optional[List[str]] = None,
                     figsize: Tuple[float, float] = (12, 8),
                     cbar_by: Optional[str] = None, cbar_label: Optional[str] = None,
                     cmap: str = 'plasma', color_by: Optional[str] = None, style_by: Optional[str] = None,
                     marker_by: Optional[str] = None, width_by: Optional[str] = None,
                     sharex: str = 'col', sharey: str = 'row',
                     wspace: float = 0.1, hspace: float = 0.1, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    # curve formatters
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'v', '^', 'd', 'p', 'h', '8', '>', '<', 'x', 'D', 'P', 'H', 'X']

    unique_colors = sorted(dataset.info_table[color_by].unique().tolist()) if color_by is not None else [None]
    unique_styles = sorted(dataset.info_table[style_by].unique().tolist()) if style_by is not None else [None]
    unique_markers = sorted(dataset.info_table[marker_by].unique().tolist()) if marker_by is not None else [None]
    unique_widths = sorted(dataset.info_table[width_by].unique().tolist()) if width_by is not None else [None]

    # setup grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, wspace=wspace, hspace=hspace)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    if axs.ndim == 1:
        axs = np.array([axs])

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
    if plot_titles is not None:
        for ax, subplot_title in zip(axs.flat, plot_titles):
            ax.set_title(subplot_title)
    # add colorbar
    if cbar_by is not None:
        plt.subplots_adjust(right=0.875)
        cax = plt.axes([0.9, 0.3, 0.014, 0.4])
        vmin = dataset.info_table[cbar_by].min()
        vmax = dataset.info_table[cbar_by].max()
        cbar_norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=cbar_norm, cmap=cmap), cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
    # add curves to axes
    for row, row_key in enumerate(row_keys):
        for col, col_key in enumerate(col_keys):
            ax = axs[row, col]
            dataset_subset = dataset[{cols_by: col_key, rows_by: row_key}]
            for dataitem in dataset_subset:
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

                # plot the curve
                try:
                    ax = dataitem.data.plot(ax=ax, x=x, y=y, linestyle=linestyle, marker=marker, linewidth=width,
                                            color=color, zorder=zorder, legend=False, **kwargs)
                    fig = ax.get_figure()
                except AttributeError as e:
                    raise AttributeError(f"Error when calling pandas.DataFrame.plot(): {e}")

    # add the legend
    handles = []
    if color_by is not None:
        for color_val in unique_colors:
            handles.append(mpatches.Patch(color=plt.get_cmap(cmap)(unique_colors.index(color_val)/len(unique_colors)),
                                          label=color_val))
    if style_by is not None:
        for style_val in unique_styles:
            handles.append(
                Line2D([0], [0], color='black', linestyle=linestyles[unique_styles.index(style_val)], label=style_val))
    if marker_by is not None:
        for marker_val in unique_markers:
            handles.append(
                Line2D([0], [0], color='black', marker=markers[unique_markers.index(marker_val)], linestyle='None',
                       label=marker_val))
    if width_by is not None:
        for width_val in unique_widths:
            handles.append(
                Line2D([0], [0], color='black', linewidth=unique_widths.index(width_val)*0.1 + 1, label=width_val))
    if len(handles) > 0:
        if cbar_by is not None:
            fig.legend(handles=handles, bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=6)
        else:
            fig.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    return fig, axs
