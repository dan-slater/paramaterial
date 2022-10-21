"""Module containing the plotting functions for the dataset class."""
from typing import Optional, Tuple, List, Any, Union

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


def dataset_colorbar_norm(dataset, cbar_by: str, cmap: str = 'plasma') -> plt.Normalize:
    vmin = dataset.info_table[cbar_by].min()
    vmax = dataset.info_table[cbar_by].max()
    return plt.Normalize(vmin=vmin, vmax=vmax)


def dataset_plot(dataset, x: str, y: str,
                 cbar_by: Optional[str] = None, cbar_label: Optional[str] = None, cmap: str = 'plasma',
                 color_by: Optional[str] = None, color_by_label: Optional[str] = None,
                 style_by: Optional[str] = None, style_by_label: Optional[str] = None,
                 marker_by: Optional[str] = None, marker_by_label: Optional[str] = None,
                 width_by: Optional[str] = None, width_by_label: Optional[str] = None, width_by_scale: float = 0.2,
                 fill_between: Optional[Tuple[str, str]] = None, cbar_norm: Optional[plt.Normalize] = None,
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
        fill_between: The columns to use for fill_between.
        **kwargs: Keyword arguments to pass to the pandas.DataFrame.plot() function.
    Returns:
        The plt.Axes object.
    """

    if 'ax' not in kwargs:  # ax can be given or created
        fig, ax = plt.subplots(figsize=kwargs['figsize'])
        kwargs['ax'] = ax
    else:
        ax = kwargs['ax']
    if 'figsize' not in kwargs:  # figsize can be given or created
        kwargs['figsize'] = (10, 6)

    # setup color function
    unique_color_vals = dataset.info_table[color_by].unique() if color_by else None
    colormap = plt.get_cmap(cmap)
    color_norm = plt.Normalize(vmin=unique_color_vals.min(), vmax=unique_color_vals.max())

    def color(val: float) -> str:
        if color_by is not None:
            return colormap(color_norm(val))
        else:
            return ax._get_lines.get_next_color()

    # setup style function
    unique_style_vals = dataset.info_table[style_by].unique() if style_by else None
    linestyles = ['-', '--', ':', '-.']

    def style(val: float) -> str:
        if style_by is not None:
            while len(linestyles) < len(unique_style_vals):
                linestyles.extend(linestyles)
            return linestyles[unique_style_vals.tolist().index(val)]
        else:
            return '-'

    # setup marker function
    unique_marker_vals = dataset.info_table[marker_by].unique() if marker_by else None
    markers = ['o', 's', 'v', '^', 'd', 'p', 'h', '8', '>', '<', 'x', 'D', 'P', 'H', 'X']

    def marker(val: float) -> Union[str, None]:
        if marker_by is not None:
            while len(markers) < len(unique_marker_vals):
                markers.extend(markers)
            return markers[unique_marker_vals.tolist().index(val)]
        else:
            return None

    # setup width function
    unique_vals = dataset.info_table[width_by].unique()
    width_norm = plt.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())  # todo: check if need +1

    def width(val: float) -> float:
        if width_by is not None:
            return width_by_scale*width_norm(val)
        else:
            return 1.0

    # make the legend # todo: check if works when color_by etc is None
    handles = []
    if color_by is not None:

    colors = [color(val) for val in dataset.info_table[color_by].unique()]
    styles = [style(val) for val in dataset.info_table[style_by].unique()]
    markers = [marker(val) for val in dataset.info_table[marker_by].unique()]
    widths = [width(val) for val in dataset.info_table[width_by].unique()]

    if color_by is not None:
        for color in unique_color_vals:
            handles.append(Line2D([], [], color=colormap(color_norm(color)), label=color_by_label))
    if style_by is not None:
        for style in unique_styles:
            handles.append(Line2D([], [], linestyle=style, label=style_by_label))
    if marker_by is not None:
        for marker in unique_markers:
            handles.append(Line2D([], [], marker=marker, label=marker_by_label))
    if width_by is not None:
        for width in unique_widths:
            handles.append(Line2D([], [], linewidth=width*width_by_scale, label=width_by_label))

    if color_by is not None:
        color_norm = plt.Normalize(vmin=dataset.info_table[color_by].min(), vmax=dataset.info_table[color_by].max())
    else:
        color_norm = plt.Normalize(vmin=0, vmax=1)
    colormap = plt.get_cmap(cmap)

    for dataitem in dataset:
        # get the curve formatters
        linestyle = linestyles[unique_styles.index(dataitem.info[style_by])] if style_by is not None else '-'
        marker = markers[unique_markers.index(dataitem.info[marker_by])] if marker_by is not None else None
        width = unique_widths.index(dataitem.info[width_by])*width_by_scale if width_by is not None else 1
        color = colormap(color_norm(dataitem.info[color_by])) if color_by is not None else 'k'
        zorder = dataitem.info[color_by] if color_by is not None else 0

        # PLOT THE CURVE
        try:
            ax = dataitem.data.plot(x=x, y=y, linestyle=linestyle, marker=marker, linewidth=width, color=color,
                                    zorder=zorder, legend=False, **kwargs)
        except Exception as e:
            raise Exception(f"Error when calling pandas.DataFrame.plot(): {e}")

        # fill between
        if fill_between is not None:
            ax.fill_between(dataitem.data[x], dataitem.data[fill_between[0]], dataitem.data[fill_between[1]],
                            color=color, alpha=0.2)

    # make the colorbar
    if cbar_by is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cbar_norm).set_array([])
        # sm.set_array([])
        cbar = plt.colorbar(sm, ax=kwargs['ax'], fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label) if cbar_label is not None else None
        cbar.ax.yaxis.set_ticks_position('right').set_label_position('right')
        # cbar.ax.yaxis.set_label_position('right')

    # if color_by is not None and cbar_norm is None:
    #     if color_by_label is not None:
    #         handles.append(mpatches.Patch(label=color_by_label, alpha=0))
    #     for color_val in unique_colors:
    #         color = plt.get_cmap(cmap)(cbar_norm(color_val))
    #         handles.append(Line2D([0], [0], color=color, marker='o', linestyle='None', label=color_val))
    if style_by is not None:
        if style_by_label is not None:
            handles.append(mpatches.Patch(label=style_by_label, alpha=0))
        for style_val in unique_styles:
            linestyle = linestyles[unique_styles.index(style_val)]
            handles.append(Line2D([0], [0], color='black', linestyle=linestyle, label=style_val))
    if marker_by is not None:
        if marker_by_label is not None:
            handles.append(mpatches.Patch(label=marker_by_label, alpha=0))
        for marker_val in unique_markers:
            handles.append(
                Line2D([0], [0], color='black', marker=markers[unique_markers.index(marker_val)], linestyle='None'))
    if width_by is not None:
        if width_by_label is not None:
            handles.append(mpatches.Patch(label=width_by_label, alpha=0))
        for width_val in unique_widths:
            handles.append(
                Line2D([0], [0], color='black', linewidth=unique_widths.index(width_val)*width_by_scale + 1))

    # add the legend
    if len(handles) > 0:
        ax.legend(handles=handles, loc='best')

    return ax


def dataset_subplots(dataset, x: str, y: str, nrows: int, ncols: int, cols_by: str, rows_by: str,
                     col_keys: List[List[Any]], row_keys: List[List[Any]], col_titles: Optional[List[str]] = None,
                     row_titles: Optional[List[str]] = None, plot_titles: Optional[List[str]] = None,
                     figsize: Tuple[float, float] = (12, 8),
                     cbar_by: Optional[str] = None, cbar_label: Optional[str] = None,
                     cmap: str = 'plasma',
                     sharex: str = 'col', sharey: str = 'row',
                     wspace: float = 0.1, hspace: float = 0.1, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """"""
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
        cax = plt.axes([0.9, 0.2, 0.014, 0.6])
        vmin = dataset.info_table[cbar_by].min()
        vmax = dataset.info_table[cbar_by].max()
        cbar_norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=cbar_norm, cmap=cmap), cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
    else:
        cbar_norm = None

    # add curves to axes
    for row, row_key in enumerate(row_keys):
        for col, col_key in enumerate(col_keys):
            ax = axs[row, col]
            dataset_subset = dataset.get_subset({cols_by: col_key, rows_by: row_key})
            # if cbar_by is not None: add it to kwargs
            if cbar_by is not None:
                kwargs['color_by'] = cbar_by
            dataset_plot(dataset_subset, x=x, y=y, cbar_norm=cbar_norm, ax=ax, **kwargs)

    return fig, axs
