"""Module containing the plotting functions for the ds class."""
import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any, Dict, Callable, Union

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import seaborn as sns

from paramaterial.plug import DataItem, DataSet


def configure_plt_formatting():
    plt.style.use('seaborn-dark')
    # mpl.rcParams['axes.facecolor'] = '#f0e6e6'
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
    mpl.rcParams["font.family"] = "Times New Roman"
    FONTSIZE = 11
    plt.rc('font', size=FONTSIZE)
    plt.rc('axes', titlesize=FONTSIZE, labelsize=FONTSIZE)
    plt.rc('xtick', labelsize=FONTSIZE - 1)
    plt.rc('ytick', labelsize=FONTSIZE - 1)
    plt.rc('legend', fontsize=FONTSIZE - 1)
    plt.rc('figure', titlesize=FONTSIZE + 1)
    mpl.rcParams.update({"axes.grid": True})
    # cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white", (85/255, 49/255, 0)])
    # mpl.rcParams['axes.facecolor'] = cmap(0.1)
    # mpl.rcParams['legend.facecolor'] = "white"
    # mpl.rcParams["grid.linewidth"] = 1
    # mpl.rcParams["text.color"] = (40/255, 40/255, 40/255)
    # cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white", (0.2124, 0.3495, 0.1692)])
    # mpl.rcParams["axes.facecolor"]= cmap(0.1)


configure_plt_formatting()


@dataclass
class Styler:
    """A class for storing plotting styles for a dataset."""
    color_by: Optional[str] = None
    linestyle_by: Optional[str] = None
    marker_by: Optional[str] = None
    width_by: Optional[str] = None
    color_by_label: Optional[str] = None
    linestyle_by_label: Optional[str] = None
    marker_by_label: Optional[str] = None
    width_by_label: Optional[str] = None
    cbar: Optional[bool] = False
    color_norm: Optional[plt.Normalize] = None
    cbar_label: Optional[str] = None
    cmap: str = 'plasma'
    handles: Optional[List[mpatches.Patch]] = None
    linestyles: List[str] = field(default_factory=lambda: ['-', '--', ':', '-.'])
    markers: List[str] = field(
        default_factory=lambda: ['s', 'H', 'd', 'v', 'D', 'p', 'X', 'o', 'd', 'h', 'H', '8', 'P', 'x'])
    color_dict: Optional[Dict[Union[str, int, float], str]] = None
    linestyle_dict: Optional[Dict[Union[str, int, float], str]] = None
    marker_dict: Optional[Dict[Union[str, int, float], str]] = None
    plot_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())
    styled_ds: Optional[DataSet] = None

    def __post_init__(self):
        self.plot_kwargs['legend'] = False
        self.plot_kwargs.update({'markeredgecolor': 'white', 'markersize': 8})
        # todo: use pandas in-built color bar
        # if self.cbar:
        #     self.plot_kwargs['cmap'] = self.cmap
        #     self.plot_kwargs['norm'] = self.color_norm
        #     self.plot_kwargs['colorbar'] = True
        #     self.plot_kwargs['colorbar_label'] = self.cbar_label

    def style_to(self, ds: DataSet):
        """Format the styles to match the dataset."""
        self.styled_ds = copy.deepcopy(ds)

        if self.color_by is not None:
            color_vals = ds.info_table[self.color_by].unique()
            if all(str(x).isnumeric() for x in color_vals):
                if self.color_norm is None:
                    self.color_norm = plt.Normalize(color_vals.min(), color_vals.max())
                self.color_dict = {x: plt.cm.get_cmap(self.cmap)(self.color_norm(x)) for x in color_vals}
            else:
                self.color_norm = plt.Normalize(0, len(color_vals))
                self.color_dict = {x: plt.cm.get_cmap(self.cmap)(self.color_norm(i)) for i, x in enumerate(color_vals)}

        if self.linestyle_by is not None:
            linestyle_vals = ds.info_table[self.linestyle_by].unique().tolist()
            while len(self.linestyles) < len(linestyle_vals):
                self.linestyles.extend(self.linestyles)
            self.linestyle_dict = dict(zip(linestyle_vals, self.linestyles))

        if self.marker_by is not None:
            marker_vals = ds.info_table[self.marker_by].unique().tolist()
            while len(self.markers) < len(marker_vals):
                self.markers.extend(self.markers)
            self.marker_dict = dict(zip(marker_vals, self.markers))

        return self

    def curve_formatters(self, di: DataItem) -> Dict[str, Any]:
        """Return the curve formatters for the dataitem curve."""
        if self.styled_ds is None:
            raise ValueError('The styler must be styled to a ds before plotting.')

        # configure_plt_formatting()
        formatters = dict()

        if self.color_by is not None:
            formatters['color'] = self.color_dict[di.info[self.color_by]]
            if all(str(x).isnumeric() for x in self.color_dict.keys()):
                formatters['zorder'] = di.info[self.color_by]
        else:
            formatters['color'] = plt.gca()._get_lines.get_next_color()

        if self.linestyle_by is not None:
            formatters['linestyle'] = self.linestyle_dict[di.info[self.linestyle_by]]

        if self.marker_by is not None:
            formatters['marker'] = self.marker_dict[di.info[self.marker_by]]

        return {k: v for k, v in formatters.items() if v is not None}

    def legend_handles(self, ds: Optional[DataSet] = None) -> List[mpatches.Patch]:
        """Return the legend handles for the dataset plot."""
        handles = list()

        if ds is None:
            ds = self.styled_ds
        if len(ds) == 0:
            return handles

        if self.color_by_label is not None:
            handles.append(mpatches.Patch(label=self.color_by_label.title(), alpha=0))

        if self.color_by is not None:
            for color_val in ds.info_table[self.color_by].unique():
                handles.append(Line2D([], [], label=color_val, color=self.color_dict[color_val], marker='o', ls=''))

        if self.linestyle_by_label is not None:
            handles.append(mpatches.Patch(label='\n' + self.linestyle_by_label.title(), alpha=0))

        if self.linestyle_by is not None:
            for ls_val in ds.info_table[self.linestyle_by].unique():
                handles.append(Line2D([], [], label=ls_val, ls=self.linestyle_dict[ls_val], c='k', marker=''))

        if self.marker_by_label is not None:
            handles.append(mpatches.Patch(label='\n' + self.marker_by_label.title(), alpha=0))

        if self.marker_by is not None:
            for marker_val in ds.info_table[self.marker_by].unique():
                handles.append(Line2D([], [], label=marker_val, marker=self.marker_dict[marker_val], c='k', ls=''))

        return handles


def dataset_plot(
        ds: DataSet,
        styler: Optional[Styler] = None,
        ax: Optional[plt.Axes] = None,
        fill_between: Optional[Tuple[str, str]] = None,
        plot_legend: bool = True,
        handletextpad: float = 0.05,
        labelspacing: float = 0.1,
        **kwargs
) -> plt.Axes:
    """Make a single combined plot from the data of every dataitem in the dataset using pandas.DataFrame.plot.

    Args:
        ds: The dataset to plot.
        styler: The styler to use for the plot.
        ax: The axis to plot on.
        fill_between: A tuple of the two columns in the data to fill between.
        plot_legend: Whether to plot the legend.
        **kwargs: Additional keyword arguments to pass to the pandas.DataFrame.plot function.

    Returns: The axis the plot was made on.
    """
    if ax is None:
        fig, (ax) = plt.subplots(1, 1, figsize=kwargs.get('figsize', (4, 3)))
    kwargs['ax'] = ax

    if ax.get_legend() is not None and plot_legend:
        ax.get_legend().remove()

    kwargs = {**styler.plot_kwargs, **kwargs}

    # plot the dataitems
    for di in ds:
        # plot the curve
        ax = di.data.plot(**styler.curve_formatters(di), **kwargs)
        # fill between curves
        if fill_between is not None:
            ax.fill_between(di.data[kwargs['x']], di.data[fill_between[0]], di.data[fill_between[1]], alpha=0.2,
                            **styler.curve_formatters(di))

    # add the legend
    handles = styler.legend_handles(ds)
    if len(handles) > 0 and plot_legend:
        ax.legend(handles=handles, loc='best', frameon=False, markerfirst=True, handletextpad=handletextpad,
                  labelspacing=labelspacing)  # handletextpad=0.05)  # , labelspacing=0.1)
        ax.get_legend().set_zorder(2000)
    # colorbar
    if styler.cbar and plot_legend:
        sm = plt.cm.ScalarMappable(cmap=styler.cmap, norm=styler.color_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=kwargs['ax'], fraction=0.046, pad=0.04)
        cbar.set_label(styler.cbar_label) if styler.cbar_label is not None else None
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')

    return ax


def info_plot(
        ds: DataSet,
        x: str,
        y: str,
        styler: Optional[Styler] = None,
        ax: Optional[plt.Axes] = None,
        plot_legend: bool = True,
        err_between: Optional[Tuple[str, str]] = None,
        **kwargs
) -> plt.Axes:
    """Make a single combined plot from the info of every dataitem in the dataset using pandas.DataFrame.plot.

    Args:
        ds: The dataset to plot.
        x: The column to plot on the x-axis.
        y: The column to plot on the y-axis.
        styler: The styler to use for the plot.
        ax: The axis to plot on.
        plot_legend: Whether to plot the legend.
        **kwargs: Additional keyword arguments to pass to the pandas.DataFrame.plot function.

    Returns: The axis the plot was made on.
    """
    if ax is None:
        fig, (ax) = plt.subplots(1, 1, figsize=kwargs.get('figsize', (6, 4)))
    kwargs['ax'] = ax

    if ax.get_legend() is not None and plot_legend:
        ax.get_legend().remove()

    kwargs = {**styler.plot_kwargs, **kwargs}

    # plot the dataitems
    for di in ds:
        # plot the curve
        df = pd.DataFrame([[di.info[x], di.info[y]]], columns=[x, y])
        ax = df.plot(x=x, y=y, **styler.curve_formatters(di), **kwargs)
        # ax = di.info.plot(x=x, y=y, **styler.curve_formatters(di), **kwargs)
        if err_between is not None:
            ax = di.info.plot(x=x, y=y, yerr=[di.info[err_between[0]], di.info[err_between[1]]],
                              **styler.curve_formatters(di), ax=ax)

    # add the legend
    handles = styler.legend_handles(ds)
    if len(handles) > 0 and plot_legend:
        ax.legend(handles=handles, loc='best', frameon=True, markerfirst=False,
                  handletextpad=0.05)  # , labelspacing=0.1)
        ax.get_legend().set_zorder(2000)
    # colorbar

    return ax


def dataset_subplots(
        ds: DataSet,
        shape: Tuple[int, int],
        rows_by: str,
        cols_by: str,
        row_vals: List[List[Any]],
        col_vals: List[List[Any]],
        styler: Optional[Styler] = None,
        axs: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (9, 6),
        sharex: str = 'col',
        sharey: str = 'row',
        wspace: float = 0.05,
        hspace: float = 0.05,
        row_titles: Optional[List[str]] = None,
        col_titles: Optional[List[str]] = None,
        plot_titles: Optional[List[str]] = None,
        subplot_legend: bool = True,
        subplot_cbar: bool = False,
        subplots_adjust: float = 0.0,
        **kwargs
) -> plt.Axes:
    """Plot a dataset as a grid of subplots, split by the 'rows_by' and 'cols_by' columns in the info_table.

    Args:
        ds: The dataset to plot.
        shape: The shape of the grid of subplots.
        rows_by: The column in the info_table to split the rows by.
        cols_by: The column in the info_table to split the columns by.
        row_vals: The values of the rows to plot.
        col_vals: The values of the columns to plot.
        styler: The styler to use for the plot.
        axs: The axes to plot on.
        figsize: The size of the figure.
        sharex: Whether to share the x axis between subplots.
        sharey: Whether to share the y axis between subplots.
        wspace: The width space between subplots.
        hspace: The height space between subplots.
        row_titles: The titles of the rows.
        col_titles: The titles of the columns.
        plot_titles: The titles of the subplots.
        subplot_legend: Whether to plot the legend in each subplot.
        subplot_cbar: Whether to plot the colorbar in each subplot.
        **kwargs: Additional keyword arguments to pass to the pandas.DataFrame.plot function.

    Returns: The axes the plot was made on.
    """
    if axs is None:
        fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize, sharex=sharex, sharey=sharey)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if shape[0] == 1 and shape[1] == 1:
        axs = np.array([[axs]])
    elif shape[0] == 1:
        axs = np.array([axs])
    elif shape[1] == 1:
        axs = np.array([[ax] for ax in axs])

    if styler is None:
        styler = Styler()

    # set the titles of the rows and columns
    if row_titles is not None:
        for ax, row_title in zip(axs[:, 0], row_titles):
            ax.annotate(row_title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90, fontsize=12)

    if col_titles is not None:
        for ax, column_title in zip(axs[0, :], col_titles):
            ax.set_title(column_title)

    if plot_titles is not None:
        for ax, subplot_title in zip(axs.flat, plot_titles):
            ax.set_title(subplot_title)

    # default kwargs
    if 'plot_legend' not in kwargs:
        kwargs['plot_legend'] = False

    # loop through the grid of axes and plot the subsets
    if rows_by == cols_by:
        for ax, row_val in zip(axs.flat, row_vals):
            subset = ds.subset({rows_by: row_val})
            dataset_plot(subset, styler=styler, ax=ax, **kwargs)
    else:
        for row, row_val in enumerate(row_vals):
            for col, col_val in enumerate(col_vals):
                ax = axs[row, col]
                subset = ds.subset({cols_by: col_val, rows_by: row_val})
                dataset_plot(subset, styler=styler, ax=ax, **kwargs)

    if subplot_cbar:
        plt.subplots_adjust(right=0.875)
        cax = plt.axes([0.9, 0.2, 0.014, 0.6])
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=styler.color_norm, cmap=styler.cmap), cax=cax)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        if styler.cbar_label is not None:
            cbar.set_label(styler.cbar_label)

    if subplot_legend:
        plt.subplots_adjust(right=0.835 + subplots_adjust)
        axs.flat[0].get_figure().legend(handles=styler.legend_handles(), loc='center right', frameon=False,
                                        bbox_to_anchor=(0.925, 0.5), markerfirst=True, handletextpad=0.05)

    return axs


def subplot_wrapper(
        ds: DataSet,
        plot_func: Callable[[DataItem, plt.axes], DataItem],
        shape: Tuple[int, int],
        rows_by: str,
        cols_by: str,
        row_vals: List[List[Any]],
        col_vals: List[List[Any]],
        axs: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (12, 8),
        sharex: str = 'col',
        sharey: str = 'row',
        wspace: float = 0.1,
        hspace: float = 0.1,
        row_titles: Optional[List[str]] = None,
        col_titles: Optional[List[str]] = None,
        plot_titles: Optional[List[str]] = None,
        **kwargs
) -> np.ndarray:
    """Plot a dataset using the given plot function as a grid of subplots,
    split by the 'rows_by' and 'cols_by' columns in the info_table.

    Args:
        ds: The dataset to plot.
        plot_func: The function to use to plot each subplot.
        shape: The shape of the grid of subplots.
        rows_by: The column in the info_table to split the rows by.
        cols_by: The column in the info_table to split the columns by.
        row_vals: The values of the rows to plot.
        col_vals: The values of the columns to plot.
        axs: The axes to plot on.
        figsize: The size of the figure.
        sharex: Whether to share the x axis between subplots.
        sharey: Whether to share the y axis between subplots.
        wspace: The width space between subplots.
        hspace: The height space between subplots.
        row_titles: The titles of the rows.
        col_titles: The titles of the columns.
        plot_titles: The titles of the subplots.
        **kwargs: Additional keyword arguments to pass to the plot function.

    Returns: The axes the plot was made on.
    """
    if axs is None:
        fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize, sharex=sharex, sharey=sharey)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    if axs.ndim == 1:
        axs = np.array([axs])

    # loop through the grid of axes and plot the subsets
    if rows_by == cols_by:
        for ax, row_val in zip(axs.flat, row_vals):
            kwargs['ax'] = ax
            subset = ds.subset({rows_by: row_val})
            for di in subset:
                plot_func(di, **kwargs)
    else:
        for row, row_val in enumerate(row_vals):
            for col, col_val in enumerate(col_vals):
                kwargs['ax'] = axs[row, col]
                subset = ds.subset({cols_by: col_val, rows_by: row_val})
                for di in subset:
                    plot_func(di, **kwargs)

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

    return axs


def matrix_plot(
        ds: DataSet,
        index: str,
        columns: str,
        x_label: str = '',
        y_label: str = '',
        titles: Union[str, List[str]] = None,
        group_by: str = None,
        group_by_vals: List[str] = None,
        axs: plt.Axes = None,
        heatmap_kwargs: Dict[str, Any] = None,
) -> plt.Axes:
    from paramaterial.preparing import make_experimental_matrix

    if axs is None:
        fig, axs = plt.subplots(1, len(group_by))

    if group_by is not None:
        ds_subsets = [ds.subset({group_by: val}) for val in group_by_vals]
    else:
        ds_subsets = [ds]

    default_heatmap_kwargs = dict(linewidths=2, cbar=False, annot=True, annot_kws={'size': 10})
    heatmap_kwargs = default_heatmap_kwargs.update(heatmap_kwargs)

    for sub_ds in ds_subsets:
        exp_matrix = make_experimental_matrix(sub_ds.info_table, index=index, columns=columns)
        sns.heatmap(exp_matrix, **heatmap_kwargs)

    for i, ax in enumerate(axs):
        ax.set_xlabel(x_label)
        if titles is not None:
            axs[i].set_title(titles[i])


    return axs

