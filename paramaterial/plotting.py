"""Module containing the plotting functions for the dataset class."""
import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any, Dict, Callable

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from paramaterial.plug import DataItem, DataSet


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


@dataclass
class Styler:
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
    markers: List[str] = field(default_factory=lambda: ['o', 's', 'v', '^'])
    color_vals: Optional[np.ndarray] = None
    linestyle_vals: Optional[List[Any]] = None
    marker_vals: Optional[List[Any]] = None
    plot_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())

    def style_to(self, ds: DataSet):
        """Style the dataset based on the styler attributes."""
        if self.color_by is not None:
            self.color_vals = ds.info_table[self.color_by].unique()
            if self.color_norm is None:
                self.color_norm = plt.Normalize(ds.info_table[self.color_by].min(), ds.info_table[self.color_by].max())

        if self.linestyle_by is not None:
            self.linestyle_vals = ds.info_table[self.linestyle_by].unique().tolist()
            while len(self.linestyles) < len(self.linestyle_vals):
                self.linestyles.extend(self.linestyles)

        if self.marker_by is not None:
            self.marker_vals = ds.info_table[self.marker_by].unique().tolist()
            while len(self.markers) < len(self.marker_vals):
                self.markers.extend(self.markers)

        return self

    def curve_formatters(self, di: DataItem) -> Dict[str, Any]:
        """Return the curve formatters for the data item."""
        configure_plt_formatting()
        formatters = dict()

        if self.color_by is not None:
            formatters['color'] = plt.get_cmap(self.cmap)(self.color_norm(di.info[self.color_by]))
            formatters['zorder'] = di.info[self.color_by]
        else:
            formatters['color'] = plt.gca()._get_lines.get_next_color()

        if self.linestyle_by is not None:
            formatters['linestyle'] = self.linestyles[self.linestyle_vals.index(di.info[self.linestyle_by])]
        else:
            formatters['linestyle'] = None

        if self.marker_by is not None:
            formatters['marker'] = self.markers[self.marker_vals.index(di.info[self.marker_by])]
        else:
            formatters['marker'] = None

        return {k: v for k, v in formatters.items() if v is not None}

    def legend_handles(self, ds: Optional[DataSet] = None) -> List[mpatches.Patch]:
        """Return the legend handles."""
        handles = list()

        if ds is not None:
            styler_copy = copy.deepcopy(self)
            styler_copy.style_to(ds)
            color_vals = styler_copy.color_vals
            linestyle_vals = styler_copy.linestyle_vals
            marker_vals = styler_copy.marker_vals
        else:
            color_vals = self.color_vals
            linestyle_vals = self.linestyle_vals
            marker_vals = self.marker_vals

        if self.color_by_label is not None:
            handles.append(mpatches.Patch(label=self.color_by_label, alpha=0))

        if self.color_by is not None:
            for color_val in color_vals:
                handles.append(Line2D([], [], label=color_val, marker='o', linestyle='',
                                      color=plt.get_cmap(self.cmap)(self.color_norm(color_val))))

        if self.linestyle_by_label is not None:
            handles.append(mpatches.Patch(label=self.linestyle_by_label, alpha=0))

        if self.linestyle_by is not None:
            for linestyle_val in linestyle_vals:
                handles.append(Line2D([], [], color='black', label=linestyle_val,
                                      linestyle=self.linestyles[linestyle_vals.index(linestyle_val)]))

        if self.marker_by_label is not None:
            handles.append(mpatches.Patch(label=self.marker_by_label, alpha=0))

        if self.marker_by is not None:
            for marker_val in marker_vals:
                handles.append(Line2D([], [], color='black', label=marker_val, linestyle='None', mfc='none',
                                      marker=self.markers[marker_vals.index(marker_val)]))

        return handles


def dataset_plot(
        ds: DataSet,
        styler: Optional[Styler] = None,
        ax: Optional[plt.Axes] = None,
        fill_between: Optional[Tuple[str, str]] = None,
        styler_legend: bool = True,
        **kwargs
) -> plt.Axes:
    """Make a single plot from the dataframe of every item in the dataset."""
    if ax is None:
        fig, (ax) = plt.subplots(1, 1, figsize=kwargs.get('figsize', (10, 6)))
    kwargs['ax'] = ax

    if ax.get_legend() is not None and styler_legend:
        ax.get_legend().remove()

    if styler is None:
        styler = Styler()

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
    if len(handles) > 0 and styler_legend:
        ax.legend(handles=handles, loc='best', frameon=True)

    # colorbar
    if styler.cbar and styler_legend:
        sm = plt.cm.ScalarMappable(cmap=styler.cmap, norm=styler.color_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=kwargs['ax'], fraction=0.046, pad=0.04)
        cbar.set_label(styler.cbar_label) if styler.cbar_label is not None else None
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')

    return ax


def dataset_subplots(
        dataset,
        shape: Tuple[int, int],
        rows_by: str,
        cols_by: str,
        row_vals: List[List[Any]],
        col_vals: List[List[Any]],
        styler: Optional[Styler] = None,
        axs: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (12, 8),
        sharex: str = 'col',
        sharey: str = 'row',
        wspace: float = 0.05,
        hspace: float = 0.05,
        row_titles: Optional[List[str]] = None,
        col_titles: Optional[List[str]] = None,
        plot_titles: Optional[List[str]] = None,
        subplot_legend: bool = True,
        subplot_cbar: bool = True,
        **kwargs
) -> plt.Axes:
    """Plot a dataset as a grid of subplots, split by the 'rows_by' and 'cols_by' columns in the info_table."""
    if axs is None:
        fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize, sharex=sharex, sharey=sharey)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    if axs.ndim == 1:
        axs = np.array([axs])

    if row_titles is not None:
        for ax, row_title in zip(axs[:, 0], row_titles):
            ax.annotate(row_title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)

    if col_titles is not None:
        for ax, column_title in zip(axs[0, :], col_titles):
            ax.set_title(column_title)

    if plot_titles is not None:
        for ax, subplot_title in zip(axs.flat, plot_titles):
            ax.set_title(subplot_title)

    # loop through the grid of axes and plot the subsets
    for row, row_val in enumerate(row_vals):
        for col, col_val in enumerate(col_vals):
            ax = axs[row, col]
            subset = dataset[{cols_by: col_val, rows_by: row_val}]
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
        axs.flat[0].get_figure().legend(handles=styler.legend_handles(), loc='right', frameon=True)

    return axs



def subplot_wrapper(
        dataset: DataSet,
        plot_func: Callable[[DataSet, ...], plt.Axes],
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
    if axs is None:
        fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize, sharex=sharex, sharey=sharey)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    if axs.ndim == 1:
        axs = np.array([axs])

    # loop through the grid of axes and plot the subsets
    for row, row_val in enumerate(row_vals):
        for col, col_val in enumerate(col_vals):
            subset = dataset[{cols_by: col_val, rows_by: row_val}]
            kwargs['ax'] = axs[row, col]
            plot_func(subset, **kwargs)

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