from typing import Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

from paramaterial import DataItem, DataSet, Styler
from paramaterial.plotting import configure_plt_formatting


def apply_ZH_regression(ds: DataSet, flow_stress_key: str = 'flow_stress_MPa', ZH_key: str = 'ZH_parameter',
                        group_by: str|List[str] = None) -> DataSet:
    """Do a linear regression for LnZ vs flow stress. #todo link

    Args:
        ds: DataSet to be fitted.
        flow_stress_key: Info key for the flow stress value.
        ZH_key: Info key for the ZH parameter value.
        group_by: Info key(s) to group by.

    Returns:
        The DataSet with the Zener-Holloman parameter and regression parameters added to the info table.
    """
    assert flow_stress_key in ds.info_table.columns, f'flow_stress_key {flow_stress_key} not in info table'

    # make dataset filters for unique combinations of group_by keys
    if group_by is not None:
        if isinstance(group_by, str):
            group_by = [group_by]
        subset_filters = []
        value_lists = [ds.info_table[col].unique() for col in group_by]
        for i in range(len(value_lists[0])):
            subset_filters.append({group_by[0]: [value_lists[0][i]]})
        for i in range(1, len(group_by)):
            new_filters = []
            for fltr in subset_filters:
                for value in value_lists[i]:
                    new_filter = fltr.copy()
                    new_filter[group_by[i]] = [value]
                    new_filters.append(new_filter)
            subset_filters = new_filters
        groups = []
        for fltr in subset_filters:
            group_ds = ds.subset(fltr)
            groups.append(group_ds)
    else:
        groups = [ds]

    # apply regression to each group
    for group_ds in groups:
        info_table = group_ds.info_table.copy()
        info_table['lnZ'] = np.log(info_table[ZH_key].values.astype(np.float64))
        result = curve_fit(lambda x, m, c: m*x + c, info_table['lnZ'], info_table[flow_stress_key])
        info_table['lnZ_fit_m'] = result[0][0]
        info_table['lnZ_fit_c'] = result[0][1]
        info_table['lnZ_fit'] = info_table['lnZ_fit_m']*info_table['lnZ'] + info_table['lnZ_fit_c']
        info_table['lnZ_fit_residual'] = info_table['lnZ_fit'] - info_table[flow_stress_key]
        info_table['lnZ_fit_r2'] = 1 - np.sum(info_table['lnZ_fit_residual']**2)/np.sum(
            (info_table[flow_stress_key] - np.mean(info_table[flow_stress_key]))**2)
        info_table['ZH_fit'] = np.exp(info_table['lnZ_fit'])
        info_table['ZH_fit_error'] = info_table['ZH_fit'] - info_table[ZH_key]
        info_table['ZH_fit_error_percent'] = info_table['ZH_fit_error']/info_table[ZH_key]
        group_ds.info_table = info_table

    group_info_tables = [group_ds.info_table for group_ds in groups]
    info_table = pd.concat(group_info_tables)
    ds.info_table = info_table
    return ds


def calculate_ZH_parameter(di: DataItem, temperature_key: str = 'temperature_K', rate_key: str = 'rate_s-1',
                           Q_key: str = 'Q_activation', gas_constant: float = 8.1345,
                           ZH_key: str = 'ZH_parameter') -> DataItem:
    """Calculate the Zener-Holloman parameter using

    $$
    Z = \\dot{\\varepsilon} \\exp \\left(\\frac{Q}{RT}\\right)
    $$

    where $\\dot{\\varepsilon}$ is the strain rate, $Q$ is the activation energy, $R$ is the gas constant,
    and $T$ is the temperature.

    Args:
        di: DataItem object with $\\dot{\\varepsilon}$, $Q$, $R$, and $T$ in info.
        temperature_key: Info key for mean temperature
        rate_key: Info key for mean strain-rate rate
        Q_key: Info key for activation energy
        gas_constant: Universal gas constant
        ZH_key: Key for Zener-Holloman parameter

    Returns: DataItem with Zener-Holloman parameter added to info.
    """
    di.info[ZH_key] = di.info[rate_key]*np.exp(di.info[Q_key]/(gas_constant*di.info[temperature_key]))
    return di


def plot_ZH_regression(ds: DataSet, flow_stress_key: str = 'flow_stress_MPa', rate_key: str = 'rate_s-1',
                       temperature_key: str = 'temperature_K', calculate: bool = True,
                       figsize: Tuple[float, float] = (6, 4),
                       ax: plt.Axes = None, cmap: str = 'plasma', styler: Styler = None, plot_legend: bool = True,
                       group_by: str|List[str] = None, color_by: str = None, marker_by: str = None,
                       linestyle_by: str = None,
                       scatter_kwargs: Dict[str, Any] = None, fit_kwargs: Dict[str, Any] = None, ):
    """Plot the Zener-Holloman regression of the flow stress vs. temperature."""
    configure_plt_formatting()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if styler is None:
        styler = Styler(color_by=color_by, color_by_label=color_by, cmap=cmap, marker_by=marker_by,
                        marker_by_label=marker_by, linestyle_by=linestyle_by, linestyle_by_label=linestyle_by
                        ).style_to(ds)

    # Calculate ZH parameter
    if calculate:
        ds = ds.apply(calculate_ZH_parameter, rate_key=rate_key, temperature_key=temperature_key)

    # make a scatter plot of lnZ vs flow stress using the styler
    for di in ds:
        updated_scatter_kwargs = styler.curve_formatters(di)
        updated_scatter_kwargs.update(scatter_kwargs) if scatter_kwargs is not None else None
        updated_scatter_kwargs.pop('linestyle') if 'linestyle' in updated_scatter_kwargs else None
        updated_scatter_kwargs.update({'color': 'k'}) if color_by is None else None
        ax.scatter(np.log(di.info['ZH_parameter']), di.info[flow_stress_key], **updated_scatter_kwargs)

    ax.set_prop_cycle(None)  # reset ax color cycle

    # make dataset filters for unique combinations of group_by keys
    if group_by is not None:
        if isinstance(group_by, str):
            group_by = [group_by]
        subset_filters = []
        value_lists = [ds.info_table[col].unique() for col in group_by]
        for i in range(len(value_lists[0])):
            subset_filters.append({group_by[0]: [value_lists[0][i]]})
        for i in range(1, len(group_by)):
            new_filters = []
            for fltr in subset_filters:
                for value in value_lists[i]:
                    new_filter = fltr.copy()
                    new_filter[group_by[i]] = [value]
                    new_filters.append(new_filter)
            subset_filters = new_filters
        groups = []
        for fltr in subset_filters:
            group_ds = ds.subset(fltr)
            group_ds = apply_ZH_regression(group_ds, flow_stress_key=flow_stress_key) if calculate else group_ds
            groups.append(group_ds)
    else:
        group_ds = apply_ZH_regression(ds, flow_stress_key=flow_stress_key) if calculate else ds
        groups = [group_ds]

    # plot the regression lines
    for group_ds in groups:
        x = np.linspace(group_ds.info_table['lnZ'].min(), group_ds.info_table['lnZ'].max(), 10)
        di = group_ds[0]
        y = di.info['lnZ_fit_m']*x + di.info['lnZ_fit_c']
        updated_fit_kwargs = styler.curve_formatters(di)
        updated_fit_kwargs.pop('marker') if 'marker' in updated_fit_kwargs else None
        updated_fit_kwargs.update(fit_kwargs) if fit_kwargs is not None else None
        ax.plot(x, y, **updated_fit_kwargs)

    # add the legend
    handles = styler.legend_handles(ds)
    if len(handles) > 0 and plot_legend:
        ax.legend(handles=handles, loc='best', frameon=True, markerfirst=False,
                  handletextpad=0.05)  # , labelspacing=0.1)
        ax.get_legend().set_zorder(2000)

    ax.set_xlabel('lnZ')
    ax.set_ylabel('Flow stress (MPa)')
    ax.set_title('Zener-Holloman Regression Plot')

    # add annotation to bottom right of axes with the regression equation
    if figsize == (6, 4):
        heights = reversed([0.05 + 0.1*i for i in range(len(groups))])
    else:
        heights = reversed([0.05 + 0.07*i for i in range(len(groups))])
    for group_ds, height in zip(groups, heights):
        di = group_ds[0]
        color = styler.color_dict[di.info[color_by]] if color_by is not None else 'k'
        info = di.info
        ax.text(0.95, height,
                f'y = {info["lnZ_fit_m"]:.2f}x + {info["lnZ_fit_c"]:.2f} | r$^2$ = {info["lnZ_fit_r2"]:.2f}',
                horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=dict(facecolor=color, alpha=0.2, edgecolor='none', boxstyle='round,pad=0.2'))

    return ax


def make_ZH_regression_table(ds: DataSet, flow_stress_key: str = 'flow_stress_MPa', rate_key: str = 'rate_s-1',
                             temperature_key: str = 'temperature_K', calculate: bool = True,
                             group_by: str|List[str] = None) -> pd.DataFrame:
    """Make a table of the Zener-Holloman regression parameters for each group."""
    if calculate:
        ds = ds.apply(calculate_ZH_parameter, rate_key=rate_key, temperature_key=temperature_key)
        ds = apply_ZH_regression(ds, flow_stress_key=flow_stress_key, group_by=group_by)
    table = ds.info_table[['ZH_fit_group', 'lnZ_fit_m', 'lnZ_fit_c', 'lnZ_fit_r2']]
    table.columns = ['Group', 'Slope', 'Intercept', 'R2']
    table = table.drop_duplicates().reset_index(drop=True)
    return table


def make_quality_matrix(info_table: pd.DataFrame, index: Union[str, List[str]], columns: Union[str, List[str]],
                        flow_stress_key: str = 'flow_stress_MPa', as_heatmap: bool = False, title: str = None,
                        xlabel: str = None, ylabel: str = None, tick_params: Dict = None,
                        **kwargs) -> pd.DataFrame|plt.Axes:
    if isinstance(index, str):
        index = [index]
    if isinstance(columns, str):
        columns = [columns]

    def calculate_quality(df):
        df['quality'] = df['lnZ_fit_residual'].abs().sum()/(df['lnZ_fit_residual'].count()*df[flow_stress_key].mean())
        return df

    quality_matrix = info_table.groupby(index + columns).apply(calculate_quality).groupby(index + columns)[
        'quality'].mean().unstack(columns).fillna(0)

    if not as_heatmap:
        return quality_matrix
    else:
        ax = sns.heatmap(quality_matrix, **kwargs)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if tick_params:
            ax.tick_params(**tick_params)
        return ax
