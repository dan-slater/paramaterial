from typing import Dict, Any, Tuple

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from paramaterial import DataItem, DataSet, Styler


def apply_ZH_regression(ds: DataSet, flow_stress_key: str = 'flow_stress_MPa', ZH_key: str = 'ZH_parameter'):
    """Do a linear regression for LnZ vs flow stress. #todo link

    Args:
        ds: DataSet to be fitted.
        flow_stress_key: Info key for the flow stress value.
        ZH_key: Info key for the ZH parameter value.

    Returns:
        The DataSet with the Zener-Holloman parameter and regression parameters added to the info table.
    """
    assert flow_stress_key in ds.info_table.columns, f'flow_stress_key {flow_stress_key} not in info table'
    info_table = ds.info_table.copy()
    info_table['lnZ'] = np.log(info_table[ZH_key].values.astype(np.float64))
    m, c = curve_fit(lambda x, m, c: m*x + c, info_table['lnZ'], info_table[flow_stress_key])[0]
    info_table['lnZ_fit_m'] = m
    info_table['lnZ_fit_c'] = c
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


def plot_ZH_regression(
        ds: DataSet,
        flow_stress_key: str = 'flow_stress_MPa',
        calculate: bool = True,
        figsize: Tuple[float, float] = (6, 4),
        ax: plt.Axes|None = None,
        scatter_kwargs: Dict[str, Any]|None = None,
        fit_kwargs: Dict[str, Any]|None = None,
        cmap: str = 'plasma',
        styler: Styler|None = None,
        group_by: str|None = None,
        color_by: str|None = None,
        marker_by: str|None = None,
        linestyle_by: str|None = None,
):
    """Plot the Zener-Holloman regression of the flow stress vs. temperature."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if styler is None:
        styler = Styler(color_by=color_by, color_by_label=color_by, cmap=cmap,
                        marker_by=marker_by, marker_by_label=marker_by,
                        linestyle_by=linestyle_by, linestyle_by_label=linestyle_by
                        ).style_to(ds)

    # Calculate ZH parameter
    if calculate:
        ds = ds.apply(calculate_ZH_parameter)

    # make a scatter plot of lnZ vs flow stress using the styler
    for di in ds:
        info = di.info
        updated_scatter_kwargs = styler.curve_formatters(di)
        updated_scatter_kwargs.update(scatter_kwargs) if scatter_kwargs is not None else None
        updated_scatter_kwargs.pop('linestyle') if 'linestyle' in updated_scatter_kwargs else None
        updated_scatter_kwargs.update({'color': 'k'}) if color_by is None else None
        ax.scatter(np.log(info['ZH_parameter']), info[flow_stress_key], **updated_scatter_kwargs)

    ax.set_prop_cycle(None)  # reset ax color cycle

    # make a line plot of the regression for each group
    if group_by is not None:
        groups = []
        for group in ds.info_table[group_by].unique():
            group_ds = ds[{group_by: [group]}]
            group_ds = apply_ZH_regression(group_ds) if calculate else group_ds
            groups.append(group_ds)
    else:
        ds = apply_ZH_regression(ds) if calculate else ds
        groups = [ds]
    for ds in groups:
        x = np.linspace(ds.info_table['lnZ'].min(), ds.info_table['lnZ'].max(), 10)
        di = ds[0]
        y = di.info['lnZ_fit_m']*x + di.info['lnZ_fit_c']
        updated_fit_kwargs = styler.curve_formatters(di)
        updated_fit_kwargs.pop('marker') if 'marker' in updated_fit_kwargs else None
        updated_fit_kwargs.update(fit_kwargs) if fit_kwargs is not None else None
        ax.plot(x, y, **updated_fit_kwargs)

    ax.set_xlabel('lnZ')
    ax.set_ylabel('Flow stress (MPa)')
    ax.legend()

    return ax
