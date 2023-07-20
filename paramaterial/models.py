from collections.abc import Callable
from functools import wraps

import numpy as np
import scipy.optimize as op





def iso_return_map(yield_stress_func: Callable, return_vec: str = 'stress'):
    """Wrapper for a yield function that describes the plastic behaviour.

    Args:
        yield_stress_func: Yield stress function.
        return_vec: Return vector. Must be one of 'stress', 'plastic strain', 'accumulated plastic strain'.

    Returns: A function that gives the return_vec (usually stress) as a function of strain.
    """

    @wraps(yield_stress_func)
    def wrapper(
            x: np.ndarray,
            mat_params
    ):
        y = np.zeros(x.shape)  # predicted stress
        x_p = np.zeros(x.shape)  # plastic strain
        aps = np.zeros(x.shape)  # accumulated plastic strain
        y_yield: callable = yield_stress_func(mat_params)  # yield stress
        E = mat_params[0]  # elastic modulus

        if not np.isclose(x[0], 0):
            y_trial_0 = E*(x[1])
            f_trial_0 = np.abs(y_trial_0) - y_yield(0)
            if f_trial_0 <= 0:
                y[0] = E*x[0]
            else:
                d_aps = op.root(lambda d: f_trial_0 - d*E - y_yield(d) + y_yield(0), 0).x[0]
                y[0] = y_trial_0*(1 - d_aps*E/np.abs(y_trial_0))

        for i in range(len(x) - 1):
            y_trial = E*(x[i + 1] - x_p[i])
            f_trial = np.abs(y_trial) - y_yield(aps[i])
            if f_trial <= 0:
                y[i + 1] = y_trial
                x_p[i + 1] = x_p[i]
                aps[i + 1] = aps[i]
            else:
                d_aps = op.root(
                    lambda d: f_trial - d*E - y_yield(aps[i] + d) + y_yield(aps[i]),
                    aps[i]
                ).x[0]
                y[i + 1] = y_trial*(1 - d_aps*E/np.abs(y_trial))
                x_p[i + 1] = x_p[i] + np.sign(y_trial)*d_aps
                aps[i + 1] = aps[i] + d_aps

        if return_vec == 'stress':
            return y
        elif return_vec == 'plastic strain':
            return x_p
        elif return_vec == 'accumulated plastic strain':
            return aps
        else:
            return None

    return wrapper


@iso_return_map
def perfect(mat_params):
    """Perfect plasticity yield function, no hardening."""
    E, s_y = mat_params
    return lambda a: s_y


@iso_return_map
def linear(mat_params):
    """Linear isotropic hardening yield function."""
    E, s_y, K = mat_params
    return lambda a: s_y + K*a


@iso_return_map
def quadratic(mat_params):
    """Quadratic isotropic hardening yield function."""
    E, s_y, Q = mat_params
    return lambda a: s_y + E*(a - Q*a**2)


@iso_return_map
def voce(mat_params):
    """Exponential isotropic hardening yield function."""
    E, s_y, s_u, d = mat_params
    return lambda a: s_y + (s_u - s_y)*(1 - np.exp(-d*a))


@iso_return_map
def ramberg(mat_params):
    """Ramberg-Osgood isotropic hardening yield function."""
    E, s_y, C, n = mat_params
    return lambda a: s_y + C*(np.sign(a)*(np.abs(a))**n)


from typing import Dict, Any, Tuple, List, Union, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

import tqdm

from paramaterial.plug import DataItem, DataSet
from paramaterial.plotting import Styler, configure_plt_formatting

class ModelSet:
    """Class that acts as model DataSet.

    Args:
        model_func: The model function to be used for fitting.
        param_names: The names of the parameters of the model function.
        var_names: The names of the variables of the model function.
        bounds: The bounds for the parameters of the model function.
        initial_guess: The initial guess for the parameters of the model function.
        scipy_func: The scipy function to be used for fitting.
        scipy_kwargs: The kwargs for the scipy function.
    """

    def __init__(self, model_func: Callable[[np.ndarray, List[float]], np.ndarray], param_names: List[str],
                 var_names: Optional[List[str]] = None, bounds: Optional[List[Tuple[float, float]]] = None,
                 initial_guess: Optional[np.ndarray] = None, scipy_func: str = 'minimize',
                 scipy_kwargs: Optional[Dict[str, Any]] = None, ):
        self.model_func = model_func
        self.params_table = pd.DataFrame(columns=['model_id'] + param_names)
        self.results_dict_list = []
        self.param_names = param_names
        self.var_names = var_names
        self.bounds = bounds
        self.initial_guess = initial_guess if initial_guess is not None else [0.0]*len(param_names)
        self.scipy_func = scipy_func
        self.scipy_kwargs = scipy_kwargs if scipy_kwargs is not None else {}
        self.sample_size: Optional[int] = None
        self.fitted_ds: Optional[DataSet] = None
        self.x_key: Optional[str] = None
        self.y_key: Optional[str] = None
        self.model_items: Optional[List] = None
        self.predicted_ds: Optional[DataSet] = None

    # @staticmethod
    # def from_info_table(info_table: pd.DataFrame, model_func: Callable[[np.ndarray, List[float]], np.ndarray],
    #                     param_names: List[str], model_id_key: str = 'model_id') -> 'ModelSet':
    #     """Create a ModelSet from an info table."""
    #     model_ids = info_table[model_id_key].values
    #     info_rows = [info_table.drop(columns=param_names).iloc[i] for i in range(len(info_table))]
    #     params_lists = [info_table[param_names].iloc[i].values for i in range(len(info_table))]
    #     model_funcs = [model_func for _ in range(len(info_table))]
    #     x
    #     x_mins = [info_table['x_min'].iloc[i] for i in range(len(info_table))]
    #     x_maxs = [info_table['x_max'].iloc[i] for i in range(len(info_table))]
    #     resolutions = [info_table['resolution'].iloc[i] for i in range(len(info_table))]
    #     ms = ModelSet(model_func, param_names)
    #     ms.model_items = list(
    #         map(ModelItem, model_ids, info_rows, params_lists, model_funcs, x_keys, y_keys, x_mins, x_maxs,
    #             resolutions))
    #     return ms

    # def fit_to(self, ds: DataSet, x_key: str, y_key: str, sample_size: int = 50) -> None:
    #     """Fit the model to the DataSet.
    #
    #     Args:
    #         ds: DataSet to fit the model to.
    #         x_key: Key of the x values in the DataSet.
    #         y_key: Key of the y values in the DataSet.
    #         sample_size: Number of samples to draw from the x-y data in the DataSet.
    #
    #     Returns: None
    #     """
    #     self.fitted_ds = ds
    #     self.x_key = x_key
    #     self.y_key = y_key
    #     self.sample_size = sample_size
    #     for _ in tqdm(map(self._fit_item, ds.data_items), unit='fits', leave=False):
    #         pass
    #     self.model_items = list(map(ModelItem.from_results_dict, self.results_dict_list))
    #
    # def predict(self, resolution: int = 50, xmin=None, xmax=None, info_table=None) -> DataSet:
    #     """Return a ds with generated data with optimised model parameters added to the info table.
    #     If an info table is provided, data items will be generated to match the rows of the info table, using the
    #     var_names and param_names and model_func.
    #
    #     Args:
    #         resolution: Number of points to generate between the x_min and x_max.
    #
    #     Returns: DataSet with generated data.
    #     """
    #     predict_ds = DataSet()
    #
    #     if info_table is not None:
    #         predict_ds.info_table = info_table
    #         for i, row in info_table.iterrows():
    #             # make a data item for each row in the info table
    #             x_data = np.linspace(row['x_min'], row['x_max'], resolution)
    #             y_data = self.model_func(x_data, row[self.param_names].values)
    #             data = {self.x_key: x_data, self.y_key: y_data}
    #             info = row
    #             test_id = row['test_id']
    #             di = DataItem()
    #
    #
    #     # predict_ds.test_id_key = 'model_id'
    #
    #     def update_resolution(mi: ModelItem):
    #         mi.resolution = resolution
    #         mi.info['resolution'] = resolution
    #         mi.info['x_min'] = xmin if xmin is not None else mi.info['x_min']
    #         mi.info['x_max'] = xmax if xmax is not None else mi.info['x_max']
    #         return mi
    #
    #     self.model_items = list(map(lambda mi: update_resolution(mi), self.model_items))
    #     predict_ds.data_items = copy.deepcopy(self.model_items)
    #     for di in predict_ds.data_items:
    #         di.info['test_id'] = di.info['model_id']
    #     predict_ds.info_table = pd.DataFrame([di.info for di in predict_ds.data_items])
    #
    #     return predict_ds

    def _objective_function(self, params: List[float], di: DataItem) -> float:
        data = di.data[di.data[self.x_key] > 0]
        if self.var_names is not None:
            variables = [di.info[var_name] for var_name in self.var_names]
            params = np.hstack([variables, params])
        x_data = data[self.x_key].values
        y_data = data[self.y_key].values
        sampling_stride = int(len(x_data)/self.sample_size)
        if sampling_stride < 1:
            sampling_stride = 1
        x_data = x_data[::sampling_stride]
        y_data = y_data[::sampling_stride]
        y_model = self.model_func(x_data, params)
        # return max((y_data - y_model)/np.sqrt(len(y_data)))
        return np.linalg.norm((y_data - y_model)/np.sqrt(len(y_data)))**2

    def _fit_item(self, di: DataItem) -> None:
        if self.scipy_func == 'minimize':
            result = op.minimize(self._objective_function, self.initial_guess, args=(di,), bounds=self.bounds,
                                 **self.scipy_kwargs)
        elif self.scipy_func == 'differential_evolution':
            result = op.differential_evolution(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'basinhopping':
            result = op.basinhopping(self._objective_function, self.initial_guess,
                                     minimizer_kwargs=dict(args=(di,), bounds=self.bounds, **self.scipy_kwargs))
        elif self.scipy_func == 'dual_annealing':
            result = op.dual_annealing(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'shgo':
            result = op.shgo(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        elif self.scipy_func == 'brute':
            result = op.brute(self._objective_function, self.bounds, args=(di,), **self.scipy_kwargs)
        else:
            raise ValueError(f'Invalid scipy_func: {self.scipy_func}\nMust be one of:'
                             f' minimize, differential_evolution, basinhopping, dual_annealing, shgo, brute')
        model_id = 'model_' + str(di.info[0])
        results_dict = {'model_id': model_id, 'info': di.info, 'params': result.x, 'param_names': self.param_names,
                        'error': result.fun,
                        'variables': [di.info[var_name] for var_name in
                                      self.var_names] if self.var_names is not None else None,
                        'variable_names': self.var_names, 'model_func': self.model_func,
                        'x_key': self.x_key, 'y_key': self.y_key,
                        'x_min': di.data[self.x_key].min(),
                        'x_max': di.data[self.x_key].max(), }
        self.results_dict_list.append(results_dict)
        self.params_table = pd.concat(
            [self.params_table, pd.DataFrame([[model_id] + list(result.x) + [result.fun]],
                                             columns=['model_id'] + self.param_names + ['fitting error'])])

    @property
    def fitting_results(self) -> pd.DataFrame:
        """Return a DataFrame with the results of the fitting."""
        # get the fitted parameters and make a table with them and include only the relevant info and the fitting error
        return self.params_table.merge(self.fitted_ds.info_table, on='model_id')


def apply_ZH_regression(ds: DataSet, flow_stress_key: str = 'flow_stress_MPa', ZH_key: str = 'ZH_parameter',
                        group_by: Union[str, List[str]] = None) -> DataSet:
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
                       group_by: Union[str, List[str]] = None, color_by: str = None, marker_by: str = None,
                       linestyle_by: str = None,
                       scatter_kwargs: Dict[str, Any] = None, fit_kwargs: Dict[str, Any] = None, eq_hscale=0.1):
    """Plot the Zener-Holloman regression of the flow stress vs. temperature."""
    # configure_plt_formatting()
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
        ax.legend(handles=handles, loc='best', handletextpad=0.05, markerfirst=False)  # , labelspacing=0.1)
        ax.get_legend().set_zorder(2000)

    ax.set_xlabel('lnZ')
    ax.set_ylabel('Flow Stress (MPa)')
    ax.set_title('Zener-Holloman Regression')

    # add annotation to bottom right of axes with the regression equation
    heights = reversed([0.05 + eq_hscale*i for i in range(len(groups))])
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
                             group_by: Union[str, List[str]] = None) -> pd.DataFrame:
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
                        **kwargs) -> Union[pd.DataFrame,plt.Axes]:
    if isinstance(index, str):
        index = [index]
    if isinstance(columns, str):
        columns = [columns]

    def calculate_quality(df):
        df['quality'] = 100 - df['lnZ_fit_residual'].abs().sum()/(df['lnZ_fit_residual'].count()*df[flow_stress_key].mean())*100
        return df

    quality_matrix = info_table.groupby(index + columns, group_keys=False).apply(calculate_quality).groupby(index + columns, group_keys=False)[
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


