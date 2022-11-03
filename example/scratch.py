import numpy as np
from matplotlib import pyplot as plt

import paramaterial as pam
from paramaterial import DataItem, DataSet


processed_set = DataSet('data/02 processed data', 'info/02 processed info.xlsx').sort_by(['temperature', 'lot'])

styler = pam.plotting.Styler(color_by='temperature', cmap='plasma', color_by_label='(°C)', plot_kwargs={'grid': True})
styler.style_to(processed_set)


def ds_plot(ds: DataSet, **kwargs):
    return pam.plotting.dataset_plot(ds, styler=styler, **kwargs)



def ds_subplots(ds: DataSet, **kwargs):
    return pam.plotting.dataset_subplots(
        ds=ds, shape=(3, 3), sharex='all',
        styler=styler, hspace=0.2, plot_legend=False,
        rows_by='lot', row_vals=[[a] for a in 'ABCDEFGHI'],
        cols_by='lot', col_vals=[[a] for a in 'ABCDEFGHI'],
        plot_titles=[f'Lot {a}' for a in 'ABCDEFGHI'],
        **kwargs
    )


def trim_at_max_force(di: DataItem):
    """Quote: The maximum force is determined by the maximum force recorded during the test. The data is then
    trimmed to this point."""
    di.info['Max_Force_idx'] = di.data['Force(kN)'].idxmax()
    di.data = di.data[:di.info['Max_Force_idx']]
    return di


def trim_at_preload(di: DataItem):
    """Quote: The data is then trimmed to the preload point."""
    pre_load = 120/100
    di.data = di.data[di.data['Force(kN)'] >= pre_load]
    return di


def determine_upper_proportional_limit(di: DataItem) -> DataItem:
    """Quote: Starting from a preload (which was chosen to be [133.5 kN] to minimize errors in the strain measurements
    caused by factors such as initial grip alignment), the upper proportional limit (UPL) is determined by linear
    regression as the point that minimizes the residuals of the slope fit between that point and the preload."""
    pre_load = 120/100
    # trim data to pre-load
    df = di.data
    df = df[df['Force(kN)'] >= pre_load]
    ax = df.plot(x='Strain', y='Stress_MPa')
    # calculate slope
    x = df['Strain'].values
    y = df['Stress_MPa'].values

    # fit line between first point and every other point
    slopes = []
    for i in range(1, len(x)):
        slopes.append((y[i] - y[0]) / (x[i] - x[0]))
    slopes = np.hstack([np.mean(slopes), slopes])
    df['Slopes'] = slopes

    ax = df.plot(x='Strain', y='Slopes', ax=ax, secondary_y=True)
    # calculate residuals
    # calculate residuals
    residuals = y - (slopes*x)
    df['Residuals'] = residuals
    ax = df.plot(x='Strain', y='Residuals', ax=ax)
    # find minimum residual
    min_residual = min(residuals)
    # find index of minimum residual
    min_residual_index = np.where(residuals == min_residual)[0][0]
    # return strain at index
    di.info['UPL_Strain'] = df['Strain'].iloc[min_residual_index]
    di.info['UPL_Stress_MPa'] = df['Stress_MPa'].iloc[min_residual_index]
    # trim data to UPL
    di.data = di.data[di.data['Strain'] <= di.info['UPL_Strain']]
    plt.show()
    return di


def determine_lower_proportional_limit(di: DataItem):
    """Quote: The lower proportional limit (LPL) is determined by linear regression as the point that minimizes the
    residuals of the slope fit between that point and the upper proportional limit (UPL)."""
    # trim data to UPL
    df = di.data
    df = df[df['Strain'] <= di.info['UPL_Strain']]
    # calculate slope
    x = df['Strain'].values
    y = df['Stress_MPa'].values
    slope = (y[-1] - y[0]) / (x[-1] - x[0])
    # calculate residuals
    residuals = y - (slope * x)
    # find minimum residual
    min_residual = min(residuals)
    # find index of minimum residual
    min_residual_index = np.where(residuals == min_residual)[0][0]
    # return strain at index
    di.info['LPL_Strain'] = df['Strain'].iloc[min_residual_index]
    di.info['LPL_Stress_MPa'] = df['Stress_MPa'].iloc[min_residual_index]
    return di


if __name__ == '__main__':
    ds = DataSet('data/02 processed data', 'info/02 processed info.xlsx')
    ds = ds.apply(trim_at_max_force)
    ax = ds_plot(ds, x='Strain', y='Stress_MPa', alpha=0.2)
    ds = ds.apply(trim_at_preload)
    ds = ds.apply(determine_upper_proportional_limit)
    ds_plot(ds, x='Strain', y='Stress_MPa', ax=ax)
    # ds = ds.apply(determine_lower_proportional_limit)
    plt.show()