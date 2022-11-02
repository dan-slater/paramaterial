import numpy as np

from paramaterial import DataItem, DataSet


def determine_upper_proportional_limit(di: DataItem, pre_load: float = 0.0) -> float:
    """Quote: Starting from a preload (which was chosen to be [133.5 kN] to minimize errors in the strain measurements
    caused by factors such as initial grip alignment), the upper proportional limit (UPL) is determined by linear
    regression as the point that minimizes the residuals of the slope fit between that point and the preload."""
    # trim data to pre-load
    df = di.data
    df = df[df['Force(kN)'] >= pre_load/100]
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
    di.info['UPL_Strain'] = df['Strain'].iloc[min_residual_index]
    di.info['UPL_Stress_MPa'] = df['Stress_MPa'].iloc[min_residual_index]
    return x[min_residual_index]


if __name__ == '__main__':
    ds = DataSet('data/02 processed data', 'info/02 processed info.xlsx')
    print(determine_upper_proportional_limit(ds[0], pre_load=133.5))