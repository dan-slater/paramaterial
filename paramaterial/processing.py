"""Module with functions for modelling a stress-strain curve."""
import os
from typing import List

import numpy as np
import pandas as pd

from paramaterial.plug import DataItem, DataSet


def determine_proportional_limits_and_elastic_modulus(
        di: DataItem, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa',
        preload: float = 0, preload_key: str = 'Stress_MPa', max_strain: float = 0.02,
        suppress_numpy_warnings: bool = False) -> DataItem:
    """Determine the upper proportional limit (UPL) and lower proportional limit (LPL) of a stress-strain curve.
    The UPL is the point that minimizes the residuals of the slope fit between that point and the specified preload.
    The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL.
    The elastic modulus is the slope between the UPL and LPL."""
    if suppress_numpy_warnings:
        np.seterr(all="ignore")

    data = di.data[di.data[strain_key] <= max_strain]

    UPL = (0, 0)
    LPL = (0, 0)

    def fit_line(_x, _y):
        n = len(_x)  # number of points
        m = (n*np.sum(_x*_y) - np.sum(_x)*np.sum(_y))/(n*np.sum(np.square(_x)) - np.square(np.sum(_x)))  # slope
        c = (np.sum(_y) - m*np.sum(_x))/n  # intercept
        S_xy = (n*np.sum(_x*_y) - np.sum(_x)*np.sum(_y))/(n - 1)  # empirical covariance
        S_x = np.sqrt((n*np.sum(np.square(_x)) - np.square(np.sum(_x)))/(n - 1))  # x standard deviation
        S_y = np.sqrt((n*np.sum(np.square(_y)) - np.square(np.sum(_y)))/(n - 1))  # y standard deviation
        r = S_xy/(S_x*S_y)  # correlation coefficient
        S_m = np.sqrt((1 - r**2)/(n - 2))*S_y/S_x  # slope standard deviation
        S_rel = S_m/m  # relative deviation of slope
        return S_rel

    x = data[strain_key].values
    y = data[stress_key].values

    x_upl = x[data[preload_key] >= preload]
    y_upl = y[data[preload_key] >= preload]

    S_min = np.inf
    for i in range(3, len(x)):
        S_rel = fit_line(x_upl[:i], y_upl[:i])  # fit a line to the first i points after the preload
        if S_rel < S_min:
            S_min = S_rel
            UPL = (x_upl[i], y_upl[i])

    x_lpl = x[x <= UPL[0]]
    y_lpl = y[x <= UPL[0]]

    S_min = np.inf
    for j in range(len(x), 3, -1):
        S_rel = fit_line(x_lpl[j:], y_lpl[j:])  # fit a line to the last i points before the UPL
        if S_rel < S_min:
            S_min = S_rel
            LPL = (x_lpl[j], y_lpl[j])

    di.info['UPL_0'] = UPL[0]
    di.info['UPL_1'] = UPL[1]
    di.info['LPL_0'] = LPL[0]
    di.info['LPL_1'] = LPL[1]
    di.info['E'] = (UPL[1] - LPL[1])/(UPL[0] - LPL[0])
    return di


def find_proof_stress(di: DataItem, proof_strain: float = 0.002, strain_key: str = 'Strain',
                      stress_key: str = 'Stress_MPa') -> DataItem:
    """Find the proof stress of a stress-strain curve."""
    E = di.info['E']
    x_data = di.data[strain_key].values
    y_data = di.data[stress_key].values
    x_shift = proof_strain
    y_line = E*(x_data - x_shift)
    cut = np.where(np.diff(np.sign(y_line - y_data)) != 0)[0][-1]
    m = (y_data[cut + 1] - y_data[cut])/(x_data[cut + 1] - x_data[cut])
    xl = x_data[cut]
    yl = y_line[cut]
    xd = x_data[cut]
    yd = y_data[cut]
    K = np.array(
        [[1, -E],
         [1, -m]]
    )
    f = np.array(
        [[yl - E*xl],
         [yd - m*xd]]
    )
    d = np.linalg.solve(K, f).flatten()
    di.info[f'YP_{proof_strain}_0'] = d[1]
    di.info[f'YP_{proof_strain}_1'] = d[0]
    return di


def calculate_statistics(ds: DataSet) -> DataSet:
    """Calculate statistics for dataset."""
    pass


def make_representative_data(
        ds: DataSet, data_dir: str, info_path: str,
        repr_col: str, repr_by_cols: List[str],
        interp_by: str, interp_res: int = 200, min_interp_val: float = 0., interp_end: str = 'max_all',
        group_info_cols: List[str]|None = None
):
    """Make representative curves of the dataset and save them to a directory.
    Args:

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # make subset filters from combinations of unique values in repr_by_cols

    # val_counts = ds.info_table.groupby(repr_by_cols).size().unstack(repr_by_cols)
    val_counts_2 = ds.info_table.value_counts(repr_by_cols)

    # todo: replace with value counts
    subset_filters = []
    value_lists = [ds.info_table[col].unique() for col in repr_by_cols]
    for i in range(len(value_lists[0])):
        subset_filters.append({repr_by_cols[0]: [value_lists[0][i]]})
    for i in range(1, len(repr_by_cols)):
        new_filters = []
        for fltr in subset_filters:
            for value in value_lists[i]:
                new_filter = fltr.copy()
                new_filter[repr_by_cols[i]] = [value]
                new_filters.append(new_filter)
        subset_filters = new_filters

    # make list of repr_ids and initialise info table for the representative data
    repr_ids = [f'repr_id_{i + 1:0>4}' for i in range(len(subset_filters))]
    repr_info_table = pd.DataFrame(columns=['repr id'] + repr_by_cols)

    # make representative curves and take means of info table columns
    for repr_id, subset_filter in zip(repr_ids, subset_filters):
        # get representative subset
        repr_subset = ds[subset_filter]
        if repr_subset.info_table.empty:
            continue
        # add row to repr_info_table
        repr_info_table = pd.concat(
            [repr_info_table, pd.DataFrame({'repr id': [repr_id], **subset_filter, 'nr averaged': [len(repr_subset)]})])

        # add means of group info columns to repr_info_table
        if group_info_cols is not None:
            for col in group_info_cols:
                repr_info_table.loc[repr_info_table['repr id'] == repr_id, 'mean_' + col] \
                    = repr_subset.info_table[col].mean()

        # find minimum of maximum interp_by vals in subset
        if interp_end == 'max_all':
            max_interp_val = max([max(dataitem.data[interp_by]) for dataitem in repr_subset])
        elif interp_end == 'min_of_maxes':
            max_interp_val = min([max(dataitem.data[interp_by]) for dataitem in repr_subset])
        else:
            raise ValueError(f'interp_end must be "max_all" or "min_of_maxes", not {interp_end}')
        # make monotonically increasing vector to interpolate by
        interp_vec = np.linspace(min_interp_val, max_interp_val, interp_res)

        # make interpolated data for averaging, staring at origin
        interp_data = pd.DataFrame(data={interp_by: interp_vec})

        for n, dataitem in enumerate(repr_subset):
            # drop columns and rows outside interp range
            data = dataitem.data[[interp_by, repr_col]].reset_index(drop=True)
            data = data[(data[interp_by] <= max_interp_val)&(data[interp_by] >= min_interp_val)]
            # interpolate the repr_by column and add to interp_data
            # add 0 to start of data to ensure interpolation starts at origin
            interp_data[f'interp_{repr_col}_{n}'] = np.interp(interp_vec, [0] + data[interp_by].tolist(),
                                                              [0] + data[repr_col].tolist())

        # make representative data from stats of interpolated data
        interp_data = interp_data.drop(columns=[interp_by])
        repr_data = pd.DataFrame({f'interp_{interp_by}': interp_vec})
        repr_data[f'mean_{repr_col}'] = interp_data.mean(axis=1)
        repr_data[f'std_{repr_col}'] = interp_data.std(axis=1)
        repr_data[f'up_std_{repr_col}'] = repr_data[f'mean_{repr_col}'] + repr_data[f'std_{repr_col}']
        repr_data[f'down_std_{repr_col}'] = repr_data[f'mean_{repr_col}'] - repr_data[f'std_{repr_col}']
        repr_data[f'up_2std_{repr_col}'] = repr_data[f'mean_{repr_col}'] + 2*repr_data[f'std_{repr_col}']
        repr_data[f'down_2std_{repr_col}'] = repr_data[f'mean_{repr_col}'] - 2*repr_data[f'std_{repr_col}']
        repr_data[f'up_3std_{repr_col}'] = repr_data[f'mean_{repr_col}'] + 3*repr_data[f'std_{repr_col}']
        repr_data[f'down_3std_{repr_col}'] = repr_data[f'mean_{repr_col}'] - 3*repr_data[f'std_{repr_col}']
        repr_data[f'min_{repr_col}'] = interp_data.min(axis=1)
        repr_data[f'max_{repr_col}'] = interp_data.max(axis=1)
        repr_data[f'q1_{repr_col}'] = interp_data.quantile(0.25, axis=1)
        repr_data[f'q3_{repr_col}'] = interp_data.quantile(0.75, axis=1)

        # make representative info from subset info

        # write the representative data and info
        repr_data.to_csv(os.path.join(data_dir, f'{repr_id}.csv'), index=False)
        repr_info_table.to_excel(info_path, index=False)


def correct_friction(
        di: DataItem,
        mu_key: str = 'mu',
        h_0_key: str = 'h_0',
        D_0_key: str = 'D_0',
        disp_key: str = 'Disp(mm)',
        F_key: str = 'Force(kN)',
) -> DataItem:
    """

    """
    mu = di.info[mu_key] # friction coefficient
    h_0 = di.info[h_0_key]  # initial height in axial direction
    D_0 = di.info[D_0_key]  # initial diameter
    h = h_0 - di.data[disp_key]  # instantaneous height
    d = D_0*np.sqrt(h_0/h)  # instantaneous diameter
    P = di.data[F_key]*1000*4/(np.pi*d**2)  # pressure (MPa)
    di.data['Pressure(MPa)'] = P
    di.data['Corrected_Stress(MPa)'] = P/(1 + (mu*d)/(3*h))  # correct stress
    return di
