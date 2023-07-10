"""
Functions for performing aggregation procedures, including making representative curves from data or finding
statistics for info.
"""
import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import pandas as pd
import scipy.optimize as op
from tqdm import tqdm

from paramaterial.plug import DataItem, DataSet


def make_representative_data(ds: DataSet, out_info_path: str, out_data_dir: str,
                             repres_col: str, interp_by: str, group_by_keys: List[str],
                             interp_res: int = 200, interp_domain: Union[str, Tuple[float, float]] = 'outer',
                             info_cols: Optional[List[str]] = None) -> None:
    """Make representative curves of the data in the dataset. 
    Write the representative curves to data files in `out_data_dir` and associated info_table to a spreadsheet at `out_info_path`.

    Args:
        ds: The dataset to make representative curves from.
        out_info_path: The path to save the info spreadsheet.
        out_data_dir: The directory to save the representative curves to.
        repres_col: The data column to aggregate for the y-axis of the representative curves.
        interp_by: The data column to interpolate for the x-axis of the representative curves.
        group_by_keys: The info categories to group by.
        interp_res: The resolution of the interpolation
        interp_domain:  Must be "outer", "inner" or a tuple of floats. Defines the domain on the x-axis to
            interpolate over. If "outer" or "inner", the domain is defined by the minimum and maximum values of the
            interpolation column in the representative subset.
        info_cols: The info categories to include in the aggregated info_table.

    Returns:
        None
    """
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    value_lists = [ds.info_table[col].unique() for col in group_by_keys]

    # make a dataset filter for each representative curve
    subset_filters = []
    for i in range(len(value_lists[0])):  # i
        subset_filters.append({group_by_keys[0]: value_lists[0][i]})
    for i in range(1, len(group_by_keys)):  # i
        new_filters = []
        for fltr in subset_filters:  # j
            for value in value_lists[i]:  # k
                new_filter = fltr.copy()
                new_filter[group_by_keys[i]] = value
                new_filters.append(new_filter)
        subset_filters = new_filters

    # make list of repres_ids and initialise info table for the representative data
    repres_ids = [f'repres_id_{i + 1:0>4}' for i in range(len(subset_filters))]
    repr_info_table = pd.DataFrame(columns=['repres_id'] + group_by_keys)

    # make representative curves and take means of info table columns
    for repres_id, subset_filter in zip(repres_ids, subset_filters):
        # get representative subset
        repres_subset = ds.subset(subset_filter)
        if repres_subset.info_table.empty:
            continue
        # add row to repr_info_table
        repr_info_table = pd.concat(
            [repr_info_table,
             pd.DataFrame({'repres_id': [repres_id], **subset_filter, 'nr averaged': [len(repres_subset)]})])

        # add means of group info columns to repr_info_table
        if info_cols is not None:
            for col in info_cols:
                df_col = repres_subset.info_table[col]
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, '' + col] = df_col.mean()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'std_' + col] = df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'upstd_' + col] = df_col.mean() + df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'downstd_' + col] = df_col.mean() - df_col.std()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'max_' + col] = df_col.max()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'min_' + col] = df_col.min()

        # find minimum of maximum interp_by vals in subset
        if interp_domain == 'outer':
            min_interp_val = min([min(dataitem.data[interp_by]) for dataitem in repres_subset])
            max_interp_val = max([max(dataitem.data[interp_by]) for dataitem in repres_subset])
        elif interp_domain == 'inner':
            min_interp_val = max([min(dataitem.data[interp_by]) for dataitem in repres_subset])
            max_interp_val = min([max(dataitem.data[interp_by]) for dataitem in repres_subset])
        elif type(interp_domain) == tuple:
            min_interp_val = interp_domain[0]
            max_interp_val = interp_domain[1]
        else:
            raise ValueError(f'interp_range must be "outer", "inner" or a tuple, not {interp_domain}')

        # make monotonically increasing vector to interpolate by
        interp_vec = np.linspace(min_interp_val, max_interp_val, interp_res)

        # make interpolated data for averaging, staring at origin
        interp_data = pd.DataFrame(data={interp_by: interp_vec})

        for n, dataitem in enumerate(repres_subset):
            # drop columns and rows outside interp range
            data = dataitem.data[[interp_by, repres_col]].reset_index(drop=True)
            data = data[(data[interp_by] <= max_interp_val) & (data[interp_by] >= min_interp_val)]
            # interpolate the repr_by column and add to interp_data
            # add 0 to start of data to ensure interpolation starts at origin
            interp_data[f'interp_{repres_col}_{n}'] = np.interp(interp_vec, data[interp_by].tolist(),
                                                                data[repres_col].tolist())

        # make representative data from stats of interpolated data
        interp_data = interp_data.drop(columns=[interp_by])
        repr_data = pd.DataFrame({f'{interp_by}': interp_vec})
        repr_data[f'{repres_col}'] = interp_data.mean(axis=1)
        repr_data[f'std_{repres_col}'] = interp_data.std(axis=1)
        repr_data[f'up_std_{repres_col}'] = repr_data[f'{repres_col}'] + repr_data[f'std_{repres_col}']
        repr_data[f'down_std_{repres_col}'] = repr_data[f'{repres_col}'] - repr_data[f'std_{repres_col}']
        repr_data[f'up_2std_{repres_col}'] = repr_data[f'{repres_col}'] + 2 * repr_data[f'std_{repres_col}']
        repr_data[f'down_2std_{repres_col}'] = repr_data[f'{repres_col}'] - 2 * repr_data[f'std_{repres_col}']
        repr_data[f'up_3std_{repres_col}'] = repr_data[f'{repres_col}'] + 3 * repr_data[f'std_{repres_col}']
        repr_data[f'down_3std_{repres_col}'] = repr_data[f'{repres_col}'] - 3 * repr_data[f'std_{repres_col}']
        repr_data[f'min_{repres_col}'] = interp_data.min(axis=1)
        repr_data[f'max_{repres_col}'] = interp_data.max(axis=1)
        repr_data[f'q1_{repres_col}'] = interp_data.quantile(0.25, axis=1)
        repr_data[f'q3_{repres_col}'] = interp_data.quantile(0.75, axis=1)

        # write the representative data and info
        repr_data.to_csv(os.path.join(out_data_dir, f'{repres_id}.csv'), index=False)
        repr_info_table.to_excel(out_info_path, index=False)


def make_representative_info(ds: DataSet, group_by_keys: List[str], group_info_cols: List[str] = None):
    """Make a table of representative info for each group in a DataSet.

    Args:
        ds: DataSet to make representative info for.
        info_path: Path to save representative info table to.
        group_by_keys: Columns to group by and make representative info for.
        group_info_cols: Columns to include in representative info table.
    """
    subset_filters = []
    value_lists = [ds.info_table[col].unique() for col in group_by_keys]
    for i in range(len(value_lists[0])):
        subset_filters.append({group_by_keys[0]: [value_lists[0][i]]})
    for i in range(1, len(group_by_keys)):
        new_filters = []
        for fltr in subset_filters:
            for value in value_lists[i]:
                new_filter = fltr.copy()
                new_filter[group_by_keys[i]] = [value]
                new_filters.append(new_filter)
        subset_filters = new_filters

    # make list of repres_ids and initialise info table for the representative data
    repres_ids = [f'repres_id_{i + 1:0>4}' for i in range(len(subset_filters))]
    repr_info_table = pd.DataFrame(columns=['repres_id'] + group_by_keys)

    for fltr, repres_id in zip(subset_filters, repres_ids):
        # get representative subset
        repr_subset = ds.subset(fltr)
        if repr_subset.info_table.empty:
            continue
        # add row to repr_info_table
        repr_info_table = pd.concat(
            [repr_info_table, pd.DataFrame({'repres_id': [repres_id], **fltr, 'nr averaged': [len(repr_subset)]})])

        # add means of group info columns to repr_info_table
        if group_info_cols is not None:
            for col in group_info_cols:
                df_col = repr_subset.info_table[col]
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, '' + col] = df_col.mean()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'std_' + col] = df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'upstd_' + col] = df_col.mean() + df_col.std()
                repr_info_table.loc[
                    repr_info_table['repres_id'] == repres_id, 'downstd_' + col] = df_col.mean() - df_col.std()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'max_' + col] = df_col.max()
                repr_info_table.loc[repr_info_table['repres_id'] == repres_id, 'min_' + col] = df_col.min()

    return repr_info_table
