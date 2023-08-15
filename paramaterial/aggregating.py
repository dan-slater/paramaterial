"""
This module provides functionalities to make representative curves from data and find statistics for metadata.

Functions:
    - `_generate_filter_permutations(info_table, group_by)` - Generates filter permutations for grouping data.
    - `make_representative_data(ds, info_path, data_dir, repres_col, group_by_keys, interp_by, interp_res, interp_range,
    group_info_cols)` - Creates representative curves from a dataset and saves them to a directory.
    - `make_representative_info(ds, group_by_keys, group_info_cols)` - Creates a table of representative information for
    each group in a DataSet.
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


def _generate_filter_permutations(info_table: pd.DataFrame, group_by: List[str]) -> List[Dict]:
    """Generate filter permutations for grouping data.

        Args:
            info_table: A DataFrame containing information to group by.
            group_by: A list of columns to group the data by.

        Returns:
            A list of dictionaries representing filter permutations.
        """
    value_lists = [info_table[col].unique() for col in group_by]
    subset_filters = []
    for i in range(len(value_lists[0])):
        subset_filters.append({group_by[0]: value_lists[0][i]})
    for i in range(1, len(group_by)):
        new_filters = []
        for fltr in subset_filters:
            for value in value_lists[i]:
                new_filter = fltr.copy()
                new_filter[group_by[i]] = value
                new_filters.append(new_filter)
        subset_filters = new_filters
    return subset_filters


def make_representative_data(ds: DataSet, info_path: str, data_dir: str, repres_col: str, group_by_keys: List[str],
                             interp_by: str, interp_res: int = 200,
                             interp_range: Union[str, Tuple[float, float]] = 'outer',
                             group_info_cols: Optional[List[str]] = None):
    """Make representative curves of the DataSet and save them to a directory.

     This function takes a DataSet, groups it by specific keys, and creates representative curves. The curves are
     then saved to a specified directory. It is useful for generating aggregated data curves that represent groups of
     similar tests.

     Args:
         ds: The DataSet to make representative curves from.
         info_path: The path to the info file where the representative information will be saved.
         data_dir: The directory to save the representative curves to.
         group_by_keys: The info columns to group the tests by.
         repres_col: The data column to aggregate for the y-axis of the representative curves.
         interp_by: The data column to interpolate for the x-axis of the representative curves.
         interp_res: The resolution of the interpolation.
         interp_range: Can be either "outer", "inner", or a tuple of floats, defining the domain on the x-axis for
         interpolation:

            - If "outer", the domain is defined by the smallest minimum and the largest maximum values of the
            interpolation column in the representative subset.
            - If "inner", the domain is defined by the largest minimum and the smallest maximum values of the
            interpolation column in the representative subset.
            - If a tuple, the domain is directly defined by the values within the tuple.
         group_info_cols: The info categories to include in the aggregated info_table.

     Returns:
         None

     Examples:
        Imagine you have performed a series of stress tests on different materials at various temperatures. You have
        collected all the data in a DataSet and want to create representative stress-strain curves for each
        combination of material and temperature. Here's how you can use this function:

        >>> import paramaterial as pam
        >>> ds = pam.DataSet('info/test_info.csv','data/tests')  # Load your dataset
        >>> pam.make_representative_data(ds, 'info/representative_info.xlsx', 'data/representative_curves',
        >>>                              repres_col='Stress_MPa', group_by_keys=['material', 'temperature'],
        interp_by='Strain')

        This will create representative curves for each material and temperature group, saving them to the specified
        directory and information to an Excel file.
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

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
        repr_info_table = pd.concat([repr_info_table, pd.DataFrame(
            {'repres_id': [repres_id], **subset_filter, 'nr averaged': [len(repres_subset)]})])

        # add means of group info columns to repr_info_table
        if group_info_cols is not None:
            for col in group_info_cols:
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
        if interp_range == 'outer':
            min_interp_val = min([min(dataitem.data[interp_by]) for dataitem in repres_subset])
            max_interp_val = max([max(dataitem.data[interp_by]) for dataitem in repres_subset])
        elif interp_range == 'inner':
            min_interp_val = max([min(dataitem.data[interp_by]) for dataitem in repres_subset])
            max_interp_val = min([max(dataitem.data[interp_by]) for dataitem in repres_subset])
        elif type(interp_range) == tuple:
            min_interp_val = interp_range[0]
            max_interp_val = interp_range[1]
        else:
            raise ValueError(f'interp_range must be "outer", "inner" or a tuple, not {interp_range}')

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
        repr_data.to_csv(os.path.join(data_dir, f'{repres_id}.csv'), index=False)
        repr_info_table.to_excel(info_path, index=False)


def make_representative_info(ds: DataSet, group_by_keys: List[str], group_info_cols: List[str] = None) -> pd.DataFrame:
    """Make a table of representative info for each group in a DataSet.

    Args:
        ds: DataSet to make representative info from.
        group_by_keys: Columns to group by and make representative info for.
        group_info_cols: Columns to include in representative info table.

    Returns:
        A pandas DataFrame containing the representative information table.

    Examples:
        To create a summary table that includes specific mechanical properties like Elastic Modulus (E), Proof Stress
        (PS), Ultimate Tensile Strength (UTS), for each temperature and material type:

        >>> import paramaterial as pam
        >>> table = pam.make_representative_info(ds, group_by_keys=['temperature', 'material'], group_info_cols=['E', 'PS', 'UTS'])
        >>> print(table.head())

        The result will be a DataFrame containing representative information for each group, including the mean, standard
        deviation, maximum, minimum, and 1st and 3rd quartiles of the specified columns.
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


def make_repres_info(info_table: pd.DataFrame, group_by: Union[str, List[str]], info_cols: List[str] = None,
                     repres_id_key: str = 'repres_id') -> pd.DataFrame:
    if type(group_by) is str:
        group_by = [group_by]

    # filters for each subset
    subset_filters = _generate_filter_permutations(info_table, group_by)

    # representative ids, one for each subset
    repres_ids = [f'{repres_id_key}_{i + 1:0>4}' for i in range(len(subset_filters))]

    # make empty representative info table
    repres_info_table = pd.DataFrame(columns=['repres_id'] + group_by + info_cols)

    # make representative info row for each subset
    for fltr, repres_id in zip(subset_filters, repres_ids):
        subset_info_table = info_table.query(' and '.join([f"`{key}` in {str(vals)}" for key, vals in fltr.items()]))

        if subset_info_table.empty:  # skip empty subsets
            continue

        # make representative info row
        repres_info_table = pd.concat(
            [repres_info_table, pd.DataFrame({'repres_id': [repres_id], 'nr_grouped': [len(subset_info_table)]})],
            **fltr)

        # add representative info to row
        if info_cols is not None:
            for col in info_cols:
                df_col = subset_info_table[col]
                repres_info_table.loc[repres_info_table['repres_id'] == repres_id, '' + col] = df_col.mean()
                repres_info_table.loc[repres_info_table['repres_id'] == repres_id, 'std_' + col] = df_col.std()
                repres_info_table.loc[repres_info_table['repres_id'] == repres_id, 'max_' + col] = df_col.max()
                repres_info_table.loc[repres_info_table['repres_id'] == repres_id, 'min_' + col] = df_col.min()
                repres_info_table.loc[repres_info_table['repres_id'] == repres_id, 'q1_' + col] = df_col.quantile(0.25)
                repres_info_table.loc[repres_info_table['repres_id'] == repres_id, 'q3_' + col] = df_col.quantile(0.75)

    return repres_info_table


def make_repres_ds(ds: DataSet, repres_col: str, interp_by: str, group_by: Union[str, List[str]], interp_res: int = 200,
                   interp_domain: Union[str, Tuple[float, float]] = 'outer', info_cols: List[str] = None,
                   repres_id_key: str = 'repres_id') -> DataSet:
    if type(group_by) is str:
        group_by = [group_by]

    # make empty representative info table
    subset_filters = _generate_filter_permutations(ds.info_table, group_by)

    # representative ids, one for each subset
    repres_ids = [f'repres_id_{i + 1:0>4}' for i in range(len(subset_filters))]

    # make full representative info table
    repres_info_table = make_repres_info(ds.info_table, group_by, info_cols, repres_id_key)

    # make representative data for each subset and make a DataItem from it
    repres_data_items = []

    for repres_id, subset_filter in zip(repres_ids, subset_filters):
        repres_subset = ds.subset(subset_filter)

        if repres_subset.info_table.empty:  # skip empty subsets
            continue

        # define interpolation domain
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

        # make interpolated dataframe for later averaging
        interp_data = pd.DataFrame(data={interp_by: interp_vec})

        # interpolate the repres_col from the data of each dataitem and add to interp_data
        for n, dataitem in enumerate(repres_subset):
            # drop columns except interp_by and repres_col
            data = dataitem.data[[interp_by, repres_col]].reset_index(drop=True)

            # drop rows outside of interpolation domain
            data = data[(data[interp_by] <= max_interp_val) & (data[interp_by] >= min_interp_val)]

            # interpolate the repres_col of the current dataitem and add to interp_data
            interp_data[f'interp_{repres_col}_{n}'] = np.interp(interp_vec, data[interp_by], data[repres_col])

        # drop interp_by column from interp_data
        interp_data = interp_data.drop(columns=[interp_by])

        # make representative data from stats of interpolated data
        repres_data = pd.DataFrame({f'{interp_by}': interp_vec})
        repres_data[f'{repres_col}'] = interp_data.mean(axis=1)
        repres_data[f'std_{repres_col}'] = interp_data.std(axis=1)
        repres_data[f'min_{repres_col}'] = interp_data.min(axis=1)
        repres_data[f'max_{repres_col}'] = interp_data.max(axis=1)
        repres_data[f'q1_{repres_col}'] = interp_data.quantile(0.25, axis=1)
        repres_data[f'q3_{repres_col}'] = interp_data.quantile(0.75, axis=1)

        # make a DataItem from the representative data and info
        repres_item = DataItem(repres_id, repres_data, repres_info_table)
        repres_data_items.append(repres_item)

    # make a new DataSet from the representative DataItems
    repres_ds = DataSet(repres_id_key)
    repres_ds.info_table = repres_info_table
    repres_ds.data_items = repres_data_items

    return repres_ds
