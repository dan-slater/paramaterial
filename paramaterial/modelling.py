
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
