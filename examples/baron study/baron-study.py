"""Module for examnple study of baron data."""

from matplotlib import pyplot as plt

import paramaterial as pam
from paramaterial.plug import DataSet, DataItem


def main():
    """Main function."""

    def ds_plot(dataset: DataSet, **kwargs) -> plt.Axes:
        return pam.plotting.dataset_plot(
            dataset, x='Strain', y='Stress(MPa)', color_by='temperature',
            cbar=True, cbar_label='Temperature ($^{\circ}$C)',
            xlim=(-0.2, 1.5), grid=True, **kwargs
        )

    proc_set = DataSet('data/02 processed data', 'info/02 processed info.xlsx')

    def multiply_neg_one(di: DataItem) -> DataItem:
        di.data['Stress(MPa)'] *= -1
        return di

    proc_set = proc_set.map_function(multiply_neg_one)

    def subset_test():
        a = proc_set[{'material': ['AC']}]

    def subset_test_2():
        a = proc_set[{'material': ['AC'], 'temperature': [25]}]

    def subset_test_3():
        a = proc_set.get_subset({'material': ['AC']})

    def subset_test_4():
        a = proc_set.get_subset({'material': ['AC'], 'temperature': [25]})

    # function to time a function
    def time_function(func, *args, **kwargs):
        import time
        start = time.time()
        for i in range(int(1e6)):
            func(*args, **kwargs)
        end = time.time()
        return end - start

    # time the functions
    print('subset_test', time_function(subset_test))
    print('subset_test_2', time_function(subset_test_2))
    print('subset_test_3', time_function(subset_test_3))
    print('subset_test_4', time_function(subset_test_4))

    # a = proc_set[{'material': ['AC']}]
    # ds_plot(proc_set.get_subset({'material': ['AC']}))
    # plt.show()



if __name__ == '__main__':
    main()

    # ds_plot = lambda dataset, **kwargs: pam.plotting.dataset_plot(
    #     dataset, x='Strain', y='Stress(MPa)', ylabel='Stress (MPa)',
    #     cbar_by='temperature', cbar_label='Temperature ($^{\circ}$C)',
    #     xlim=(-0.2, 1.5), grid=True, **kwargs
    # )
    # ds_plot(dataset)

    # ds_subplot = lambda dataset: pam.plotting.dataset_subplots(
    #     dataset, x='Strain', y='Stress(MPa)', ylabel='Stress (MPa)',
    #     nrows=3, ncols=4, rows_by='material', cols_by='rate',
    #     row_keys=[['AC'], ['H560'], ['H580']], col_keys=[[1], [10], [50], [100]],
    #     row_titles=['AC', 'H560', 'H580'], col_titles=['1 s$^{-1}$', '10 s$^{-1}$', '50 s$^{-1}$', '100 s$^{-1}$'],
    #     cbar_by='temperature', cbar_label='Temperature ($^{\circ}$C)',
    #     xlim=(-0.2, 1.5), grid=True, wspace=0.05, hspace=0.05,
    # )
    # ds_subplot(dataset)

    #
    # # load processed data
    # proc_set = DataSet('data/02 processed data', 'info/02 processed info.xlsx')
    #
    # # make representative data from processed data
    # # pam.processing.make_representative_curves(
    # #     proc_set, 'data/03 representative curves', 'info/03 representative info.xlsx',
    # #     repr_col='Stress(MPa)', repr_by_cols=['material', 'temperature', 'rate'],
    # #     interp_by='Strain'
    # # )
    #
    # # load representative data
    # repr_set = DataSet('data/03 representative curves', 'info/03 representative info.xlsx', 'repr id')
    #
    # # setup screening plot
    # color_by = 'temperature'
    # color_norm = plt.Normalize(vmin=proc_set.info_table[color_by].min(), vmax=proc_set.info_table[color_by].max())
    #
    # def screening_plot(di: DataItem) -> None:
    #     """Screening plot function."""
    #
    #     # get other similar data items
    #     subset_filter = {'material': di.info['material'], 'temperature': di.info['temperature'],
    #                      'rate': di.info['rate']}
    #     similar_set = proc_set[subset_filter]
    #
    #     # get associated representative data
    #     repr_item_set = repr_set[subset_filter]
    #     assert len(repr_item_set) == 1  # should be only one repr item
    #
    #     # plot dataitem
    #     ax = di.data.plot(x='Strain', y='Stress(MPa)', color='k', legend=False)
    #
    #     # plot representative curve
    #     ax = pam.plotting.dataset_plot(
    #         repr_item_set, x='interp_Strain', y='mean_Stress(MPa)', ylabel='Stress (MPa)', ax=ax,
    #         color_by='temperature', color_by_label=r'Temp (${^\circ}$C):', color_norm=color_norm,
    #         style_by='material', style_by_label='Material:', width_by='rate', width_by_label='Rate (s$^{-1}$):',
    #         xlim=(-0.2, 1.5), grid=True, fill_between=('min_Stress(MPa)', 'max_Stress(MPa)'), alpha=0.2,
    #         figsize=(10, 5.8),
    #     )
    #
    #     # plot similar curves
    #     ax = pam.plotting.dataset_plot(
    #         similar_set, x='Strain', y='Stress(MPa)', ax=ax, color_by='temperature', color_by_label=r'Temp (${^\circ}$C):',
    #         color_norm=color_norm, style_by='material', style_by_label='Material:', width_by='rate',
    #         width_by_label='Rate (s$^{-1}$):', xlim=(-0.2, 1.5), grid=True,
    #         alpha=0.5, figsize=(10, 5.8),
    #     )
    #
    #     # add title with number of similar curves
    #     ax.set_title(f'{len(similar_set)} similar curves')
    #     ax.set_title(f'{di.info["material"]} at {di.info["temperature"]} C and {di.info["rate"]} s$^{{-1}}$. '
    #                  f'One of {len(similar_set)}.')
    #
    # # todo: run screening
    # pam.processing.make_screening_pdf(proc_set, screening_plot, 'data/04 screening.pdf')
