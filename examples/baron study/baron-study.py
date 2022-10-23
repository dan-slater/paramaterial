"""Module for examnple study of baron data."""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from paramaterial.plug import DataSet, DataItem


def main():
    """Main function."""

    reference_table = pd.read_excel('report/baron quoted results.xlsx')
    reference_table = reference_table.rename(columns={
        'Strain-rate (s^-1)': 'rate', 'Nominal temperature': 'temperature', 'Material condition': 'material',
        'Mean temperature': 'mean temperature', 'Mean flow stress (MPa)': 'mean flow stress'
    })
    reference_table = reference_table.set_index(['material', 'rate', 'temperature']).sort_index().reset_index()

    analysis_set = DataSet('data/04 screened trimmed data', 'info/04 screened trimmed info.xlsx')[{'rate': [1, 10]}]

    def read_temp_and_stress(di: DataItem) -> DataItem:
        # interpolate the temperature and stress at 0.1 strain and 0.3 strain
        di.info['0.1 temperature'] = np.interp(0.1, di.data['Strain'], di.data['TC1(C)'])
        di.info['0.3 temperature'] = np.interp(0.3, di.data['Strain'], di.data['TC1(C)'])
        di.info['0.1 stress'] = np.interp(0.1, di.data['Strain'], di.data['Stress(MPa)'])
        di.info['0.3 stress'] = np.interp(0.3, di.data['Strain'], di.data['Stress(MPa)'])
        return di

    analysis_set = analysis_set.apply_function(read_temp_and_stress)
    analysis_set.info_table.head()
    analysis_table = analysis_set.info_table[
        ['material', 'temperature', 'rate', '0.1 temperature', '0.3 temperature', '0.1 stress', '0.3 stress']]
    analysis_table = analysis_table.groupby(['material', 'temperature', 'rate']).mean().reset_index()

    # drop outlier
    analysis_table = analysis_table.drop(analysis_table[analysis_table['0.3 temperature'] > 1000].index)

    merged_table = pd.merge(reference_table, analysis_table, on=['material', 'rate', 'temperature'])

    # transform table so that it can be used in seaborn
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey='row')
    melt_df = pd.melt(merged_table, id_vars=['material', 'rate', 'temperature'],
                      value_vars=['mean temperature', '0.1 temperature', '0.3 temperature', 'mean flow stress',
                                  '0.1 stress', '0.3 stress'])

    # get a df with only the mean temperature and mean flow stress
    melt_df_mean = melt_df[melt_df['variable'].isin(['mean temperature', 'mean flow stress'])]

    # plot the mean temperature and 0.1 temperature
    sns.violinplot(ax=axs[0, 0], x='material', y='value', hue='variable', split=True,
                   data=melt_df[melt_df['variable'].isin(['mean temperature', '0.1 temperature'])])
    # plot the mean temperature and 0.3 temperature
    sns.violinplot(ax=axs[0, 1], x='material', y='value', hue='variable', split=True,
                   data=melt_df[melt_df['variable'].isin(['mean temperature', '0.3 temperature'])])
    # plot the mean flow stress and 0.1 stress
    sns.violinplot(ax=axs[1, 0], x='material', y='value', hue='variable', split=True,
                   data=melt_df[melt_df['variable'].isin(['mean flow stress', '0.1 stress'])])
    # plot the mean flow stress and 0.3 stress
    sns.violinplot(ax=axs[1, 1], x='material', y='value', hue='variable', split=True,
                   data=melt_df[melt_df['variable'].isin(['mean flow stress', '0.3 stress'])])
    plt.show()


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

    # def ds_plot(dataset: DataSet, **kwargs) -> plt.Axes:
    #     return pam.plotting.dataset_plot(
    #         dataset, x='Strain', y='Stress(MPa)', color_by='temperature',
    #         cbar=True, cbar_label='Temperature ($^{\circ}$C)',
    #         xlim=(-0.2, 1.5), grid=True, **kwargs
    #     )
    #
    # proc_set = DataSet('data/02 processed data', 'info/02 processed info.xlsx')
    #
    # def multiply_neg_one(di: DataItem) -> DataItem:
    #     di.data['Stress(MPa)'] *= -1
    #     return di
    #
    # proc_set = proc_set.map_function(multiply_neg_one)
    #
    # def subset_test():
    #     a = proc_set[{'material': ['AC']}]
    #
    # def subset_test_2():
    #     a = proc_set[{'material': ['AC'], 'temperature': [25]}]
    #
    # def subset_test_3():
    #     a = proc_set.get_subset({'material': ['AC']})
    #
    # def subset_test_4():
    #     a = proc_set.get_subset({'material': ['AC'], 'temperature': [25]})
    #
    # # function to time a function
    # def time_function(func, *args, **kwargs):
    #     import time
    #     start = time.time()
    #     for i in range(int(5e3)):
    #         func(*args, **kwargs)
    #     end = time.time()
    #     return end - start
    #
    # # time the functions
    # print('subset_test', time_function(subset_test))
    # print('subset_test_2', time_function(subset_test_2))
    # print('subset_test_3', time_function(subset_test_3))
    # print('subset_test_4', time_function(subset_test_4))
    #
    # # a = proc_set[{'material': ['AC']}]
    # # ds_plot(proc_set.get_subset({'material': ['AC']}))
    # # plt.show()


def compare_results():
    ref_results_table = pd.read_excel('report/baron quoted results.xlsx')
    ref_results_table = ref_results_table.rename(columns={
        'Strain-rate (s^-1)': 'rate', 'Nominal temperature': 'temperature', 'Material condition': 'material',
        'Mean temperature': 'mean temperature', 'Mean flow stress (MPa)': 'mean flow stress'
    })
    ref_results_table = ref_results_table.set_index(['material', 'rate', 'temperature']).sort_index().reset_index()
    print(ref_results_table)

    analysis_set = DataSet('data/04 screened trimmed data', 'info/04 screened trimmed info.xlsx')
    analysis_set = analysis_set[{'rate': [1, 10]}]

    def read_temp_and_stress(di: DataItem) -> DataItem:
        # interpolate the temperature and stress at 0.1 strain and 0.3 strain
        di.info['0.1 temperature'] = np.interp(0.1, di.data['Strain'], di.data['TC1(C)'])
        di.info['0.3 temperature'] = np.interp(0.3, di.data['Strain'], di.data['TC1(C)'])
        di.info['0.1 stress'] = np.interp(0.1, di.data['Strain'], di.data['Stress(MPa)'])
        di.info['0.3 stress'] = np.interp(0.3, di.data['Strain'], di.data['Stress(MPa)'])
        return di

    analysis_set = analysis_set.apply_function(read_temp_and_stress, update_info=True)
    print(analysis_set.info_table.columns)
    analysis_table = analysis_set.info_table

    # get the mean temperature and mean flow stress at every unique combination of material, rate, and temperature
    analysis_table = analysis_table.groupby(['material', 'rate', 'temperature']).mean().reset_index()
    analysis_table = analysis_table.rename(columns={
        '0.1 temperature': 'mean 0.1 temperature', '0.3 temperature': 'mean 0.3 temperature',
        '0.1 stress': 'mean 0.1 stress', '0.3 stress': 'mean 0.3 stress'
    })
    print(analysis_table)

    # merge the reference results and the analysis results
    merged_table = pd.merge(ref_results_table, analysis_table, on=['material', 'rate', 'temperature'])
    print(merged_table)

    # plot the mean temperature and mean flow stress at 0.1 strain and 0.3 strain
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    for i, material in enumerate(['AC', 'H560', 'H580']):
        ax[0, 0].plot(merged_table.loc[merged_table['material'] == material, 'mean 0.1 temperature'],
                      merged_table.loc[merged_table['material'] == material, 'mean 0.1 stress'],
                      'o', label=material)
        ax[0, 1].plot(merged_table.loc[merged_table['material'] == material, 'mean 0.3 temperature'],
                      merged_table.loc[merged_table['material'] == material, 'mean 0.3 stress'],
                      'o', label=material)
        ax[1, 0].plot(merged_table.loc[merged_table['material'] == material, 'mean 0.1 temperature'],
                      merged_table.loc[merged_table['material'] == material, 'mean flow stress'],
                      'o', label=material)
        ax[1, 1].plot(merged_table.loc[merged_table['material'] == material, 'mean 0.3 temperature'],
                      merged_table.loc[merged_table['material'] == material, 'mean flow stress'],
                      'o', label=material)
    ax[0, 0].set_xlabel('Mean 0.1 temperature (C)')
    ax[0, 0].set_ylabel('Mean 0.1 stress (MPa)')
    ax[0, 1].set_xlabel('Mean 0.3 temperature (C)')
    ax[0, 1].set_ylabel('Mean 0.3 stress (MPa)')
    ax[1, 0].set_xlabel('Mean 0.1 temperature (C)')
    ax[1, 0].set_ylabel('Quoted mean flow stress (MPa)')
    ax[1, 1].set_xlabel('Mean 0.3 temperature (C)')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    plt.show()
