"""Module for examnple study of baron data."""

from matplotlib import pyplot as plt

import paramaterial as pam
from paramaterial.plug import DataSet


def main():
    """Main function."""

    dataset = DataSet('data/02 processed data', 'info/02 processed info.xlsx')

    ds_cbar_norm = pam.plotting.dataset_colorbar_norm(dataset, 'temperature')
    ds_plot = lambda ds, **kwargs: pam.plotting.dataset_plot(
        ds, x='Strain', y='Stress(MPa)', ylabel='Stress (MPa)',
        color_by='temperature', cbar_norm=ds_cbar_norm,
        xlim=(-0.2, 1.5), grid=True, **kwargs
    )

    pam.processing.make_representative_curves(
        dataset, 'data/03 representative curves', 'info/03 representative info.xlsx',
        repr_col='Stress(MPa)', repr_by_cols=['material', 'temperature', 'rate'],
        interp_by='Strain'
    )

    repr_set = DataSet('data/03 representative curves', 'info/03 representative info.xlsx', 'repr id')

    # for the dataitem in the dataset
    # find the corresponding dataitem in the representative dataset based on temperature, material and rate
    # plot the dataitem and the corresponding dataitem in the representative dataset
    def screening_plot(dataitem):
        repr_filter = {'material': dataitem.info.material, 'temperature': dataitem.info.temperature,
                       'rate': dataitem.info.rate}
        repr_dataitem_set = repr_set[repr_filter]
        assert len(repr_dataitem_set) == 1
        repr_dataitem = repr_dataitem_set[0]

        ax = ds_plot()

    pam.processing.make_screening_pdf(dataset, screening_plot, 'data/04 screening.pdf')

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
