"""Module for examnple study of baron data."""

from matplotlib import pyplot as plt

import paramaterial as pam
from paramaterial.plug import DataSet, DataItem


def main():
    """Main function."""

    # load processed data
    proc_set = DataSet('data/02 processed data', 'info/02 processed info.xlsx')

    # make representative data from processed data
    # pam.processing.make_representative_curves(
    #     proc_set, 'data/03 representative curves', 'info/03 representative info.xlsx',
    #     repr_col='Stress(MPa)', repr_by_cols=['material', 'temperature', 'rate'],
    #     interp_by='Strain'
    # )

    # load representative data
    repr_set = DataSet('data/03 representative curves', 'info/03 representative info.xlsx', 'repr id')

    # setup screening plot
    cbar_norm = pam.plotting.dataset_colorbar_norm(proc_set, 'temperature')

    def screening_plot(di: DataItem) -> None:
        """Screening plot function."""

        # get representative data corresponding to dataitem
        repr_filter = {'material': di.info.material,
                       'temperature': di.info.temperature,
                       'rate': di.info.rate}
        repr_item_set = repr_set[repr_filter]
        assert len(repr_item_set) == 1  # should be only one repr item

        # plot dataitem
        ax = di.data.plot(x='Strain', y='Stress(MPa)', color='k')

        # plot representative curve
        pam.plotting.dataset_plot(
            repr_item_set, x='interp_Strain', y='mean_Stress(MPa)', ylabel='Stress (MPa)', ax=ax,
            color_by='temperature', cbar_norm=cbar_norm,
            style_by='material', style_by_label='Material', width_by='rate', width_by_label='Rate (s$^{-1}$)',
            xlim=(-0.2, 1.5), grid=True, fill_between=('down_std_Stress(MPa)', 'up_std_Stress(MPa)'), alpha=0.5,
        )



    # todo: screening plot formatting
    # todo: comment box
    # todo: run screening
    # make screening plot
    screening_plot(proc_set[0])
    plt.show()
    # pam.processing.make_screening_pdf(proc_set[0:1], screening_plot, 'data/04 screening.pdf')

    # repr_set.fit_model()


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
