"""Functions for post-processing material test data. (Stress-strain)"""
import os
from io import BytesIO
from typing import Callable
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.graphics import renderPDF
from reportlab.lib.colors import magenta, pink, blue
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg

from paramaterial.plug import DataSet, DataItem


def make_screening_pdf(
        dataset: DataSet,
        plot_func: Callable[[DataItem], None],
        pdf_path: str = 'dataset_plots.pdf'
) -> None:
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=(820, 600))
    for dataitem in dataset:
        plot_func(dataitem)
        imgdata = BytesIO()
        plt.savefig(imgdata, format='svg')
        imgdata.seek(0)
        drawing = svg2rlg(imgdata)
        renderPDF.draw(drawing, pdf_canvas, 5, 5)
        form = pdf_canvas.acroForm
        pdf_canvas.setFont("Courier", 22)
        pdf_canvas.drawString(20, 555, 'SELECT TO REJECT:')
        form.checkbox(name=f'reject_box_{dataitem.test_id}', x=252, y=552, buttonStyle='check',
                      borderColor=magenta, fillColor=pink, textColor=blue, forceBorder=True)
        pdf_canvas.showPage()
        plt.close()
    pdf_canvas.save()
    print(f'Screening pdf saved to {pdf_path}.')


def make_representative_curves(
        dataset: DataSet, data_dir: str, info_path: str,
        repr_col: str, repr_by_cols: List[str],
        interp_by: str, interp_res: int = 200, min_interp_val: float = 0., interp_end: str = 'max'
):
    """Make representative curves of the dataset and save them to a directory.
    Args:

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # make subset filters from combinations of unique values in repr_by_cols
    # todo: replace with value counts
    subset_filters = []
    value_lists = [dataset.info_table[col].unique() for col in repr_by_cols]
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

    # make representative curves
    for repr_id, subset_filter in zip(repr_ids, subset_filters):
        # get representative subset
        repr_subset = dataset[subset_filter]
        if repr_subset.info_table.empty:
            continue
        # add row to repr_info_table
        repr_info_table = pd.concat(
            [repr_info_table, pd.DataFrame({'repr id': [repr_id], **subset_filter, 'nr averaged': [len(repr_subset)]})])

        # find minimum of maximum interp_by vals in subset
        if interp_end == 'max_all':
            max_interp_val = max([max(subset[interp_by]) for subset in repr_subset])
        elif interp_end == 'min_of_max':
            max_interp_val = min([max(dataitem.data[interp_by]) for dataitem in repr_subset])
        else:
            raise ValueError(f'interp_end must be "max_all" or "min_of_max", not {interp_end}')
        # make monotonically increasing vector to interpolate by
        interp_vec = np.linspace(min_interp_val, max_interp_val, interp_res)

        # make interpolated data for averaging
        interp_data = pd.DataFrame(data={interp_by: interp_vec})
        for n, dataitem in enumerate(repr_subset):
            # drop columns and rows outside interp range
            data = dataitem.data[[interp_by, repr_col]].reset_index(drop=True)
            data = data[(data[interp_by] <= max_interp_val) & (data[interp_by] >= min_interp_val)]
            # interpolate the repr_by column and add to interp_data
            interp_data[f'interp_{repr_col}_{n}'] = np.interp(interp_vec, data[interp_by], data[repr_col])

        # make representative data from stats of interpolated data
        interp_data = interp_data.drop(columns=[interp_by])
        repr_data = pd.DataFrame({f'interp_{interp_by}': interp_vec})
        repr_data[f'mean_{repr_col}'] = interp_data.mean(axis=1)
        repr_data[f'std_{repr_col}'] = interp_data.std(axis=1)
        repr_data[f'min_{repr_col}'] = interp_data.min(axis=1)
        repr_data[f'max_{repr_col}'] = interp_data.max(axis=1)
        repr_data[f'q1_{repr_col}'] = interp_data.quantile(0.25, axis=1)
        repr_data[f'q3_{repr_col}'] = interp_data.quantile(0.75, axis=1)

        # write the representative data and info
        repr_data.to_csv(os.path.join(data_dir, f'{repr_id}.csv'), index=False)
        repr_info_table.to_excel(info_path, index=False)
