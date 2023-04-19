"""Module with functions for generating and reading a screening pdf file."""

import os
from io import BytesIO
from typing import Callable
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PyPDF2 import PdfReader
from reportlab.graphics import renderPDF
from reportlab.lib.colors import black
from reportlab.lib.colors import magenta, pink, blue
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg

from paramaterial.plug import DataSet, DataItem


def make_screening_pdf(
        ds: DataSet,
        plot_func: Callable[[DataItem], None],
        pdf_path: str = 'screening_pdf.pdf',
        pagesize: Tuple[float, float] = (900, 600),
) -> None:
    """Make a screening pdf where each page contains a plot, a check-box and a comment-box.
    Scaling the plot is still under development. Currently, the plot_func should produce a figure with figsize=(10., 5.8).

    Args:
        ds: DataSet object
        plot_func: Function that takes a DataItem and generates a plot
        pdf_path: Path where pdf will be saved
        pagesize: Size of the pdf pages

    Returns: None
    """
    # setup canvas
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=(pagesize[0], pagesize[1]))

    # loop through dataitems
    for di in ds:
        # make plot for dataitem
        plot_func(di)

        # add plot to page
        imgdata = BytesIO()
        plt.savefig(imgdata, format='svg')
        imgdata.seek(0)
        drawing = svg2rlg(imgdata)
        renderPDF.draw(drawing, pdf_canvas, 0.001*pagesize[0], 0.15*pagesize[1])

        # setup form
        form = pdf_canvas.acroForm
        pdf_canvas.setFont("Courier", plt.rcParams['font.size'] + 6)

        # add test_id
        pdf_canvas.drawString(0.05*pagesize[0], 0.95*pagesize[1], f'{di.test_id}')

        # add checkbox
        pdf_canvas.drawString(0.05*pagesize[0], 0.14*pagesize[1], 'REJECT?:')
        form.checkbox(name=f'reject_box_{di.test_id}', buttonStyle='check',
                      x=0.15*pagesize[0], y=0.13*pagesize[1],
                      borderColor=magenta, fillColor=pink, textColor=blue, forceBorder=True)

        # add text field
        pdf_canvas.drawString(0.05*pagesize[0], 0.08*pagesize[1], 'COMMENT:')
        form.textfield(name=f'comment_box_{di.test_id}', maxlen=10000,
                       x=0.15*pagesize[0], y=0.05*pagesize[1], width=0.7*pagesize[0], height=0.05*pagesize[1],
                       borderColor=magenta, fillColor=pink, textColor=black, forceBorder=True, fieldFlags='multiline')

        # add page to canvas and close plot
        pdf_canvas.showPage()
        plt.close()

    pdf_canvas.save()
    print(f'Screening pdf saved to {pdf_path}.')


def read_screening_pdf(ds: DataSet, pdf_path: str) -> DataSet:
    """Read the values from the checkbox and comment fields in the screening pdf and add them to the DataSet's info_table.
    The info_table will have two new columns: 'reject' and 'comment'. The 'reject' column will contain either 'True' or 'False'.
    The 'comment' column will contain the comment string entered into the comment field.

    Args:
        ds: DataSet object
        pdf_path: Path to screening pdf file

    Returns: DataSet object with checkbox and comment fields added to each DataItem's info
    """
    new_ds = ds.copy()
    _info_table = new_ds.copy().info_table
    test_id_key = ds.test_id_key

    # drop reject and comment cols if they exist
    if 'reject' in _info_table.columns:
        _info_table.drop(columns=['reject'], inplace=True)
    if 'comment' in _info_table.columns:
        _info_table.drop(columns=['comment'], inplace=True)

    # dataframe for screening results
    screening_df = pd.DataFrame(columns=[test_id_key, 'reject', 'comment'])

    with open(pdf_path, 'rb') as f:
        pdf_fields = PdfReader(f).get_fields()

    # get comment and reject fields
    comment_fields = [field for field in pdf_fields if 'comment' in field]
    reject_fields = [field for field in pdf_fields if 'reject' in field]

    # get test_ids from comment fields
    test_ids = [field.split('_box_')[1] for field in comment_fields]

    # get comments and rejects
    comments = [pdf_fields[field]['/V'] for field in comment_fields]
    rejects = [pdf_fields[field]['/V'] for field in reject_fields]

    # add to dataframe
    screening_df[test_id_key] = test_ids
    screening_df['reject'] = rejects
    screening_df['comment'] = comments

    # replace reject /Yes values with True, and /Off with False
    screening_df['reject'] = screening_df['reject'].replace('/Yes', 'True')
    screening_df['reject'] = screening_df['reject'].replace('/Off', 'False')

    _info_table = _info_table.merge(screening_df, on=test_id_key, how='left')

    new_ds.info_table = _info_table

    return new_ds


def remove_rejected_items(ds: DataSet, reject_key: str = 'reject') -> DataSet:
    """Remove DataItems from the DataSet that were marked as rejected in the screening pdf.
    DataItems will be removed if the value in the reject_key column of the ds.info_table is 'True'.

    Args:
        ds: DataSet object
        reject_key: Column name in the info_table that contains the reject values
    """
    new_ds = ds.copy()
    new_ds.info_table = new_ds.info_table[new_ds.info_table['reject'] != 'True']
    # print a list of the rejected items with a detailed message
    rejected_items = new_ds.info_table[new_ds.info_table['reject'] == 'True']
    for i, row in rejected_items.iterrows():
        print(f'Item {row[new_ds.test_id_key]} was rejected because {row["comment"]}')
    return new_ds

