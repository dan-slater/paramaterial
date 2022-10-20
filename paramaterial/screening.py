import copy
import os
import shutil
from typing import Dict, Callable

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.colors import magenta, pink, blue
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from PyPDF2 import PdfFileReader

from paramaterial.plug import DataSet, DataItem


def make_dataset_plots_pdf(
        dataset: DataSet,
        plot_func: Callable[[DataItem, Dict], None],
        plot_path: str = 'dataset_plots.pdf',
        plot_cfg: Dict = None,
        pdf_kwargs: Dict = None
) -> None:
    with PdfPages(plot_path) as pdf:
        for dataitem in copy.deepcopy(dataset.datamap):
            plot_func(dataitem, plot_cfg)
            pdf.savefig(**pdf_kwargs)
            plt.close()


def make_dataset_plots_screening_pdf(
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


def make_screening_pdf(data_dir: str, pdf_path: str, df_plot_kwargs: Dict):
    # make page
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=(820, 600))
    # loop through files to screen
    for filename in os.listdir(data_dir):
        # plot data
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        df = pd.read_csv(f'{data_dir}/{filename}')
        df.plot(ax=ax, **df_plot_kwargs)
        plt.suptitle(filename)
        # add plot to page
        imgdata = BytesIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)  # rewind the data
        drawing = svg2rlg(imgdata)
        renderPDF.draw(drawing, pdf_canvas, 5, 5)
        # make check box
        form = pdf_canvas.acroForm
        pdf_canvas.setFont("Courier", 22)
        pdf_canvas.drawString(20, 555, 'SELECT TO REJECT:')
        form.checkbox(name=f'reject_box_{filename}', x=252, y=552, buttonStyle='check',
                      borderColor=magenta, fillColor=pink, textColor=blue, forceBorder=True)
        # end page
        pdf_canvas.showPage()
        plt.close()
        # save pdf
    pdf_canvas.save()
    print(f'Screening pdf saved to {pdf_path}.')


def copy_with_screening(data_dir: str, info_path: str, screening_pdf: str, new_data: str, new_info: str):
    pdf_reader = PdfFileReader(open(screening_pdf, 'rb'))
    keep_list = []
    if 'test id' not in pd.read_excel(info_path).columns:
        raise ValueError(f'No column called "test id" in {info_path}.')
    # read checkboxes
    for key, value in pdf_reader.get_fields().items():
        filename = key[11:]
        check_box = value['/V']
        if check_box == '/Off':
            keep_list.append(filename)
    # keep tests if checkboxes not ticked
    dataset = DataSet(data_dir, info_path)
    dataset.get_subset({'filename': keep_list})
    dataset.write_output(new_data, new_info)


if __name__ == '__main__':
    copy_with_screening('../examples/baron study/data/01 raw data', '../examples/baron study/info/01 raw info.xlsx',
                        '../examples/baron study/data/01 raw data screening.pdf',
                        '../examples/baron study/data/02 prepared data', '../examples/baron study/info/03 prepared info.xlsx', )


def make_screening_pdf_old(dataset: DataSet, pdf_path: str = 'screening.pdf', x: str = 'Strain', y: str = 'Stress(MPa)',
                           x_lims: Dict = None):
    # make page
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=(820, 600))
    # loop through files to screen
    for dataitem in dataset:
        test_id = dataitem.test_id
        data = dataitem.data
        info = dataitem.info
        # plot data
        fig, (ax) = plt.subplots(1, 1, figsize=(9, 6))
        plt.title(f'{" ".join(test_id.split("_"))}, {info.reference}, {info["test type"]}, {info.temperature}C, '
                  f'{info.rate}/s', loc='left')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if x_lims is not None:
            ax.set_xlim(**x_lims)
        ax.plot(data[x], data[y], label=y, lw=1, color='b', alpha=0.8)
        plt.tight_layout()
        # add plot to page
        imgdata = BytesIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)  # rewind the data
        drawing = svg2rlg(imgdata)
        renderPDF.draw(drawing, pdf_canvas, 5, 5)
        # make check box
        form = pdf_canvas.acroForm
        pdf_canvas.setFont("Courier", 22)
        pdf_canvas.drawString(20, 555, 'SELECT TO REJECT:')
        form.checkbox(name=f'reject_box_{test_id}', x=252, y=552, buttonStyle='check',
                      borderColor=magenta, fillColor=pink, textColor=blue, forceBorder=True)
        # end page
        pdf_canvas.showPage()
        plt.close()
    # save pdf
    pdf_canvas.save()
    print(f'Screening pdf saved to {pdf_path}.')


def copy_screened_data(dataset: DataSet, screening_pdf: str, new_data: str, new_info: str):
    pdf_reader = PdfFileReader(open(screening_pdf, 'rb'))
    keep_list = []
    # read checkboxes
    for key, value in pdf_reader.get_fields().items():
        test_id = key[11:]
        check_box = value['/V']
        if check_box == '/Off':
            keep_list.append(test_id)
    # keep tests if checkboxes not ticked
    dataset.get_subset({'test id': keep_list})
    dataset.write_output(new_data, new_info)
