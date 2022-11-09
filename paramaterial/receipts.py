import copy
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Callable

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import yaml
from PyPDF2 import PdfFileMerger
from jinja2 import Environment, FileSystemLoader, meta
from matplotlib import pyplot as plt

import paramaterial.plotting as plotting
from paramaterial.plug import DataSet, DataItem

ENV = Environment(
    variable_start_string=r'\VAR{',
    variable_end_string='}',
    autoescape=False,
    loader=FileSystemLoader(os.path.abspath('..')),
    comment_start_string='%',
    comment_end_string='#'
)


def make_receipts(cfg: Dict):
    """
    Use latex template to make pdf receipts for each test.
    """
    template = ReceiptTemplate()
    template.load_config(receipts_cfg=cfg)
    template.parse_vars_from_latex_template(template_path=cfg['template_path'])
    if not template.satisfies_config():
        raise TemplateConfigMismatchError

    # loop through datasets and make output folders, plots and filled tex files
    for dataset_cfg in cfg['dataset_cfgs']:
        dataset = DataSet()
        dataset.load_data(**dataset_cfg['load_args'])
        for dataitem in dataset.data_items:
            receipt = TestReceipt(test_id=dataitem.test_id, dataitem=dataitem, template_path=cfg['template_path'],
                                  receipts_dir=cfg['receipts_dir'], receipt_vars=template.template_vars)
            receipt.make_output_folder()
            receipt.make_plots(dataset_cfg['plot_cfgs'])
            receipt.fill_receipt_vars(dataset_cfg['plot_cfgs'], dataset_cfg['table_cfgs'])
            receipt.write_filled_tex_file()

    # loop output folders and compile test receipt pdfs
    for folder_name in os.listdir(cfg['receipts_dir']):
        TestReceipt.make_latex_pdf(receipts_dir=cfg['receipts_dir'], test_id=folder_name)


@dataclass
class ReceiptTemplate:
    """Handles interfacing with config and template. Provides info to TestReceipt class."""
    cfg: Dict = None
    template_vars: Dict = None

    def load_config(self, receipts_cfg: Dict) -> None:
        self.cfg = receipts_cfg

    def parse_vars_from_latex_template(self, template_path: str) -> Dict:
        template_source = ENV.loader.get_source(ENV, template_path)
        parsed_content = ENV.parse(template_source)
        self.template_vars = {key: None for key in meta.find_undeclared_variables(parsed_content)}
        return self.template_vars

    def satisfies_config(self) -> bool:
        # todo
        return True


class TemplateConfigMismatchError(Exception):
    """Raised when template vars are requested but not specified in config."""
    pass


@dataclass
class TestReceipt:
    """Generates receipt for individual test info from ReceiptTemplate class."""
    test_id: str = None
    dataitem: DataItem = None
    template_path: str = None
    receipts_dir: str = None
    receipt_vars: Dict = None

    def make_output_folder(self):
        test_folder_path = f'{self.receipts_dir}/{self.test_id}'
        if not os.path.exists(test_folder_path):
            os.makedirs(test_folder_path)
        else:
            shutil.rmtree(test_folder_path)
            os.makedirs(test_folder_path)

    def make_plots(self, plot_cfgs: Dict):
        for plot_cfg in plot_cfgs:
            fig, ax = plt.subplots(1, 1, figsize=plot_cfg['figsize'])
            make_plot_from_config(ax, self.dataitem, plot_cfg, f'{self.receipts_dir}/{self.test_id}')
            plt.close()

    def fill_receipt_vars(self, plot_cfgs: Dict, table_cfgs: Dict, **custom_kwargs) -> None:
        for var_key in self.receipt_vars.keys():
            if var_key.startswith('plot'):
                label = var_key.split('_')[1]
                plot_cfg = [cfg for cfg in plot_cfgs if cfg['label'] == label][0]
                self.receipt_vars[var_key] = self._latex_plot_string(plot_cfg)
            elif var_key.startswith('table'):
                label = var_key.split('_')[1]
                table_cfg = [cfg for cfg in table_cfgs if cfg['label'] == label][0]
                self.receipt_vars[var_key] = self._latex_table_string(table_cfg, self.dataitem.info)
            else:
                self.custom_fill_vars_func(**custom_kwargs)

    def custom_fill_vars_func(self, **custom_kwargs) -> None:
        # user defined custom_var_dict
        custom_var_dict = {
            'test_id': ' '.join((self.dataitem.test_id).split('_'))
        }
        for var_key in self.receipt_vars.keys():
            if var_key.startswith('plot') or var_key.startswith('table'):
                continue
            for key, item in custom_var_dict.items():
                self.receipt_vars[key] = item

    def write_filled_tex_file(self):
        template = ENV.get_template(self.template_path)
        document = template.render(**self.receipt_vars)
        with open(f'{self.receipts_dir}/{self.test_id}/{self.test_id}_receipt.tex', 'w') as out_file:
            out_file.write(document)

    @staticmethod
    def _latex_plot_string(cfg: Dict) -> str:
        plot_string = ''
        plot_string += r'\begin{figure}[!htb]' + '\n'
        plot_string += r'\centering' + '\n'
        plot_string += r'\label{plot:' + cfg['label'] + '}\n'
        plot_string += r'\includegraphics[width=\textwidth]{' + cfg['label'] + '}\n'
        plot_string += r'\caption{' + cfg['caption'] + '}\n'
        plot_string += r'\end{figure}' + '\n'
        return plot_string

    @staticmethod
    def _latex_table_string(cfg: Dict, info: pd.Series) -> str:
        table_df = pd.DataFrame(info).loc[cfg['info_keys']]
        return table_df.to_latex(header=False, longtable=True, caption=cfg['caption'])

    @staticmethod
    def make_latex_pdf(receipts_dir: str, test_id: str):
        src_wd = os.getcwd()
        os.chdir(f'{receipts_dir}/{test_id}')
        cmd = ['pdflatex', '-interaction', 'nonstopmode', f'{test_id}_receipt.tex']
        proc = subprocess.Popen(cmd)
        proc.communicate()
        retcode = proc.returncode
        if not retcode == 0:
            os.unlink(f'{test_id}_receipt.pdf')
            raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))
        os.chdir(src_wd)

    @staticmethod
    def combine_receipts():
        pdfs = []
        test_receipts_dir = '/plots/receipts/tests'
        for folder in os.listdir(test_receipts_dir):
            for file in os.listdir(f'{test_receipts_dir}/{folder}'):
                if file.startswith('testID') and file.endswith('.pdf'):
                    # print(f'{test_receipts_dir}/{folder}/{file}')
                    pdfs.append(f'{test_receipts_dir}/{folder}/{file}')

        merger = PdfFileMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write(f"../paramaterial/plots/receipts/combined-receipts.pdf")
        merger.close()


def make_plot_from_config(ax: plt.Axes, dataitem: DataItem, plot_cfg: Dict, plot_dir: str) -> None:
    plot_func = getattr(plotting, plot_cfg['func'])
    plot_func(ax=ax, dataitem=dataitem, **plot_cfg['kwargs'])
    plt.savefig(fname=f'{plot_dir}/{plot_cfg["label"]}.pdf')


if __name__ == '__main__':
    make_receipts(yaml.load_data(open(r'../CONFIG.yaml'), Loader=yaml.Loader)['make_receipts'])
