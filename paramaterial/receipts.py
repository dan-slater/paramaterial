"""Module containing the class for generating test receipts."""

import os
import subprocess
from pathlib import Path
import shutil
from typing import Dict, List, Any, Union, Callable

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PyPDF2 import PdfFileMerger
from jinja2 import Environment, FileSystemLoader, meta
from matplotlib import pyplot as plt

from paramaterial.plug import DataSet, DataItem


class TestReceipts:
    def __init__(self, template_path: str, jinja_env_kwargs: Dict = None):
        self.template_path = template_path
        self.placeholders: List[str] = ['']
        default_jinja_env_kwargs = dict(variable_start_string=r'\VAR{',
                                        variable_end_string='}',
                                        autoescape=False,
                                        loader=FileSystemLoader(os.path.abspath('.')),
                                        comment_start_string='%',
                                        comment_end_string='#')
        if jinja_env_kwargs is None:
            env_kwargs = default_jinja_env_kwargs
        else:
            env_kwargs = default_jinja_env_kwargs.update(jinja_env_kwargs)
        self.jinja_env = Environment(**env_kwargs)

    def parse_template_vars(self, as_dict: bool = False) -> Union[List[str], Dict[str, Any]]:
        template_source = self.jinja_env.loader.get_source(self.jinja_env, self.template_path)
        parsed_content = self.jinja_env.parse(template_source)
        self.placeholders = list(meta.find_undeclared_variables(parsed_content))
        if as_dict:
            return {key: None for key in self.placeholders}
        return self.placeholders

    def generate_receipts(self, ds: DataSet, receipts_path: str, replace_dict: Dict[str, Any],
                          receipts_dir: str = './receipts', clean: bool = True):
        # make receipts folder
        if not os.path.exists(receipts_dir):
            os.mkdir(receipts_dir)

        for di in ds.data_items:
            # make di folder
            di_folder = os.path.join(receipts_dir, di.test_id)
            if not os.path.exists(di_folder):
                os.mkdir(di_folder)

            # change to di folder
            src_wd = os.getcwd()

            # try make receipt, if error, go back to src_wd
            try:
                os.chdir(f'{receipts_dir}/{di.test_id}')
                # call replacer functions and fill template
                template = self.jinja_env.get_template(self.template_path)
                replace_dict_strings = {}
                for placeholder, replacer in replace_dict.items():
                    if callable(replacer):
                        replace_dict_strings.update({placeholder: replacer(di)})
                        plt.close()
                    else:
                        replace_dict_strings.update({placeholder: replacer})
                filled_template = template.render(**replace_dict_strings).replace('_', '\_')

                # write filled template to file
                with open(f'{di.test_id}_receipt.tex', 'w') as f:
                    f.write(filled_template)

                # compile pdf
                cmd = ['pdflatex', '-interaction', 'nonstopmode', f'{di.test_id}_receipt.tex']
                proc = subprocess.Popen(cmd)
                proc.communicate()
                retcode = proc.returncode
                if not retcode == 0:
                    os.unlink(f'{di.test_id}_receipt.pdf')
                    raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))
            except Exception as e:
                print(e)
            finally:
                os.chdir(src_wd)

            # change back to receipts folder
            os.chdir(src_wd)

        # merge pdfs
        pdfs = [os.path.join(receipts_dir, di.test_id, f'{di.test_id}_receipt.pdf') for di in ds.data_items]
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(receipts_path)
        merger.close()

        # delete receipts folder
        if clean:
            shutil.rmtree(receipts_dir)



if __name__ == '__main__':
    tr = TestReceipts(template_path='./receipts.tex')
    tr.parse_template_vars()
