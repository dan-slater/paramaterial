"""Module containing the class for generating test receipts."""

import os
import subprocess
from pathlib import Path
import shutil
from typing import Dict, List, Any, Union, Callable

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PyPDF2 import PdfMerger
from jinja2 import Environment, FileSystemLoader, meta
from matplotlib import pyplot as plt

from paramaterial.plug import DataSet, DataItem


class TestReceipts:
    """Class for generating test receipts.

    Args:
        template_path: Path to the template file to be used for generating the receipts.
        jinja_env_kwargs: Keyword arguments to be passed to the jinja2.Environment constructor.
    """

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

    def parse_placeholders(self, as_dict: bool = False) -> Union[List[str], Dict[str, Any]]:
        """Parse the template file for placeholders. With the default setup for the jinja2.Environment, placeholders are
        defined as \VAR{placeholder_name}. The placeholders are returned as a list of strings, or as a dictionary with
        the placeholder names as keys and None as values.

        Args:
            as_dict: If True, return a dictionary with the placeholder names as keys and None as values. If False,
            return a list of strings.
        """
        template_source = self.jinja_env.loader.get_source(self.jinja_env, self.template_path)
        parsed_content = self.jinja_env.parse(template_source)
        self.placeholders = list(meta.find_undeclared_variables(parsed_content))
        if as_dict:
            return {key: None for key in self.placeholders}
        return self.placeholders

    def generate_receipts(self, ds: DataSet, receipts_path: str, replace_dict: Dict[str, Any],
                          receipts_dir: str = './receipts', clean: bool = True):
        """Generate receipts for the tests in the DataSet. The receipts are saved as pdf files in the receipts_dir
        directory. The directory structure is receipts_dir/test_id/(files for test_id receipt). The receipts are merged
        into a single pdf file and saved at receipts_path. The replace_dict dictionary is used to replace the
        placeholders
        in the template file. The keys of the dictionary are the placeholders, and the values are the replacement
        strings.
        The values can also be functions that take a DataItem as input and return a string. The functions are called
        with the DataItem corresponding to the test_id of the receipt being generated. If the function generates a
        plot, it should save the plot in the current directory and return the name of the saved plot-file"""
        # make receipts folder
        if not os.path.exists(receipts_dir):
            os.mkdir(receipts_dir)

        page_num = 1
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

                # set page number
                filled_template = filled_template.replace('\end{document}',
                                                          f'\setcounter{{page}}{{{page_num}}}\n\end{{document}}')
                page_num += 1

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
        merger = PdfMerger()
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
