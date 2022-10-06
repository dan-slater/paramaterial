""" Main module for running experiments and configuring io paths. [danslater. 22march2022] """
import datetime
import shutil
from typing import Callable, Dict

import yaml

from paramaterial.plug import DataSet, IO_Paths
from paramaterial.preparing import prepare_data
from paramaterial.processing import process_data
from paramaterial.fitting import fit_models
from paramaterial.plotting import make_plots

CONFIG = 'CONFIG.yaml'


def main() -> None:
    config = yaml.load_data(open(CONFIG), Loader=yaml.Loader)
    active_functions = config['active functions']
    for mode in active_functions:
        function = eval(mode)
        if function in [prepare_data, process_data, fit_models]:
            run(function, config[mode])
            # store_run()
        elif function is make_plots:
            make_plots(config[mode])


def run(func: Callable, cfg: Dict) -> None:
    io = IO_Paths(*cfg['io'])
    dataset = DataSet()
    subset_filters = cfg['filters']
    dataset.load_data(io.input_data, io.input_info, subset_filters)  # map input
    dataset.datamap = map(lambda o: func(o, cfg), dataset.datamap)  # map function
    dataset.output(io.output_data, io.output_info)  # execute function and write output


def store_run():
    time_string = datetime.datetime.now().strftime("%m-%d-%Y %Hh%Mm%Ss")
    shutil.copytree('info', f'log/{time_string}')
    shutil.copy('CONFIG.yaml', f'log/{time_string}/CONFIG.yaml')


if __name__ == '__main__':
    main()
