# todo: prepare data for processing
# todo: extract test ids from folder names into info
# todo: extract gleeble output measurements, format as csv and rename by test id
# todo: develop test report template
import os
from pathlib import Path
from enum import Enum


def main():
    prepare_data()
    trim_data()
    correct_data()


def prepare_data():
    raw_data = 'data/01 raw data'
    # extract info from folder names
    for filename in os.listdir(raw_data):
        print(filename)


def trim_data():
    pass


def correct_data():
    pass


if __name__ == '__main__':
    main()