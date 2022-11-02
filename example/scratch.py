import pandas as pd
import numpy as np
import paramaterial as pam
from paramaterial.plug import DataSet, DataItem
import matplotlib.pyplot as plt


prepared_set = DataSet('data/01 prepared data', 'info/01 prepared info.xlsx').sort_by(['temperature', 'lot'])

prepared_set = prepared_set[{'test type': ['UT']}]

assert(all(type(di.info) == pd.Series for di in prepared_set))

def test(di: DataItem):
    di.info['test'] = di.info['test id'][-3:]
    return di


prepared_set = prepared_set.apply(test)
assert(all(type(di.info) == pd.Series for di in prepared_set))

for di in prepared_set:
    print(di.info['test'])

assert(all(type(di.info) == pd.Series for di in prepared_set))


def trim(di: DataItem) -> DataItem:
    di.data = di.data[:-1]
    return di

assert(all(type(di.info) == pd.Series for di in prepared_set))

trimmed_set = prepared_set.apply(trim)
a = trimmed_set[0].data
assert(all(type(di.info) == pd.Series for di in trimmed_set))

trimmed_set = pam.processing.read_screening_pdf_to(trimmed_set, '02 trimming screening marked.pdf')
assert(all(type(di.info) == pd.Series for di in trimmed_set))

print(trimmed_set.info_table['comment'].unique())