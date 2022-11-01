import pandas as pd
import numpy as np
import paramaterial as pam
from paramaterial.plug import DataSet, DataItem
import matplotlib.pyplot as plt


prepared_set = DataSet('data/01 prepared data', 'info/01 prepared info.xlsx').sort_by(['temperature', 'lot'])\


def test(di: DataItem):
    di.info['test'] = di.info['test id'][-3:]
    return di


prepared_set = prepared_set.apply_function(test)

for di in prepared_set:
    print(di.info['test'])

trimmed_set = pam.processing.read_screening_pdf_to(prepared_set, '02 trimming screening marked.pdf')
print(trimmed_set)


for di in trimmed_set:
    print(di.test_id, di.info['comment'])

