import pandas as pd
import numpy as np
import paramaterial as pam
from paramaterial.plug import DataSet, DataItem
import matplotlib.pyplot as plt


prepared_set = DataSet('data/01 prepared data', 'info/01 prepared info.xlsx').sort_by(['temperature', 'lot'])

trimmed_set = pam.processing.read_screening_pdf_to(prepared_set, '02 trimming screening marked.pdf')
print(trimmed_set)


for di in trimmed_set:
    print(di.test_id, di.info['comment'])

