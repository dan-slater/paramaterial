import os
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from paramaterial import DataItem, DataSet, ModelItem, ModelSet


def linear_model(x, params):
    return params[0] + params[1] * x

def main():
    x1 = np.linspace(0, 10, 100)
    x2 = np.linspace(0, 10, 100)
    y1 = linear_model(x1, [2, 3]) + np.random.normal(0, 0.1, 100)
    y2 = linear_model(x2, [2, 3]) + np.random.normal(0, 0.1, 100)

    data_dir = './test_data'
    info_path = './test_data/info.xlsx'
    test_id_key = 'test id'

    data1 = pd.DataFrame({'x': x1, 'y': y1})
    data2 = pd.DataFrame({'x': x2, 'y': y2})

    info1 = pd.Series({'test id': 'id_001', 'a': 1, 'b': 4})
    info2 = pd.Series({'test id': 'id_002', 'a': 2, 'b': 5})

    info_table = pd.DataFrame({'test id': ['id_001', 'id_002'], 'a': [1, 2], 'b': [4, 5]})
    data_items = list(map(DataItem, ['id_001', 'id_002'], [data1, data2], [info1, info2]))

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir('./test_data')
    info_table.to_excel('./test_data/info.xlsx', index=False)
    data1.to_csv('./test_data/id_001.csv', index=False)
    data2.to_csv('./test_data/id_002.csv', index=False)

    ds = DataSet(data_dir, info_path, test_id_key)
    ms = ModelSet(linear_model, param_names=['p1', 'p2'], bounds=[(0, 10), (0, 10)], initial_guess=[2, 4])

    ms.fit(ds, x_key='x', y_key='y')

    ds_p = ms.predict()

    fig, axs = plt.subplots(1,2)
    axs[0].plot(ds[0].data['x'], ds[0].data['y'], 'x')
    axs[0].plot(ds_p[0].data['x'], ds_p[0].data['y'], '-')
    axs[1].plot(ds[1].data['x'], ds[1].data['y'], 'x')
    axs[1].plot(ds_p[1].data['x'], ds_p[1].data['y'], '-')
    plt.show()




if __name__ == '__main__':
    main()