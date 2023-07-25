import os
import pandas as pd
import shutil

from paramaterial.plug import DataSet, DataItem

if not os.path.exists('./test_data'):
    os.mkdir('./test_data')
else:
    shutil.rmtree('./test_data')
    os.mkdir('./test_data')

data_dir = './test_data'
info_path = './test_data/info.xlsx'
test_id_key = 'test_id'

data1 = pd.DataFrame({'x': [1.1, 2, 3], 'y': [4.1, 5, 6]})
data2 = pd.DataFrame({'x': [1, 2.2, 3], 'y': [4, 5.2, 6]})
data3 = pd.DataFrame({'x': [1, 2, 3.3], 'y': [4, 5, 6.3]})

info1 = pd.Series({'test_id': 'id_001', 'a': 1, 'b': 4})
info2 = pd.Series({'test_id': 'id_002', 'a': 2, 'b': 5})
info3 = pd.Series({'test_id': 'id_003', 'a': 3, 'b': 6})

info_table = pd.DataFrame({'test_id': ['id_001', 'id_002', 'id_003'],
                                'a': [1, 2, 3],
                                'b': [4, 5, 6]})

data_items = [DataItem('id_001', data1, info_table),
                   DataItem('id_002', data2, info_table),
                   DataItem('id_003', data3, info_table)]

info_table.to_excel('./test_data/info.xlsx', index=False)

data1.to_csv('./test_data/id_001.csv', index=False)
data2.to_csv('./test_data/id_002.csv', index=False)
data3.to_csv('./test_data/id_003.csv', index=False)


ds = DataSet(info_path=info_path, data_dir=data_dir, test_id_key=test_id_key)

def di_func(di: DataItem) -> DataItem:
    di.data['x'] = di.data['x'] * 2
    di.info['a'] = di.info['a'] * 2
    return di

print(ds.info_table)

ds = ds.apply(di_func)

print(ds.info_table)
