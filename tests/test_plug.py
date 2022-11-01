"""Tests for the plug module."""
import os
import shutil
import unittest
from pathlib import Path

import pandas as pd

from paramaterial.plug import DataItem, DataSet


class TestDataItem(unittest.TestCase):
    """Tests for the DataItem class."""

    def setUp(self):
        os.mkdir('./test_data')
        self.test_id = 'id_001'
        self.data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 3, 4]})
        self.info = pd.Series({'test id': 'id_001', 'a': 1, 'b': 2})
        self.data.to_csv('./test_data/id_001.csv', index=False)

    def tearDown(self):
        shutil.rmtree('./test_data')

    def test_read_data_from_csv(self):
        file_path = './test_data/id_001.csv'
        dataitem = DataItem.read_data_from_csv(file_path)
        self.assertEqual(dataitem.test_id, self.test_id)
        self.assertTrue(dataitem.data.equals(self.data))

    def test_update_info_from_table(self):
        file_path = './test_data/id_001.csv'
        dataitem = DataItem.read_data_from_csv(file_path)
        info_table = pd.DataFrame({'test id': ['id_001'], 'a': [1], 'b': [2]})
        dataitem.set_info(info_table, 'test id')
        self.assertTrue(dataitem.info.equals(self.info))

    def test_write_data_to_csv(self):
        file_path = './test_data/id_001.csv'
        dataitem = DataItem.read_data_from_csv(file_path)
        os.remove(file_path)
        dataitem.write_data_to_csv('./test_data')
        self.assertTrue(Path('./test_data/id_001.csv').exists())
        data = pd.read_csv('./test_data/id_001.csv')
        self.assertTrue(data.equals(self.data))


class TestDataSet(unittest.TestCase):
    """Tests for the DataSet class."""

    def setUp(self):
        self.data_dir = './test_data'
        self.info_path = './test_data/info.xlsx'
        self.test_id_key = 'test id'

        self.data1 = pd.DataFrame({'x': [1.1, 2, 3], 'y': [4.1, 5, 6]})
        self.data2 = pd.DataFrame({'x': [1, 2.2, 3], 'y': [4, 5.2, 6]})
        self.data3 = pd.DataFrame({'x': [1, 2, 3.3], 'y': [4, 5, 6.3]})

        self.info1 = pd.Series({'test id': 'id_001', 'a': 1, 'b': 4})
        self.info2 = pd.Series({'test id': 'id_002', 'a': 2, 'b': 5})
        self.info3 = pd.Series({'test id': 'id_003', 'a': 3, 'b': 6})

        self.info_table = pd.DataFrame({'test id': ['id_001', 'id_002', 'id_003'],
                                        'a': [1, 2, 3],
                                        'b': [4, 5, 6]})

        self.dataitems = map(DataItem,
                             ['id_001', 'id_002', 'id_003'],
                             [self.data1, self.data2, self.data3],
                             [self.info1, self.info2, self.info3])

        os.mkdir('./test_data')

        self.info_table.to_excel('./test_data/info.xlsx', index=False)

        self.data1.to_csv('./test_data/id_001.csv', index=False)
        self.data2.to_csv('./test_data/id_002.csv', index=False)
        self.data3.to_csv('./test_data/id_003.csv', index=False)

    def tearDown(self):
        shutil.rmtree('./test_data')

    def test_init(self):
        dataset = DataSet(self.data_dir, self.info_path, self.test_id_key)
        self.assertTrue(dataset.info_table.convert_dtypes().equals(self.info_table.convert_dtypes()))
        self.assertEqual(len(list(dataset.dataitems)), 3)
        # todo:

    def test_get_info_table(self):
        dataset = DataSet(self.data_dir, self.info_path, self.test_id_key)
        self.assertTrue(dataset.info_table.convert_dtypes().equals(self.info_table.convert_dtypes()))

    def test_set_info_table(self):
        dataset = DataSet(self.data_dir, self.info_path, self.test_id_key)
        info_table = pd.DataFrame({'test id': ['id_001', 'id_002', 'id_003'],
                                   'a': [9, 2, 3],
                                   'b': [9, 5, 6]})
        dataset.info_table = info_table
        self.assertTrue(dataset.info_table.convert_dtypes().equals(info_table.convert_dtypes()))

    # def test_init(self):
    #     dataset = DataSet(self.data_dir, self.info_path, self.test_id_key)
    #     print(dataset.info_table)
    #     self.assertEqual(dataset.data_dir, self.data_dir)
    #     self.assertEqual(dataset.info_path, self.info_path)
    #     self.assertEqual(dataset.test_id_key, self.test_id_key)
    #     self.assertEqual(len(list(dataset.dataitems)), 3)
    #     print(dataset.info_table)
    #     self.assertTrue(dataset.info_table.equals(self.info_table))

    # def test_iter(self):
    #     dataset = DataSet(self.data_dir, self.info_path, self.test_id_key)
    #
    #     # update info table and check if dataitems are updated
    #     self.info_table['c'] = [7, 8, 9]
    #
    #     for di in dataset:
    #         self.assertTrue(di.info.equals(self.info_table.loc[di.test_id]))
    #
    # def test_apply_function(self):
    #     dataset = DataSet(self.data_dir, self.info_path, self.test_id_key)
    #
    #     def test(di: DataItem):
    #         di.info['test'] = di.info['test id'][-3:]
    #         return di
    #
    #     dataset = dataset.apply_function(test)
    #
    #     self.assertEqual(list(dataset.dataitems)[0].info['test'], '001')
    #     self.assertEqual(list(dataset.dataitems)[1].info['test'], '002')
    #     self.assertEqual(list(dataset.dataitems)[2].info['test'], '003')
    #
    #     self.assertEqual(dataset.info_table['test'], ['001', '002', '003'])
