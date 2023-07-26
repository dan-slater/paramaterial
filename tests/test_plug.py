"""Tests for the plug module."""
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paramaterial.plug import DataItem, DataSet


class TestDataSet(unittest.TestCase):
    """Tests for the DataSet class."""

    def setUp(self):
        # create test data if it does not exist, otherwise overwrite it
        if not os.path.exists('./test_data'):
            os.mkdir('./test_data')
        else:
            shutil.rmtree('./test_data')
            os.mkdir('./test_data')

        self.data_dir = './test_data'
        self.info_path = './test_data/info.xlsx'
        self.test_id_key = 'test_id'

        # create test data
        self.data1 = pd.DataFrame({'x': [1.1, 2, 3], 'y': [4.1, 5, 6]})
        self.data2 = pd.DataFrame({'x': [1, 2.2, 3], 'y': [4, 5.2, 6]})
        self.data3 = pd.DataFrame({'x': [1, 2, 3.3], 'y': [4, 5, 6.3]})

        # create test info
        self.info1 = pd.Series({'test_id': 'id_001', 'a': 1, 'b': 4})
        self.info2 = pd.Series({'test_id': 'id_002', 'a': 2, 'b': 5})
        self.info3 = pd.Series({'test_id': 'id_003', 'a': 3, 'b': 6})

        # create test info table
        self.info_table = pd.DataFrame({'test_id': ['id_001', 'id_002', 'id_003'],
                                        'a': [1, 2, 3],
                                        'b': [4, 5, 6]})

        # create test data items
        self.data_items = [DataItem('id_001', self.data1, self.info1),
                           DataItem('id_002', self.data2, self.info2),
                           DataItem('id_003', self.data3, self.info3)]

        # write test info table and data items to files
        self.info_table.to_excel('./test_data/info.xlsx', index=False)
        self.data1.to_csv('./test_data/id_001.csv', index=False)
        self.data2.to_csv('./test_data/id_002.csv', index=False)
        self.data3.to_csv('./test_data/id_003.csv', index=False)

    def tearDown(self):
        # delete test data files
        shutil.rmtree('./test_data')

    def test_init(self):
        """Test the initialization of the DataSet."""
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)

        # Assert if the class is correctly initialized
        self.assertIsInstance(dataset, DataSet)
        self.assertEqual(dataset.info_path, self.info_path)
        self.assertEqual(dataset.data_dir, self.data_dir)
        self.assertEqual(dataset.test_id_key, self.test_id_key)
        self.assertEqual(len(dataset.data_items), 3)

    def test_set_info_table(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)
        info_table = pd.DataFrame({'test_id': ['id_001', 'id_002', 'id_003'],
                                   'a': [9, 2, 3],
                                   'b': [9, 5, 6]})
        dataset.info_table = info_table
        self.assertTrue(dataset.info_table.convert_dtypes().equals(info_table.convert_dtypes()))

        # test if an applied function is still applied after setting the info table
        def test(di: DataItem):
            di.data = di.data[:-1]
            return di

        dataset = dataset.apply(test)
        dataset.info_table = info_table
        self.assertTrue(np.equal(dataset[0].data['x'], self.data1['x'][:-1]).all())

        # test if the info in the data items is updated when the info table is updated
        new_info_table = pd.DataFrame({'test_id': ['id_001', 'id_002', 'id_003'],
                                       'a': [1, 2, 3],
                                       'b': [9, 5, 6],
                                       'c': [7, 8, 9]})
        dataset.info_table = new_info_table
        self.assertTrue(dataset[0].info['c'] == 7)
        self.assertEqual(dataset.data_items[0].info['a'], 1)

        # test if data items are removed when the info table is updated
        new_info_table = pd.DataFrame({'test_id': ['id_001', 'id_002'],
                                        'a': [1, 2],
                                        'b': [9, 5],
                                        'c': [7, 8]})
        dataset.info_table = new_info_table
        self.assertEqual(len(dataset), 2)

    def test_get_info_table(self):
        # test if get info table works before and after applying a function and before and after updating the info table
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)
        pd.testing.assert_frame_equal(dataset.info_table, self.info_table)

        def test(di: DataItem):
            di.data['x'] = di.data['x'] * 2
            di.info['a'] = di.info['a'] + 1
            return di

        dataset = dataset.apply(test)
        new_info_table = pd.DataFrame({'test_id': ['id_001', 'id_002', 'id_003'],
                                       'a': [2, 3, 4],
                                       'b': [4, 5, 6]})
        pd.testing.assert_frame_equal(dataset.info_table, new_info_table)
        new_info_table = pd.DataFrame({'test_id': ['id_001', 'id_002', 'id_003'],
                                       'a': [1, 2, 3],
                                       'b': [4, 5, 6],
                                       'c': [7, 8, 9]})
        dataset.info_table = new_info_table
        pd.testing.assert_frame_equal(dataset.info_table, new_info_table)

    def test_apply(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)

        def apply_func(di: DataItem) -> DataItem:
            di.data['x'] = di.data['x'] + 1
            di.data['z'] = di.data['x'] + di.data['y']
            di.info['new'] = 1
            di.info['a'] = di.info['a'] + 1
            di.info['c'] = di.info['a'] + di.info['b']
            return di

        new_ds = dataset.apply(apply_func)
        self.assertTrue(new_ds[0].info['new'] == 1)
        self.assertTrue('new' in new_ds.info_table.columns)
        self.assertTrue(new_ds.info_table['new'][0] == 1)

        self.assertEqual(dataset.data_items[0].test_id, 'id_001')
        self.assertTrue(np.equal(dataset.data_items[0].data['x'], self.data1['x'] + 1).all())
        self.assertTrue(np.equal(dataset.data_items[0].data['y'], self.data1['y']).all())
        self.assertTrue(np.equal(dataset.data_items[0].data['z'], self.data1['x'] + self.data1['y'] + 1).all())

        self.assertEqual(dataset.data_items[0].info['a'], self.info1['a'] + 1)
        self.assertEqual(dataset.data_items[0].info['b'], self.info1['b'])
        self.assertEqual(dataset.data_items[0].info['c'], self.info1['a'] + self.info1['b'] + 1)

    def test_write_output(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)

        def apply_func(di: DataItem) -> DataItem:
            di.data['x'] = di.data['x'] + 1
            di.data['z'] = di.data['x'] + di.data['y']
            di.info['a'] = di.info['a'] + 1
            di.info['c'] = di.info['a'] + di.info['b']
            return di

        dataset = dataset.apply(apply_func)
        dataset.write_output('./test_data/output/info.xlsx', './test_data/output')

        self.assertTrue(os.path.exists('./test_data/output'))
        self.assertTrue(os.path.exists('./test_data/output/info.xlsx'))
        self.assertTrue(os.path.exists('./test_data/output/id_001.csv'))
        self.assertTrue(os.path.exists('./test_data/output/id_002.csv'))
        self.assertTrue(os.path.exists('./test_data/output/id_003.csv'))

        self.assertTrue(pd.read_csv('./test_data/output/id_001.csv').convert_dtypes().equals(
            dataset.data_items[0].data.convert_dtypes()))
        self.assertTrue(pd.read_csv('./test_data/output/id_002.csv').convert_dtypes().equals(
            dataset.data_items[1].data.convert_dtypes()))
        self.assertTrue(pd.read_csv('./test_data/output/id_003.csv').convert_dtypes().equals(
            dataset.data_items[2].data.convert_dtypes()))
        print(pd.read_excel('./test_data/output/info.xlsx').convert_dtypes())
        print(dataset.info_table.convert_dtypes())

        self.assertTrue(pd.read_excel('./test_data/output/info.xlsx').convert_dtypes().equals(
            dataset.info_table.convert_dtypes()))

    def test_sort_by(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)
        df = pd.DataFrame({'test_id': ['id_001', 'id_002', 'id_003'],
                           'a': [9, 2, 3],
                           'b': [9, 5, 6]})
        dataset.info_table = df
        print(dataset.info_table)
        dataset = dataset.sort_by('a')
        print(dataset.info_table)
        print(df)
        df = df.sort_values('a').reset_index(drop=True)
        print(df)
        self.assertTrue(dataset.info_table.convert_dtypes().equals(df.convert_dtypes()))
        self.assertEqual(dataset.data_items[0].test_id, 'id_002')
        self.assertTrue(dataset.data_items[0].data.convert_dtypes().equals(self.data2.convert_dtypes()))

        # test if an applied function is still applied after sorting
        def test(di: DataItem):
            di.data = di.data[:-1]
            return di

        dataset = dataset.apply(test)
        dataset = dataset.sort_by('a')
        self.assertTrue(dataset.data_items[0].data.convert_dtypes().equals(self.data2[:-1].convert_dtypes()))

    def test_iter(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)

        info_table = dataset.info_table
        info_table['c'] = [7, 8, 9]
        dataset.info_table = info_table

        for di in dataset:
            self.assertTrue(di.info['c'] in [7, 8, 9])

    def test_len(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)
        self.assertEqual(len(dataset), 3)

    def test_subset(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)

        subset = dataset.subset({'a': [1, 2], 'b': [4]})

        pd.testing.assert_frame_equal(subset.info_table, self.info_table[:1])
        pd.testing.assert_frame_equal(subset.data_items[0].data, self.data1)

        self.assertTrue(type(dataset.subset({'a': [1, 2], 'b': [4]}) is DataSet))
        self.assertTrue(len(dataset.subset({'a': [1, 2], 'b': [4]})) == 1)
        self.assertTrue(len(dataset.subset({'a': [1, 2], 'b': [4, 5]})) == 2)

        def trim(di: DataItem) -> DataItem:
            di.data = di.data[:-1]
            return di

        dataset = dataset.apply(trim)
        self.assertTrue(dataset.subset({'a': [1, 2], 'b': [4, 5]}).data_items[0].data.convert_dtypes().equals(
            self.data1[:-1].convert_dtypes()))

    def test_getitem(self):
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)

        self.assertTrue(dataset[0].data.convert_dtypes().equals(self.data1.convert_dtypes()))
        self.assertTrue(dataset[1].info.convert_dtypes().equals(self.info2.convert_dtypes()))

        self.assertTrue(dataset['id_001'].data.convert_dtypes().equals(self.data1.convert_dtypes()))
        self.assertTrue(dataset[1:].data_items[0].data.convert_dtypes().equals(self.data2.convert_dtypes()))

        self.assertTrue(type(dataset[:1]) is DataSet)
        self.assertTrue(dataset.subset({'a': [1, 2], 'b': [4]}).data_items[0].data.convert_dtypes().equals(
            self.data1.convert_dtypes()))

        self.assertTrue(type(dataset.subset({'a': [1, 2], 'b': [4]})) is DataSet)
        self.assertTrue(len(dataset.subset({'a': [1, 2], 'b': [4]})) == 1)
        self.assertTrue(len(dataset.subset({'a': [1, 2], 'b': [4, 5]})) == 2)

        def trim(di: DataItem) -> DataItem:
            di.data = di.data[:-1]
            return di

        dataset = dataset.apply(trim)
        self.assertTrue(dataset.subset({'a': [1, 2], 'b': [4, 5]}).data_items[0].data.convert_dtypes().equals(
            self.data1[:-1].convert_dtypes()))

    def test_copy(self):
        # test if an applied function is still applied after copying
        dataset = DataSet(self.info_path, self.data_dir, self.test_id_key)

        def trim(di: DataItem) -> DataItem:
            di.data = di.data[:-1]
            return di

        dataset = dataset.apply(trim)
        dataset_copy = dataset.copy()
        self.assertTrue(dataset_copy.data_items[0].data.convert_dtypes().equals(
            self.data1[:-1].convert_dtypes()))
