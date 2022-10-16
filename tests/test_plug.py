"""Tests for the plug module."""
import os
import shutil
import unittest
from pathlib import Path
from typing import Dict

import pandas as pd
from paramaterial.plug import DataItem, DataSet


class TestDataItem(unittest.TestCase):
    """Tests for the DataItem class."""

    def setUp(self):
        self.test_id = 'test_id'
        self.data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.info = pd.Series({'test id': 'test_id', 'a': 1, 'b': 2})
        self.dataitem = DataItem(self.test_id, self.data, self.info)

    def test_read_from_csv(self):
        file_path = './test_data/test_data.csv'
        dataitem = DataItem.read_from_csv(file_path)
        self.assertEqual(dataitem.test_id, 'test_data')
        self.assertTrue(dataitem.data.equals(self.data))
        self.assertTrue(dataitem.info.equals(self.info))

    def test_get_row_from_info_table(self):
        info_table = pd.DataFrame({'test id': ['test_id', 'test_id2'], 'a': [1, 2], 'b': [2, 3]})
        dataitem = DataItem(self.test_id, self.data)
        dataitem.get_row_from_info_table(info_table)
        self.assertTrue(dataitem.info.equals(self.info))

    def test_write_to_csv(self):
        output_dir = './test_data/test_data_out'
        dataitem = DataItem(self.test_id, self.data, self.info)
        dataitem.write_to_csv(output_dir)
        file_path = output_dir + '/test_id.csv'
        dataitem = DataItem.read_from_csv(file_path)
        self.assertEqual(dataitem.test_id, 'test_id')
        self.assertTrue(dataitem.data.equals(self.data))
        self.assertTrue(dataitem.info.equals(self.info))
        shutil.rmtree(output_dir)

    def test_contains(self):
        dataitem = DataItem(self.test_id, self.data, self.info)
        dataitem_subset = DataItem(self.test_id, self.data.iloc[0:1, :], self.info)
        self.assertTrue(dataitem_subset in dataitem)


class TestDataSet(unittest.TestCase):
    """Tests for the DataSet class."""

    def setUp(self):
        self.data_dir = './test_data/test_data_dir'
        self.info_path = './test_data/test_data_info.csv'
        self.dataitem1 = DataItem('test_id1', pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
                                  pd.Series({'test id': 'test_id1', 'a': 1, 'b': 2}))
        self.dataitem2 = DataItem('test_id2', pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
                                  pd.Series({'test id': 'test_id2', 'a': 1, 'b': 2}))
        self.dataitem3 = DataItem('test_id3', pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
                                  pd.Series({'test id': 'test_id3', 'a': 1, 'b': 2}))
        self.dataitem1.write_to_csv(self.data_dir)
        self.dataitem2.write_to_csv(self.data_dir)
        self.dataitem3.write_to_csv(self.data_dir)
        info_table = pd.DataFrame({'test id': ['test_id1', 'test_id2', 'test_id3'], 'a': [1, 2, 3], 'b': [2, 3, 4]})
        info_table.to_csv(self.info_path, index=False)

    def test_init(self):
        dataset = DataSet(self.data_dir, self.info_path)
        self.assertEqual(dataset.data_dir, self.data_dir)
        self.assertEqual(dataset.info_path, self.info_path)
        self.assertEqual(len(dataset), 3)
        self.assertTrue(self.dataitem1 in dataset)
        self.assertTrue(self.dataitem2 in dataset)
        self.assertTrue(self.dataitem3 in dataset)

    def test_get_item(self):
        dataset = DataSet(self.data_dir, self.info_path)
        dataitem = dataset[{'test id': ['test_id1']}]
        self.assertTrue(dataitem.data.equals(self.dataitem1.data))
        self.assertTrue(dataitem.info.equals(self.dataitem1.info))

    def test_get_item_not_found(self):
        dataset = DataSet(self.data_dir, self.info_path)
        with self.assertRaises(KeyError):
            dataset[{'test id': ['test_id4']}]

    def test_get_subset(self):
        dataset = DataSet(self.data_dir, self.info_path)
        subset = dataset.get_subset({'a': 1, 'b': 2})
        self.assertEqual(len(subset), 1)
        self.assertTrue(self.dataitem1 in subset)
