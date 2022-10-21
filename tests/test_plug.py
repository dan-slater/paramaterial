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
        dataitem.write_data_to_csv(output_dir)
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
        # make temporary data directory
        self.data_dir = 'test_data'
        os.mkdir(self.data_dir)
        # make temporary data files
        self.dataitem1 = DataItem('test_id1', pd.DataFrame({'datacol1': [1.1, 2, 3], 'datacol2': [4, 5.1, 6]}),
                                  pd.Series({'test id': 'test_id1', 'info1': 1.4, 'info2': 'i1'}))
        self.dataitem2 = DataItem('test_id2', pd.DataFrame({'datacol1': [1, 2.2, 3], 'datacol2': [4, 5, 6.2]}),
                                  pd.Series({'test id': 'test_id2', 'info1': 1.5, 'info2': 'i2'}))
        self.dataitem3 = DataItem('test_id3', pd.DataFrame({'datacol1': [1, 2, 3.3], 'datacol2': [4, 5, 6.3]}),
                                  pd.Series({'test id': 'test_id3', 'info1': 1.6, 'info2': 'i3'}))
        # write data files
        self.dataitem1.write_data_to_csv(self.data_dir)
        self.dataitem2.write_data_to_csv(self.data_dir)
        self.dataitem3.write_data_to_csv(self.data_dir)
        # make temporary info file
        self.info_path = './test_info.xlsx'
        info_table = pd.DataFrame(
            {'test id': ['test_id1', 'test_id2', 'test_id3'], 'info1': [1.4, 1.5, 1.6], 'info2': ['i1', 'i2', 'i3']})
        info_table.to_csv(self.info_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.data_dir)
        os.remove(self.info_path)

    def test_init(self):
        dataset = DataSet(self.data_dir, self.info_path)
        self.assertEqual(dataset.data_dir, self.data_dir)
        self.assertEqual(dataset.info_path, self.info_path)
        self.assertEqual(len(dataset), 3)
        self.assertTrue(self.dataitem1 in dataset)
        self.assertTrue(self.dataitem2 in dataset)
        self.assertTrue(self.dataitem3 in dataset)

    def test_apply_function(self):
        dataset = DataSet(self.data_dir, self.info_path)

        def function(dataitem: DataItem) -> DataItem:
            dataitem.data['datacol1 + datacol2'] = dataitem.data['datacol1'] + dataitem.data['datacol2']
            return dataitem

        dataset.map_function(function)
        self.assertEqual(dataset[0].data['datacol1 + datacol2'].iloc[0], 5.1)
        self.assertEqual(dataset[1].data['datacol1 + datacol2'].iloc[1], 7.2)
        self.assertEqual(dataset[2].data['datacol1 + datacol2'].iloc[2], 9.3)


