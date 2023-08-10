import unittest
import os
import pandas as pd
from paramaterial.plug import DataItem, DataSet


class TestDataSet(unittest.TestCase):
    """Tests for the DataSet class."""

    def setUp(self):
        # Create synthetic test data
        self.data1 = pd.DataFrame({'strain': [0, 0.1, 0.2], 'stress': [0, 100, 200]})
        self.data2 = pd.DataFrame({'strain': [0, 0.15, 0.3], 'stress': [0, 150, 300]})
        self.data3 = pd.DataFrame({'strain': [0, 0.05, 0.1], 'stress': [0, 50, 100]})

        # Create synthetic test info
        self.info1 = pd.Series({'test_id': 'id_001', 'temperature': 20, 'strain_rate': 0.01})
        self.info2 = pd.Series({'test_id': 'id_002', 'temperature': 25, 'strain_rate': 0.02})
        self.info3 = pd.Series({'test_id': 'id_003', 'temperature': 30, 'strain_rate': 0.03})

        # Create synthetic test info table
        self.info_table = pd.DataFrame({
            'test_id': ['id_001', 'id_002', 'id_003'],
            'temperature': [20, 25, 30],
            'strain_rate': [0.01, 0.02, 0.03]
        })

        # Create synthetic test data items
        self.data_items = [
            DataItem('id_001', self.data1, self.info1),
            DataItem('id_002', self.data2, self.info2),
            DataItem('id_003', self.data3, self.info3)
        ]

    def test_init(self):
        """Test the initialization of the DataSet."""
        dataset = DataSet(info_path=None, data_dir=None, test_id_key='test_id')
        dataset.data_items = self.data_items

        self.assertIsInstance(dataset, DataSet)
        self.assertEqual(len(dataset.data_items), 3)

    def test_info_table(self):
        """Test the info_table property of the DataSet."""
        dataset = DataSet(info_path=None, data_dir=None, test_id_key='test_id')
        dataset.data_items = self.data_items
        pd.testing.assert_frame_equal(dataset.info_table, self.info_table)

    def test_apply(self):
        """Test the apply method of the DataSet."""
        dataset = DataSet(info_path=None, data_dir=None, test_id_key='test_id')
        dataset.data_items = self.data_items

        def apply_func(di: DataItem) -> DataItem:
            di.data['stress'] = di.data['stress'] * 2
            return di

        new_ds = dataset.apply(apply_func)
        self.assertEqual(new_ds[0].data['stress'][1], 200)

    def test_subset(self):
        """Test the subset method of the DataSet."""
        dataset = DataSet(info_path=None, data_dir=None, test_id_key='test_id')
        dataset.data_items = self.data_items
        subset = dataset.subset({'temperature': [20, 25]})
        self.assertEqual(len(subset), 2)

    def test_sort_by(self):
        """Test the sort_by method of the DataSet."""
        dataset = DataSet(info_path=None, data_dir=None, test_id_key='test_id')
        dataset.data_items = self.data_items

        # Sort by 'temperature' in descending order
        sorted_dataset = dataset.sort_by('temperature', ascending=False)

        # Expected result after sorting by 'temperature'
        expected_info_table = self.info_table.sort_values(by='temperature', ascending=False).reset_index(drop=True)

        # Assert if the info_table is sorted correctly
        pd.testing.assert_frame_equal(sorted_dataset.info_table, expected_info_table)

        # Assert if the corresponding data_items are also sorted correctly
        self.assertTrue(sorted_dataset.data_items[0].data.equals(self.data3))
        self.assertTrue(sorted_dataset.data_items[1].data.equals(self.data2))
        self.assertTrue(sorted_dataset.data_items[2].data.equals(self.data1))
