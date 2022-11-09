"""Tests for models.py"""
import os
import shutil

import numpy as np
import pandas as pd
from paramaterial import ModelSet, DataSet, DataItem, ModelItem
import unittest

linear_model = lambda x, p1, p2: p1*x + p2


class TestModelSet(unittest.TestCase):
    """Tests for ModelSet class."""

    def SetUp(self):
        # make fake data with noise from a linear model
        x1 = np.linspace(0, 10, 100)
        x2 = np.linspace(0, 0.001, 100)
        y1 = linear_model(x1, 2, 3) + np.random.normal(0, 0.1, 100)
        y2 = linear_model(x2, 2, 3) + np.random.normal(0, 0.1, 100)

        self.data_dir = './test_data'
        self.info_path = './test_data/info.xlsx'
        self.test_id_key = 'test id'

        self.data1 = pd.DataFrame({'x': x1, 'y': y1})
        self.data2 = pd.DataFrame({'x': x2, 'y': y2})

        self.info1 = pd.Series({'test id': 'id_001', 'a': 1, 'b': 4})
        self.info2 = pd.Series({'test id': 'id_002', 'a': 2, 'b': 5})

        self.info_table = pd.DataFrame({'test id': ['id_001', 'id_002'], 'a': [1, 2], 'b': [4, 5]})
        self.data_items = list(map(DataItem, ['id_001', 'id_002'], [self.data1, self.data2], [self.info1, self.info2]))

        os.mkdir('./test_data')
        self.info_table.to_excel('./test_data/info.xlsx', index=False)
        self.data1.to_csv('./test_data/id_001.csv', index=False)
        self.data2.to_csv('./test_data/id_002.csv', index=False)

        # make params table
        self.param_names = ['p1', 'p2']
        self.bounds = [(0, 10), (0, 10)]
        self.initial_guess = [1, 1]

        self.params1 = pd.Series({'p1': 2, 'p2': 3})
        self.params2 = pd.Series({'p1': 2, 'p2': 3})

        self.params_table = pd.DataFrame({'test id': ['id_001', 'id_002'], 'p1': [2, 2], 'p2': [3, 3]})
        self.params_table.to_excel('./test_data/params.xlsx', index=False)

        # make results table
        self.results1 = pd.Series({'fun': 0.1, 'nfev': 100, 'success': True})
        self.results2 = pd.Series({'fun': 0.1, 'nfev': 100, 'success': True})

        self.results_table = pd.DataFrame({'test id': ['id_001', 'id_002'], 'fun': [0.1, 0.1], 'success': [True, True]})
        self.results_table.to_excel('./test_data/results.xlsx', index=False)

        # make model items
        self.model_items = list(map(
            ModelItem, ['id_001', 'id_002'],
            [self.info1, self.info2],
            [self.params1, self.params2],
            [self.results1, self.results2],
            [self.data1, self.data2]
        ))

    def TearDown(self):
        shutil.rmtree('./test_data')

    def test_init(self):
        """Test ModelSet.__init__"""
        model_set = ModelSet(
            model_func=linear_model,
            param_names=['p1', 'p2'],
            bounds=[(0, 10), (0, 10)]
        )
        self.assertEqual(model_set.model_func, linear_model)
        self.assertEqual(model_set.param_names, ['p1', 'p2'])
        self.assertEqual(model_set.bounds, [(0, 10), (0, 10)])

    def test_fit(self):
        """Test ModelSet.fit"""
        dataset = DataSet('./test_data', './test_data/info.xlsx', 'test id')
        model_set = ModelSet(
            model_func=linear_model,
            param_names=['p1', 'p2'],
            bounds=[(0, 10), (0, 10)]
        )
        model_set.fit(dataset, 'x', 'y')
        self.assertEqual(model_set.model_items, self.model_items)

    def test_predict(self):
        """Test ModelSet.predict"""
        dataset = DataSet('./test_data', './test_data/info.xlsx', 'test id')
        model_set = ModelSet(
            model_func=linear_model,
            param_names=['p1', 'p2'],
            bounds=[(0, 10), (0, 10)]
        )
        model_set.fit(dataset, 'x', 'y')
        predicted_ds = model_set.predict()
        self.assertEqual(predicted_ds.data_items[0].data, self.data_items[0].data)
        self.assertEqual(predicted_ds.data_items[1].data, self.data_items[1].data)
        self.assertEqual(predicted_ds.data_items[0].info, self.data_items[0].info)
        self.assertEqual(predicted_ds.data_items[1].info, self.data_items[1].info)





