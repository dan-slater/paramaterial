"""Tests for the processing module."""

import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import paramaterial as pam
from paramaterial.plug import DataItem, DataSet


class TestProcessing(unittest.TestCase):
    """Tests for the processing module."""

    def setUp(self):
        self.ds = DataSet('processing test data/test data/01 prepared data',
                          'processing test data/test info/01 prepared info.xlsx')

    def test_find_upl_and_lpl(self):
        """Test the find_upl_and_lpl function."""
        ds = self.ds.apply(pam.processing.find_upl_and_lpl, preload=36, preload_key='Stress_MPa',
                           suppress_numpy_warnings=True)

        def trim_small(di):
            di.data = di.data[di.data['Strain'] < 0.01]
            return di

        def make_strain_percent(di):
            di.data['Strain'] = di.data['Strain']*100
            return di

        ds = ds.apply(trim_small).apply(make_strain_percent)
        ds.write_output('test data/02 find_upl_and_lpl_data', 'test info/02 find_upl_and_lpl_info.xlsx')
