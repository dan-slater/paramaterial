"""Module for testing Zener-Holloman regression plot."""

import unittest

from matplotlib import pyplot as plt

import paramaterial as pam

class TestZenerHollomanPlot(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = pam.DataSet('info/03 screened info.xlsx', 'data/03 screened data', test_id_key='test id')
        self.dataset.apply(pam.find_flow_stress_values, stress_key='Stress(MPa)', flow_strain=0.3)

    def test_calculate_ZH_parameter(self):
        ds = self.dataset
        ds = ds.apply(pam.calculate_ZH_parameter, temperature_key='temperature', rate_key='rate', Q_key='Q')
        ds.info_table.to_csv('info/04 ZH parameter info.csv')

    def test_apply_ZH_regression(self):
        ds = self.dataset
        ds = ds.apply(pam.find_flow_stress_values, stress_key='Stress(MPa)', flow_strain=0.3)
        ds = ds.apply(pam.calculate_ZH_parameter, temperature_key='temperature', rate_key='rate', Q_key='Q')
        ds = pam.apply_ZH_regression(ds=ds)
        ds.info_table.to_csv('info/04 ZH regression info.csv')

    def test_zener_holloman_plot(self):
        ds = self.dataset
        ds = ds.apply(pam.find_flow_stress_values, stress_key='Stress(MPa)', flow_strain=0.3)
        ds = ds.apply(pam.calculate_ZH_parameter, temperature_key='temperature', rate_key='rate', Q_key='Q')
        ds = pam.apply_ZH_regression(ds=ds)
        pam.plot_ZH_regression(ds=ds)
        plt.show()
