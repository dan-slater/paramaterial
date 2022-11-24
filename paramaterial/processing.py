"""Module with functions for modelling a stress-strain curve."""
import os
from typing import List

import numpy as np
import pandas as pd

from paramaterial.plug import DataItem, DataSet


def determine_proportional_limits_and_elastic_modulus(
        di: DataItem, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa',
        preload: float = 0, preload_key: str = 'Stress_MPa', max_strain: float = 0.02,
        suppress_numpy_warnings: bool = False) -> DataItem:
    """Determine the upper proportional limit (UPL) and lower proportional limit (LPL) of a stress-strain curve.
    The UPL is the point that minimizes the residuals of the slope fit between that point and the specified preload.
    The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL.
    The elastic modulus is the slope between the UPL and LPL."""
    if suppress_numpy_warnings:
        np.seterr(all="ignore")

    data = di.data[di.data[strain_key] <= max_strain]

    UPL = (0, 0)
    LPL = (0, 0)

    def fit_line(_x, _y):
        n = len(_x)  # number of points
        m = (n*np.sum(_x*_y) - np.sum(_x)*np.sum(_y))/(n*np.sum(np.square(_x)) - np.square(np.sum(_x)))  # slope
        c = (np.sum(_y) - m*np.sum(_x))/n  # intercept
        S_xy = (n*np.sum(_x*_y) - np.sum(_x)*np.sum(_y))/(n - 1)  # empirical covariance
        S_x = np.sqrt((n*np.sum(np.square(_x)) - np.square(np.sum(_x)))/(n - 1))  # x standard deviation
        S_y = np.sqrt((n*np.sum(np.square(_y)) - np.square(np.sum(_y)))/(n - 1))  # y standard deviation
        r = S_xy/(S_x*S_y)  # correlation coefficient
        S_m = np.sqrt((1 - r**2)/(n - 2))*S_y/S_x  # slope standard deviation
        S_rel = S_m/m  # relative deviation of slope
        return S_rel

    x = data[strain_key].values
    y = data[stress_key].values

    x_upl = x[data[preload_key] >= preload]
    y_upl = y[data[preload_key] >= preload]

    S_min = np.inf
    for i in range(3, len(x)):
        S_rel = fit_line(x_upl[:i], y_upl[:i])  # fit a line to the first i points after the preload
        if S_rel < S_min:
            S_min = S_rel
            UPL = (x_upl[i], y_upl[i])

    x_lpl = x[x <= UPL[0]]
    y_lpl = y[x <= UPL[0]]

    S_min = np.inf
    for j in range(len(x), 3, -1):
        S_rel = fit_line(x_lpl[j:], y_lpl[j:])  # fit a line to the last i points before the UPL
        if S_rel < S_min:
            S_min = S_rel
            LPL = (x_lpl[j], y_lpl[j])

    di.info['UPL_0'] = UPL[0]
    di.info['UPL_1'] = UPL[1]
    di.info['LPL_0'] = LPL[0]
    di.info['LPL_1'] = LPL[1]
    di.info['E'] = (UPL[1] - LPL[1])/(UPL[0] - LPL[0])
    return di


def determine_proof_stress(di: DataItem, proof_strain: float = 0.002, strain_key: str = 'Strain',
                           stress_key: str = 'Stress_MPa') -> DataItem:
    """Find the proof stress of a stress-strain curve."""
    E = di.info['E']
    x_data = di.data[strain_key].values
    y_data = di.data[stress_key].values
    x_shift = proof_strain
    y_line = E*(x_data - x_shift)
    cut = np.where(np.diff(np.sign(y_line - y_data)) != 0)[0][-1]
    m = (y_data[cut + 1] - y_data[cut])/(x_data[cut + 1] - x_data[cut])
    xl = x_data[cut]
    yl = y_line[cut]
    xd = x_data[cut]
    yd = y_data[cut]
    K = np.array(
        [[1, -E],
         [1, -m]]
    )
    f = np.array(
        [[yl - E*xl],
         [yd - m*xd]]
    )
    d = np.linalg.solve(K, f).flatten()
    di.info[f'YP_{proof_strain}_0'] = d[1]
    di.info[f'YP_{proof_strain}_1'] = d[0]
    return di


def determine_ultimate_strength():
    pass


def determine_fracture_point():
    pass


def determine_flow_stress():
    pass


def correct_uniaxial_compression_friction(
        di: DataItem,
        mu_key: str = 'mu',
        h_0_key: str = 'h_0',
        D_0_key: str = 'D_0',
        disp_key: str = 'Disp(mm)',
        F_key: str = 'Force(kN)',
) -> DataItem:
    """

    """
    mu = di.info[mu_key] # friction coefficient
    h_0 = di.info[h_0_key]  # initial height in axial direction
    D_0 = di.info[D_0_key]  # initial diameter
    h = h_0 - di.data[disp_key]  # instantaneous height
    d = D_0*np.sqrt(h_0/h)  # instantaneous diameter
    P = di.data[F_key]*1000*4/(np.pi*d**2)  # pressure (MPa)
    di.data['Pressure(MPa)'] = P
    di.data['Corrected_Stress(MPa)'] = P/(1 + (mu*d)/(3*h))  # correct stress
    return di


def correct_plane_strain_compression_friction():
    pass


def smooth_load_cell_ringing():
    pass


def correct_for_compliance():
    pass


def correct_for_thermal_expansion():
    pass


def correct_foot():
    pass
