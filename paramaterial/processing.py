"""Module with functions for processing materials test data. This includes functionality for finding properties
like yield strength Young's modulus from stress-strain curves, as well as post-processing functions for cleaning
and correcting experimental measurements."""
from typing import Tuple, Optional, Union

import numpy as np

from paramaterial.plug import DataItem, DataSet


def find_UTS(ds: DataSet, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa',
             max_strain: Optional[float] = None) -> DataSet:
    """Find the ultimate tensile strength (UTS) of an engineering stress-strain curves in the DataSet. The UTS is
    defined
    as the maximum stress in the curve. The UTS is added to the DataSet.info dictionary as 'UTS_1' and the strain at the
    UTS is added as 'UTS_0'.

    Args:
        ds: DataSet containing the stress-strain curves.
        strain_key: Key for the strain data in the DataItem.
        stress_key: Key for the stress data in the DataItem.
        max_strain: Maximum strain to consider when finding the UTS. If None, the maximum strain in the curve is used.

    Returns: DataSet with UTS added to info_table.
    """
    ds = ds.copy()

    def find_di_UTS(di):
        data = di.data[di.data[strain_key] <= max_strain] if max_strain is not None else di.data
        x = data[strain_key].values
        y = data[stress_key].values
        di.info['UTS_1'] = np.max(y)
        di.info['UTS_0'] = x[np.argmax(y)]
        return di

    return ds.apply(find_di_UTS)


def find_fracture_point(ds: DataSet, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa') -> DataSet:
    """Find the fracture point for the stress-strain curves in the DataSet. The fracture point is defined as the
    maximum strain in the curve. The fracture point is added to the DataSet.info dictionary as 'FP_1' and the stress at
    the fracture point is added as 'FP_0'.

    Args:
        ds: DataSet with stress-strain curves
        strain_key: Key for strain data
        stress_key: Key for stress data

    Returns: DataSet with fracture point added to info_table.
    """
    ds = ds.copy()

    def find_di_fracture_point(di):
        idx_max = di.data[strain_key].idxmax()
        di.info['FP_0'] = di.data[strain_key][idx_max]
        di.info['FP_1'] = di.data[stress_key][idx_max]
        return di

    return ds.apply(find_di_fracture_point)


def find_flow_stress_values(ds: DataSet, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa',
                            temperature_key: str = None, rate_key: str = None,
                            flow_strain: Union[int, float, Tuple[float, float]] = None) -> DataSet:
    """

    Args:
        di: DataItem with stress-strain curve
        strain_key: Data key for reading strain.
        stress_key: Data key for reading stress.
        flow_strain_key: Info key for writing flow strain.
        flow_stress_key: Info key for storing flow stress.
        flow_strain: Strain at which to find the flow stress. If None, the maximum stress is used.

    Returns: DataItem with flow stress added to info.
    """
    def find_di_flow_stress_values(di, flow_strain):
        if flow_strain is None:
            flow_strain = di.data[strain_key].max()
        if (type(flow_strain) is float) or (type(flow_strain) is int):
            di.info[f'flow_{strain_key}'] = flow_strain
            di.info[f'flow_{stress_key}'] = di.data[stress_key][di.data[strain_key] <= flow_strain].max()
            if temperature_key is not None:
                di.info[f'flow_{temperature_key}'] = di.data[temperature_key][di.data[strain_key] <= flow_strain].max()
            if rate_key is not None:
                di.info[f'flow_{rate_key}'] = di.data[rate_key][di.data[strain_key] <= flow_strain].max()
        elif type(flow_strain) is tuple:
            # average the flow stress over a range of strains
            di.info[f'flow_{strain_key}'] = np.mean(flow_strain)
            di.info[f'flow_{stress_key}'] = di.data[stress_key][
                (di.data[strain_key] >= flow_strain[0])&(di.data[strain_key] <= flow_strain[1])].mean()
            if temperature_key is not None:
                di.info[f'flow_{temperature_key}'] = di.data[temperature_key][
                    (di.data[strain_key] >= flow_strain[0])&(di.data[strain_key] <= flow_strain[1])].mean()
            if rate_key is not None:
                di.info[f'flow_{rate_key}'] = di.data[rate_key][
                    (di.data[strain_key] >= flow_strain[0])&(di.data[strain_key] <= flow_strain[1])].mean()
        return di

    return ds.apply(find_di_flow_stress_values, flow_strain=flow_strain)


def find_upl_and_lpl(ds: DataSet, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa', preload: float = 0,
                     preload_key: str = 'Stress_MPa', max_strain: Optional[float] = None,
                     suppress_numpy_warnings: bool = True) -> DataSet:
    """Determine the upper proportional limit (UPL) and lower proportional limit (LPL) of a stress-strain curve.
    The UPL is the point that minimizes the residuals of the slope fit between that point and the specified preload.
    The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL.
    The elastic modulus is the slope between the UPL and LPL.

    Args:
        di: DataItem with stress-strain curve
        strain_key: key for strain data
        stress_key: key for stress data
        preload: preload value
        preload_key: key for preload data
        max_strain: maximum strain to consider
        suppress_numpy_warnings: suppress numpy warnings

    Returns:
        DataItem with UPL, LPL, and E added to info.
    """
    if suppress_numpy_warnings:
        np.seterr(all="ignore")

    ds = ds.copy()

    def _find_upl_and_lpl(di: DataItem) -> DataItem:
        data = di.data[di.data[strain_key] <= max_strain] if max_strain is not None else di.data

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
        for i in range(3, len(x_upl)):
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
                LPL = [x_lpl[j], y_lpl[j]]

        # if LPL is less than preload, then find the first data point with stress greater than preload
        if LPL[1] < preload:
            LPL = (x[data[preload_key] >= preload][0], y[data[preload_key] >= preload][0])


        di.info['UPL_0'] = UPL[0]
        di.info['UPL_1'] = UPL[1]
        di.info['LPL_0'] = LPL[0]
        di.info['LPL_1'] = LPL[1]
        di.info['E'] = (UPL[1] - LPL[1])/(UPL[0] - LPL[0])
        return di

    return ds.apply(_find_upl_and_lpl)


def correct_foot(ds: DataSet, strain_key: str = 'Strain', LPL_key: str = 'LPL', UPL_key: str = 'UPL') -> DataSet:
    def correct_di_foot(di):
        UPL = di.info['UPL_0'], di.info['UPL_1']
        E = di.info['E']
        strain_shift = UPL[0] - UPL[1]/E  # x-intercept of line through UPL & LPL
        di.info['foot correction'] = -strain_shift
        di.data[strain_key] = di.data[strain_key].values - strain_shift
        di.info[UPL_key + '_0'] = di.info[UPL_key + '_0'] - strain_shift
        di.info[LPL_key + '_0'] = di.info[LPL_key + '_0'] - strain_shift
        return di

    return ds.apply(correct_di_foot)


def find_E(ds: DataSet, LPL_stress: float, UPL_stress: float, strain_key: str = 'Strain',
                      stress_key: str = 'Stress_MPa', E_key: str = 'E'):
    """Find the elastic modulus of a stress-strain curve by fitting a line to the points between the specified stresses.

Args:
        di: DataItem with stress-strain curve
        LPL_stress: Lower stress bound
        UPL_stress: Upper stress bound
        strain_key: Key for strain data
        stress_key: Key for stress data
        E_key: Key to store elastic modulus in info
    """
    ds = ds.copy()

    def find_di_E(di):
        x_data = di.data[strain_key].values
        y_data = di.data[stress_key].values
        x = x_data[(y_data >= LPL_stress) & (y_data <= UPL_stress)]
        y = y_data[(y_data >= LPL_stress) & (y_data <= UPL_stress)]
        E = np.polyfit(x, y, 1)[0]
        di.info[E_key] = E
        return di

    return ds.apply(find_di_E)


def find_proof_stress(ds: DataSet, proof_strain: float = 0.002, strain_key: str = 'Strain',
                      stress_key: str = 'Stress_MPa', E_key: str = 'E') -> DataSet:
    """Find the proof stress of a stress-strain curve.

    Args:
        di: DataItem with stress-strain curve
        proof_strain: Strain at which to find the proof stress
        strain_key: Key for strain data
        stress_key: Key for stress data

    Returns: DataItem with proof stress added to info.
    """

    def find_di_proof_stress(di):
        E = di.info[E_key]
        x_data = di.data[strain_key].values
        y_data = di.data[stress_key].values
        x_shift = proof_strain
        y_line = E*(x_data - x_shift)
        try:
            cut = np.where(np.diff(np.sign(y_line - y_data)) != 0)[0][-1]
            m = (y_data[cut + 1] - y_data[cut])/(x_data[cut + 1] - x_data[cut])
            xl = x_data[cut]
            yl = y_line[cut]
            xd = x_data[cut]
            yd = y_data[cut]
            K = np.array([[1, -E], [1, -m]])
            f = np.array([[yl - E*xl], [yd - m*xd]])
            d = np.linalg.solve(K, f).flatten()
            di.info[f'PS_{proof_strain}_0'] = d[1]
            di.info[f'PS_{proof_strain}_1'] = d[0]
        except IndexError:
            di.info[f'PS_{proof_strain}_0'] = np.nan
            di.info[f'PS_{proof_strain}_1'] = np.nan
        return di

    return ds.apply(find_di_proof_stress)


def calculate_strain_rate(ds: DataSet, strain_key: str = 'Strain', time_key: str = 'Time_s',
                          strain_rate_key: str = 'Strain_Rate') -> DataSet:
    """Calculate the strain rate of a stress-strain curve.

    Args:
        di: DataItem with stress-strain curve
        strain_key: Key for strain data
        time_key: Key for time data
        strain_rate_key: Key for strain rate data

    Returns: DataItem with strain rate added to data.
    """
    def _calculate_di_strain_rate(di):
        gradient = np.gradient(di.data[strain_key], di.data[time_key])
        di.data[strain_rate_key] = gradient
        di.data[f'Smoothed_{strain_rate_key}'] = np.convolve(gradient, np.ones(5)/60, mode='same')
        return di

    return ds.apply(_calculate_di_strain_rate)


def correct_friction_UC(di: DataItem, mu_key: str = 'mu', h0_key: str = 'h_0', D0_key: str = 'D_0',
                        disp_key: str = 'Disp(mm)', force_key: str = 'Force(kN)') -> DataItem:
    """
    Calculate the pressure and corrected stress for a uniaxial compression test with friction.

    Args:
        di: DataItem with uniaxial compression test data
        mu_key: Key for friction coefficient in info
        h0_key: Key for initial height in info
        D0_key: Key for initial diameter in info
        disp_key: Key for displacement data
        force_key: Key for force data

    Returns: DataItem with corrected stress and pressure added to data.
    """
    mu = di.info[mu_key]  # friction coefficient
    h_0 = di.info[h0_key]  # initial height in axial direction
    D_0 = di.info[D0_key]  # initial diameter
    h = h_0 - di.data[disp_key]  # instantaneous height
    d = D_0*np.sqrt(h_0/h)  # instantaneous diameter
    P = di.data[force_key]*1000*4/(np.pi*d**2)  # pressure (MPa)
    di.data['Pressure(MPa)'] = P
    di.data['Corrected_Stress(MPa)'] = P/(1 + (mu*d)/(3*h))  # correct stress
    return di


def correct_friction_PSC(di: DataItem, mu_key: str = 'mu', h0_key: str = 'h_0',
                         hf_key: str = 'h_f', b0_key: str = 'b_0', bf_key: str = 'b_f',
                         w_key: str = 'w', spread_exponent_key: str = 'spread_exponent',
                         disp_key: str = 'Displacement_mm', force_key: str = 'Force_kN',
                         stress_key: str = 'Stress_MPa') -> DataItem:
    # store uncorrected stress
    di.data[f'Uncorrected_{stress_key}'] = di.data[stress_key]

    # calculate final height if not in info
    try:
        di.info[hf_key]
    except KeyError:
        di.info[hf_key] = di.info[h0_key] - di.data[disp_key].max()

    # get info
    mu = di.info[mu_key]  # friction coefficient
    h_0 = di.info[h0_key]  # initial height in normal direction
    h_f = di.info[hf_key]  # final height in normal direction
    b_0 = di.info[b0_key]  # initial width in transverse direction
    b_f = di.info[bf_key]  # final width in transverse direction
    w = di.info[w_key]  # initial width in rolling direction

    # breadth spread correction
    n = di.info[spread_exponent_key]  # spread exponent
    C = (b_f/b_0 - 1)/(1 - (h_f/h_0)**n)  # breadth spread coefficient
    h = h_0 - di.data[disp_key]  # instantaneous height
    b = b_0*(1 + C*(1 - (h/h_0)**n))  # instantaneous breadth

    # calculate strains
    eps_3 = np.log(h/h_0)  # normal strain
    eps_2 = np.log(b/b_0)  # transverse strain
    eps_bar = 2*(eps_2**2 + eps_2*eps_3 + eps_3**2)/np.sqrt(3)  # equivalent strain
    di.data['Normal_Strain'] = eps_3
    di.data['Transverse_Strain'] = eps_2
    di.data['Equivalent_Strain'] = eps_bar

    # calculate spread factor
    f = 1 + eps_bar/eps_3
    di.data['Spread_Factor'] = f

    # calculate average pressure
    P = di.data[force_key]*1000/(b*w)  # pressure (MPa)
    di.data['Pressure_MPa'] = P

    # calculate shear yield stress
    z_0 = (h/(2*mu))*np.log(1/(2*mu))  # sticking-sliding transition distance from centre (mm)
    k = np.zeros(len(di.data))  # shear yield stress (MPa)
    for i in range(len(di.data)):
        if i < 50:
            continue
        if 2*z_0[i] > w:  # transition distance is greater than the width, full sliding
            k[i] = P[i]/2*((2*h[i]**2/mu**2 + (b[i] - w)*h[i]/mu)*(np.exp(mu*w/h[i]) - 1)/b[i]*w - 2*h[i]/mu*b[i])
        elif w > 2*z_0[i] > 0:  # transition distance is less than the width, partial sliding
            k[i] = P[i]/2*((w/2 - z_0[i])/mu*w + (w/2 - z_0[i])**2/h[i]*w + h[i]*(1/2*mu - 1)/mu*w
                           + (z_0[i]**2 - 4*z_0[i]**3/3*w - w**2/12)/h[i]*b
                           + (2*z_0[i]**2/w - z_0[i] - 2*h*z_0[i]/mu*w
                              + h[i]/2*mu - h[i] + h[i]**2/w*mu**2 - 2*h[i]**2/mu*w)/mu*b[i])
        elif 0 > 2*z_0[i]:  # transition distance is less than zero, full sticking
            k[i] = P[i]/2*(1 + w/4*h[i] - w**2/12*h[i]*b[i])
        else:
            raise ValueError('Invalid value for z_0, the friction transition distance.'
                             '\nConsider changing the sign of the displacement and force data.')

    di.data['Shear_Yield_Stress_MPa'] = k  # shear yield stress (MPa)

    # calculate equivalent flow stress
    di.data[stress_key] = 2*k/f  # corrected stress

    return di


def smooth_load_cell_ringing(di: DataItem):
    return di


def correct_compliance(di: DataItem):
    return di


def correct_thermal_expansion(di: DataItem):
    return di


def trim_leading_data():
    pass


def trim_trailing_data():
    pass


def trim_after_max_force(di: DataItem, force_key: str = 'Force_kN'):
    di.data = di.data.loc[:di.data[force_key].idxmax()]
    return di
