"""Module with functions for processing materials test data. This includes functionality for finding properties
like yield strength Young's modulus from stress-strain curves, as well as post-processing functions for cleaning
and correcting experimental measurements."""

import numpy as np

from paramaterial.plug import DataItem


def find_upl_and_lpl(di: DataItem, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa', preload: float = 0,
                     preload_key: str = 'Stress_MPa', max_strain: float|None = None,
                     suppress_numpy_warnings: bool = False) -> DataItem:
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


def find_proof_stress(di: DataItem, proof_strain: float = 0.002, strain_key: str = 'Strain',
                      stress_key: str = 'Stress_MPa') -> DataItem:
    """Find the proof stress of a stress-strain curve.

    Args:
        di: DataItem with stress-strain curve
        proof_strain: Strain at which to find the proof stress
        strain_key: Key for strain data
        stress_key: Key for stress data

    Returns: DataItem with proof stress added to info.
    """
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
    K = np.array([[1, -E], [1, -m]])
    f = np.array([[yl - E*xl], [yd - m*xd]])
    d = np.linalg.solve(K, f).flatten()
    di.info[f'YP_{proof_strain}_0'] = d[1]
    di.info[f'YP_{proof_strain}_1'] = d[0]
    return di


def find_ultimate_strength(di: DataItem, force_key: str = 'Force_kN', strain_key: str = 'Strain',
                           stress_key: str = 'Stress_MPa') -> DataItem:
    """Find the ultimate strength of a stress-strain curve, defined as the strain and stress at the maximum force.

    Args:
        di: DataItem with stress-strain curve
        force_key: Key for force data
        strain_key: Key for strain data
        stress_key: Key for stress data

    Returns: DataItem with ultimate strength added to info.
    """
    idx_max = di.data[force_key].idxmax()
    di.info['Ultimate_Strength_0'] = di.data[strain_key][idx_max]
    di.info['Ultimate_Strength_1'] = di.data[stress_key][idx_max]
    return di


def find_fracture_point(di: DataItem, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa') -> DataItem:
    """Find the fracture point of a stress-strain curve, defined as the point at which the strain is maximum.

    Args:
        di: DataItem with stress-strain curve
        strain_key: Key for strain data
        stress_key: Key for stress data

    Returns: DataItem with fracture point added to info.
    """
    idx_max = di.data[strain_key].idxmax()
    di.info['Fracture_Point_0'] = di.data[strain_key][idx_max]
    di.info['Fracture_Point_1'] = di.data[stress_key][idx_max]
    return di


def find_flow_stress(di: DataItem, strain_key: str = 'Strain', stress_key: str = 'Stress_MPa',
                     flow_strain_key: str = 'flow_strain', flow_stress_key: str = 'flow_stress_MPa',
                     flow_strain: float|None = None) -> DataItem:
    """Find the flow stress of a stress-strain curve, defined as the point at which the stress is maximum,
    or as the point at a specified strain.

    Args:
        di: DataItem with stress-strain curve
        strain_key: Data key for reading strain.
        stress_key: Data key for reading stress.
        flow_strain_key: Info key for writing flow strain.
        flow_stress_key: Info key for storing flow stress.
        flow_strain: Strain at which to find the flow stress. If None, the maximum stress is used.

    Returns: DataItem with flow stress added to info.
    """
    if flow_strain is None:
        idx_max = di.data[stress_key].idxmax()
        di.info[f'{flow_strain_key}'] = di.data[strain_key][idx_max]
        di.info[f'{flow_stress_key}'] = di.data[stress_key][idx_max]
    else:
        di.info[f'{flow_strain_key}'] = flow_strain
        di.info[f'{flow_stress_key}'] = di.data[stress_key][di.data[strain_key] <= flow_strain].max()
    return di


def calculate_ZH_parameter(di: DataItem, temperature_key: str = 'temperature_K', rate_key: str = 'rate_s-1',
                           Q_key: str = 'Q_activation', gas_constant: float = 8.1345,
                           ZH_key: str = 'ZH_parameter') -> DataItem:
    """Calculate the Zener-Holloman parameter using

    $$
    Z = \\dot{\\varepsilon} \\exp \\left(\\frac{Q}{RT}\\right)
    $$

    where $\\dot{\\varepsilon}$ is the strain rate, $Q$ is the activation energy, $R$ is the gas constant,
    and $T$ is the temperature.

    Args:
        di: DataItem object with $\\dot{\\varepsilon}$, $Q$, $R$, and $T$ in info.
        temperature_key: Info key for mean temperature
        rate_key: Info key for mean strain-rate rate
        Q_key: Info key for activation energy
        gas_constant: Universal gas constant
        ZH_key: Key for Zener-Holloman parameter

    Returns: DataItem with Zener-Holloman parameter added to info.
    """
    di.info[ZH_key] = di.info[rate_key]*np.exp(
        di.info[Q_key]/(gas_constant*di.info[temperature_key]))
    return di


def correct_foot(di: DataItem):
    UPL = di.info['UPL_0'], di.info['UPL_1']
    E = di.info['E']
    strain_shift = UPL[0] - UPL[1]/E  # x-intercept of line through UPL & LPL
    di.info['foot correction'] = -strain_shift
    di.data['Strain'] = di.data['Strain'].values - strain_shift
    di.info['UPL_0'] = di.info['UPL_0'] - strain_shift
    di.info['LPL_0'] = di.info['LPL_0'] - strain_shift
    return di


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
                         disp_key: str = 'Displacement_mm', force_key: str = 'Force_kN') -> DataItem:
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
    f = -eps_bar/eps_3

    # calculate average pressure
    P = di.data[force_key]*1000/(b*w)  # pressure (MPa)
    di.data['Pressure_MPa'] = P

    # calculate shear yield stress
    z_0 = (h/(2*mu))*np.log(1/(2*mu))  # sticking-sliding transition distance from centre (mm)
    if 2*z_0 > w:  # transition distance is greater than the width, full sliding
        k = P/2*((2*h**2/mu**2 + (b - w)*h/mu)*(np.exp(mu*w/h) - 1)/b*w - 2*h/mu*b)  # shear yield stress (MPa)
    elif w > 2*z_0 > 0:  # transition distance is less than the width, partial sliding
        k = P/2*((w/2 - z_0)/mu*w + (w/2 - z_0)**2/h*w + h*(1/2*mu - 1)/mu*w + (z_0**2 - 4*z_0**3/3*w - w**2/12)/h*b
                 + (2*z_0**2/w - z_0 - 2*h*z_0/mu*w + h/2*mu - h + h**2/w*mu**2 - 2*h**2/mu*w)/mu*b)
    elif 0 > 2*z_0:  # transition distance is less than zero, full sticking
        k = P/2*(1 + w/4*h - w**2/12*h*b)
    else:
        raise ValueError('Invalid value for z_0, the friction transition distance.'
                         '\nConsider changing the sign of the displacement and force data.')

    di.data['Shear_Yield_Stress_MPa'] = k  # shear yield stress (MPa)

    # calculate equivalent flow stress
    di.data['Equivalent_Stress_MPa'] = 2*k/f  # corrected stress

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
