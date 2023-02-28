from collections.abc import Callable
from functools import wraps

import numpy as np
import scipy.optimize as op


def iso_return_map(yield_stress_func: Callable, return_vec: str = 'stress'):
    """Wrapper for a yield function that describes the plastic behaviour.

    Args:
        yield_stress_func: Yield stress function.
        return_vec: Return vector. Must be one of 'stress', 'plastic strain', 'accumulated plastic strain'.

    Returns: A function that gives the return_vec (usually stress) as a function of strain.
    """

    @wraps(yield_stress_func)
    def wrapper(
            x: np.ndarray,
            mat_params
    ):
        y = np.zeros(x.shape)  # predicted stress
        x_p = np.zeros(x.shape)  # plastic strain
        aps = np.zeros(x.shape)  # accumulated plastic strain
        y_yield: callable = yield_stress_func(mat_params)  # yield stress
        E = mat_params[0]  # elastic modulus

        # if not np.isclose(x[0], 0):
        #     y_trial_0 = E*(x[1])
        #     f_trial_0 = np.abs(y_trial_0) - y_yield(0)
        #     if f_trial_0 <= 0:
        #         y[0] = E*x[0]
        #     else:
        #         d_aps = op.root(lambda d: f_trial_0 - d*E - y_yield(d) + y_yield(0), 0).x[0]
        #         y[0] = y_trial_0*(1 - d_aps*E/np.abs(y_trial_0))

        for i in range(len(x) - 1):
            y_trial = E*(x[i + 1] - x_p[i])
            f_trial = np.abs(y_trial) - y_yield(aps[i])
            if f_trial <= 0:
                y[i + 1] = y_trial
                x_p[i + 1] = x_p[i]
                aps[i + 1] = aps[i]
            else:
                d_aps = op.root(
                    lambda d: f_trial - d*E - y_yield(aps[i] + d) + y_yield(aps[i]),
                    aps[i]
                ).x[0]
                y[i + 1] = y_trial*(1 - d_aps*E/np.abs(y_trial))
                x_p[i + 1] = x_p[i] + np.sign(y_trial)*d_aps
                aps[i + 1] = aps[i] + d_aps

        if return_vec == 'stress':
            return y
        elif return_vec == 'plastic strain':
            return x_p
        elif return_vec == 'accumulated plastic strain':
            return aps
        else:
            return None

    return wrapper


@iso_return_map
def perfect(mat_params):
    """Perfect plasticity yield function, no hardening."""
    E, s_y = mat_params
    return lambda a: s_y


@iso_return_map
def linear(mat_params):
    """Linear isotropic hardening yield function."""
    E, s_y, K = mat_params
    return lambda a: s_y + K*a


@iso_return_map
def quadratic(mat_params):
    """Quadratic isotropic hardening yield function."""
    E, s_y, Q = mat_params
    return lambda a: s_y + E*(a - Q*a**2)


@iso_return_map
def voce(mat_params):
    """Exponential isotropic hardening yield function."""
    E, s_y, s_u, d = mat_params
    return lambda a: s_y + (s_u - s_y)*(1 - np.exp(-d*a))


@iso_return_map
def ramberg(mat_params):
    """Ramberg-Osgood isotropic hardening yield function."""
    E, s_y, C, n = mat_params
    return lambda a: s_y + C*(np.sign(a)*(np.abs(a))**n)


