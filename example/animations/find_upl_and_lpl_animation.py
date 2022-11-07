"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

import numpy as np

from paramaterial import DataSet, DataItem

PRELOAD = 36.0  # MPa

dataset = DataSet('../data/02 processed data', '../info/02 processed info.xlsx')
di = dataset[1]

FONT = 13
plt.style.use('seaborn-whitegrid')
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
mpl.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=FONT)
plt.rc('axes', titlesize=FONT, labelsize=FONT)
plt.rc('xtick', labelsize=0.9*FONT)
plt.rc('ytick', labelsize=0.9*FONT)
plt.rc('legend', fontsize=0.9*FONT)
plt.rc('figure', titlesize=1.1*FONT)


# trimming functions
def trim_at_max_force(di: DataItem):
    """Quote: The maximum force is determined by the maximum force recorded during the test. The data is then
    trimmed to this point."""
    di.info['Max_Force_idx'] = di.data['Force(kN)'].idxmax()
    di.data = di.data[:di.info['Max_Force_idx']]
    return di


def trim_at_strain(di: DataItem):
    """Quote: The data is then trimmed to the preload point."""
    di.data = di.data[di.data['Strain'] <= 0.01]
    return di


def trim_at_preload(di: DataItem):
    """Quote: The data is then trimmed to the preload point."""
    di.info['preload_stress'] = PRELOAD
    di.info['preload_strain'] = np.interp(PRELOAD, di.data['Stress_MPa'], di.data['Strain'])
    di.data = di.data[di.data['Stress_MPa'] >= PRELOAD]
    return di


di = trim_at_max_force(di)
di = trim_at_strain(di)


def fit_line(_x, _y):
    n = len(_x)  # number of points
    m = (n*np.sum(_x*_y) - np.sum(_x)*np.sum(_y))/(n*np.sum(np.square(_x)) - np.square(np.sum(_x)))  # slope
    c = (np.sum(_y) - m*np.sum(_x))/n  # intercept
    S_xy = (n*np.sum(_x*_y) - np.sum(_x)*np.sum(_y))/(n - 1)  # empirical covariance
    S_x = np.sqrt((n*np.sum(np.square(_x)) - np.square(np.sum(_x)))/(n - 1))  # x standard deviation
    S_y = np.sqrt((n*np.sum(np.square(_y)) - np.square(np.sum(_y)))/(n - 1))  # y standard deviation
    r = S_xy/(S_x*S_y)  # correlation coefficient
    S_m = np.sqrt((1 - r ** 2)/(n - 2))*S_y/S_x  # slope standard deviation
    S_rel = S_m/m  # relative deviation of slope
    return m, c, S_m, S_rel


fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

# plotting kwargs
data_kwargs = dict(marker='o', c='b', mfc='none', lw=0, alpha=0.5, label='Tensile test data')
fitting_sample_kwargs = dict(marker='x', color='r', mfc='none', lw=0, alpha=0.6, label='Fitting sample')
linear_fit_kwargs = dict(c='r', ls='--', alpha=0.7, label='Linear fit')
proportional_region_kwargs = dict(c='k', ls='-', alpha=1, label='Proportional region')
UPL_kwargs = dict(marker=4, color='k', mfc='m', markersize=10, label='Upper proportional limit', lw=0)
LPL_kwargs = dict(marker=5, color='k', mfc='c', markersize=10, label='Lower proportional limit', lw=0)
preload_kwargs = dict(marker='s', color='g', mfc='none', markersize=10, lw=0, label='Preload')

x_data = di.data['Strain'].values
y_data = di.data['Stress_MPa'].values
preload = x_data[y_data >= PRELOAD][0], y_data[y_data >= PRELOAD][0]
S_min = np.inf
UPL = np.zeros(2)
LPL = np.zeros(2)


def animate(i):
    global S_min, UPL, LPL, x_data, y_data

    ax.cla()
    ax.set_xlim(-0.0006, 0.0105)
    ax.set_ylim(-15, 270)
    ax.grid()
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress (MPa)')
    ax.plot(x_data, y_data, **data_kwargs)

    def plot_fitting_sample():
        ax.plot(x, y, **fitting_sample_kwargs)
        ax.plot(x, m*x + c, **linear_fit_kwargs)

    if i <= 2:
        title = ''
        x, y, m, c = np.zeros(1), np.zeros(1), 0, 0
    elif 2 < i < len(x_data):
        title = 'Finding Upper Proportional Limit'
        a = len(x_data)
        x_data_ups = x_data[y_data >= PRELOAD]
        y_data_ups = y_data[y_data >= PRELOAD]
        x = x_data_ups[:i]
        y = y_data_ups[:i]
        m, c, S_m, S_rel = fit_line(x, y)
        if S_rel < S_min:
            S_min = S_rel
            UPL[0] = x[-1]
            UPL[1] = y[-1]
        LPL = x[0], y[0]
        plot_fitting_sample()
    elif i == len(x_data):
        title = ''
        x, y, m, c = np.zeros(1), np.zeros(1), 0, 0
        S_min = np.inf
        LPL = UPL
    elif len(x_data) < i < int(1.7*len(x_data)):
        title = 'Finding Lower Proportional Limit'
        x_data_lps = x_data[y_data < UPL[1]]
        y_data_lps = y_data[y_data < UPL[1]]
        j = len(x_data_lps) - (i - len(x_data))
        if not j > 2:
            j = 3
        x = x_data_lps[j:]
        y = y_data_lps[j:]
        m, c, S_m, S_rel = fit_line(x, y)
        if S_rel < S_min:
            S_min = S_rel
            LPL = (x[0], y[0])
        plot_fitting_sample()
    else:
        title = 'Proportional Region Found'
        x, y, m, c = 0, 0, 0, 0

    ax.set_title(title)
    ax.plot(*preload, **preload_kwargs)
    ax.plot([LPL[0], UPL[0]], [LPL[1], UPL[1]], **proportional_region_kwargs)
    ax.plot(UPL[0], UPL[1], **UPL_kwargs)
    ax.plot(LPL[0], LPL[1], **LPL_kwargs)
    ax.legend(frameon=True, ncol=2)
    return fig,


anim = animation.FuncAnimation(fig=fig, func=animate, frames=int(2.1*len(x_data)), repeat=False)

anim.save(f'find upl and lpl animation.mp4', writer='ffmpeg', fps=10, dpi=300)
