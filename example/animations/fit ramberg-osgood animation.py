"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

import numpy as np

from paramaterial import DataSet, DataItem

PRELOAD = 30.0  # MPa

dataset = DataSet('../data/02 trimmed small data', 'info/02 trimmed small info.xlsx')
di = dataset[1]


FONT = 13
plt.style.use('seaborn-dark')
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
mpl.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=FONT)
plt.rc('axes', titlesize=FONT, labelsize=FONT)
plt.rc('xtick', labelsize=0.9*FONT)
plt.rc('ytick', labelsize=0.9*FONT)
plt.rc('legend', fontsize=0.9*FONT)
plt.rc('figure', titlesize=1.1*FONT)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)

# plotting kwargs
data_kwargs = dict(marker='o', c='b', mfc='none', alpha=0.5, lw=0, label='Tensile test data')
proportional_region_kwargs = dict(c='k', ls='-', alpha=0.9, label='Proportional region')
UPL_kwargs = dict(marker=4, color='k', mfc='m', markersize=10, label='UPL', lw=0)
LPL_kwargs = dict(marker=5, color='k', mfc='c', markersize=10, label='LPL', lw=0)
proof_line_kwargs = dict(c='k', ls='--', alpha=0.9, label='Proof line')
proof_point_kwargs = dict(marker='*', c='k', mfc='none', lw=0, markersize=10, label='Proof point')

x_data = di.data['Strain'].values
y_data = di.data['Stress_MPa'].values
preload = x_data[y_data >= PRELOAD][0], y_data[y_data >= PRELOAD][0]
proof = np.zeros(2)
UPL = np.zeros(2)
LPL = np.zeros(2)


def animate(i):
    global proof, UPL, x_data, y_data
    # foot correction
    # move line to the right
    # find intersection with the line
    ax.cla()
    ax.set_xlim(-0.0006, 0.0105)
    ax.set_ylim(-15, 270)
    ax.grid()
    title=''
    ax.set_title(title)
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress (MPa)')
    ax.plot(x_vec, y_vec, **data_kwargs)
    ax.plot([LPL[0], UPL[0]], [LPL[1], UPL[1]], **proportional_region_kwargs)
    ax.plot(UPL[0], UPL[1], **UPL_kwargs)
    ax.plot(LPL[0], LPL[1], **LPL_kwargs)
    ax.legend()
    return fig,


anim = animation.FuncAnimation(fig=fig, func=animate, frames=1000, repeat=False)

anim.save(f'find_upl_and_lpl_animation.mp4', writer='ffmpeg', fps=10, dpi=300)

