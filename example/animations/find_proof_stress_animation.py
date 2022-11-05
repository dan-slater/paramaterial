"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

import numpy as np

from paramaterial import DataSet, DataItem

dataset = DataSet('../data/02 trimmed small data', '../info/02 trimmed small info.xlsx')
di = dataset[1]

FONT = 13
plt.style.use('seaborn-whitegrid')
mpl.rcParams['text.usetex'] = True
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
elastic_line_kwargs = dict(c='k', ls=':', label='Elastic line')
proof_line_kwargs = dict(c='k', ls='--', alpha=0.9, label='Proof line')
proof_point_kwargs = dict(marker='*', c='k', mfc='none', lw=0, markersize=10, label='Proof point')
strain_intercept_kwargs = dict(marker='o', mfc='none', c='k', lw=0)

x_data = di.data['Strain'].values
y_data = di.data['Stress_MPa'].values
x_line = np.linspace(min(x_data), max(x_data), len(x_data))
proof = (0, 0)
# proof = di.info['YS_0'], di.info['YS_1']
print(di.info)
E = di.info['E']
UPL = di.info['UPL_0'], di.info['UPL_1']
LPL = di.info['LPL_0'], di.info['LPL_1']


def plot_proporional_region(ax, shift: float):
    ax.plot([LPL[0] + shift, UPL[0]], [LPL[1], UPL[1]], **proportional_region_kwargs)
    ax.plot(UPL[0] + shift, UPL[1], **UPL_kwargs)
    ax.plot(LPL[0] + shift, LPL[1], **LPL_kwargs)
    return ax

def animate(i):
    global proof, UPL, LPL, E, x_data, y_data, x_line
    ax.cla()
    ax.set_xlim(-0.0006, 0.0105)
    ax.set_ylim(-15, 270)
    ax.grid()
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress (MPa)')
    strain_shift = UPL[0] - UPL[1]/E
    if i <= 20:  # extend line to intercept x_axis
        x_shift = strain_shift
        ax.plot(x_data, y_data, **data_kwargs)
        plot_proporional_region(ax, 0)
        ax.axline((x_shift, 0), slope=E, **elastic_line_kwargs)
        ax.plot(x_shift, 0,  label=f'$\\varepsilon_{{shift}} = {x_shift:.3f}$', **strain_intercept_kwargs)
    elif 20 < i <= 40:  # shift line to the origin
        x_shift = strain_shift*(i - 20)/20
        ax.plot(x_data, y_data, **data_kwargs)
        plot_proporional_region(ax, 0)
        ax.axline((x_shift, 0), slope=E, **elastic_line_kwargs)
        ax.plot(x_shift, 0,  label=f'$\\varepsilon_{{shift}} = {x_shift:.3f}$', **strain_intercept_kwargs)
    elif 40 < i <= 60:  # shift data the same amount as the line (apply foot correction)
        x_shift = strain_shift*(i - 40)/20
        ax.plot(x_data - x_shift, y_data, **data_kwargs)
        plot_proporional_region(ax, -x_shift)
        ax.axline((0, 0), slope=E, **elastic_line_kwargs)
        ax.plot(0, 0,  label=f'$\\varepsilon_{{shift}} = {x_shift:.3f}$', **strain_intercept_kwargs)
    elif 60 < i <= 80:  # shift line so that the x-intercept is at 0.0002 (proof strain), mark line-data intersection
        x_shift = 0.002*(i - 60)/20
        ax.plot(x_data - strain_shift, y_data, **data_kwargs)
        plot_proporional_region(ax, -strain_shift)
        ax.axline((x_shift, 0), slope=E, **elastic_line_kwargs)
        y_line = E*(x_line - x_shift)
        cut_idx = np.argwhere(np.diff(np.sign(y_line - y_data))).flatten()
        ax.plot(x_data[cut_idx], y_data[cut_idx], **proof_point_kwargs)
        ax.plot(0, 0,  label=f'$\\varepsilon_{{shift}} = {x_shift:.3f}$', **strain_intercept_kwargs)
    else:  # add annotation to proof point with cool entrance animation
        assert 80 < i <= 100, f'i must be between 0 and 100, not {i}'
        pass
        # x_shift = 0.002
        # ax.plot(x_vec_shifted, y_data, **data_kwargs)
        # x_line = np.linspace(min(x_data), max(x_data), 40)
        # y_line = E*(x_line - 0.0002*(1 - (i - 80)/20))
        # ax.plot(x_line, y_line, **proof_line_kwargs)
        # x_intersect = (y_data - y_line[0])/(y_line[-1] - y_line[0])*(x_line[-1] - x_line[0]) + x_line[0]
        # y_intersect = y_line[0] + (y_line[-1] - y_line[0])/(x_line[-1] - x_line[0])*(x_intersect - x_line[0])
        # ax.plot(x_intersect, y_intersect, **proof_point_kwargs)
        # # ax.annotate(f'Proof stress: {proof} MPa', xy=(x_intersect, y_intersect), xytext=(0.005, 200),
        #             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=14)
    ax.legend()
    ax.set_title(f'Frame {i}')
    return fig,


anim = animation.FuncAnimation(fig=fig, func=animate, frames=100, repeat=False)

anim.save(f'foot correction and proof stress animation.mp4', writer='ffmpeg', fps=5, dpi=150)
