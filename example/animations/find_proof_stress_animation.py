"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from paramaterial import DataSet

dataset = DataSet('../data/02 trimmed small data', '../info/02 trimmed small info.xlsx')
di = dataset[1]

FONT = 13
plt.style.use('seaborn-whitegrid')
# mpl.rcParams['text.usetex'] = True
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

x_data = di.data['Strain'].values*100
y_data = di.data['Stress_MPa'].values
x_line = np.linspace(min(x_data), max(x_data), len(x_data))

offset = 0.2  # Strain (%)

E = di.info['E']/100
UPL = di.info['UPL_0']*100, di.info['UPL_1']
LPL = di.info['LPL_0']*100, di.info['LPL_1']
YP = np.zeros(2)


def plot_proporional_region(ax, shift: float):
    ax.plot([LPL[0] + shift, UPL[0] + shift], [LPL[1], UPL[1]], **proportional_region_kwargs)
    ax.plot(UPL[0] + shift, UPL[1], **UPL_kwargs)
    ax.plot(LPL[0] + shift, LPL[1], **LPL_kwargs)
    return ax


strain_shift = UPL[0] - UPL[1]/E


# ##1
# plt.plot(x_data, y_data, **data_kwargs)
# plt.plot(strain_shift, 0, **strain_intercept_kwargs)
# plt.plot(x_data, E*(x_data - strain_shift), **elastic_line_kwargs)
#
# plt.legend()
# plt.show()
#
# ##2
# plt.plot(x_data, y_data, **data_kwargs)
# plt.plot(x_data, E*x_data, **elastic_line_kwargs)
#
# plt.legend()
# plt.show()
#
# ##3
# plt.plot(x_data - strain_shift, y_data, **data_kwargs)
# plt.plot(x_data, E*x_data, **elastic_line_kwargs)
#
# plt.legend()
# plt.show()
#
# ##4
# plt.plot(x_data - strain_shift, y_data, **data_kwargs)
# plt.plot(x_data+offset, E*x_data, **elastic_line_kwargs)
#
# plt.legend()
# plt.show()
#
# #5
# plt.plot(x_data, y_data, **data_kwargs)
# plt.plot(x_data+offset+strain_shift, E*x_data, **elastic_line_kwargs)
#
# plt.legend()
# plt.show()

# 6
# plt.plot(x_data - strain_shift, y_data, **data_kwargs)
#
#
# y_line = E*(x_data - offset - strain_shift)
# cut_idx = np.where(np.diff(np.sign(y_line - y_data)) != 0)
# plt.plot(x_data - strain_shift, y_line, **elastic_line_kwargs)
#
# plt.plot(x_data[cut_idx] - strain_shift, y_data[cut_idx], **proof_point_kwargs)
#
#
# plt.legend()
# plt.show()

def calculate_proof_point(x_data, y_data, x_shift):
    y_line = E*(x_data - x_shift)
    cut = np.where(np.diff(np.sign(y_line - y_data)) != 0)[0][0]
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
    d = np.linalg.solve(K, f)
    return np.array([d[1], d[0]])


def animate(i):
    global YP, UPL, LPL, E, x_data, y_data
    ax.cla()
    ax.set_xlim(-0.1, 1)
    ax.set_ylim(-15, 270)
    ax.set_xlabel('Strain ( %)')
    ax.set_ylabel('Stress (MPa)')
    strain_shift = UPL[0] - UPL[1]/E
    if i <= 100:  # extend line to intercept x_axis
        title = 'Extend proportional region to intercept x-axis'
        x_shift = strain_shift
        ax.plot(x_data, y_data, **data_kwargs)
        plot_proporional_region(ax, 0)
        ax.axline((x_shift, 0), slope=E, **elastic_line_kwargs)
        ax.plot(x_shift, 0, label=f'$\\varepsilon_{{shift}} = {x_shift:.3f}$', **strain_intercept_kwargs)
    elif 100 < i <= 300:  # shift line to the origin
        title = 'Shift line to the origin'
        x_shift = strain_shift*(1 - (i - 300)/200)
        ax.plot(x_data, y_data, **data_kwargs)
        plot_proporional_region(ax, 0)
        ax.axline((x_shift, 0), slope=E, **elastic_line_kwargs)
        ax.plot(x_shift, 0, label=f'$\\varepsilon_{{shift}} = {x_shift:.3f}$', **strain_intercept_kwargs)
    elif 300 < i <= 500:  # shift data the same amount as the line (apply foot correction)
        title = 'Shift data (apply foot correction)'
        x_shift = strain_shift*(i - 300)/200
        ax.plot(x_data - x_shift, y_data, **data_kwargs)
        plot_proporional_region(ax, -x_shift)
        ax.axline((0, 0), slope=E, **elastic_line_kwargs)
        ax.plot(0, 0, label=f'$\\varepsilon_{{shift}} = {0:.3f}$', **strain_intercept_kwargs)
        if i == 500:
            # update data
            x_data = x_data - strain_shift
    elif 500 < i <= 800:  # shift line so that the x-intercept is at 0.02 % (proof strain), mark line-data intersection
        title = 'Find 0.02 % proof stress'
        x_shift = offset*(i - 500)/300
        ax.plot(x_data, y_data, **data_kwargs)
        plot_proporional_region(ax, -strain_shift)
        ax.axline((x_shift, 0), slope=E, **elastic_line_kwargs)
        YP = calculate_proof_point(x_data, y_data, x_shift)
        ax.plot(YP[0], YP[1], **proof_point_kwargs)
        ax.plot(x_shift, 0, label=f'$\\varepsilon_{{shift}} = {x_shift:.2f}$', **strain_intercept_kwargs)
    else:  # add annotation to proof point with cool entrance animation
        assert 800 < i <= 1000, f'i must be between 0 and 100, not {i}'
        title = '0.2% proof stress found'
        x_shift = offset
        ax.plot(x_data, y_data, **data_kwargs)
        plot_proporional_region(ax, -strain_shift)
        ax.axline((x_shift, 0), slope=E, **elastic_line_kwargs)
        ax.plot(YP[0], YP[1], **proof_point_kwargs)
        ax.plot(x_shift, 0, label=f'$\\varepsilon_{{shift}} = {x_shift:.3f}$', **strain_intercept_kwargs)
    ax.legend(frameon=True)
    ax.set_title(title)
    ax.grid()
    return fig,


anim = animation.FuncAnimation(fig=fig, func=animate, frames=1000, repeat=False)
anim.save(f'foot correction and proof stress animation.mp4', writer='ffmpeg', fps=50, dpi=200)
