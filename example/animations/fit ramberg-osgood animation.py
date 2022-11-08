"""Module for developing code to find the upper proportional limit (UPL) and lower proportional limit (LPL) of a
stress-strain curve. The UPL is the point that minimizes the residuals of the slope fit between that point and the
specified preload. The LPL is the point that minimizes the residuals of the slope fit between that point and the UPL."""
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

import numpy as np

from paramaterial import DataSet, DataItem


class Model:
    def __init__(self, E, bounds):
        self.E = E
        self.bounds = bounds
        self.fitted_params = None
        self.fitting_error = None
        self.x_data = None
        self.y_data = None

    def model(self, y, K, n):
        x = y/self.E + K*((y/self.E) ** n)
        return x

    def fit(self, x_data, y_data, **de_kwargs):
        self.x_data = x_data
        self.y_data = y_data
        result = list(self.de(self.rmse, self.bounds, **de_kwargs))
        # self.fitted_params = result[0]
        # self.fitting_error = result[1]
        return result

    def predict(self, y):
        return self.model(y, *self.fitted_params)

    def rmse(self, w):
        """Root mean squared error."""
        x_pred = self.model(self.y_data, *w)
        # return np.sqrt(sum((self.x_data - x_pred) ** 2)/len(self.x_data))
        return np.sqrt(max((self.x_data - x_pred) ** 2)/len(self.x_data))

    @staticmethod
    def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=5000):
        """Differential evolution algorithm."""
        dimensions = len(bounds)
        pop = np.random.rand(popsize, dimensions)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop*diff
        fitness = np.asarray([fobj(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        for i in range(its):
            for j in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mut*(b - c), 0, 1)
                cross_points = np.random.rand(dimensions) < crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial*diff
                f = fobj(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
            # yield best, fitness[best_idx]
            yield min_b + pop*diff, fitness, best_idx


def style_plt():
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
    return fig, ax


fig, ax = style_plt()

data_kwargs = dict(marker='o', c='b', mfc='none', alpha=0.5, lw=0, label='Tensile test data')
UPL_kwargs = dict(marker=4, color='k', mfc='m', markersize=10, label='UPL', lw=0)
YP_kwargs = dict(marker=4, color='k', mfc='m', markersize=10, label='YP', lw=0)
fitted_curve_kwargs = dict(color='r', lw=1, label='Fitted curve')
sample_kwargs = dict(marker='o', c='r', mfc='none', alpha=0.5, lw=0, label='Fitting sample')

dataset = DataSet('../data/02 fitting data', '../info/02 fitting info.xlsx')
di = dataset[2]


def main():
    eps_scale = 1000

    E = di.info['E']/eps_scale  # MPa/..

    eps_data = di.data['Strain'].values*eps_scale  # ..
    sig_data = di.data['Stress_MPa'].values  # MPa

    # shift data to start at 0
    eps_data -= eps_data[0]
    sig_data -= sig_data[0]

    # plt.plot(sig_data, eps_data, **data_kwargs)

    model = Model(E, bounds=[(2, 2000), (1, 100)])
    result = model.fit(eps_data, sig_data, popsize=20, its=2000, mut=0.7, crossp=0.8)

    def animate(i):
        ax.clear()
        ax.plot(eps_data, sig_data, **data_kwargs)
        # ax.set_ylim((-10, 70))
        # ax.set_xlim((-1, 8))
        pop, fit, idx = result[i]
        print(pop[idx], fit[idx])
        for ind in pop:
            eps_model = model.model(sig_data, *ind)
            ax.plot(eps_model, sig_data, alpha=0.3)

    # anim = animation.FuncAnimation(fig, animate, frames=2000, interval=20)
    anim = animation.FuncAnimation(fig=fig, func=animate, frames=300, repeat=False)
    anim.save(f'fit ramberg osgood animation.mp4', writer='ffmpeg', fps=20, dpi=100)


if __name__ == '__main__':
    main()
