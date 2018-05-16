import time

import matplotlib
from openmdao.core.explicitcomponent import ExplicitComponent

import matplotlib.pyplot as plt
import numpy as np


def mypause(interval):
    # pause without show
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


class PlotComp(ExplicitComponent):
    colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey'] * 100

    def __init__(self, memory=10, delay=0.001, plot_initial=True):
        ExplicitComponent.__init__(self)
        self.memory = memory
        self.delay = delay
        self.plot_initial = plot_initial
        self.history = []
        self.counter = 0

    def show(self):
        plt.show()

    def setup(self):
        self.add_input('turbineX', np.zeros(self.n_wt), units='m')
        self.add_input('turbineY', np.zeros(self.n_wt), units='m')
        self.add_input('cost', 0.)
        self.add_input('boundary', np.zeros((self.n_vertices, 2)), units='m')

    def init_plot(self, boundary):
        plt.cla()
        plt.axis('equal')
        b = np.r_[boundary[:], boundary[:1]]
        plt.plot(b[:, 0], b[:, 1], 'k')
        mi = b.min(0)
        ma = b.max(0)
        ra = ma - mi
        ext = .1
        xlim, ylim = np.array([mi - ext * ra, ma + ext * ra]).T
        plt.xlim(xlim)
        plt.ylim(ylim)

    def compute(self, inputs, outputs):
        x = inputs['turbineX']
        y = inputs['turbineY']
        if not hasattr(self, "initial"):
            self.initial = np.array([x, y]).T
        cost = inputs['cost']
        self.history = [(x.copy(), y.copy())] + self.history[:self.memory]

        boundary = inputs['boundary']
        self.init_plot(boundary)
        plt.title(cost)

        history_arr = np.array(self.history)
        for i, c, x_, y_ in zip(range(len(x)), self.colors, x, y):
            if self.plot_initial:
                plt.plot([self.initial[i, 0], x_], [self.initial[i, 1], y_], '-', color=c, lw=1)
            plt.plot(history_arr[:, 0, i], history_arr[:, 1, i], '.--', color=c, lw=1)
            plt.plot(x_, y_, 'o', color=c, ms=5)
            plt.plot(x_, y_, 'x' + 'k', ms=4)

        if self.counter == 0:
            plt.pause(.01)
        mypause(self.delay)

        self.counter += 1


class NoPlot(PlotComp):
    def __init__(self, *args, **kwargs):
        ExplicitComponent.__init__(self)

    def show(self):
        pass

    def setup(self):
        self.add_input('turbineX', np.zeros(self.n_wt), units='m')
        self.add_input('turbineY', np.zeros(self.n_wt), units='m')
        self.add_input('cost', 0.)
        self.add_input('boundary', np.zeros((self.n_vertices, 2)), units='m')

    def compute(self, inputs, outputs):
        pass
