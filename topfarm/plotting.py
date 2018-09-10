import matplotlib
from openmdao.core.explicitcomponent import ExplicitComponent
import os
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

    def __init__(self, memory=10, delay=0.001, plot_initial=True, ax=None):
        ExplicitComponent.__init__(self)
        self.ax_ = ax
        self.memory = memory
        self.delay = max([delay, 1e-6])
        self.plot_initial = plot_initial
        self.history = []
        self.counter = 0
        self.by_pass = False

    @property
    def ax(self):
        return self.ax_ or plt.gca()

    def show(self):
        plt.show()

    def setup(self):
        self.add_input('turbineX', np.zeros(self.n_wt), units='m')
        self.add_input('turbineY', np.zeros(self.n_wt), units='m')
        self.add_input('cost', 0.)
        if hasattr(self, 'n_vertices'):
            self.add_input('boundary', np.zeros((self.n_vertices, 2)), units='m')

    def init_plot(self, limits):
        self.ax.cla()
        self.ax.axis('equal')
        mi = limits.min(0)
        ma = limits.max(0)
        ra = ma - mi
        ext = .1
        xlim, ylim = np.array([mi - ext * ra, ma + ext * ra]).T
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def plot_boundary(self, boundary):
        b = np.r_[boundary[:], boundary[:1]]
        plt.plot(b[:, 0], b[:, 1], 'k')

    def plot_history(self, x, y):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            x = np.r_[rec['turbineX'][-self.memory:], [x]]
            y = np.r_[rec['turbineY'][-self.memory:], [y]]
            for c, x_, y_ in zip(self.colors, x.T, y.T):
                self.ax.plot(x_, y_, '--', color=c)

    def plot_initial2current(self, x0, y0, x, y):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            x0 = rec['turbineX'][0]
            y0 = rec['turbineY'][0]
            for c, x0_, y0_, x_, y_ in zip(self.colors, x0, y0, x, y):
                self.ax.plot((x0_, x_), (y0_, y_), '-', color=c)

    def plot_current_position(self, x, y):
        for c, x_, y_ in zip(self.colors, x, y):
            self.ax.plot(x_, y_, 'o', color=c, ms=5)
            self.ax.plot(x_, y_, 'xk', ms=4)

    def set_title(self, cost0, cost):
        rec = self.problem.recorder
        if cost0 != 0:
            self.ax.set_title("%d: %f (%.2f%%)" % (rec.num_cases, cost, (cost0 - cost) / cost0 * 100))
        else:
            self.ax.set_title("%d: %f" % (rec.num_cases, cost))

    def get_initial(self):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            x0 = rec['turbineX'][0]
            y0 = rec['turbineY'][0]
            cost0 = rec['cost'][0]
            return x0, y0, cost0

    def compute(self, inputs, outputs):
        if self.by_pass is False:
            x = inputs['turbineX']
            y = inputs['turbineY']
            cost = inputs['cost'][0]

            if 'boundary' in inputs:
                boundary = inputs['boundary']
                self.init_plot(boundary)
                self.plot_boundary(boundary)
            else:
                self.init_plot(np.array([x, y]).T)

            initial = self.get_initial()

            if initial is not None:
                x0, y0, cost0 = initial
                if self.plot_initial:
                    self.plot_initial2current(x0, y0, x, y)
                if self.memory > 0:
                    self.plot_history(x, y)
            else:
                cost0 = cost

            self.plot_current_position(x, y)
            self.set_title(cost0, cost)

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
        self.add_input('cost', 0.)

    def compute(self, inputs, outputs):
        pass


class TurbineTypePlotComponent(PlotComp):
    colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey'] * 100
    markers = np.array(list(".ov^<>12348spP*hH+xXDd|_"))

    def __init__(self, turbine_type_names, memory=10, delay=0.001, plot_initial=True):
        self.turbine_type_names = turbine_type_names
        PlotComp.__init__(self, memory=memory, delay=delay, plot_initial=plot_initial)

    def setup(self):
        PlotComp.setup(self)
        self.add_input('turbineType', np.zeros(self.n_wt, dtype=np.int))

    def compute(self, inputs, outputs):
        self.types = np.asarray(inputs['turbineType'], dtype=np.int)
        PlotComp.compute(self, inputs, outputs)

    def init_plot(self, limits):
        PlotComp.init_plot(self, limits)
        for m, n in zip(self.markers, self.turbine_type_names):
            self.ax.plot([], [], m + 'k', label=n)
        self.ax.legend()

    def plot_current_position(self, x, y):
        for m, c, x_, y_ in zip(self.markers[self.types], self.colors, x, y):
            #self.ax.plot(x_, y_, 'o', color=c, ms=5)
            self.ax.plot(x_, y_,  m + 'k', markeredgecolor=c, markeredgewidth=1, ms=8)
