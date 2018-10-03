import matplotlib
from openmdao.core.explicitcomponent import ExplicitComponent
import matplotlib.pyplot as plt
import numpy as np
import topfarm
import sys


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


class NoPlot(ExplicitComponent):
    def __init__(self, *args, **kwargs):
        ExplicitComponent.__init__(self)

    def show(self):
        pass

    def setup(self):
        self.add_input('cost', 0.)

    def compute(self, inputs, outputs):
        pass


class XYPlotComp(NoPlot):
    # colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey'] * 100
    colors = [c['color'] for c in iter(matplotlib.rcParams['axes.prop_cycle'])] * 100

    def __init__(self, memory=10, delay=0.001, plot_initial=True, plot_improvements_only=False, ax=None, legendloc=1):
        ExplicitComponent.__init__(self)
        self.delay = delay
        self.plot_improvements_only = plot_improvements_only
        self._ax = ax
        self.memory = memory
        self.delay = max([delay, 1e-6])
        self.plot_initial = plot_initial
        self.history = []
        self.counter = 0
        self.by_pass = False
        self.legendloc = legendloc

    @property
    def ax(self):
        return self._ax or plt.gca()

    def show(self):
        plt.show()

    def setup(self):
        NoPlot.setup(self)
        self.add_input(topfarm.x_key, np.zeros(self.n_wt))
        self.add_input(topfarm.y_key, np.zeros(self.n_wt))
        if hasattr(self.problem, 'xy_boundary'):
            self.xy_boundary = self.problem.xy_boundary
        if hasattr(self.problem.cost_comp, 'output_key'):
            self.cost_key = self.problem.cost_comp.output_key
            self.cost_unit = self.problem.cost_comp.output_unit
        else:
            self.cost_key = "Cost"
            self.cost_unit = ""
        self.add_input(self.cost_key, 0.)

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

#     def plot_boundary(self):
#         b = np.r_[self.xy_boundary[:], self.xy_boundary[:1]]
#         plt.plot(b[:, 0], b[:, 1], 'k')

    def plot_constraints(self):
        for constr in self.problem.model.constraint_components:
            constr.plot(self.ax)

    def plot_history(self, x, y):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            def get(xy, xy_key):
                rec_xy = rec[xy_key][-self.memory:]
                if len(rec_xy.shape) == 1:
                    rec_xy = rec_xy[:, np.newaxis]
                return np.r_[rec_xy, [xy]]
            x = get(x, topfarm.x_key)
            y = get(y, topfarm.y_key)
            for c, x_, y_ in zip(self.colors, x.T, y.T):
                self.ax.plot(x_, y_, '--', color=c)

    def plot_initial2current(self, x0, y0, x, y):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            x0 = np.atleast_1d(rec[topfarm.x_key][0])
            y0 = np.atleast_1d(rec[topfarm.y_key][0])
            for c, x0_, y0_, x_, y_ in zip(self.colors, x0, y0, x, y):
                self.ax.plot(x0_, y0_, '>', markerfacecolor=c, markeredgecolor='k')
                self.ax.plot((x0_, x_), (y0_, y_), '-', color=c)
            self.ax.plot([], [], '>k', markerfacecolor="#00000000", markeredgecolor='k', label='Initial position')

    def plot_current_position(self, x, y):
        for c, x_, y_ in zip(self.colors, x, y):
            self.ax.plot(x_, y_, 'o', color=c, ms=5)
            self.ax.plot(x_, y_, 'xk', ms=4)
        self.ax.plot([], [], 'xk', label='Current position')

    def set_title(self, cost0, cost):
        rec = self.problem.recorder
        title = "%d) %s %f %s" % (rec.num_cases, self.cost_key, cost, self.cost_unit)
        if cost0 != 0:
            title += " (%+.2f%%)" % ((cost - cost0) / cost0 * 100)
        self.ax.set_title(title)

    def get_initial(self):
        rec = self.problem.recorder
        if rec.num_cases > 0:
            x0 = rec[topfarm.x_key][0]
            y0 = rec[topfarm.y_key][0]
            c = rec[self.cost_key]
            cost0 = np.r_[c[c != 0], 0][0]  # first non-zero if exists
            return x0, y0, cost0

    def compute(self, inputs, outputs):
        if self.by_pass is False:
            cost = inputs[self.cost_key][0]

            if (self.plot_improvements_only and
                'cost' in self.problem.recorder.driver_iteration_dict and
                len(self.problem.recorder['cost']) and
                    cost > self.problem.recorder['cost'].min()):
                return

            # find limits

            if (topfarm.x_key in self.problem.design_vars and
                    isinstance(self.problem.design_vars[topfarm.x_key], tuple)):
                min_x, max_x = self.problem.design_vars[topfarm.x_key][1:]
            else:
                min_x, max_x = min(inputs[topfarm.x_key]), max(inputs[topfarm.x_key])
            if (topfarm.y_key in self.problem.design_vars and
                    isinstance(self.problem.design_vars[topfarm.y_key], tuple)):
                min_y, max_y = self.problem.design_vars[topfarm.y_key][1:]
            else:
                min_y, max_y = min(inputs[topfarm.y_key]), max(inputs[topfarm.y_key])

            self.init_plot(np.array([[min_x, min_y], [max_x, max_y]]))

            self.plot_constraints()

            initial = self.get_initial()

            x = inputs[topfarm.x_key]
            y = inputs[topfarm.y_key]
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
            self.ax.legend(loc=self.legendloc)

            if self.counter == 0:
                plt.pause(1e-6)
            mypause(self.delay)

            self.counter += 1


class PlotComp(XYPlotComp):
    def __init__(self, memory=10, delay=0.001, plot_initial=True, plot_improvements_only=False, ax=None):
        XYPlotComp.__init__(self, memory=memory, delay=delay, plot_initial=plot_initial, plot_improvements_only=plot_improvements_only, ax=ax)
        sys.stderr.write("%s is deprecated. Use XYPlotComp instead\n" % self.__class__.__name__)


class TurbineTypePlotComponent(XYPlotComp):
    colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey'] * 100
    markers = np.array(list("123v^<>.o48spP*hH+xXDd|_"))

    def __init__(self, turbine_type_names, **kwargs):
        self.turbine_type_names = turbine_type_names
        XYPlotComp.__init__(self, **kwargs)

    def setup(self):
        XYPlotComp.setup(self)
        self.add_input('type', np.zeros(self.n_wt, dtype=np.int))

    def compute(self, inputs, outputs):
        self.types = np.asarray(inputs['type'], dtype=np.int)
        XYPlotComp.compute(self, inputs, outputs)

    def init_plot(self, limits):
        XYPlotComp.init_plot(self, limits)
        for m, n in zip(self.markers, self.turbine_type_names):
            self.ax.plot([], [], m + 'k', label=n)
        self.ax.legend()

    def plot_current_position(self, x, y):
        for m, c, x_, y_ in zip(self.markers[self.types], self.colors, x, y):
            # self.ax.plot(x_, y_, 'o', color=c, ms=5)
            self.ax.plot(x_, y_, m + 'k', markeredgecolor=c, markeredgewidth=1, ms=8)
