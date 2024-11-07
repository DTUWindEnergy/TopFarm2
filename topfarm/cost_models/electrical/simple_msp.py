import topfarm
import openmdao.api as om
from topfarm.cost_models.utils.spanning_tree import mst
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.plotting import XYPlotComp, NoPlot
from IPython.display import display

import numpy as np
import matplotlib.pylab as plt


class ElNetComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)

    def setup(self):
        self.add_input(topfarm.x_key, np.zeros(self.options['n_wt']), units='m')
        self.add_input(topfarm.y_key, np.zeros(self.options['n_wt']), units='m')


class ElNetLength(ElNetComp):
    def setup(self):
        super().setup()
        self.add_output('elnet_length', 0.0, units='m')

        # Note that this component is NOT derivative friendly, as the
        # electrical layout will jump from one turbine to another in a non-
        # continuous manner if you move the turbines
        self.declare_partials('elnet_length', [topfarm.x_key, topfarm.y_key],
                              method='fd')

    def compute(self, inputs, outputs):
        x, y = inputs[topfarm.x_key], inputs[topfarm.y_key]
        elnet_layout = mst(x, y)
        outputs['elnet_length'] = sum(list(elnet_layout.values()))


class PlotElNet(ElNetComp):
    def compute(self, inputs, outputs):
        x, y = inputs[topfarm.x_key], inputs[topfarm.y_key]
        elnet_layout = mst(x, y)
        indices = np.array(list(elnet_layout.keys())).T
        plt.plot(x[indices], y[indices], color='r')
        plt.plot(x, y, 'o')
        plt.axis('equal')
        plt.title(f'Total Length = {sum(list(elnet_layout.values()))}')


class ElNetCost(CostModelComponent):
    def __init__(self, n_wt, length_key='elnet_length', **kwargs):
        self.n_wt = n_wt
        self.length_key = length_key
        CostModelComponent.__init__(self, [(self.length_key, 0.0)], self.n_wt,
                                    self.cost, self.grad, objective=False, input_units=['m'],
                                    **kwargs)

    def initialize(self):
        self.options.declare('cost_per_meter')

    def cost(self, **kwargs):
        return self.options['cost_per_meter'] * kwargs[self.length_key]

    def grad(self, **kwargs):
        return [self.options['cost_per_meter']]


class XYElPlotComp(XYPlotComp):
    """Plotting component for turbine locations"""

    def plot_current_position(self, x, y):
        elnet_layout = mst(x, y)
        indices = np.array(list(elnet_layout.keys())).T
        plt.plot(x[indices], y[indices], color='r')
        for c, x_, y_ in zip(self.colors, x, y):
            self.ax.plot(x_, y_, 'o', color=c, ms=5)
            self.ax.plot(x_, y_, 'xk', ms=4)
        self.ax.plot([], [], 'xk', label='Current position')


class XYCablePlotComp(XYPlotComp):
    def __init__(self, memory=10, delay=0.001, plot_initial=True, plot_improvements_only=False, ax=None, legendloc=1):
        if ax is None:
            self.fig, ax = plt.subplots(1, 1)
        else:
            self.fig = plt.gcf()
        self.hdisplay = display("", display_id=True)
        XYPlotComp.__init__(self, ax=ax)

    def plot_current_position(self, x, y):
        elnet_layout = mst(x, y)
        indices = np.array(list(elnet_layout.keys())).T
        self.ax.plot(x[indices], y[indices], color='r')
        XYPlotComp.plot_current_position(self, x, y)

    def compute(self, inputs, outputs):
        XYPlotComp.compute(self, inputs, outputs)
        self.hdisplay.update(self.fig)
