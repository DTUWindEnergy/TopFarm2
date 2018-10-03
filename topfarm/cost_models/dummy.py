import matplotlib.pyplot as plt
import numpy as np
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
import topfarm


class DummyCost(CostModelComponent):
    """Sum of squared error between current and optimal state
       Evaluates the equation
       f(x,..) = SUM((x_i - optx_i)^2 + ...).
    """

    def __init__(self, optimal_state, inputs=[topfarm.x_key, topfarm.y_key]):
        """
        Parameters
        ----------
        optimal_state : array_like, dim=(#wt,#inputs)
            optimal state array
        inputs : array_like
            list of input names
        """
        self.optimal_state = np.array(optimal_state)
        self.n_wt = self.optimal_state.shape[0]

        CostModelComponent.__init__(self, inputs, self.n_wt, self.cost, self.grad)

    def cost(self, **kwargs):
        opt = self.optimal_state
        return np.sum([(kwargs[n] - opt[:, i])**2 for i, n in enumerate(self.input_keys)])

    def grad(self, **kwargs):
        opt = self.optimal_state
        return [(2 * kwargs[n] - 2 * opt[:, i]) for i, n in enumerate(self.input_keys)]


class TurbineTypeDummyCost(DummyCost):
    def __init__(self, optimal):
        if len(np.array(optimal).shape) == 1:
            optimal = np.array([optimal]).T
        DummyCost.__init__(self, optimal, inputs=[topfarm.type_key])


class DummyCostPlotComp(XYPlotComp):
    def __init__(self, optimal, memory=10, delay=0.001, **kwargs):
        super().__init__(memory, delay, **kwargs)
        self.optimal = optimal

    def init_plot(self, boundary):
        opt_x, opt_y = np.array(self.optimal).T[0:2]
        XYPlotComp.init_plot(self, boundary)
        for c, optx, opty in zip(self.colors, opt_x, opt_y):
            plt.plot(optx, opty, 'ko', ms=6)
            plt.plot(optx, opty, 'o', color=c, ms=4)
        plt.plot([], [], 'o', color='k', markeredgecolor='k',
                 markerfacecolor='#00000000', ms=6, label='Optimal position')
