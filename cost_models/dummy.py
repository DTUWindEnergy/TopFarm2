from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np
from core.plotting import PlotComp
import matplotlib.pyplot as plt


class DummyCost(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-optimal_x)^2 + (y+optimal_y)^2 - 3.
    """

    def __init__(self, optimal_positions):
        ExplicitComponent.__init__(self)
        self.optimal = np.array(optimal_positions)
        self.N = self.optimal.shape[0]
        self.history = []

    def cost(self, x, y, optimal=None):
        if optimal is not None:
            opt_x, opt_y = optimal
        else:
            opt_x, opt_y = np.array(self.optimal).T
        #return (x - opt_x)**2 + (x - opt_x) * (y - opt_y) + (y - opt_y)**2 - 3.0
        return (x - opt_x)**2 +  (y - opt_y)**2 - 3.0

    def setup(self):
        self.add_input('turbineX', val=np.zeros(self.N), units='m')
        self.add_input('turbineY', val=np.zeros(self.N), units='m')

        self.add_output('cost', val=0.0)

        # Finite difference all partials.
        self.declare_partials('cost', '*')
        #self.declare_partials('*', '*', method='fd')

        
    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['turbineX']
        y = inputs['turbineY']
        
        #print(x, y, sum(self.cost(x, y)))
        outputs['cost'] = np.sum(self.cost(x, y))

    def compute_partials(self, inputs, J):
        x = inputs['turbineX']
        y = inputs['turbineY']
        #J['aep', 'turbineX'] = -(2 * x - 2 * np.array(self.optimal)[:, 0] + y - np.array(self.optimal)[:, 1])
        #J['aep', 'turbineY'] = -(2 * y - 2 * np.array(self.optimal)[:, 1] + x - np.array(self.optimal)[:, 0])
        J['cost', 'turbineX'] = (2 * x - 2 * np.array(self.optimal)[:, 0])
        J['cost', 'turbineY'] = (2 * y - 2 * np.array(self.optimal)[:, 1])

class DummyCostPlotComp(PlotComp):
    def __init__(self, optimal):
        super().__init__()
        self.optimal = optimal
        
    def init_plot(self, boundary):
        PlotComp.init_plot(self, boundary)
        for c, (optx, opty) in zip(self.colors, self.optimal):
            plt.plot(optx, opty, 'ko', ms=10)
            plt.plot(optx, opty, 'o',color=c, ms=8)