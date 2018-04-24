from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np
import matplotlib.pyplot as plt


class PlotComp(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """
    colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey'] * 100

    def __init__(self, memory=10):
        ExplicitComponent.__init__(self)
        self.memory = memory
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
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['turbineX']
        y = inputs['turbineY']
        cost = inputs['cost']
        self.history = [(x.copy(), y.copy())] + self.history[:self.memory]
        
        boundary = inputs['boundary']
        self.init_plot(boundary)
        plt.title(cost)

        history_arr = np.array(self.history)
        for i, c, x_, y_ in zip(range(len(x)), self.colors, x, y):
            plt.plot(history_arr[:, 0, i], history_arr[:, 1, i], '.-', color=c, lw=1)
            plt.plot(x_, y_, 'o', color=c, ms=5)
            plt.plot(x_, y_, 'x' + 'k', ms=4)

        plt.pause(.01)
        self.counter += 1
