'''
Created on 17. maj 2018

@author: mmpe
'''
from topfarm import TopFarm
import unittest

import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.cost_models.cost_model_wrappers import CostModelComponent


class TestTopFarm(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
        self.initial = [[6, 0], [6, -8], [1, 1], [-1, -8]]  # initial turbine layouts
        self.optimal_with_constraints = np.array([[2.5, -3], [6, -7], [4.5, -3], [3, -7]])  # optimal turbine layout
        self.min_spacing = 2  # min distance between turbines
        self.optimal = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine layouts

    def cost(self, pos):
        x, y = pos.T
        opt_x, opt_y = self.optimal.T
        return np.sum((x - opt_x)**2 + (y - opt_y)**2)

    def gradients(self, pos):
        x, y = pos.T
        return (2 * x - 2 * self.optimal[:, 0]), (2 * y - 2 * self.optimal[:, 1])

    def wrong_gradients(self, pos):
        x, y = pos.T
        return (2 * x - 2 * self.optimal[:, 0] + 1), (2 * y - 2 * self.optimal[:, 1])

    def testTopFarm_default_plotcomp(self):
        tf = TopFarm(self.initial, CostModelComponent(4, self.cost, self.gradients), self.min_spacing, boundary=self.boundary, plot_comp='default')

    def testTopFarm_check_gradients(self):
        tf = TopFarm(self.initial, CostModelComponent(4, self.cost, self.gradients), self.min_spacing, boundary=self.boundary)
        tf.check(True)

        tf = TopFarm(self.initial, CostModelComponent(4, self.cost, self.wrong_gradients), self.min_spacing, boundary=self.boundary)
        self.assertRaisesRegex(Warning, "Mismatch between finite difference derivative of 'cost' wrt. 'turbineX' and derivative computed in 'cost_comp' is", tf.check)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
