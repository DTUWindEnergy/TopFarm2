'''
Created on 17. maj 2018

@author: mmpe
'''
import unittest
import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.cost_models.cost_model_wrappers import CostModelComponent,\
    AEPCostModelComponent
from topfarm import TopFarm


class TestCostModelWrappers(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
        self.initial = [[6, 0], [6, -8], [1, 1], [-1, -8]]  # initial turbine layouts
        self.optimal_with_constraints = np.array([[2.5, -3], [6, -7], [4.5, -3], [3, -7]])  # optimal turbine layout
        self.min_spacing = 2  # min distance between turbines
        self.optimal = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine layouts

    def cost(self, tb):
        x, y = tb[:, :2].T
        opt_x, opt_y = self.optimal.T
        return np.sum((x - opt_x)**2 + (y - opt_y)**2)

    def aep_cost(self, tb):
        x, y = tb[:, :2].T
        opt_x, opt_y = self.optimal.T
        return -np.sum((x - opt_x)**2 + (y - opt_y)**2)

    def gradients(self, tb):
        x, y = tb[:, :2].T
        return (2 * x - 2 * self.optimal[:, 0]), (2 * y - 2 * self.optimal[:, 1])

    def aep_gradients(self, tb):
        x, y = tb[:, :2].T
        return -(2 * x - 2 * self.optimal[:, 0]), -(2 * y - 2 * self.optimal[:, 1])

    def testCostModelComponent(self):
        tf = TopFarm(self.initial, CostModelComponent(4, self.cost, self.gradients), self.min_spacing, boundary=self.boundary, record_id=None)
        tf.optimize()
        np.testing.assert_array_almost_equal(tf.turbine_positions, self.optimal_with_constraints, 5)

    def testCostModelComponent_no_gradients(self):
        tf = TopFarm(self.initial, CostModelComponent(4, self.cost), self.min_spacing, boundary=self.boundary, record_id=None)
        tf.optimize()
        np.testing.assert_array_almost_equal(tf.turbine_positions, self.optimal_with_constraints, 5)

    def testAEPCostModelComponent(self):
        tf = TopFarm(self.initial, AEPCostModelComponent(4, self.aep_cost, self.aep_gradients),
                     self.min_spacing, boundary=self.boundary, record_id=None)
        tf.optimize()
        np.testing.assert_array_almost_equal(tf.turbine_positions, self.optimal_with_constraints, 5)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
