'''
Created on 16. apr. 2018

@author: mmpe
'''
import unittest
import numpy as np
from topfarm.topfarm import TopFarm


class Test(unittest.TestCase):

    def test_topfarm_with_dummy(self):
        from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp

        eps = 1e-6
        optimal = [(3, -3), (7, -7), (4, -3), (3, -7)]
        initial = [[6, 0], [6, -8], [1, 1], [-1, -8]]
        boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]
        plot = True
        plot_comp = None
        if plot:
            plot_comp = DummyCostPlotComp(optimal)
        tf = TopFarm(initial, DummyCost(optimal), 2, boundary=boundary, plot_comp=plot_comp)
        tf.evaluate()
        tf.optimize()

        tb_pos = tf.turbine_positions
        self.assertGreater(sum((tb_pos[2] - tb_pos[0])**2), 2**2 - eps)
        np.testing.assert_array_almost_equal(tb_pos[3], optimal[3], 3)
        self.assertLess(tb_pos[1][0], 6 + eps)
        if plot:
            plot_comp.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_topfarm']
    unittest.main()
