"""Tests for TOPFARM
"""
import warnings

import numpy as np
from topfarm.topfarm import TopFarm

from topfarm.cost_models.dummy import DummyCost_v2


def test_optimize_4tb():
    """Optimize 4-turbine layout and check final positions

        The farm boundaries and min spacing are chosen such that the desired
        turbine positions are not within the boundaries or constraints.
    """

    # test options
    dec_prec = 4  # decimal precision for comparison

    # given
    boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
    initial = [[6, 0], [6, -8], [1, 1], [-1, -8]]  # initial turbine layouts
    desired = [[3, -3], [7, -7], [4, -3], [3, -7]]  # desired turbine layouts
    optimal = np.array([[2.5, -3], [6, -7],
                        [4.5, -3], [3, -7]])  # optimal turbine layout
    min_spacing = 2  # min distance between turbines

    # when
    tf = TopFarm(initial, DummyCost_v2(desired), min_spacing,
                 boundary=boundary)
    tf.evaluate()
    with warnings.catch_warnings():  # suppress OpenMDAO/SLSQP warnings
        warnings.simplefilter('ignore')
        tf.optimize()
    tb_pos = tf.turbine_positions

    # # then
    np.testing.assert_array_almost_equal(tb_pos, optimal, dec_prec)


# class Test(unittest.TestCase):  # unittest version

    # def test_topfarm_with_dummy(self):
        # from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp

        # eps = 1e-6
        # optimal = [(3, -3), (7, -7), (4, -3), (3, -7)]
        # initial = [[6, 0], [6, -8], [1, 1], [-1, -8]]
        # boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]
        # plot = True
        # plot_comp = None
        # if plot:
            # plot_comp = DummyCostPlotComp(optimal)
        # tf = TopFarm(initial, DummyCost(optimal), 2, boundary=boundary, plot_comp=plot_comp)
        # tf.evaluate()
        # tf.optimize()

        # tb_pos = tf.turbine_positions
        # self.assertGreater(sum((tb_pos[2] - tb_pos[0])**2), 2**2 - eps)
        # np.testing.assert_array_almost_equal(tb_pos[3], optimal[3], 3)
        # self.assertLess(tb_pos[1][0], 6 + eps)
        # if plot:
            # plot_comp.show()


# if __name__ == "__main__":
    # #import sys;sys.argv = ['', 'Test.test_topfarm']
    # unittest.main()
