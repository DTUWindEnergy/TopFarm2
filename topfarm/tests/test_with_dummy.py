import os
import numpy as np
from topfarm.cost_models.dummy import DummyCostPlotComp
from topfarm.tests import uta
import pytest
from topfarm.tests.test_files import xy3tb


def test_optimize_3tb():
    """Optimize 3-turbine layout and check final positions

        The farm boundaries and min spacing are chosen such that the desired
        turbine positions are not within the boundaries or constraints.
    """

    dec_prec = 4  # decimal precision for comparison

    tf = xy3tb.get_tf()
    tf.optimize()
    tb_pos = tf.turbine_positions

    tol = 1e-6
    uta.assertGreater(sum((tb_pos[2] - tb_pos[0])**2), 2**2 - tol)  # check min spacing
    uta.assertLess(tb_pos[1][0], 6 + tol)  # check within border
    np.testing.assert_array_almost_equal(tb_pos[:, :2], xy3tb.optimal, dec_prec)


def testDummyCostPlotComp():
    tf = xy3tb.get_tf(plot_comp=DummyCostPlotComp(xy3tb.desired))
    tf.evaluate()
    tf.optimize()
