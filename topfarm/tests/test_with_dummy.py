"""Tests for TOPFARM
"""
import os
from topfarm import TopFarm
import unittest
import pytest
import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.tests import uta


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
    optimal = np.array([[2.5, -3], [6, -7], [4.5, -3], [3, -7]])  # optimal turbine layout
    min_spacing = 2  # min distance between turbines

    # when
    tf = TopFarm(initial, DummyCost(desired, ['turbineX', 'turbineY']), min_spacing,
                 boundary=boundary, record_id=None)
    tf.optimize()
    tb_pos = tf.turbine_positions

    # then
    tol = 1e-6
    uta.assertGreater(sum((tb_pos[2] - tb_pos[0])**2), 2**2 - tol)  # check min spacing
    uta.assertLess(tb_pos[1][0], 6 + tol)  # check within border
    np.testing.assert_array_almost_equal(tb_pos[:, :2], optimal, dec_prec)


def testDummyCostPlotComp():
    if os.name == 'posix' and "DISPLAY" not in os.environ:
        pytest.xfail("No display")

    desired = [[3, -3], [7, -7], [4, -3], [3, -7]]
    tf = TopFarm(turbines=[[6, 0], [6, -8], [1, 1], [-1, -8]],
                 cost_comp=DummyCost(desired, ['turbineX', 'turbineY']),
                 min_spacing=2,
                 boundary=[(0, 0), (6, 0), (6, -10), (0, -10)],
                 plot_comp=DummyCostPlotComp(desired),
                 record_id=None)
    tf.evaluate()
    tf.optimize()
