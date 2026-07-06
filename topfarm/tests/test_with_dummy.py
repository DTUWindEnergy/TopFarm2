import os
import numpy as np
from topfarm.cost_models.dummy import DummyCostPlotComp
import pytest
from topfarm.tests.test_files import xy3tb
from topfarm._topfarm import TopFarmGroup, TopFarmProblem
from topfarm.recorders import TopFarmListRecorder
import topfarm


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
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0] < 6 + tol  # check within border
    np.testing.assert_array_almost_equal(tb_pos[:, :2], xy3tb.optimal, dec_prec)


def test_dummy_cost_plot_comp():
    tf = xy3tb.get_tf(plot_comp=DummyCostPlotComp(xy3tb.desired))
    tf.evaluate()
    tf.optimize()


def test_topfarm_group_empty_instantiation():
    # Regression: TopFarmGroup() raised TypeError when comps=[] because
    # the loop iterated over the parameter `comps` before super() could
    # normalize it into self.comps.
    group = TopFarmGroup()
    assert group.comps == []


def test_setup_with_existing_driver_recorder():
    # Regression: TopFarmProblem.setup() raised UnboundLocalError when the
    # driver already had a recorder attached, because tmp_recorder was only
    # assigned in one branch of the if/else but referenced in finally.
    tf = xy3tb.get_tf()
    existing_recorder = TopFarmListRecorder()
    tf.driver.add_recorder(existing_recorder)
    tf.setup()  # must not raise UnboundLocalError
    # verify original recorder is still attached
    assert existing_recorder in tf.driver._rec_mgr._recorders
