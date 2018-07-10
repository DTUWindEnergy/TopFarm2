from topfarm import TopFarm

import numpy as np
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
import pytest

initial = np.zeros((4, 4))
initial[:, :2] = [[6, 0], [6, -8], [1, 1], [-1, -8]]  # initial turbine layouts
initial[:, 2] = range(70, 74)
initial[:, 3] = range(4)

optimal = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine layouts
boundary = [(0, 0), (0, -10), (6, -10), (6, 0)]  # turbine boundaries


@pytest.fixture
def topfarm_generator():
    def _topfarm_obj(gradients, **kwargs):
        return TopFarm(initial, CostModelComponent(4, cost, gradients), min_spacing=2, boundary=boundary, record_id=None, **kwargs)
    return _topfarm_obj


def cost(pos):
    x, y = pos[:, :2].T
    opt_x, opt_y = optimal.T
    return np.sum((x - opt_x)**2 + (y - opt_y)**2)


def gradients(pos):
    x, y = pos[:, :2].T
    return (2 * x - 2 * optimal[:, 0]), (2 * y - 2 * optimal[:, 1])


def wrong_gradients(pos):
    x, y = pos[:, :2].T
    return (2 * x - 2 * optimal[:, 0] + 1), (2 * y - 2 * optimal[:, 1])


def testTopFarm_default_plotcomp(topfarm_generator):
    """Check that setting plot_comp to 'default' does not fails"""
    topfarm_generator(gradients, plot_comp='default')


def testTopFarm_check_gradients(topfarm_generator):
    # Check that gradients check does not raise exception for correct gradients
    tf = topfarm_generator(gradients)
    tf.check(True)

    # Check that gradients check raises an exception for incorrect gradients
    tf = topfarm_generator(wrong_gradients)
    with pytest.raises(Warning, match="Mismatch between finite difference derivative of 'cost' wrt. 'turbineX' and derivative computed in 'cost_comp' is"):
        tf.check()


def testTopFarm_evaluate(topfarm_generator):
    # check that evaluate function does not fail
    tf = topfarm_generator(gradients)
    cost, tb = tf.evaluate()
    assert cost == 62
    np.testing.assert_array_equal(tb[:, :2], initial)


def testTopFarm_evaluate_gradients(topfarm_generator):
    # check taht evalueate_gradients does not fail
    tf = topfarm_generator(gradients)
    np.testing.assert_array_equal(tf.evaluate_gradients()['cost']['turbineX'], [[-6., -14., -8., -6.]])


# ------------------- Topfarm methods and properties ---------------------------

def testTopfarm_turbines(topfarm_generator):
    tf = topfarm_generator(gradients)
    np.testing.assert_array_equal(tf.turbines, initial)


def testTopfarm_turbine_positions(topfarm_generator):
    tf = topfarm_generator(gradients)
    np.testing.assert_array_equal(tf.turbine_positions, initial[:, :2])


def testTopfarm_boundary(topfarm_generator):
    tf = topfarm_generator(gradients)
    np.testing.assert_array_equal(tf.boundary, np.r_[boundary, boundary[:1]])

    # for no xy-boundary return empty list
    tf = TopFarm(initial, CostModelComponent(4, cost, gradients), min_spacing=2, boundary=None, record_id=None)
    assert tf.boundary == []
