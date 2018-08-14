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
    def _topfarm_obj(**kwargs):
        return TopFarm(initial, CostModelComponent(['turbineX', 'turbineY'], 4, cost, gradients),
                       min_spacing=2, boundary=boundary, record_id=None, **kwargs)
    return _topfarm_obj


def cost(turbineX, turbineY):
    opt_x, opt_y = optimal.T
    return np.sum((turbineX - opt_x)**2 + (turbineY - opt_y)**2)


def gradients(turbineX, turbineY):
    x, y = turbineX, turbineY
    return (2 * x - 2 * optimal[:, 0]), (2 * y - 2 * optimal[:, 1])


def testTopFarm(topfarm_generator):
    tf = topfarm_generator()
    cost, state = tf.evaluate()
    assert cost == 62
    np.testing.assert_array_equal(state['turbineX'], initial[:, 0])
    np.testing.assert_array_equal(state['turbineY'], initial[:, 1])
