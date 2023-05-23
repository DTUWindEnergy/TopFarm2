from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import FullFactorialGenerator

from topfarm import TurbineTypeOptimizationProblem, ProblemComponent
from topfarm.cost_models.dummy import TurbineTypeDummyCost
import numpy as np
import pytest
from topfarm.tests import npt
from topfarm import TopFarm
from topfarm.cost_models.cost_model_wrappers import CostModelComponent


"""Test methods in TopFarmProblem
cost
state
state_array
update state
evaluate
optimize
check_gradients
as_component
get_DOE_list
get_DOE_array
"""


@pytest.fixture
def turbineTypeOptimizationProblem():
    return TurbineTypeOptimizationProblem(
        TurbineTypeDummyCost([2, 0, 1]),
        turbineTypes=[0, 0, 0],
        lower=[0, 0, 0],
        upper=[2, 2, 2],
        driver=DOEDriver(FullFactorialGenerator(3)))


def test_cost(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    cost, _, = tf.evaluate()
    npt.assert_equal(tf.cost, cost)
    assert tf.cost == 5


def test_state(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    npt.assert_equal(tf.state, {'type': [0, 0, 0]})


def test_state_array(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    arr = tf.state_array(['type', 'type'])
    npt.assert_equal(arr.shape, [3, 2])
    npt.assert_array_equal(arr, [[0, 0],
                                 [0, 0],
                                 [0, 0]])


@pytest.mark.parametrize('types,cost', [([0, 0, 0], 5),
                                        ([2, 0, 2], 1)])
def test_update_state(turbineTypeOptimizationProblem, types, cost):
    tf = turbineTypeOptimizationProblem
    c, state = tf.evaluate({'type': types})
    npt.assert_equal(c, cost)
    npt.assert_array_equal(state['type'], types)
    # wrong shape
    c, state = tf.evaluate({'type': [types]})
    npt.assert_equal(c, cost)
    npt.assert_array_equal(state['type'], types)
    # missing key
    c, state = tf.evaluate({'missing': types})


def test_evaluate(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    cost, state = tf.evaluate()
    assert cost == 5
    np.testing.assert_array_equal(state['type'], [0, 0, 0])


def test_optimize(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    cost, state, recorder = tf.optimize()
    assert cost == 0
    np.testing.assert_array_equal(state['type'], [2, 0, 1])
    doe_list = np.squeeze(tf.get_DOE_array())
    np.testing.assert_array_almost_equal(recorder.get('cost'), np.sum((doe_list - [2, 0, 1])**2, 1))


initial = [[6, 0, 70, 0],
           [6, -8, 71, 1],
           [1, 1, 72, 2],
           [-1, -8, 73, 3]]  # initial turbine layouts
optimal = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine layouts
boundary = [(0, 0), (0, -10), (6, -10), (6, 0)]  # turbine boundaries


@pytest.fixture
def turbineXYZOptimizationProblem_generator():
    def _topfarm_obj(gradients, **kwargs):
        return TopFarm(initial, CostModelComponent(['x', 'y'], 4, cost, gradients),
                       min_spacing=2, boundary=boundary, record_id=None, **kwargs)
    return _topfarm_obj


def cost(x, y):
    return np.sum((x - optimal[:, 0])**2 + (y - optimal[:, 1])**2)


def gradients(x, y):
    return ((2 * x - 2 * optimal[:, 0]),
            (2 * y - 2 * optimal[:, 1]))


def wrong_gradients(x, y):
    return ((2 * x - 2 * optimal[:, 0]) + 1,
            (2 * y - 2 * optimal[:, 1]))


def testTopFarmProblem_check_gradients(turbineXYZOptimizationProblem_generator):
    # Check that gradients check does not raise exception for correct gradients
    tf = turbineXYZOptimizationProblem_generator(gradients)
    tf.check_gradients(True)

    # Check that gradients check raises an exception for incorrect gradients
    tf = turbineXYZOptimizationProblem_generator(wrong_gradients)
    with pytest.raises(Warning, match="Mismatch between finite difference derivative of 'cost' wrt. 'x' and derivative computed in 'cost_comp' is"):
        tf.check_gradients()


def testTopFarmProblem_evaluate_gradients(turbineXYZOptimizationProblem_generator):
    tf = turbineXYZOptimizationProblem_generator(gradients)
    np.testing.assert_array_equal(tf.evaluate_gradients()['final_cost']['x'], [[-6., -14., -8., -6.]])


def testTopFarmProblem_as_component(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    c = tf.as_component()
    npt.assert_equal(c.__class__, ProblemComponent)
    assert c.problem == tf


def testTopFarmProblem_get_DOE_list(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    npt.assert_array_equal(len(tf.get_DOE_list()), 27)
    (k, v), = tf.get_DOE_list()[1]
    assert k == "indeps.type"
    npt.assert_array_equal(v, [1, 0, 0])

    # npt.assert_array_equal(tf.get_DOE_list()[1], [[('indeps.turbineType', array([0., 0., 0.]))], [('indeps.turbineType', array([1., 0., 0.]))]])


def testTopFarmProblem_get_DOE_array(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    npt.assert_array_equal(tf.get_DOE_array().shape, (27, 1, 3))
    npt.assert_array_equal(tf.get_DOE_array()[:5], [[[0, 0, 0]],
                                                    [[1, 0, 0]],
                                                    [[2, 0, 0]],
                                                    [[0, 1, 0]],
                                                    [[1, 1, 0]]])
