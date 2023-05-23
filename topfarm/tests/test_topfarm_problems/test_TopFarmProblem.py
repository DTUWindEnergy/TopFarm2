from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import FullFactorialGenerator
import pytest
import numpy as np
from topfarm import ProblemComponent
from topfarm import TopFarmProblem
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.tests.test_files import xy3tb

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
turbine_positions
smart_start
"""


@pytest.fixture
def turbineTypeOptimizationProblem():
    return TopFarmProblem(
        design_vars={'type': ([0, 0, 0], 0, 2)},
        cost_comp=DummyCost(np.array([[2, 0, 1]]).T, ['type']),
        driver=DOEDriver(FullFactorialGenerator(3)))


@pytest.mark.parametrize('design_vars', [{'type': ([0, 0, 0], 0, 2)},
                                         [('type', ([0, 0, 0], 0, 2))],
                                         (('type', ([0, 0, 0], 0, 2)),),
                                         zip(['type'], [([0, 0, 0], 0, 2)]),
                                         ])
def test_design_var_list(turbineTypeOptimizationProblem, design_vars):
    tf = TopFarmProblem(
        design_vars=design_vars,
        cost_comp=DummyCost(np.array([[2, 0, 1]]).T, ['type']),
        driver=DOEDriver(FullFactorialGenerator(3)))
    cost, _, = tf.evaluate()
    npt.assert_equal(tf.cost, cost)
    assert tf.cost == 5


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
    tf.evaluate(disp=True)  # test that disp=True does not fail


def test_optimize(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    cost, state, recorder = tf.optimize()
    assert cost == 0
    np.testing.assert_array_equal(state['type'], [2, 0, 1])
    doe_list = np.squeeze(tf.get_DOE_array())
    np.testing.assert_array_almost_equal(recorder.get('cost'), np.sum((doe_list - [2, 0, 1])**2, 1))
    tf.optimize(disp=True)  # test that disp=True does not fail


initial = np.array([[6, 0, 70, 0],
                    [6, -8, 71, 1],
                    [1, 1, 72, 2],
                    [-1, -8, 73, 3]])  # initial turbine layouts
optimal = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine layouts
boundary = [(0, 0), (0, -10), (6, -10), (6, 0)]  # turbine boundaries


@pytest.fixture
def turbineXYZOptimizationProblem_generator():
    def _topfarm_obj(gradients, cost_comp=None, **kwargs):

        return TopFarmProblem(
            {'x': initial[:, 0], 'y': initial[:, 1]},
            cost_comp=cost_comp or CostModelComponent(['x', 'y'], 4, cost, gradients),
            constraints=[SpacingConstraint(2), XYBoundaryConstraint(boundary)],
            driver=EasyScipyOptimizeDriver(),
            **kwargs)
    return _topfarm_obj


def cost(x, y):
    return np.sum((x - optimal[:, 0])**2 + (y - optimal[:, 1])**2)


def income(x, y):
    return -np.sum((x - optimal[:, 0])**2 + (y - optimal[:, 1])**2)


def income_gradients(x, y):
    return (-(2 * x - 2 * optimal[:, 0]),
            -(2 * y - 2 * optimal[:, 1]))


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


def testTopFarmProblem_check_gradients_Income(turbineXYZOptimizationProblem_generator):
    # Check that gradients check does not raise exception for correct gradients
    cost_comp = CostModelComponent('xy', 4, income, income_gradients, maximize=True)
    tf = turbineXYZOptimizationProblem_generator(None, cost_comp)
    tf.check_gradients(True)

    # Check that gradients check raises an exception for incorrect gradients
    cost_comp = CostModelComponent('xy', 4, income, wrong_gradients, maximize=True)
    tf = turbineXYZOptimizationProblem_generator(None, cost_comp)
    with pytest.raises(Warning, match="Mismatch between finite difference derivative of 'cost' wrt. 'y' and derivative computed in 'cost_comp' is"):
        tf.check_gradients()


def testTopFarmProblem_evaluate_gradients(turbineXYZOptimizationProblem_generator):
    tf = turbineXYZOptimizationProblem_generator(gradients)
    np.testing.assert_array_equal(tf.evaluate_gradients(disp=True)['final_cost']['x'], [[-6., -14., -8., -6.]])


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


def testTopFarmProblem_turbine_positions():
    tf = xy3tb.get_tf()
    np.testing.assert_array_equal(tf.turbine_positions, xy3tb.initial)


def test_smart_start():
    xs_ref = [1.6, 1.6, 3.7]
    ys_ref = [1.6, 3.7, 1.6]

    x = np.arange(0, 5, 0.1)
    y = np.arange(0, 5, 0.1)
    YY, XX = np.meshgrid(y, x)
    ZZ = np.sin(XX) + np.sin(YY)
    min_spacing = 2.1
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(min_spacing)])

    tf.smart_start(XX, YY, ZZ, seed=0, plot=True)
    try:
        npt.assert_array_almost_equal(tf.turbine_positions, np.array([xs_ref, ys_ref]).T)
    except AssertionError:
        # wt2 and wt3 may switch
        npt.assert_array_almost_equal(tf.turbine_positions, np.array([ys_ref, xs_ref]).T)
    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, ZZ, 100)
        for x, y in tf.turbine_positions:
            circle = plt.Circle((x, y), min_spacing / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(x, y, 'rx')
        plt.axis('equal')
        plt.show()


def test_smart_start_boundary():
    xs_ref = [1.6, 1.6, 3.6]
    ys_ref = [1.6, 3.7, 2.3]

    x = np.arange(0, 5.1, 0.1)
    y = np.arange(0, 5.1, 0.1)
    YY, XX = np.meshgrid(y, x)
    ZZ = np.sin(XX) + np.sin(YY)
    min_spacing = 2.1
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(min_spacing),
                                   XYBoundaryConstraint([(0, 0), (5, 3), (5, 5), (0, 5)])])
    tf.smart_start(XX, YY, ZZ)
    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, ZZ, 100)
        plt.plot(tf.xy_boundary[:, 0], tf.xy_boundary[:, 1], 'k')
        for x, y in tf.turbine_positions:
            circle = plt.Circle((x, y), min_spacing / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(x, y, 'rx')

        plt.axis('equal')
        plt.show()
    npt.assert_array_almost_equal(tf.turbine_positions, np.array([xs_ref, ys_ref]).T)


def test_smart_start_polygon_boundary():
    xs_ref = [1.6, 1.6, 3.6]
    ys_ref = [1.6, 3.7, 2.3]

    x = np.arange(0, 5.1, 0.1)
    y = np.arange(0, 5.1, 0.1)
    YY, XX = np.meshgrid(y, x)
    ZZ = np.sin(XX) + np.sin(YY)
    min_spacing = 2.1
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(min_spacing),
                                   XYBoundaryConstraint([(0, 0), (5, 3), (5, 5), (0, 5)], 'polygon')])
    tf.smart_start(XX, YY, ZZ)
    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, ZZ, 100)
        plt.plot(tf.xy_boundary[:, 0], tf.xy_boundary[:, 1], 'k')
        for x, y in tf.turbine_positions:
            circle = plt.Circle((x, y), min_spacing / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(x, y, 'rx')

        plt.axis('equal')
        plt.show()
    npt.assert_array_almost_equal(tf.turbine_positions, np.array([xs_ref, ys_ref]).T)


def testTopFarmProblem_approx_totols():
    tf = xy3tb.get_tf(approx_totals=True)
    np.testing.assert_array_equal(tf.turbine_positions, xy3tb.initial)


def testTopFarmProblem_expected_cost():
    tf = xy3tb.get_tf(expected_cost=None)
    np.testing.assert_array_equal(tf.turbine_positions, xy3tb.initial)


def testTopFarmProblem_update_reports(turbineTypeOptimizationProblem):
    tf = turbineTypeOptimizationProblem
    tf._update_reports(DOEDriver(FullFactorialGenerator(3)))
