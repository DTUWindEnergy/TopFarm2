import pytest
import numpy as np
from topfarm.cost_models.cost_model_wrappers import CostModelComponent, \
    AEPCostModelComponent, AEPMaxLoadCostModelComponent
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint, CircleBoundaryComp, CircleBoundaryConstraint
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySimpleGADriver
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.tests import npt
from topfarm.tests.test_files import xy3tb


boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
initial = np.array([[6, 0], [6, -8], [1, 1], [-1, -8]])  # initial turbine layouts
optimal_with_constraints = np.array([[2.5, -3], [6, -7], [4.5, -3], [3, -7]])  # optimal turbine layout
min_spacing = 2  # min distance between turbines
optimal = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine layouts


def get_tf(cost_comp, plot_comp=NoPlot()):
    return TopFarmProblem(
        dict(zip('xy', initial.T)),
        cost_comp=cost_comp,
        plot_comp=plot_comp,
        constraints=[SpacingConstraint(min_spacing), XYBoundaryConstraint(boundary)],
        driver=EasyScipyOptimizeDriver(disp=False),
        expected_cost=1e-1)


def cost(x, y):
    opt_x, opt_y = optimal.T
    return np.sum((x - opt_x)**2 + (y - opt_y)**2)


def aep_cost(x, y):
    opt_x, opt_y = optimal.T
    return -np.sum((x - opt_x)**2 + (y - opt_y)**2)


def gradients(x, y):
    return (2 * x - 2 * optimal[:, 0]), (2 * y - 2 * optimal[:, 1])


def aep_gradients(x, y):
    return -(2 * x - 2 * optimal[:, 0]), -(2 * y - 2 * optimal[:, 1])


def test_CostModelComponent():
    tf = get_tf(CostModelComponent(['x', 'y'], 4, cost, gradients))
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal_with_constraints, 5)


def testCostModelComponent_no_gradients():
    tf = get_tf(CostModelComponent(['x', 'y'], 4, cost, None))
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal_with_constraints, 4)


def testAEPCostModelComponent():
    tf = get_tf(AEPCostModelComponent(['x', 'y'], 4, aep_cost, aep_gradients))
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal_with_constraints, 5)


@pytest.mark.skip(reason="This test is not deterministic")
def test_maxiter_CostModelComponent():
    tf = get_tf(AEPCostModelComponent(['x', 'y'], 4, aep_cost, aep_gradients, max_eval=10))
    cost, state, recorder = tf.optimize()
    assert (
        10 <= tf.cost_comp.counter <= 100
    ), tf.cost_comp.counter  # this is not deterministic, cannot set it to 12
    npt.assert_array_equal(recorder['AEP'][tf.cost_comp.n_func_eval], recorder['AEP'][tf.cost_comp.n_func_eval:])


def testCostModelComponentDiffShapeInput():
    def aep_cost(x, y, h):
        opt_x, opt_y = optimal.T
        return -np.sum((x - opt_x)**2 + (y - opt_y)**2) + h, {'add_out': sum(x)}
    cost_comp = AEPCostModelComponent(['x', 'y', ('h', 0)], 4, aep_cost, additional_output=[('add_out', 0)])
    tf = TopFarmProblem(
        dict(zip('xy', initial.T)),
        cost_comp=cost_comp,
        constraints=[SpacingConstraint(min_spacing), XYBoundaryConstraint(boundary)],
        driver=EasyScipyOptimizeDriver(disp=False),
        ext_vars={'h': 0})
    cost0, _, _ = tf.optimize(state={'h': 0})
    cost10, _, _ = tf.optimize(state={'h': 10})
    npt.assert_almost_equal(cost10, cost0 - 10)


def testCostModelComponentAdditionalOutput():
    def aep_cost(x, y):
        opt_x, opt_y = optimal.T
        return -np.sum((x - opt_x)**2 + (y - opt_y)**2), {'add_out': sum(x)}
    tf = get_tf(AEPCostModelComponent(['x', 'y'], 4, aep_cost, aep_gradients, additional_output=[('add_out', 0)]))
    _, state, _ = tf.optimize()
    npt.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal_with_constraints, 5)
    npt.assert_equal(sum(state['x']), state['add_out'])


def test_AEPMaxLoadCostModelComponent_as_penalty():
    tf = xy3tb.get_tf(
        design_vars={'x': ([0]), 'y': ([0])},
        cost_comp=AEPMaxLoadCostModelComponent(
            input_keys='xy', n_wt=1,
            aep_load_function=lambda x, y: (-np.sin(np.hypot(x, y)), np.hypot(x, y)),
            max_loads=3),
        constraints=[],
        driver=EasySimpleGADriver(),
        plot_comp=None)

    # check normal result that satisfies the penalty
    assert tf.evaluate({'x': np.pi / 2})[0] == 1
    # check penalized result if capacity constraint is not satisfied
    for x, y in [(4, 0), (0, 4)]:
        assert tf.evaluate({'x': x, 'y': y})[0] == 1e10 + 1


def test_AEPMaxLoadCostModelComponent_as_penalty_multi_wt():
    tf = xy3tb.get_tf(
        design_vars={'x': ([0, 1]), 'y': ([0, 0])},
        cost_comp=AEPMaxLoadCostModelComponent(
            input_keys='xy', n_wt=2,
            output_keys=["AEP", ('loads', [3, 3])],
            aep_load_function=lambda x, y: (-np.sin(np.hypot(x, y)).sum(), np.hypot(x, y)),
            max_loads=[3, 3]),
        constraints=[],
        driver=EasySimpleGADriver(),
        plot_comp=None)

    # check normal result that satisfies the penalty
    assert tf.evaluate({'x': [np.pi / 2, -np.pi / 2]})[0] == 2
    # check penalized result if capacity constraint is not satisfied
    for x, y in [([4, 0], [0, 0]), ([0, 0], [4, 0]), ([4, 4], [0, 0])]:
        assert tf.evaluate({'x': x, 'y': y})[0] == 1e10 + 1


def test_AEPMaxLoadCostModelComponent_constraint():

    tf = TopFarmProblem(
        design_vars={'x': ([1]), 'y': (.1, 0, 2.5)},
        # design_vars={'x': ([2.9], [1], [3])},
        cost_comp=AEPMaxLoadCostModelComponent(
            input_keys='xy', n_wt=1,
            aep_load_function=lambda x, y: (np.hypot(x, y), x),
            max_loads=3),
        constraints=[CircleBoundaryConstraint((0, 0), 7)],
    )

    tf.evaluate()
    cost, state, recorder = tf.optimize()
    npt.assert_allclose(state['x'], 3)  # constrained by max_loads
    npt.assert_allclose(state['y'], 2.5)  # constrained by design var lim
