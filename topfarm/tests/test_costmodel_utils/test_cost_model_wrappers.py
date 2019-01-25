import numpy as np
from topfarm.cost_models.cost_model_wrappers import CostModelComponent,\
    AEPCostModelComponent
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.tests import npt

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
        driver=EasyScipyOptimizeDriver(disp=False))


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
    tf = get_tf(CostModelComponent(['x', 'y'], 4, cost, gradients),)
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal_with_constraints, 5)


def testCostModelComponent_no_gradients():
    tf = get_tf(CostModelComponent(['x', 'y'], 4, cost, None))
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal_with_constraints, 5)


def testAEPCostModelComponent():
    tf = get_tf(AEPCostModelComponent(['x', 'y'], 4, aep_cost, aep_gradients))
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal_with_constraints, 5)


def test_maxiter_CostModelComponent():
    tf = get_tf(AEPCostModelComponent(['x', 'y'], 4, aep_cost, aep_gradients, max_eval=10))
    cost, state, recorder = tf.optimize()
    assert 10 <= tf.cost_comp.counter <= 11, tf.cost_comp.counter
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
