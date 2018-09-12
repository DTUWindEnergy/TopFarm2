import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp

from topfarm._topfarm import TurbineXYZOptimizationProblem
from topfarm.constraint_components.boundary_component import BoundaryComp
from topfarm.plotting import NoPlot
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
import warnings

initial = np.array([[0, 0, 0], [6, 0, 0], [6, -10, 0]])  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # desired turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3, 1], [7, -7, 2], [4, -3, 3]])  # desired turbine layouts


def test_setup_as_constraint_xy():
    from topfarm.cost_models.dummy import DummyCostPlotComp

    # plot_comp = DummyCostPlotComp(desired)
    plot_comp = NoPlot()

    tf = TurbineXYZOptimizationProblem(DummyCost(desired[:, :2], ['turbineX', 'turbineY']), initial[:, :2],
                                       boundary_comp=BoundaryComp(len(initial), boundary, None),
                                       plot_comp=plot_comp)

    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert tb_pos[1][0] < 6 + tol  # check within border


def test_setup_as_constraint_z():
    tf = TurbineXYZOptimizationProblem(DummyCost(desired[:, 2:], ['turbineZ']), initial,
                                       boundary_comp=BoundaryComp(len(initial), None, [0, 2]))
    tf.optimize()
    assert np.all(tf.turbine_positions[:, 2]) <= 2  # check within height limit


def test_setup_as_constraint_xyz():
    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['turbineX', 'turbineY', 'turbineZ']), initial,
                                       boundary_comp=BoundaryComp(len(initial), boundary, [0, 2]))
    tf.optimize()
    tb_pos = tf.turbine_positions
    tol = 1e-4
    assert tb_pos[1][0] < 6 + tol  # check within border
    assert np.all(tf.turbine_positions[:, 2]) <= 2  # check within height limit


def test_setup_as_penalty_xy():
    driver = SimpleGADriver()
    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['turbineX', 'turbineY']), initial,
                                       boundary_comp=BoundaryComp(len(initial), boundary, None),
                                       driver=driver)

    # check normal result if boundary constraint is satisfied
    assert tf.evaluate()[0] == 121
    # check penalized result if boundary constraint is not satisfied
    assert tf.evaluate({'turbineX': [2.5, 7, 4.5], 'turbineY': [-3., -7., -3.], 'turbineZ': [0., 0., 0.]})[0] == 1e10 + 1


def test_setup_as_penalty_z():
    driver = SimpleGADriver()
    tf = TurbineXYZOptimizationProblem(DummyCost(desired[:, 2:], ['turbineZ']), initial,
                                       boundary_comp=BoundaryComp(3, None, [0, 2]),
                                       driver=driver)

    # check normal result if height constraint is satisfied
    assert tf.evaluate()[0] == 14
    # check penalized result if height constraint is not satisfied
    assert tf.evaluate({'turbineX': [2.5, 7, 4.5], 'turbineY': [-3., -7., -3.], 'turbineZ': [0., 0., 3.]})[0] == 1e10 + 1
    assert tf.evaluate({'turbineX': [2.5, 7, 4.5], 'turbineY': [-3., -7., -3.], 'turbineZ': [-1., 0., 0.]})[0] == 1e10 + 1


def test_setup_as_penalty_xyz():
    driver = SimpleGADriver()
    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['turbineX', 'turbineY', 'turbineZ']), initial,
                                       boundary_comp=BoundaryComp(len(initial), boundary, [0, 2]),
                                       driver=driver)

    # check normal result if height and boundary constraint is satisfied
    assert tf.evaluate()[0] == 135
    # check penalized result if height and boundary constraint is not satisfied
    assert tf.evaluate({'turbineX': [2.5, 7, 4.5], 'turbineY': [-3., -7., -3.], 'turbineZ': [0., 0., 0.]})[0] == 1e10 + 1  # outside border
    assert tf.evaluate({'turbineX': [2.5, 7, 4.5], 'turbineY': [-3., -7., -3.], 'turbineZ': [0., 0., 3.]})[0] == 1e10 + 2  # above limit and outside
    assert tf.evaluate({'turbineX': [2.5, 7, 4.5], 'turbineY': [-3., -7., -3.], 'turbineZ': [-2., 0., 0.]})[0] == 1e10 + 3  # below limit and outside


def test_setup_as_penalty_none():
    driver = SimpleGADriver()
    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['turbineX', 'turbineY']), initial,
                                       boundary_comp=BoundaryComp(len(initial), None, None),
                                       driver=driver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # check that it does not fail if xy and z is not specified
        assert tf.evaluate()[0] == 121
        assert tf.evaluate({'turbineX': [2.5, 7, 4.5], 'turbineY': [-3., -7., -3.], 'turbineZ': [0., 0., 0.]})[0] == .5
