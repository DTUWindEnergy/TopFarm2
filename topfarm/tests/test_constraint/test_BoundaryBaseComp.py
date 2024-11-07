import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot
from topfarm.tests import npt
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm import TopFarmProblem
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver


initial = np.array([[0, 0, 0], [6, 0, 0], [6, -10, 0]])  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # desired turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3, 1], [7, -7, 2], [4, -3, 3]])  # desired turbine layouts


def test_setup_as_constraint_xy():
    # plot_comp = DummyCostPlotComp(desired)
    plot_comp = NoPlot()

    tf = TopFarmProblem(
        {'x': initial[:, 0], 'y': initial[:, 1]},
        DummyCost(desired[:, :2]),
        constraints=[XYBoundaryConstraint(boundary)],
        driver=EasyScipyOptimizeDriver(disp=False),
        plot_comp=plot_comp)

    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert tb_pos[1][0] < 6 + tol  # check within border


def test_setup_as_constraint_z():
    tf = TopFarmProblem(
        {'z': (initial[:, 2], 0, 2)},
        DummyCost(desired[:, :2], 'z'),
        driver=EasyScipyOptimizeDriver(disp=False),
    )

    tf.optimize()
    npt.assert_array_less(tf['z'], 2 + 1e-10)


def test_setup_as_constraint_xyz():
    desvar = dict(zip('xy', initial.T))
    desvar['z'] = (initial[:, 2], 0, 2)
    tf = TopFarmProblem(
        desvar,
        DummyCost(desired, 'xyz'),
        driver=EasyScipyOptimizeDriver(disp=False),
        constraints=[XYBoundaryConstraint(boundary)])
    tf.optimize()
    tb_pos = tf.turbine_positions
    tol = 1e-4
    assert tb_pos[1][0] < 6 + tol  # check within border
    npt.assert_array_less(tf['z'], 2 + tol)  # check within height limit


def test_setup_as_penalty_xy():
    driver = SimpleGADriver()
    tf = TopFarmProblem(
        dict(zip('xy', initial.T)),
        DummyCost(desired),
        constraints=[XYBoundaryConstraint(boundary)],
        driver=driver)

    # check normal result if boundary constraint is satisfied
    assert tf.evaluate()[0] == 121
    # check penalized result if boundary constraint is not satisfied
    assert tf.evaluate({'x': [2.5, 7, 4.5], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == 1e10 + 1


def test_setup_as_penalty_none():
    driver = SimpleGADriver()
    design_vars = dict(zip("xy", initial.T))
    for k, v in design_vars.items():
        design_vars[k] = (v, 0, 10)
    tf = TopFarmProblem(design_vars, DummyCost(desired), driver=driver)

    # check that it does not fail if xy and z is not specified
    assert tf.evaluate()[0] == 121
    assert tf.evaluate({'x': [2.5, 7, 4.5], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == .5
