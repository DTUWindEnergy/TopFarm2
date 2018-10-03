import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp

from topfarm import TurbineXYZOptimizationProblem
from topfarm.constraint_components.boundary_component import BoundaryComp
from topfarm.plotting import NoPlot
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
import warnings
from topfarm.tests import npt


initial = np.array([[0, 0, 0], [6, 0, 0], [6, -10, 0]])  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # desired turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3, 1], [7, -7, 2], [4, -3, 3]])  # desired turbine layouts


def test_setup_as_constraint_xy():
    from topfarm.cost_models.dummy import DummyCostPlotComp

    # plot_comp = DummyCostPlotComp(desired)
    plot_comp = NoPlot()

    tf = TurbineXYZOptimizationProblem(DummyCost(desired[:, :2], ['x', 'y']), initial[:, :2],
                                       boundary_comp=BoundaryComp(len(initial), boundary, None),
                                       plot_comp=plot_comp)

    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert tb_pos[1][0] < 6 + tol  # check within border


def test_setup_as_penalty_xy():
    driver = SimpleGADriver()
    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['x', 'y']), initial,
                                       boundary_comp=BoundaryComp(len(initial), boundary, None),
                                       driver=driver)

    # check normal result if boundary constraint is satisfied
    assert tf.evaluate()[0] == 121
    # check penalized result if boundary constraint is not satisfied
    assert tf.evaluate({'x': [2.5, 7, 4.5], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == 1e10 + 1


def test_setup_as_penalty_none():
    driver = SimpleGADriver()
    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['x', 'y']), initial,
                                       boundary_comp=BoundaryComp(len(initial), None, None),
                                       driver=driver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # check that it does not fail if xy and z is not specified
        assert tf.evaluate()[0] == 121
        assert tf.evaluate({'x': [2.5, 7, 4.5], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == .5
