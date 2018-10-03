import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot

from topfarm import TurbineXYZOptimizationProblem
from topfarm.constraint_components.boundary_component import BoundaryComp
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
import warnings


initial = np.array([[6, 0], [6, -8], [1, 1]])  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # desired turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3], [7, -7], [4, -3]])  # desired turbine layouts


def test_spacing():
    from topfarm.cost_models.dummy import DummyCostPlotComp

    # plot_comp = DummyCostPlotComp(desired)
    plot_comp = NoPlot()

    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['x', 'y']), initial,
                                       boundary_comp=BoundaryComp(len(initial), boundary, None),
                                       min_spacing=2, plot_comp=plot_comp)

    tf.evaluate()
    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing


def test_spacing_as_penalty():

    driver = SimpleGADriver()

    tf = TurbineXYZOptimizationProblem(DummyCost(desired, ['x', 'y']), initial,
                                       boundary_comp=BoundaryComp(len(initial), None, None),
                                       min_spacing=2, driver=driver)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # check normal result if spacing constraint is satisfied
        assert tf.evaluate()[0] == 45
        # check penalized result if spacing constraint is not satisfied
        assert tf.evaluate({'x': [3, 7, 4.], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == 1e10 + 3
