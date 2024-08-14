from topfarm import TurbineTypeOptimizationProblem, \
    TurbineXYZOptimizationProblem, InitialXYZOptimizationProblem
from openmdao.drivers.doe_generators import FullFactorialGenerator, \
    ListGenerator
from openmdao.drivers.doe_driver import DOEDriver
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.plotting import NoPlot
import numpy as np
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary_component import BoundaryComp
from topfarm.tests import npt

optimal = [(0, 2, 4, 1), (4, 2, 1, 0)]


def get_boundary_comp():
    return BoundaryComp(2, xy_boundary=[(0, 0), (4, 4)],
                        z_boundary=(0, 4),
                        xy_boundary_type='square')


def test_turbineType_and_XYZ_optimization():
    plot_comp = DummyCostPlotComp(optimal)
    plot_comp = NoPlot()
    cost_comp = DummyCost(
        optimal_state=optimal,
        inputs=['x', 'y', 'z', 'type'])
    xyz_opt_problem = TurbineXYZOptimizationProblem(
        cost_comp,
        turbineXYZ=[(0, 0, 0), (1, 1, 1)],
        min_spacing=2,
        boundary_comp=get_boundary_comp(),
        plot_comp=plot_comp,
        driver=EasyScipyOptimizeDriver(disp=False))
    tf = TurbineTypeOptimizationProblem(
        cost_comp=xyz_opt_problem,
        turbineTypes=[0, 0], lower=0, upper=1,
        driver=DOEDriver(FullFactorialGenerator(2)))
    cost = tf.optimize()[0]
    npt.assert_almost_equal(cost, 0)


def test_turbine_Type_multistart_XYZ_optimization():
    plot_comp = DummyCostPlotComp(optimal, delay=.5)
    plot_comp = NoPlot()
    xyz = [(0, 0, 0), (1, 1, 1)]

    p1 = DummyCost(optimal_state=optimal,
                   inputs=['x', 'y', 'z', 'type'])

    p2 = TurbineXYZOptimizationProblem(
        cost_comp=p1,
        turbineXYZ=xyz,
        min_spacing=2,
        boundary_comp=get_boundary_comp(),
        plot_comp=plot_comp,
        driver=EasyScipyOptimizeDriver(disp=True, optimizer='COBYLA', maxiter=10))
    p3 = InitialXYZOptimizationProblem(
        cost_comp=p2,
        turbineXYZ=xyz, min_spacing=2,
        boundary_comp=get_boundary_comp(),
        driver=DOEDriver(ListGenerator([[('x', [0, 4]), ('y', [2, 2]), ('z', [4, 1])]])))
    tf = TurbineTypeOptimizationProblem(
        cost_comp=p3,
        turbineTypes=[0, 0], lower=0, upper=1,
        driver=DOEDriver(FullFactorialGenerator(1)))

    case_gen = tf.driver.options['generator']
    cost, state, recorder = tf.optimize()
    print(cost)
    # print (state)
    print(recorder.get('type'))
    print(recorder.get('cost'))
    best_index = np.argmin(recorder.get('cost'))
    initial_xyz_recorder = recorder['recorder'][best_index]
    xyz_recorder = initial_xyz_recorder.get('recorder')[0]
    npt.assert_almost_equal(xyz_recorder['cost'][-1], cost)


if __name__ == '__main__':
    test_turbine_Type_multistart_XYZ_optimization()
