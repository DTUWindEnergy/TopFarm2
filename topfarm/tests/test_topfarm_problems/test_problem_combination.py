from topfarm._topfarm import TurbineTypeOptimizationProblem,\
    TurbineXYZOptimizationProblem, InitialXYZOptimizationProblem
from openmdao.drivers.doe_generators import FullFactorialGenerator,\
    ListGenerator
from openmdao.drivers.doe_driver import DOEDriver
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.plotting import NoPlot
import numpy as np
from topfarm.easy_drivers import EasyScipyOptimizeDriver
optimal = [(0, 2, 4, 1), (4, 2, 1, 0)]


def test_turbineType_and_XYZ_optimization():
    plot_comp = DummyCostPlotComp(optimal)
    plot_comp = NoPlot()
    cost_comp = DummyCost(
        optimal_state=optimal,
        inputs=['turbineX', 'turbineY', 'turbineZ', 'turbineType'])
    xyz_opt_problem = TurbineXYZOptimizationProblem(
        cost_comp,
        turbineXYZ=[(0, 0, 0), (1, 1, 1)],
        min_spacing=2,
        xy_boundary=[(0, 0), (4, 4)],
        z_boundary=(0, 4),
        xy_boundary_type='square',
        plot_comp=plot_comp,
        driver=EasyScipyOptimizeDriver(disp=False))
    tf = TurbineTypeOptimizationProblem(
        cost_comp=xyz_opt_problem,
        turbineTypes=[0, 0], lower=0, upper=1,
        driver=DOEDriver(FullFactorialGenerator(2)))
    print(tf.optimize())


def test_turbine_Type_multistart_XYZ_optimization():
    plot_comp = DummyCostPlotComp(optimal, delay=.5)
    plot_comp = NoPlot()
    xyz = [(0, 0, 0), (1, 1, 1)]
    xy_boundary = [(0, 0), (4, 4)]
    z_boundary = (0, 4)
    p1 = DummyCost(optimal_state=optimal,
                   inputs=['turbineX', 'turbineY', 'turbineZ', 'turbineType'])
    p2 = TurbineXYZOptimizationProblem(
        cost_comp=p1,
        turbineXYZ=xyz,
        min_spacing=2,
        xy_boundary=xy_boundary,
        z_boundary=z_boundary,
        xy_boundary_type='square',
        plot_comp=plot_comp,
        driver=EasyScipyOptimizeDriver(disp=True, optimizer='COBYLA', maxiter=10))
    p3 = InitialXYZOptimizationProblem(
        cost_comp=p2,
        turbineXYZ=xyz, min_spacing=2, xy_boundary=xy_boundary, z_boundary=z_boundary, xy_boundary_type='square',
        driver=DOEDriver(ListGenerator([[('turbineX', [0, 4]), ('turbineY', [2, 2]), ('turbineZ', [4, 1])]])))
    tf = TurbineTypeOptimizationProblem(
        cost_comp=p3,
        turbineTypes=[0, 0], lower=0, upper=1,
        driver=DOEDriver(FullFactorialGenerator(1)))

    case_gen = tf.driver.options['generator']
    cost, state, recorder = tf.optimize()
    print(cost)
    #print (state)
    #print (recorder.driver_iteration_lst)
    print(recorder.get('turbineType'))
    print(recorder.get('cost'))
    best_index = np.argmin(recorder.get('cost'))
    initial_xyz_recorder = recorder.driver_iteration_lst[best_index][-1]
    xyz_recorder = initial_xyz_recorder.get('recorder')[0]
    print(xyz_recorder.get(['turbineX', 'turbineY', 'turbineZ']))


if __name__ == '__main__':
    test_turbine_Type_multistart_XYZ_optimization()
