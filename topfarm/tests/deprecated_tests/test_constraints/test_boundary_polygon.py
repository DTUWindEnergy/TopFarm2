import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm import TopFarm
from topfarm.constraint_components.boundary_component import PolygonBoundaryComp
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver


def testPolygonConcave():
    optimal = [(1.5, 1.3), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()  # DummyCostPlotComp(optimal)
    initial = [(-0, .1), (4, 1.5)][::-1]
    tf = TopFarm(initial, DummyCost(optimal, inputs=['x', 'y']), 0,
                 boundary=boundary, boundary_type='polygon', plot_comp=plot_comp,
                 driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False))
    tf.evaluate()
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
    plot_comp.show()


def testPolygonTwoRegionsStartInWrong():
    optimal = [(1, 1), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()
    # plot_comp = DummyCostPlotComp(optimal, delay=.1)
    initial = [(3.5, 1.5), (0.5, 1.5)]
    tf = TopFarm(initial, DummyCost(optimal, inputs=['x', 'y']), 0,
                 boundary=boundary, boundary_type='polygon', plot_comp=plot_comp,
                 driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False))
    tf.optimize()
    plot_comp.show()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
