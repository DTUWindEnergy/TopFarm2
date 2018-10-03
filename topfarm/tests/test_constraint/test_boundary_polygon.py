import unittest

import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm import TopFarm

from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint,\
    PolygonBoundaryComp
from topfarm._topfarm import TopFarmProblem


def get_tf(initial, optimal, boundary, plot_comp=NoPlot()):
    initial, optimal = map(np.array, [initial, optimal])
    return TopFarmProblem(
        {'x': initial[:, 0], 'y': initial[:, 1]},
        DummyCost(optimal),
        constraints=[XYBoundaryConstraint(boundary, 'polygon')],
        driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False),
        plot_comp=plot_comp)


def testPolygon():
    boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
    b = PolygonBoundaryComp(0, boundary)
    np.testing.assert_array_equal(b.xy_boundary[:, :2], [[0, 0],
                                                         [1, 1],
                                                         [2, 0],
                                                         [2, 2],
                                                         [0, 2],
                                                         [0, 0]])


def testPolygonConcave():
    optimal = [(1.5, 1.3), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()  # DummyCostPlotComp(optimal)
    initial = [(-0, .1), (4, 1.5)][::-1]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
    plot_comp.show()


def testPolygonTwoRegionsStartInWrong():
    optimal = [(1, 1), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()  # DummyCostPlotComp(optimal, delay=.1)
    initial = [(3.5, 1.5), (0.5, 1.5)]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    plot_comp.show()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
