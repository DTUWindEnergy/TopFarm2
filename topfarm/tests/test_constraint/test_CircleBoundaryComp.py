import os
import numpy as np

from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
import pytest

from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint, CircleBoundaryComp
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot, PlotComp
from topfarm.tests import npt


def testCircle():
    b = CircleBoundaryComp(3, [1, 2], 3)
    np.testing.assert_array_almost_equal(((b.xy_boundary - [1, 2])**2).sum(1), 3**2)


def test_TopFarmProblem_with_cirleboundary_penalty():
    optimal = np.array([(0, 0)])
    desvar = dict(zip('xy', optimal.T))
    b = CircleBoundaryConstraint([1, 2], 3)
    driver = SimpleGADriver()
    tf = TopFarmProblem(
        desvar,
        DummyCost(optimal, 'xy'),
        constraints=[b],
        driver=driver)
    tf.evaluate()
    tf.plot_comp.show()
    np.testing.assert_array_almost_equal(
        ((b.constraintComponent.xy_boundary - [1, 2])**2).sum(1), 3**2)
    npt.assert_array_less(tf.evaluate({'x': [3.9], 'y': [2]})[0], 1e10)
    npt.assert_array_less(1e10, tf.evaluate({'x': [4.1], 'y': [2]})[0])


def test_TopFarmProblem_with_cirleboundary_constraint():
    optimal = np.array([(0, 0)])
    desvar = dict(zip('xy', optimal.T))

    b = CircleBoundaryConstraint([2, 2], 2)
    tf = TopFarmProblem(
        desvar,
        DummyCost(optimal, 'xy'),
        constraints=[b])
    _, state, _ = tf.optimize()

    npt.assert_array_less((state['x'] - 2)**2 + (state['y'] - 2)**2, 9)


def test_TopFarmProblem_with_cirleboundary_constraint_and_limits():
    optimal = np.array([(0, 0)])
    desvar = {'x': ([0], 1, 4), 'y': [0]}

    b = CircleBoundaryConstraint([2, 2], 2)
    tf = TopFarmProblem(
        desvar,
        DummyCost(optimal, 'xy'),
        constraints=[b])
    _, state, _ = tf.optimize()

    npt.assert_array_less((state['x'] - 2)**2 + (state['y'] - 2)**2, 9)
    npt.assert_array_less(.9999999, state['x'])


def test_TopFarmProblem_with_cirleboundary_plot():
    optimal = np.array([(0, 0)])
    desvar = dict(zip('xy', optimal.T))
    plot_comp = PlotComp()
    b = CircleBoundaryConstraint([1, 2], 3)
    tf = TopFarmProblem(
        desvar,
        DummyCost(optimal, 'xy'),
        constraints=[b],
        plot_comp=plot_comp)
    tf.evaluate()


def test_TopFarmProblem_with_cirleboundary_gradients():
    optimal = np.array([(0, 0)])
    desvar = dict(zip('xy', optimal.T + 1.5))
    plot_comp = NoPlot()
    b = CircleBoundaryConstraint([1, 2], 3)
    tf = TopFarmProblem(
        desvar,
        DummyCost(optimal, 'xy'),
        constraints=[b],
        plot_comp=plot_comp,
        driver=SimpleGADriver())
    tf.check_gradients(True)


def test_check_gradients():
    center = [0.1, 0]
    points = np.array([(-2, 0), (-1, 0), (-.5, 0), (.5, 0), (1, 0), (2, 0),
                       (0, -2), (0, -1), (0, -.5), (0, .5), (0, 1), (0, 2),
                       (-1, -.5), (-.5, -.25), (.5, .25), (1, .5),
                       (-1, .5), (-.5, .25), (.5, -.25), (1, -.5)
                       ])
    cb = CircleBoundaryComp(3, center, 1)
    d = cb.distances(*points.T)
    np.testing.assert_array_almost_equal(d, 1 - np.sqrt(((points - center)**2).sum(1)))
    dx, dy = [np.diagonal(d) for d in cb.gradients(*points.T)]
    eps = 1e-7
    d1 = cb.distances(points[:, 0] + eps, points[:, 1])
    np.testing.assert_array_almost_equal((d1 - d) / eps, dx)
    d2 = cb.distances(points[:, 0], points[:, 1] + eps)
    np.testing.assert_array_almost_equal((d2 - d) / eps, dy)


def test_move_inside():
    pbc = CircleBoundaryComp(1, (2, 1), 3)
    x0, y0 = [3, 3, 3, 12, 12, 12], [3, 5, 10, 8, 10, 12]
    state = pbc.satisfy({'x': x0, 'y': y0})
    x, y = state['x'], state['y']
    if 0:
        import matplotlib.pyplot as plt
        b = np.r_[pbc.xy_boundary, pbc.xy_boundary[:1]]
        plt.plot(b[:, 0], b[:, 1], 'k')
        for x0_, x_, y0_, y_ in zip(x0, x, y0, y):
            plt.plot([x0_, x_], [y0_, y_], '.-')
        plt.show()
    npt.assert_array_less((x - 2)**2 + (y - 1)**2, 3**2)
