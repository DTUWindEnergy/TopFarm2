from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.tests import npt
from topfarm.tests.test_files import xy3tb
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
import numpy as np
import scipy


def test_TopFarmProblem():
    tf = xy3tb.get_tf(design_vars={'x': [3, 7, 4], 'y': [-3, -7, -3]},
                      constraints=[])

    cost, state, _ = tf.optimize()
    npt.assert_almost_equal(cost, 0)
    npt.assert_array_almost_equal(state['x'], xy3tb.desired[:, 0])
    npt.assert_array_almost_equal(state['y'], xy3tb.desired[:, 1])


def test_TopFarmProblemLimits():

    tf = xy3tb.get_tf(design_vars={'x': (xy3tb.initial[:, 0], -3, 3),
                                   'y': (xy3tb.initial[:, 1], [-4, -3, -2], [2, 3, 4])},
                      driver=EasyRandomSearchDriver(RandomizeTurbinePosition(1), max_iter=100),
                      constraints=[])
    tf.evaluate()
    desvars = tf.driver._designvars
    npt.assert_equal(desvars['indeps.x']['lower'], -3)
    npt.assert_equal(desvars['indeps.x']['upper'], 3)
    npt.assert_array_equal(desvars['indeps.y']['lower'], [-4, -3, -2])
    npt.assert_array_equal(desvars['indeps.y']['upper'], [2, 3, 4])


def test_TopFarmProblemSpacingConstraint():
    tf = xy3tb.get_tf(design_vars={'x': [3, 7, 4], 'y': [-3, -7, -3]},
                      constraints=[SpacingConstraint(2)])
    tf.evaluate({'x': xy3tb.desired[:, 0], 'y': xy3tb.desired[:, 1]})
    npt.assert_array_equal(tf['wtSeparationSquared'], [32, 1, 25])

    _, state, _ = tf.optimize()
    npt.assert_array_almost_equal(state['x'], [2.5, 7, 4.5])
    npt.assert_array_almost_equal(state['y'], xy3tb.optimal[:, 1])


def test_TopFarmProblemSpacingPenalty():
    tf = xy3tb.get_tf(design_vars={'x': [3, 7, 4], 'y': [-3, -7, -3]},
                      driver=EasyRandomSearchDriver(RandomizeTurbinePosition(1), 10),
                      constraints=[SpacingConstraint(2)])
    # spacing violated
    cost, _ = tf.evaluate({'x': xy3tb.desired[:, 0], 'y': xy3tb.desired[:, 1]})
    npt.assert_array_less(1e10, cost)

    # spacing satisfied
    cost, _ = tf.evaluate({'x': xy3tb.optimal[:, 0], 'y': xy3tb.optimal[:, 1]})
    npt.assert_equal(1.5, cost)


def test_TopFarmProblemXYBoundaryConstraint():
    tf = xy3tb.get_tf(design_vars={'x': [3, 7, 4], 'y': [-3, -7, -3]},
                      constraints=[XYBoundaryConstraint(xy3tb.boundary)])
    tf.evaluate({'x': xy3tb.desired[:, 0], 'y': xy3tb.desired[:, 1]})
    npt.assert_equal(tf['boundaryDistances'][1, 3], -1)

    _, state, _ = tf.optimize()
    npt.assert_array_almost_equal(state['x'], [3, 6, 4])
    npt.assert_array_almost_equal(state['y'], xy3tb.optimal[:, 1])

    desvars = tf.driver._designvars
    if tuple(map(int, scipy.__version__.split("."))) < (1, 5, 0):
        for xy in 'xy':
            for lu in ['lower', 'upper']:
                npt.assert_equal(desvars['indeps.' + xy][lu], np.nan)
    else:
        for i, xy in enumerate('xy'):
            if tf.driver._has_scaling:
                for lu, z in zip(['lower', 'upper'], (0, 1)):
                    npt.assert_equal(desvars['indeps.' + xy][lu], z)
            else:
                for lu, func in zip(['lower', 'upper'], (np.min, np.max)):
                    npt.assert_equal(desvars['indeps.' + xy][lu], func(xy3tb.boundary[:, i]))


def test_TopFarmProblemXYBoundaryConstraintPolygon():
    tf = xy3tb.get_tf(design_vars={'x': [3, 7, 4], 'y': [-3, -7, -3]},
                      constraints=[XYBoundaryConstraint(xy3tb.boundary, 'polygon')])
    # constraint violated
    tf.evaluate({'x': xy3tb.desired[:, 0], 'y': xy3tb.desired[:, 1]})
    npt.assert_equal(tf['boundaryDistances'][1], -1)

    _, state, _ = tf.optimize()
    npt.assert_array_almost_equal(state['x'], [3, 6, 4])
    npt.assert_array_almost_equal(state['y'], xy3tb.optimal[:, 1])


def test_TopFarmProblemXYBoundaryPenalty():
    tf = xy3tb.get_tf(design_vars={'x': [3, 7, 4], 'y': [-3, -7, -3]},
                      driver=EasyRandomSearchDriver(RandomizeTurbinePosition(1), 10),
                      constraints=[XYBoundaryConstraint(xy3tb.boundary)])
    # spacing violated
    cost, _ = tf.evaluate({'x': xy3tb.desired[:, 0], 'y': xy3tb.desired[:, 1]})
    npt.assert_array_less(1e10, cost)

    # spacing satisfied
    cost, _ = tf.evaluate({'x': xy3tb.optimal[:, 0], 'y': xy3tb.optimal[:, 1]})
    npt.assert_equal(1.5, cost)


def test_TopFarmProblemXYBoundaryPenaltyAndLimits():
    tf = xy3tb.get_tf(design_vars={'x': ([3, 7, 4], -1, 5), 'y': ([-3, -7, -3], -9, -1)},
                      driver=EasyRandomSearchDriver(RandomizeTurbinePosition(1), 10),
                      constraints=[XYBoundaryConstraint(xy3tb.boundary)])
    tf.evaluate({'x': xy3tb.desired[:, 0], 'y': xy3tb.desired[:, 1]})
    npt.assert_equal(tf['boundaryDistances'][1, 3], -1)

    desvars = tf.driver._designvars
    npt.assert_equal(desvars['indeps.x']['lower'], 0)
    npt.assert_equal(desvars['indeps.x']['upper'], 5)
    npt.assert_array_equal(desvars['indeps.y']['lower'], -9)
    npt.assert_array_equal(desvars['indeps.y']['upper'], -1)
