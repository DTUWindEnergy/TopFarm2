
from openmdao.drivers.doe_generators import FullFactorialGenerator
from openmdao.drivers.doe_driver import DOEDriver
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt

import numpy as np
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.easy_drivers import EasyScipyOptimizeDriver


def test_turbineType_optimization():
    optimal = np.array([[1], [0]])
    tf = TopFarmProblem(
        design_vars={'type': (optimal[:, 0], 0, 1)},
        cost_comp=DummyCost(optimal_state=optimal, inputs=['type']),
        driver=DOEDriver(FullFactorialGenerator(2)))
    cost, state, _ = tf.optimize()
    assert cost == 0
    npt.assert_array_equal(state['type'], [1, 0])


def test_turbineXYZ_optimization():
    optimal = np.array([(5, 4, 3),
                        (3, 2, 1)])
    turbineXYZ = np.array([[0, 0, 0],
                           [2, 2, 2]])
    design_vars = {k: v for k, v in zip('xy', turbineXYZ.T)}
    design_vars['z'] = (turbineXYZ[:, 2], 1, 4)

    xy_boundary = [(0, 0), (5, 5)]
    tf = TopFarmProblem(
        design_vars=design_vars,
        cost_comp=DummyCost(optimal, 'xyz'),
        driver=EasyScipyOptimizeDriver(disp=False),
        constraints=[XYBoundaryConstraint(xy_boundary, 'square')])

    cost, state = tf.evaluate()
    assert cost == 52
    np.testing.assert_array_equal(state['x'], [0, 2])

    cost = tf.optimize()[0]
    assert cost < 1e6
    np.testing.assert_array_almost_equal(tf.turbine_positions, optimal[:, :2], 3)
