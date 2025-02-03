from openmdao.drivers.doe_generators import FullFactorialGenerator
from openmdao.drivers.doe_driver import DOEDriver
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt
import numpy as np
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import InclusionZone, ExclusionZone


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
    assert cost < 1e-6
    np.testing.assert_array_almost_equal(tf.turbine_positions, optimal[:, :2], 3)


def test_turbineXYZ_optimization_with_incl_excl():
    turbineXY = np.array([[0, 0]])
    optimal = np.array([[50, 50]])
    design_vars = {k: v for k, v in zip("xy", turbineXY.T)}

    dist2wt = 5
    zones = [
        InclusionZone(
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]]), dist2wt=lambda: dist2wt
        ),
        ExclusionZone(
            np.array([[25, 25], [75, 25], [75, 75], [25, 75]]), dist2wt=lambda: dist2wt
        ),
    ]

    tf = TopFarmProblem(
        design_vars=design_vars,
        cost_comp=DummyCost(optimal, "xy"),
        driver=EasyScipyOptimizeDriver(disp=False),
        constraints=[XYBoundaryConstraint(zones, "multi_polygon")],
    )
    state = tf.optimize()[1]

    # check that optimized state is outside the boundary of
    # exclusion plus the dist2wt expansion of the zone
    assert (
        np.all(state["x"] < (25 - dist2wt)) or
        np.all(state["x"] > (75 + dist2wt)) or
        np.all(state["y"] < (25 - dist2wt)) or
        np.all(state["y"] > (75 + dist2wt))
    )

    assert np.all(state["x"] >= 0)
    assert np.all(state["x"] <= 100)
    assert np.all(state["y"] >= 0)
    assert np.all(state["y"] <= 100)

    optimal = np.array([[120, 120]])
    tf = TopFarmProblem(
        design_vars=design_vars,
        cost_comp=DummyCost(optimal, "xy"),
        driver=EasyScipyOptimizeDriver(disp=False),
        constraints=[XYBoundaryConstraint(zones, "multi_polygon")],
    )
    state = tf.optimize()[1]

    # check that optimized state is within the boundary of
    # inclusion zone plus the dist2wt contraction;
    assert np.all(state["x"] >= (0 + dist2wt))
    assert np.all(state["x"] <= (100 - dist2wt))
    assert np.all(state["y"] >= (0 + dist2wt))
    assert np.all(state["y"] <= (100 - dist2wt))
