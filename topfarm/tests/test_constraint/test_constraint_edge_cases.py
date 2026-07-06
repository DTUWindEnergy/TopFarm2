"""Edge case tests for spacing and boundary constraints."""
import numpy as np
import pytest
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver

from topfarm._topfarm import TopFarmProblem
from topfarm.cost_models.dummy import DummyCost
from topfarm.constraint_components.spacing import SpacingConstraint, SpacingComp
from topfarm.constraint_components.boundary import ConvexBoundaryComp


# ---------------------------------------------------------------------------
# SpacingConstraint edge cases
# ---------------------------------------------------------------------------

def test_spacing_single_turbine_produces_no_pairs():
    """SpacingComp with n_wt=1 produces an empty output — no pairs to compute."""
    sc = SpacingComp(n_wt=1, min_spacing=2)
    outputs = {}
    sc.compute({'x': np.array([0.0]), 'y': np.array([0.0])}, outputs)
    assert outputs['wtSeparationSquared'].shape == (0,)


def test_spacing_identical_positions_penalized():
    """Two turbines at the same position should receive a penalty (GA penalty mode)."""
    sc = SpacingConstraint(min_spacing=2)
    optimal = np.array([[0.0, 0.0], [3.0, 0.0]])
    tf = TopFarmProblem(
        design_vars={'x': (optimal[:, 0], -10.0, 10.0), 'y': (optimal[:, 1], -10.0, 10.0)},
        cost_comp=DummyCost(optimal, ['x', 'y']),
        constraints=[sc],
        driver=SimpleGADriver(),
    )
    cost = tf.evaluate({'x': [1.0, 1.0], 'y': [1.0, 1.0]})[0]
    assert cost >= 1e10


def test_spacing_min_spacing_exactly_met_no_penalty():
    """Two turbines separated by exactly min_spacing should not be penalized."""
    sc = SpacingConstraint(min_spacing=2)
    optimal = np.array([[0.0, 0.0], [2.0, 0.0]])
    tf = TopFarmProblem(
        design_vars={'x': (optimal[:, 0], -10.0, 10.0), 'y': (optimal[:, 1], -10.0, 10.0)},
        cost_comp=DummyCost(optimal, ['x', 'y']),
        constraints=[sc],
        driver=SimpleGADriver(),
    )
    cost = tf.evaluate({'x': [0.0, 2.0], 'y': [0.0, 0.0]})[0]
    assert cost < 1e10


# ---------------------------------------------------------------------------
# ConvexBoundaryComp distances — negative means outside
# ---------------------------------------------------------------------------

def test_convex_boundary_distances_inside():
    """Points well inside the boundary should have non-negative distances."""
    b = ConvexBoundaryComp(1, [(0, 0), (10, 0), (10, 10), (0, 10)], 'rectangle')
    distances = b.distances(np.array([5.0]), np.array([5.0]))
    assert np.all(distances >= 0)


def test_convex_boundary_distances_outside():
    """Points outside the boundary should have at least one negative distance."""
    b = ConvexBoundaryComp(1, [(0, 0), (10, 0), (10, 10), (0, 10)], 'rectangle')
    distances = b.distances(np.array([15.0]), np.array([5.0]))
    assert np.any(distances < 0)
