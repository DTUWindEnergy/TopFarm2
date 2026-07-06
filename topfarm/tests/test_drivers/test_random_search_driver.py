import numpy as np
import pytest
from topfarm._topfarm import TopFarmProblem
from topfarm.cost_models.dummy import DummyCost
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import (
    RandomizeAllUniform,
    RandomizeNUniform,
    RandomizeTurbinePosition,
    RandomizeTurbinePosition_Circle,
    RandomizeTurbinePosition_Square,
    RandomizeTurbinePosition_Normal,
)
import topfarm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem(initial, optimal, randomize_func, max_iter=20):
    design_vars = {topfarm.x_key: (initial[:, 0], -10.0, 10.0),
                   topfarm.y_key: (initial[:, 1], -10.0, 10.0)}
    return TopFarmProblem(
        design_vars=design_vars,
        cost_comp=DummyCost(optimal[:, :2], [topfarm.x_key, topfarm.y_key]),
        driver=EasyRandomSearchDriver(randomize_func=randomize_func, max_iter=max_iter, disp=False),
    )


# ---------------------------------------------------------------------------
# EasyRandomSearchDriver integration
# ---------------------------------------------------------------------------

def test_random_search_improves_cost():
    np.random.seed(0)
    initial = np.array([[0.0, 0.0], [5.0, 5.0]])
    optimal = np.array([[1.0, 1.0], [4.0, 4.0]])
    randomize = RandomizeAllUniform()
    tf = _make_problem(initial, optimal, randomize, max_iter=50)
    cost_initial, _ = tf.evaluate()
    cost_final, _, _ = tf.optimize()
    assert cost_final <= cost_initial


def test_random_search_returns_state():
    np.random.seed(1)
    initial = np.array([[0.0, 0.0]])
    optimal = np.array([[2.0, 3.0]])
    randomize = RandomizeAllUniform()
    tf = _make_problem(initial, optimal, randomize, max_iter=30)
    _, state, _ = tf.optimize()
    assert topfarm.x_key in state
    assert topfarm.y_key in state


def test_random_search_respects_bounds():
    np.random.seed(2)
    initial = np.array([[0.0, 0.0], [1.0, 1.0]])
    optimal = np.array([[0.5, 0.5], [1.5, 1.5]])
    randomize = RandomizeAllUniform()
    tf = _make_problem(initial, optimal, randomize, max_iter=30)
    _, state, _ = tf.optimize()
    assert np.all(state[topfarm.x_key] >= -10.0)
    assert np.all(state[topfarm.x_key] <= 10.0)
    assert np.all(state[topfarm.y_key] >= -10.0)
    assert np.all(state[topfarm.y_key] <= 10.0)


# ---------------------------------------------------------------------------
# RandomizeAllUniform
# ---------------------------------------------------------------------------

def test_randomize_all_uniform_stays_within_bounds():
    np.random.seed(3)
    randomize = RandomizeAllUniform()
    values = np.array([0.0, 0.0])
    lbound = np.array([-5.0, -5.0])
    ubound = np.array([5.0, 5.0])
    desvar_dict = {'x': (values.copy(), lbound, ubound)}
    for _ in range(20):
        desvar_dict = randomize(desvar_dict)
        assert np.all(desvar_dict['x'][0] >= lbound)
        assert np.all(desvar_dict['x'][0] <= ubound)


def test_randomize_all_uniform_integer_var():
    np.random.seed(4)
    randomize = RandomizeAllUniform(int_design_vars_lst=['type'])
    values = np.array([0, 0, 0])
    lbound = np.array([0, 0, 0])
    ubound = np.array([3, 3, 3])
    desvar_dict = {'type': (values.copy(), lbound, ubound)}
    for _ in range(20):
        desvar_dict = randomize(desvar_dict)
        assert np.all(desvar_dict['type'][0] >= 0)
        assert np.all(desvar_dict['type'][0] <= 3)
        assert desvar_dict['type'][0].dtype.kind == 'i' or np.all(
            desvar_dict['type'][0] == desvar_dict['type'][0].astype(int))


# ---------------------------------------------------------------------------
# RandomizeNUniform
# ---------------------------------------------------------------------------

def test_randomize_n_uniform_stays_within_bounds():
    np.random.seed(5)
    randomize = RandomizeNUniform(N=1)
    values = np.array([0.0, 0.0, 0.0])
    lbound = np.array([-2.0, -2.0, -2.0])
    ubound = np.array([2.0, 2.0, 2.0])
    desvar_dict = {'x': (values.copy(), lbound, ubound)}
    for _ in range(20):
        desvar_dict = randomize(desvar_dict)
        assert np.all(desvar_dict['x'][0] >= lbound)
        assert np.all(desvar_dict['x'][0] <= ubound)


def test_randomize_n_uniform_changes_only_n_elements():
    np.random.seed(6)
    N = 1
    randomize = RandomizeNUniform(N=N)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    lbound = np.full(4, 0.0)
    ubound = np.full(4, 10.0)
    original = values.copy()
    desvar_dict = {'x': (values, lbound, ubound)}
    desvar_dict = randomize(desvar_dict)
    changed = np.sum(desvar_dict['x'][0] != original)
    assert changed <= N


# ---------------------------------------------------------------------------
# RandomizeTurbinePosition variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cls', [
    RandomizeTurbinePosition_Circle,
    RandomizeTurbinePosition_Square,
    RandomizeTurbinePosition_Normal,
])
def test_randomize_turbine_position_stays_within_bounds(cls):
    np.random.seed(7)
    randomize = cls(max_step=1.0)
    x = np.array([0.0, 3.0])
    y = np.array([0.0, 3.0])
    lbound = np.array([-5.0, -5.0])
    ubound = np.array([5.0, 5.0])
    desvar_dict = {
        topfarm.x_key: (x.copy(), lbound.copy(), ubound.copy()),
        topfarm.y_key: (y.copy(), lbound.copy(), ubound.copy()),
    }
    for _ in range(20):
        desvar_dict = randomize(desvar_dict)
        assert np.all(desvar_dict[topfarm.x_key][0] >= lbound)
        assert np.all(desvar_dict[topfarm.x_key][0] <= ubound)
        assert np.all(desvar_dict[topfarm.y_key][0] >= lbound)
        assert np.all(desvar_dict[topfarm.y_key][0] <= ubound)
