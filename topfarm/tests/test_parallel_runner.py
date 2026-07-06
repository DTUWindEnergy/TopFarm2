import numpy as np
import pytest
from topfarm.parallel_runner import ParallelRunner


# Top-level function required for pickle compatibility (Windows multiprocessing uses spawn)
def _simple_seq_runner(state_lst):
    """Return (cost, state, recorder) for each numeric state value."""
    return [(float(s), {'val': float(s)}, []) for s in state_lst]


@pytest.fixture
def pool_cleanup():
    """Collect pools and terminate them after each test."""
    pools = []
    yield pools
    for pool in pools:
        pool.terminate()
        pool.join()


def test_returns_best_with_minimum_cost(pool_cleanup):
    runner = ParallelRunner(processes=2)
    pool_cleanup.append(runner.pool)
    state_lst = [3.0, 1.0, 4.0, 2.0]
    best, _ = runner(state_lst, _simple_seq_runner)
    assert best[0] == pytest.approx(1.0)


def test_returns_all_results(pool_cleanup):
    runner = ParallelRunner(processes=2)
    pool_cleanup.append(runner.pool)
    state_lst = [3.0, 1.0, 4.0, 2.0]
    _, results = runner(state_lst, _simple_seq_runner)
    assert len(results) == len(state_lst)


def test_best_matches_argmin_over_all_results(pool_cleanup):
    runner = ParallelRunner(processes=2)
    pool_cleanup.append(runner.pool)
    state_lst = [5.0, 2.0, 8.0, 1.0, 3.0, 7.0]
    best, results = runner(state_lst, _simple_seq_runner)
    costs = [r[0] for r in results]
    assert best[0] == pytest.approx(min(costs))


def test_single_process(pool_cleanup):
    runner = ParallelRunner(processes=1)
    pool_cleanup.append(runner.pool)
    state_lst = [2.0, 0.5, 1.0]
    best, results = runner(state_lst, _simple_seq_runner)
    assert best[0] == pytest.approx(0.5)
    assert len(results) == len(state_lst)


def test_result_tuple_structure(pool_cleanup):
    runner = ParallelRunner(processes=2)
    pool_cleanup.append(runner.pool)
    state_lst = [1.0, 2.0]
    best, results = runner(state_lst, _simple_seq_runner)
    cost, state, recorder = best
    assert isinstance(cost, float)
    assert isinstance(state, dict)
    assert isinstance(recorder, list)


def test_results_contain_all_states(pool_cleanup):
    runner = ParallelRunner(processes=2)
    pool_cleanup.append(runner.pool)
    state_lst = [10.0, 20.0, 30.0, 40.0]
    _, results = runner(state_lst, _simple_seq_runner)
    returned_costs = sorted(r[0] for r in results)
    assert returned_costs == pytest.approx(sorted(state_lst))
