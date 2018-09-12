from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import UniformGenerator
import pytest
from topfarm._topfarm import InitialXYZOptimizationProblem
import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt
from topfarm.constraint_components.boundary_component import BoundaryComp
from topfarm.parallel_runner import ParallelRunner


@pytest.fixture("module")
def parallelRunner():
    return ParallelRunner()


def get_InitialXYZOptimizationProblem(driver):
    return InitialXYZOptimizationProblem(
        cost_comp=DummyCost([(1, 0, 4),
                             (0, 1, 3)]),
        min_spacing=None,
        turbineXYZ=[[0, 0, 0],
                    [2, 2, 2]],
        boundary_comp=BoundaryComp(n_wt=2,
                                   xy_boundary=[(10, 6), (11, 8)],
                                   xy_boundary_type='rectangle',
                                   z_boundary=[3, 4]),
        driver=driver)


@pytest.fixture
def lst():
    return get_InitialXYZOptimizationProblem(driver=UniformGenerator(200)).get_DOE_list()


def seq_runner_example(lst):
    return get_InitialXYZOptimizationProblem(lst).optimize()


def test_parallel_run(lst, parallelRunner):
    # run sequential
    s_cost, s_state, s_recorder = seq_runner_example(lst)

    # run in parallel
    (p_cost, p_state, p_recorder), results = parallelRunner(lst, seq_runner_example)
    npt.assert_equal(s_cost, p_cost)
