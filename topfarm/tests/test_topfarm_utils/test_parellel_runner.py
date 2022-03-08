from openmdao.drivers.doe_generators import UniformGenerator
import pytest
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt
from topfarm.parallel_runner import ParallelRunner
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm import TopFarmProblem


@pytest.fixture(scope="module")
def parallelRunner():
    return ParallelRunner()


def get_InitialXYZOptimizationProblem(driver):
    return TopFarmProblem(
        {'x': [0, 2], 'y': [0, 2], 'z': ([0, 2], 3, 4)},
        cost_comp=DummyCost([(1, 0, 4),
                             (0, 1, 3)], 'xyz'),
        constraints=[XYBoundaryConstraint([(10, 6), (11, 8)], 'rectangle')],
        driver=driver)


@pytest.fixture
def lst():
    return get_InitialXYZOptimizationProblem(driver=UniformGenerator(200)).get_DOE_list()


def seq_runner_example(lst):
    return [get_InitialXYZOptimizationProblem(lst).optimize(recorder_as_list=True)]


def test_parallel_run(lst, parallelRunner):
    # run sequential
    s_cost, s_state, s_recorder = seq_runner_example(lst)[0]

    # run in parallel
    (p_cost, p_state, p_recorder), results = parallelRunner(lst, seq_runner_example)
    npt.assert_equal(s_cost, p_cost)
