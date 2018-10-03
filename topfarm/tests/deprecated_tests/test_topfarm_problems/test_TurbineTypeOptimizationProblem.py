from topfarm import TurbineTypeOptimizationProblem
from openmdao.drivers.doe_generators import FullFactorialGenerator
from openmdao.drivers.doe_driver import DOEDriver
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt


optimal = [[1], [0]]


def test_turbineType_optimization():
    cost_comp = DummyCost(
        optimal_state=optimal,
        inputs=['type'])
    tf = TurbineTypeOptimizationProblem(
        cost_comp=cost_comp,
        turbineTypes=[0, 0], lower=0, upper=1,
        driver=DOEDriver(FullFactorialGenerator(2)))
    cost, state, _ = tf.optimize()
    assert cost == 0
    npt.assert_array_equal(state['type'], [1, 0])
