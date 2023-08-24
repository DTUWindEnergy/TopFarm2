from topfarm.constraint_components.capacity import CapacityConstraint
import numpy as np
import topfarm
from topfarm.tests.test_files import xy3tb
from topfarm._topfarm import TopFarmProblem
from topfarm.easy_drivers import EasySimpleGADriver


def test_capacity_as_penalty():
    tf = xy3tb.get_tf(design_vars={topfarm.type_key: ([0, 0, 0], 0, 2)},
                      constraints=[CapacityConstraint(5, rated_power_array=[100, 10000, 10])],
                      driver=EasySimpleGADriver(),
                      plot_comp=None)

    # check normal result that satisfies the penalty
    assert tf.evaluate()[0] == 141.0
    # check penalized result if capacity constraint is not satisfied
    assert tf.evaluate({'type': np.array([0, 1, 1])})[0] == 1e10 + 15.1


def test_capacity_tf():
    # 15 turbines, 5 different types, 50MW max installed capacity
    n_wt = 15
    rated_power_array_kW = np.linspace(1, 10, int(n_wt / 3)) * 1e3

    inputtypes = np.tile(np.array([range(int(n_wt / 3))]), 3).flatten()
    tf = TopFarmProblem({'type': inputtypes},
                        constraints=[CapacityConstraint(max_capacity=50, rated_power_array=rated_power_array_kW)],
                        driver=EasySimpleGADriver()
                        )

    tf.evaluate()
    # case above the maximum allowed installed capacity, yes penalty
    assert tf["totalcapacity"] == 82.5
    assert tf['constraint_violation_comp.constraint_violation'] == 32.5

    # set all turbines type 0, still 15 turbines and re-run the problem
    tf.evaluate({'type': inputtypes * 0})
    # case below the maximum allowed installed capacity, no penalty
    assert tf["totalcapacity"] == 15
    assert tf['constraint_violation_comp.constraint_violation'][0] == 0.0
