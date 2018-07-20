from topfarm.cost_models.dummy import DummyCost
from topfarm._topfarm import TopFarm
import numpy as np
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
import pytest
from openmdao.drivers.doe_generators import FullFactorialGenerator
from openmdao.drivers.doe_driver import DOEDriver


xy, z = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)], (70, 90)


def get_cost(xyz):
    def cost(tb):
        return np.sum([(tb[:, i] - xyz)**2 for i, xyz in enumerate(xyz) if xyz is not None])
    return CostModelComponent(1, cost)


def get_cost_grad(xyz):
    def cost(tb):
        return np.sum([(tb[:, i] - xyz_)**2 for i, xyz_ in enumerate(xyz) if xyz_ is not None])

    def grad(tb):
        res = [None, None, None]
        for i, xyz_ in enumerate(xyz):
            if xyz_ is not None:
                res[i] = 2 * tb[:, i] - 2 * xyz_
        return res
    return CostModelComponent(1, cost, grad)


@pytest.mark.parametrize("optimal,boundary,res",
                         [((1, .5), xy, [[1, .5, 0, 0]]),  # opt xy inside boundary
                          ((3, .5), xy, [[2, .5, 0, 0]]),  # opt xy outside boundary
                          ((None, None, 85), [None, z], [[3, 3, 85, 0]]),  # opt z inside boundary
                          ((None, None, 95), [None, z], [[3, 3, 90, 0]]),  # opt z outside boundary
                          ((1, .5, 85), [xy, z], [[1, .5, 85, 0]]),  # opt xy and z inside boundary
                          ((3, .5, 95), [xy, z], [[2, .5, 90, 0]]),  # opt xy and z outside boundary
                          ((3, .5, 95), None, [[3, .5, 95, 0]]),  # opt xy and z outside boundary
                          ][-1:])
@pytest.mark.parametrize("get_cost", [get_cost,  # gradients by finite difference
                                      get_cost_grad  # analytical gradients
                                      ])
def test_optimize_xy(get_cost, optimal, boundary, res):
    # check that optimization works for xy and/or z with and w/o analytical gradients
    tf = TopFarm([(3, 3)], get_cost(optimal), 2, boundary=boundary, driver=EasyScipyOptimizeDriver(tol=1e-7, disp=False), record_id=None)
    state = tf.optimize()[1]
    np.testing.assert_array_almost_equal(state, res, 4)


def test_optimize_type():
    turbines = np.zeros((3, 4))
    cost = CostModelComponent(3, lambda tb: np.sum((tb[:, 3] - [0, 1, 2])**2))
    tf = TopFarm(turbines, cost, min_spacing=None, boundary=None, driver=EasyScipyOptimizeDriver(tol=1e-7, disp=False), record_id=None)
    lst = []
    for i1 in range(3):
        for i2 in range(3):
            for i3 in range(3):
                tb = np.zeros((3, 4))
                tb[:, 3] = [i1, i2, i3]
                lst.append(tb)
    best, _ = tf.multistart(lst)
    np.testing.assert_array_equal(best[:, 3], [0, 1, 2])


def test_generators():
    turbines = np.zeros((3, 4))
    cost = CostModelComponent(3, lambda tb: np.sum((tb[:, 3] - [0, 1, 2])**2))

    tf = TopFarm(turbines, cost, min_spacing=None, boundary=None, driver=EasyScipyOptimizeDriver(tol=1e-7, disp=False), record_id=None,
                 turbine_type_options = (0,2,FullFactorialGenerator(5)))
    print (tf.turbineTypeProblem.get_DOE_list().shape)
    print (tf.turbineTypeProblem.optimize())