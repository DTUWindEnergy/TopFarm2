from topfarm import TopFarm
import numpy as np
import pytest
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot, TurbineTypePlotComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasyPyOptSparseIPOPT,\
    EasySimpleGADriver, EasyRandomSearchDriver
from topfarm.tests import uta
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition,\
    RandomizeTurbinePosition_Circle, RandomizeTurbinePosition_Square,\
    RandomizeTurbineTypeAndPosition, RandomizeTurbinePosition_Normal
from topfarm.drivers import random_search_driver
from topfarm._topfarm import TurbineTypeXYZOptimizationProblem
from topfarm.constraint_components.boundary_component import BoundaryComp


initial = np.array([[6, 0, 0], [6, -8, 0], [1, 1, 0]])  # initial turbine layouts
optimal = np.array([[2.5, -3, 1], [6, -7, 2], [4.5, -3, 3]])  # optimal turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3, 1], [7, -7, 2], [4, -3, 4]])  # desired turbine layouts


@pytest.fixture
def topfarm_generator_scalable():
    def _topfarm_obj(driver, xy_scale=[1, 1], cost_scale=1, cost_offset=0, spacing=2):
        from topfarm.cost_models.dummy import DummyCostPlotComp

        # plot_comp = DummyCostPlotComp(desired[:,:2] * xy_scale, plot_improvements_only=True)
        plot_comp = NoPlot()

        class DummyCostScaled(DummyCost):
            def cost(self, **kwargs):
                opt = self.optimal_state
                return np.sum([(kwargs[n] - opt[:, i])**2 for i, n in enumerate(self.input_keys)]) * cost_scale + cost_offset

            def grad(self, **kwargs):
                opt = self.optimal_state
                return [(2 * cost_scale * (kwargs[n] - opt[:, i])) for i, n in enumerate(self.input_keys)]

        return TopFarm(initial[:, :2] * xy_scale, DummyCostScaled(desired[:, :2] * xy_scale, ['turbineX', 'turbineY']),
                       spacing * xy_scale[0], plot_comp=plot_comp, boundary=boundary * xy_scale, driver=driver,
                       expected_cost=1.5 * cost_scale)
    return _topfarm_obj


@pytest.fixture
def topfarm_generator():
    def _topfarm_obj(driver, spacing=2, keys=['turbineX', 'turbineY']):
        # from topfarm.cost_models.dummy import DummyCostPlotComp
        # plot_comp = DummyCostPlotComp(desired[:,:len(keys)], plot_improvements_only=True)
        plot_comp = NoPlot()

        return TopFarm(initial[:, :len(keys)], DummyCost(desired[:, :len(keys)], keys),
                       spacing, plot_comp=plot_comp, boundary=boundary, driver=driver,
                       expected_cost=1.5)
    return _topfarm_obj


@pytest.mark.parametrize('driver,tol', [
    (EasyScipyOptimizeDriver(disp=False), 1e-4),
    (EasyScipyOptimizeDriver(tol=1e-3, disp=False), 1e-2),
    (EasyScipyOptimizeDriver(maxiter=14, disp=False), 1e-1),
    (EasyScipyOptimizeDriver(optimizer='COBYLA', tol=1e-3, disp=False), 1e-2),
    (EasySimpleGADriver(max_gen=10, pop_size=100, bits={'turbineX': [12] * 3, 'turbineY':[12] * 3}, random_state=1), 1e-1),
    (EasyPyOptSparseIPOPT(), 1e-4),
][:])
def test_optimizers(driver, tol, topfarm_generator_scalable):
    if driver.__class__.__name__ == "PyOptSparseMissingDriver":
        pytest.xfail("Driver missing")
    tf = topfarm_generator_scalable(driver)
    tf.evaluate()
    cost, _, recorder = tf.optimize()
    print(recorder.driver_cases.num_cases)
    tb_pos = tf.turbine_positions[:, :2]
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0] < 6 + tol  # check within border
    if isinstance(driver, EasySimpleGADriver):
        assert cost == recorder['cost'].min()
    else:
        np.testing.assert_array_almost_equal(tb_pos, optimal[:, :2], -int(np.log10(tol)))


@pytest.mark.parametrize('driver,tol,N', [
    (EasyScipyOptimizeDriver(disp=False), 1e-4, 28),
    (EasyScipyOptimizeDriver(optimizer='COBYLA', tol=1e-3, disp=False), 1e-2, 103),
    # (EasyPyOptSparseIPOPT(), 1e-4, 25),
][:])
@pytest.mark.parametrize('cost_scale,cost_offset', [(1, 0),
                                                    (1000, 0),
                                                    (1, 200),
                                                    (0.001, 200),
                                                    (0.001, -200),
                                                    (1000, 200),
                                                    (1000, -200)
                                                    ])
def test_optimizers_scaling(driver, tol, N, cost_scale, cost_offset, topfarm_generator_scalable):
    if driver.__class__.__name__ == "PyOptSparseMissingDriver":
        pytest.xfail("Driver missing")

    tf = topfarm_generator_scalable(driver, cost_scale=cost_scale, cost_offset=cost_offset)
    _, _, recorder = tf.optimize()
    uta.assertLessEqual(recorder.driver_cases.num_cases, N)

    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()

    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0] < 6 + tol  # check within border
    np.testing.assert_array_almost_equal(tb_pos, optimal[:, :2], -int(np.log10(tol)))


def find_optimal_scaling(topfarm_generator_scalable):
    i = 0.000001
    res = []
    while i < 100000:
        driver = EasyScipyOptimizeDriver(disp=False, tol=1e-6 * i)
        tf = topfarm_generator_scalable(driver)
        tf.model.get_objectives()['cost_comp.cost']['scaler'] = i

        cost, _, recorder = tf.optimize()
        N = recorder.driver_cases.num_cases
        res.append((i, N, cost))
        print(i, N, cost)
        i *= 2

    import matplotlib.pyplot as plt
    plt.figure()
    res = np.array(res)
    plt.plot(res[:, 0], res[:, 1])
    plt.xscale('log')
    plt.show()


@pytest.mark.parametrize('randomize_func', [RandomizeTurbinePosition_Circle(1),
                                            RandomizeTurbinePosition_Square(1),
                                            RandomizeTurbinePosition_Normal(1)])
def test_random_search_driver_position(topfarm_generator, randomize_func):
    np.random.seed(1)
    driver = EasyRandomSearchDriver(randomize_func, max_iter=1000)
    tf = topfarm_generator(driver, spacing=1)
    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tol = 1e-1
    assert tb_pos[1][0] < 6 + tol  # check within border
    np.testing.assert_array_almost_equal(tb_pos, [[3, -3], [6, -7], [4, -3]], -int(np.log10(tol)))


def test_random_search_driver_type_and_position(topfarm_generator):
    np.random.seed(1)

    tf = TurbineTypeXYZOptimizationProblem(
        cost_comp=DummyCost(desired, ['turbineX', 'turbineY', 'turbineType']),
        turbineTypes=[0, 0, 0], lower=0, upper=3,
        turbineXYZ=np.array([[1, 1], [6, 0], [6, -8]]),
        boundary_comp=BoundaryComp(3, boundary),
        min_spacing=1,
        driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbineTypeAndPosition(1), max_iter=2000),
    )
    _, state, _ = tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tol = 1e-1
    assert tb_pos[1][0] < 6 + tol  # check within border

    np.testing.assert_array_almost_equal(tb_pos, [[3, -3], [6, -7], [4, -3]], -int(np.log10(tol)))
    np.testing.assert_array_equal(state['turbineType'], [1, 2, 3])
