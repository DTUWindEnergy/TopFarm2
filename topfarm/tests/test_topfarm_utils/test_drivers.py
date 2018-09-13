from topfarm import TopFarm
import numpy as np
import pytest
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasyPyOptSparseIPOPT,\
    EasySimpleGADriver, EasyRandomSearchDriver
from topfarm.tests import uta
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_DirStep,\
    RandomizeTurbinePosition_Uniform


initial = np.array([[6, 0], [6, -8], [1, 1]])  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # desired turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3], [7, -7], [4, -3]])  # desired turbine layouts


@pytest.fixture
def topfarm_generator():
    def _topfarm_obj(driver, xy_scale=[1, 1], cost_scale=1, cost_offset=0, spacing=2):
        from topfarm.cost_models.dummy import DummyCostPlotComp

        plot_comp = DummyCostPlotComp(desired * xy_scale, plot_improvements_only=True)
        plot_comp = NoPlot()

        class DummyCostScaled(DummyCost):
            def cost(self, **kwargs):
                opt = self.optimal_state
                return np.sum([(kwargs[n] - opt[:, i])**2 for i, n in enumerate(self.input_keys)]) * cost_scale + cost_offset

            def grad(self, **kwargs):
                opt = self.optimal_state
                return [(2 * cost_scale * (kwargs[n] - opt[:, i])) for i, n in enumerate(self.input_keys)]

        return TopFarm(initial * xy_scale, DummyCostScaled(desired * xy_scale, ['turbineX', 'turbineY']),
                       spacing * xy_scale[0], plot_comp=plot_comp, boundary=boundary * xy_scale, driver=driver,
                       expected_cost=1.5 * cost_scale)
    return _topfarm_obj


@pytest.mark.parametrize('driver,tol', [
    (EasyScipyOptimizeDriver(disp=False), 1e-4),
    (EasyScipyOptimizeDriver(tol=1e-3, disp=False), 1e-2),
    (EasyScipyOptimizeDriver(maxiter=14, disp=False), 1e-1),
    (EasyScipyOptimizeDriver(optimizer='COBYLA', tol=1e-3, disp=False), 1e-2),
    (EasySimpleGADriver(max_gen=10, pop_size=100, bits={'turbineX': [12] * 3, 'turbineY':[12] * 3}, random_state=1), 1e-1),
    (EasyPyOptSparseIPOPT(), 1e-4),
][:])
def test_optimizers(driver, tol, topfarm_generator):
    if driver.__class__.__name__ == "PyOptSparseMissingDriver":
        pytest.xfail("Driver missing")
    tf = topfarm_generator(driver)
    tf.evaluate()
    cost, state, recorder = tf.optimize()
    print(recorder.driver_cases.num_cases)
    tb_pos = tf.turbine_positions[:, :2]
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0] < 6 + tol  # check within border
    if isinstance(driver, EasySimpleGADriver):
        assert cost == recorder['cost'].min()
    else:
        np.testing.assert_array_almost_equal(tb_pos, optimal, -int(np.log10(tol)))


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
def test_optimizers_scaling(driver, tol, N, cost_scale, cost_offset, topfarm_generator):
    if driver.__class__.__name__ == "PyOptSparseMissingDriver":
        pytest.xfail("Driver missing")

    tf = topfarm_generator(driver, cost_scale=cost_scale, cost_offset=cost_offset)
    cost, _, recorder = tf.optimize()
    uta.assertLessEqual(recorder.driver_cases.num_cases, N)

    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()

    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0] < 6 + tol  # check within border
    np.testing.assert_array_almost_equal(tb_pos, optimal, -int(np.log10(tol)))


def find_optimal_scaling(topfarm_generator):
    i = 0.000001
    res = []
    while i < 100000:
        driver = EasyScipyOptimizeDriver(disp=False, tol=1e-6 * i)
        tf = topfarm_generator(driver)
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


@pytest.mark.parametrize('randomize_func', [RandomizeTurbinePosition_DirStep(1),
                                            RandomizeTurbinePosition_Uniform()])
def test_random_search_driver(topfarm_generator, randomize_func):

    driver = EasyRandomSearchDriver(randomize_func, max_iter=2000)
    tf = topfarm_generator(driver, spacing=1)
    cost, _, recorder = tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tol = 1e-1
    assert tb_pos[1][0] < 6 + tol  # check within border
    if isinstance(driver, EasySimpleGADriver):
        assert cost == recorder['cost'].min()
    else:
        np.testing.assert_array_almost_equal(tb_pos, [[3, -3], [6, -7], [4, -3]], -int(np.log10(tol)))
