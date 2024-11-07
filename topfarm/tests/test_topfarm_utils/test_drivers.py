import pytest

import numpy as np

from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle, RandomizeTurbinePosition_Square, \
    RandomizeTurbineTypeAndPosition, RandomizeTurbinePosition_Normal, \
    RandomizeAllUniform, RandomizeAllRelativeMaxStep, RandomizeNUniform
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasyRandomSearchDriver, EasyPyOptSparseSNOPT, EasyPyOptSparseIPOPT
from topfarm.plotting import NoPlot
from topfarm.tests import uta, npt
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm._topfarm import TopFarmProblem
from topfarm.cost_models.cost_model_wrappers import CostModelComponent


initial = np.array([[6, 0, 0], [6, -8, 0], [1, 1, 0]])  # initial turbine layouts
optimal = np.array([[2.5, -3, 1], [6, -7, 2], [4.5, -3, 3]])  # optimal turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3, 1], [7, -7, 2], [4, -3, 4]])  # desired turbine layouts


@pytest.fixture
def topfarm_generator_scalable():
    def _topfarm_obj(driver, xy_scale=[1, 1], cost_scale=1, cost_offset=0, spacing=2):
        # from topfarm.cost_models.dummy import DummyCostPlotComp
        # plot_comp = DummyCostPlotComp(desired[:, :2] * xy_scale, plot_improvements_only=False)
        plot_comp = NoPlot()

        class DummyCostScaled(DummyCost):
            def cost(self, **kwargs):
                opt = self.optimal_state
                return np.sum([(kwargs[n] - opt[:, i])**2 for i, n in enumerate(self.input_keys)]) * \
                    cost_scale + cost_offset

            def grad(self, **kwargs):
                opt = self.optimal_state
                return [(2 * cost_scale * (kwargs[n] - opt[:, i])) for i, n in enumerate(self.input_keys)]

        return TopFarmProblem(
            dict(zip('xy', (initial[:, :2] * xy_scale).T)),
            DummyCostScaled(desired[:, :2] * xy_scale),
            constraints=[SpacingConstraint(spacing * xy_scale[0]),
                         XYBoundaryConstraint(boundary * xy_scale)],
            driver=driver,
            plot_comp=plot_comp,
            expected_cost=1.5 * cost_scale)
    return _topfarm_obj


@pytest.fixture
def topfarm_generator():
    def _topfarm_obj(driver, spacing=2, keys='xy'):
        # from topfarm.cost_models.dummy import DummyCostPlotComp
        # plot_comp = DummyCostPlotComp(desired[:,:len(keys)], plot_improvements_only=True)
        plot_comp = NoPlot()

        return TopFarmProblem(
            dict(zip(keys, initial.T[:len(keys)])),
            DummyCost(desired[:, :len(keys)], keys),
            constraints=[SpacingConstraint(spacing),
                         XYBoundaryConstraint(boundary)],
            plot_comp=plot_comp,
            driver=driver,
            expected_cost=1.5)
    return _topfarm_obj


@pytest.mark.parametrize(
    "driver,tol",
    [
        (EasyScipyOptimizeDriver(disp=False), 1e-4),
        (EasyScipyOptimizeDriver(tol=1e-3, disp=False), 1e-2),
        (EasyScipyOptimizeDriver(maxiter=14, disp=False), 1e-1),
        (EasyScipyOptimizeDriver(optimizer="COBYLA", tol=1e-3, disp=False), 1e-2),
        (EasyPyOptSparseIPOPT(), 1e-4),
        (EasyPyOptSparseSNOPT(), 1e-4),
    ][:],
)
def test_optimizers(driver, tol, topfarm_generator_scalable):
    if driver is None or driver.__class__.__name__ == "PyOptSparseMissingDriver":
        pytest.xfail("Driver missing")

    tf = topfarm_generator_scalable(driver)
    tf.evaluate()
    cost, _, recorder = tf.optimize({'x': [6., 6., 1.], 'y': [-.01, -8., 1.]})

    tb_pos = tf.turbine_positions[:, :2]

    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0] < 6 + tol  # check within border
    tf.plot_comp.show()
    np.testing.assert_array_almost_equal(tb_pos, optimal[:, :2], -int(np.log10(tol)))


@pytest.mark.parametrize('driver,tol,N', [
    (EasyScipyOptimizeDriver(disp=False), 1e-4, 30),
    (EasyPyOptSparseSNOPT(), 1e-4, 39),
    # COBYLA no longer works with scaling.
    # See issue on Github: https://github.com/OpenMDAO/OpenMDAO/issues/942
    # It can therefore requires 120 iterations instead of 104
    #    (EasyScipyOptimizeDriver(optimizer='COBYLA', tol=1e-3, disp=False), 1e-2, 104),
    (EasyScipyOptimizeDriver(optimizer='COBYLA', tol=1e-3, disp=False), 1e-2, 120),
    (EasyPyOptSparseIPOPT(), 1e-4, 25),
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
    uta.assertLessEqual(recorder.num_cases, N)

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
        N = recorder.num_cases
        res.append((i, N, cost))
        print(i, N, cost)
        i *= 2

    if 0:
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

    tf = TopFarmProblem(
        {'x': [1, 6, 6], 'y': [1, 0, -8], 'type': ([0, 0, 0], 0, 3)},
        cost_comp=DummyCost(desired, ['x', 'y', 'type']),
        constraints=[SpacingConstraint(1),
                     XYBoundaryConstraint(boundary)],
        driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbineTypeAndPosition(1), max_iter=2000),
    )
    _, state, _ = tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tol = 1e-1
    assert tb_pos[1][0] < 6 + tol  # check within border

    np.testing.assert_array_almost_equal(tb_pos, [[3, -3], [6, -7], [4, -3]], -int(np.log10(tol)))
    np.testing.assert_array_equal(state['type'], [1, 2, 3])


def test_random_search_driver_randomize_all_uniform():
    np.random.seed(1)

    class Cost():
        i = 0

        def __call__(self, *args, **kwargs):
            self.i += 1
            return self.i

    cost_comp = CostModelComponent(
        input_keys=['x', 'y', 'type'],
        n_wt=2,
        cost_function=Cost(),
        maximize=True)

    tf = TopFarmProblem(
        {'x': ([1, 6], [0, 1], [5, 6]), 'y': ([-1., 0], -6, 0), 'type': ([3, 3], 3, 8)},
        cost_comp=cost_comp,
        constraints=[],
        driver=EasyRandomSearchDriver(randomize_func=RandomizeAllUniform(['x', 'type']), max_iter=600, disp=False),
    )
    _, state, recorder = tf.optimize()

    # check that integer design variables are somewhat evenly distributed
    x, y, t = recorder['x'], recorder['y'], recorder['type']
    for arr, l, u in [(x[:, 0], 0, 5), (x[:, 1], 1, 6), (t[:, 0], 3, 8)]:
        count = [(arr == i).sum() for i in range(l, u + 1)]
        npt.assert_equal(601, sum(count))
        npt.assert_array_less(600 / len(count) * .70, count)

    count, _ = np.histogram(y[:, 0], np.arange(-6, 1))
    npt.assert_equal(y.shape[0], sum(count))
    npt.assert_array_less(600 / len(count) * .70, count)


def test_random_search_driver_RandomizeAllRelativeMaxStep(topfarm_generator):
    np.random.seed(1)

    tf = TopFarmProblem(
        {'x': [1, 6, 6], 'y': [1, 0, -8], 'type': ([0, 0, 0], 0, 3)},
        cost_comp=DummyCost(desired, ['x', 'y', 'type']),
        constraints=[SpacingConstraint(1),
                     XYBoundaryConstraint(boundary)],
        driver=EasyRandomSearchDriver(randomize_func=RandomizeAllRelativeMaxStep(.01), max_iter=2000),
    )
    _, state, _ = tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tol = 1e-1
    assert tb_pos[1][0] < 6 + tol  # check within border

    np.testing.assert_array_almost_equal(tb_pos, [[3, -3], [6, -7], [4, -3]], -int(np.log10(tol)))
    np.testing.assert_array_equal(np.round(state['type']), [1, 2, 3])


def test_random_search_NUniform():
    np.random.seed(1)

    tf = TopFarmProblem(
        {'type': ([0, 2, 3], 0, 3)},
        cost_comp=DummyCost(desired, ['x', 'y', 'type']),
        driver=EasyRandomSearchDriver(RandomizeNUniform(1, ['type']), max_iter=20),
        ext_vars={'x': [1, 6, 6], 'y': [1, 0, -8]}

    )
    cost, state, recorder = tf.optimize()
    np.testing.assert_array_equal(np.round(state['type']), [1, 2, 3])


def test_EasyScipyOptimizeDriver_used_only_with_right_opts():
    with pytest.raises(RuntimeError):
        EasyScipyOptimizeDriver(optimizer="IPOPT")
    with pytest.raises(RuntimeError):
        EasyScipyOptimizeDriver(optimizer="SGD")
    with pytest.raises(RuntimeError):
        EasyScipyOptimizeDriver(optimizer="NonExistent")
    EasyScipyOptimizeDriver(optimizer="COBYLA")
    EasyScipyOptimizeDriver(optimizer="SLSQP")
