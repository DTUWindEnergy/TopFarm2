import matplotlib.pyplot as plt
import numpy as np
import pytest

from topfarm.utils import smart_start, SmoothMax, SmoothMin, SoftMax, StrictMax, StrictMin, LogSumExpMax, LogSumExpMin
from topfarm.tests import npt
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint, SpacingTypeConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent

from py_wake.examples.data import hornsrev1
from py_wake.deficit_models.noj import NOJ
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.site._site import UniformSite
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt


def egg_tray_map():
    x = np.arange(0, 20, 0.1)
    y = np.arange(0, 10, 0.1)
    YY, XX = np.meshgrid(y, x)
    val = np.sin(XX) + np.sin(YY)
    return XX, YY, val


def types_map():
    x = np.arange(0, 40, 0.1)
    y = np.arange(0, 20, 0.1)
    YY, XX = np.meshgrid(y, x)
    val = np.ones((4, ) + XX.shape) * (np.sin(XX) + np.sin(YY))
    val[0, :] *= 16
    val[1, :] *= 9
    val[2, :] *= 4
    return XX, YY, val


def tests_smart_start():
    xs_ref = [1.6, 14.1, 1.6, 7.9, 14.1, 7.9, 19.9, 19.9, 7.8, 5.8, 14.2,
              5.8, 1.5, 16.2, 16.2, 1.6, 3.7, 14.1, 7.9, 3.7]
    ys_ref = [1.6, 1.6, 7.9, 1.6, 7.9, 7.9, 1.6, 7.9, 5.8, 7.8, 5.8, 1.5, 5.8, 7.8, 1.5, 3.7, 1.6, 3.7, 3.7, 7.9]
    N_WT = 20
    min_space = 2.1

    XX, YY, val = egg_tray_map()
    xs, ys = smart_start(XX, YY, val, N_WT, min_space, seed=0)

    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, val, 100)
        for i in range(N_WT):
            circle = plt.Circle((xs[i], ys[i]), min_space / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(xs[i], ys[i], 'rx')
        print(np.round(xs, 1).tolist())
        print(np.round(ys, 1).tolist())

        plt.axis('equal')
        plt.show()
    npt.assert_array_almost_equal([xs, ys], [xs_ref, ys_ref])


def tests_smart_start_types():
    xs_ref = [26.7, 26.7, 1.6, 39.3, 1.6, 14.1, 39.3, 14.1, 33.0, 20.4, 7.9, 33.0, 20.4, 7.9, 20.4, 20.4, 20.4, 20.4, 33.0, 33.0]
    ys_ref = [1.6, 14.1, 1.6, 1.6, 14.1, 1.6, 14.1, 14.1, 7.9, 7.9, 7.9, 19.9, 19.9, 19.9, 15.7, 0.0, 3.2, 12.5, 15.8, 3.3]
    N_WT = 20
    min_space = np.array([8, 6, 4, 2]) * 1.3
    XX, YY, val = types_map()
    types = [0, 1, 2, 3]
    xs, ys, type_i = smart_start(XX, YY, val, N_WT, min_space, seed=0, types=types)
    npt.assert_array_almost_equal([xs, ys], [xs_ref, ys_ref])
    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, val[0], 100)
        for i in range(N_WT):
            circle = plt.Circle((xs[i], ys[i]), min_space[type_i[i]] / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(xs[i], ys[i], 'rx')
        plt.axis('equal')


def tests_smart_start_random():
    xs_ref = [14.2, 1.6, 8.1, 7.7, 14.1, 1.1, 19.8, 19.4, 3.2, 14.4,
              6.2, 16.1, 12.1, 1.1, 9.9, 16.3, 7.9, 7.9, 3.3, 12.0]
    ys_ref = [1.6, 8.2, 1.8, 7.8, 7.5, 1.6, 7.9, 2.0, 1.0, 9.6, 0.7, 8.2, 1.7, 6.1, 8.1, 1.7, 9.9, 5.7, 6.9, 7.7]

    N_WT = 20
    min_space = 2.1
    XX, YY, val = egg_tray_map()
    np.random.seed(0)

    with pytest.raises(expected_exception=AssertionError):
        xs, ys = smart_start(XX, YY, val, N_WT, min_space, random_pct=101, seed=0)
    xs, ys = smart_start(XX, YY, val, N_WT, min_space, random_pct=1, seed=0, plot=False)

    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, val, 100)
        plt.plot(XX, YY, ',k')
        for i in range(N_WT):
            circle = plt.Circle((xs[i], ys[i]), min_space / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(xs[i], ys[i], 'rx')
        print(np.round(xs, 1).tolist())
        print(np.round(ys, 1).tolist())

        plt.axis('equal')
        plt.show()
    npt.assert_array_almost_equal([xs, ys], [xs_ref, ys_ref])


def tests_smart_start_no_feasible():
    XX, YY, val = egg_tray_map()
    N_WT = 20
    min_space = 5.1

    with pytest.raises(Exception, match="No feasible positions for wt 8"):
        xs, ys = smart_start(XX, YY, val, N_WT, min_space)


@pytest.mark.parametrize('seed,radius,resolution,tol', [(1, 500, 10, 5),
                                                        (0, 2000, 100, 3)])
def test_smart_start_aep_map(seed, radius, resolution, tol):
    site = IEA37Site(16)
    n_wt = 4
    x, y = site.initial_position[:n_wt].T
    wd_lst = np.arange(0, 360, 45)
    ws_lst = [10]
    turbines = hornsrev1.HornsrevV80()
    site = UniformSite([1], .75)
    site.default_ws = ws_lst
    site.default_wd = wd_lst
    wfm = NOJ(site, turbines)
    aep_comp = PyWakeAEPCostModelComponent(wfm, n_wt=n_wt)
    aep_1wt = wfm([0], [0]).aep().sum()

    tf = TopFarmProblem(
        design_vars={'x': x, 'y': y},
        cost_comp=aep_comp,
        driver=EasyScipyOptimizeDriver(),
        constraints=[SpacingConstraint(160), CircleBoundaryConstraint((0, 0), radius)]
    )
    x = np.arange(-radius, radius, resolution)
    y = np.arange(-radius, radius, resolution)
    XX, YY = np.meshgrid(x, y)

    tf.smart_start(XX, YY, aep_comp.get_aep4smart_start(wd=wd_lst, ws=ws_lst), radius=40, plot=0, seed=seed)
    tf.evaluate()

    if 0:
        wt_x, wt_y = tf['x'], tf['y']
        for i, _ in enumerate(wt_x, 1):
            print(wfm(wt_x[:i], wt_y[:i]).aep().sum(['wd', 'ws']))
        aep_comp.windFarmModel(wt_x, wt_y, ws=ws_lst, wd=wd_lst).flow_map().aep_xy().plot()
        print(tf.evaluate())
        # import matplotlib.pyplot as plt
        plt.plot(wt_x, wt_y, '2r')
        for c in tf.model.constraint_components:
            c.plot()
        plt.axis('equal')
        plt.show()
    npt.assert_almost_equal(aep_1wt * n_wt, tf['AEP'], tol)


def test_smart_start_aep_map_types(seed=1, radius=750, resolution=10):
    x_ref = [0.0, -240.0, 700.0, -730.0, -550.0, 390.0, -600.0, 570.0, -130.0, 220.0, -180.0, 20.0, -510.0, 230.0, 670.0, 160.0]
    y_ref = [-690.0, 700.0, -70.0, -100.0, -490.0, 470.0, 450.0, -480.0, 120.0, -80.0, -300.0, 450.0, 140.0, -460.0, 300.0, 730.0]
    ts_ref = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0]
    aep_ref = 96.86027158
    site = IEA37Site(16)
    n_wt = 16
    x, y = site.initial_position.T
    wd_lst = np.arange(0, 360, 20)
    ws_lst = [10]
    types = [0, 1, 2]
    init_types = n_wt * [0]
    turbines = WindTurbines(names=['T1', 'T2', 'T3'],
                            diameters=[60, 80, 100],
                            hub_heights=[60, 80, 100],
                            powerCtFunctions=[CubePowerSimpleCt(power_rated=200 * 60 ** 2, power_unit='W'),
                                              CubePowerSimpleCt(power_rated=200 * 80 ** 2, power_unit='W'),
                                              CubePowerSimpleCt(power_rated=200 * 100 ** 2, power_unit='W')],)
    min_space = 4 * np.array([60, 80, 100])
    site.default_ws = ws_lst
    site.default_wd = wd_lst
    wfm = NOJ(site, turbines)
    aep_comp = PyWakeAEPCostModelComponent(wfm, n_wt=n_wt, additional_input=[('type', init_types)], grad_method=None)

    tf = TopFarmProblem(
        design_vars={'x': x, 'y': y},
        cost_comp=aep_comp,
        driver=EasyScipyOptimizeDriver(),
        constraints=[SpacingTypeConstraint(min_space), CircleBoundaryConstraint((0, 0), radius)],
        ext_vars={'type': init_types}
    )
    xs = np.arange(-radius, radius, resolution)
    ys = np.arange(-radius, radius, resolution)
    XX, YY = np.meshgrid(xs, ys)

    xs, ys, ts = tf.smart_start(XX, YY, aep_comp.get_aep4smart_start(wd=wd_lst, ws=ws_lst), seed=seed, types=types)
    tf.evaluate()

    if 0:
        wt_x, wt_y = tf['x'], tf['y']
        plt.figure()
        aep_comp.windFarmModel(wt_x, wt_y, ws=ws_lst, wd=wd_lst, type=ts).flow_map().aep_xy(type=1).plot()
        plt.plot(wt_x, wt_y, '2r')
        for c in tf.model.constraint_components:
            c.plot()
        plt.axis('equal')
        plt.show()
    npt.assert_array_almost_equal([xs, ys, ts], [x_ref, y_ref, ts_ref])
    assert tf['AEP'] >= aep_ref


@pytest.mark.parametrize('max_func,scaling,ref',
                         [
                             (StrictMax(), 1, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
                             (StrictMin(), 1, [0, .1, .2, .3, .4, .5]),
                             (SmoothMax(.1), 1, [1.0, 0.9, 0.8, 0.69, 0.58, 0.5]),
                             (SmoothMax(.2), 1, [0.99, 0.89, 0.77, 0.65, 0.55, 0.5]),
                             (SmoothMax(100), 1000, [999.95, 899.73, 798.52, 692.81, 576.16, 500.0]),
                             (SmoothMax(50), 1000, [1000.0, 900.0, 800.0, 699.87, 596.4, 500.0]),
                             (SmoothMax(1), 1000, [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0]),
                             (SmoothMin(.1), 1, [0.0, 0.1, 0.2, 0.31, 0.42, 0.5]),
                             (SmoothMin(100), 1000, [0.05, 100.27, 201.48, 307.19, 423.84, 500.0]),
                             (SmoothMin(1), 1000, [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]),
                             (LogSumExpMax(.1), 1, [1.0, 0.9, 0.8, 0.7, 0.61, 0.57]),
                             (LogSumExpMax(.2), 1, [1.0, 0.9, 0.81, 0.73, 0.66, 0.64]),
                             (LogSumExpMax(200), 1000, [1001.34, 903.63, 809.72, 725.39, 662.65, 638.63]),
                             (LogSumExpMin(.1), 1, [-0.0, 0.1, 0.2, 0.3, 0.39, 0.43]),

                         ])
def test_max_funcs(max_func, scaling, ref):

    def abmax(x):
        a, b = x * scaling, (1 - x) * scaling
        return a, b, max_func([a, b], 0), max_func.gradient([a, b], 0)

    def dmax_da_fd(x):
        a, b = x * scaling, (1 - x) * scaling
        step = 1e-6
        return (max_func([a + step, b], 0) - max_func([a, b], 0)) / step

    if 0:
        x = np.arange(0, 1, .01)
        a, b, m, (dm_da, _) = abmax(x)
        plt.title(str(max_func))
        plt.plot(x, a, label='a')
        plt.plot(x, b, label='b')
        plt.plot(x, m, label='max')
        dmax_fd = dmax_da_fd(x)
        plt.plot(x, dmax_fd * scaling, label=f'fd*{scaling}')
        plt.plot(x, dm_da * scaling, '--', label=f'dmax da*{scaling}')

        x = np.arange(0, .6, .1)
        a, b, m, (dm_da, _) = abmax(x)
        plt.plot(x, m, '.')
        print(list(np.round(m, 2)))
        plt.legend()
        plt.show()

    x = np.arange(0, .6, .1)
    a, b, m, _ = abmax(x)
    npt.assert_array_almost_equal(ref, m, 2)

    x = np.arange(0, 1, .01)
    a, b, m, (dm_da, _) = abmax(x)
    if max_func.__class__.__name__.startswith('Strict'):
        mask = (x < .49) | (x > .51)
    else:
        mask = slice(None)
    npt.assert_array_almost_equal(dmax_da_fd(x)[mask], dm_da[mask], 4)
