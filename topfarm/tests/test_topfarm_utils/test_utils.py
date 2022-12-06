from topfarm.utils import smart_start, SmoothMax, SmoothMin, SoftMax, StrictMax, StrictMin, LogSumExpMax, LogSumExpMin
import numpy as np
from topfarm.tests import npt
from topfarm import TopFarmProblem

from py_wake.examples.data import hornsrev1
from py_wake.deficit_models.noj import NOJ
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent, PyWakeAEP
from py_wake.examples.data.iea37._iea37 import IEA37Site
import matplotlib.pyplot as plt
from py_wake.site._site import UniformSite
import pytest


def egg_tray_map():
    x = np.arange(0, 20, 0.1)
    y = np.arange(0, 10, 0.1)
    YY, XX = np.meshgrid(y, x)
    val = np.sin(XX) + np.sin(YY)
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
        import matplotlib.pyplot as plt
        plt.plot(wt_x, wt_y, '2r')
        for c in tf.model.constraint_components:
            c.plot()
        plt.axis('equal')
        plt.show()
    npt.assert_almost_equal(aep_1wt * n_wt, tf['AEP'], tol)


def test_smart_start_aep_map_PyWakeAEP():
    site = IEA37Site(16)
    n_wt = 4

    x, y = site.initial_position[:n_wt].T
    wd_lst = np.arange(0, 360, 45)
    ws_lst = [10]
    turbines = hornsrev1.HornsrevV80()
    site = UniformSite([1], .75)
    site.default_ws = ws_lst
    site.default_wd = wd_lst

    aep = PyWakeAEP(wake_model=NOJ(site, turbines))
    aep_1wt = aep.calculate_AEP([0], [0]).sum()

    tf = TopFarmProblem(
        design_vars={'x': x, 'y': y},
        cost_comp=aep.get_TopFarm_cost_component(n_wt),
        driver=EasyScipyOptimizeDriver(),
        constraints=[SpacingConstraint(160), CircleBoundaryConstraint((0, 0), 500)]
    )
    x = np.arange(-500, 500, 10)
    y = np.arange(-500, 500, 10)
    XX, YY = np.meshgrid(x, y)

    tf.smart_start(XX, YY, aep.get_aep4smart_start(wd=wd_lst, ws=ws_lst), radius=40, seed=1)
    tf.evaluate()

    if 0:
        wt_x, wt_y = tf['x'], tf['y']
        for i, _ in enumerate(wt_x, 1):
            print(aep.calculate_AEP(wt_x[:i], wt_y[:i]).sum((1, 2)))
        X_j, Y_j, aep_map = aep.aep_map(x, y, 0, wt_x, wt_y, ws=ws_lst, wd=wd_lst)
        print(tf.evaluate())
        import matplotlib.pyplot as plt
        c = plt.contourf(X_j, Y_j, aep_map, 100)
        plt.colorbar(c)
        plt.plot(wt_x, wt_y, '2r')
        for c in tf.model.constraint_components:
            c.plot()
        plt.axis('equal')
        plt.show()
    npt.assert_almost_equal(aep_1wt * n_wt, tf['AEP'], 5)


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
