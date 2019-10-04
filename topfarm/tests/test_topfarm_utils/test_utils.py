'''
Created on 17. jul. 2018

@author: mmpe
'''
from topfarm.utils import smart_start
import numpy as np
from topfarm.tests import npt
from topfarm import TopFarmProblem

from py_wake.examples.data import hornsrev1
from py_wake.wake_models.noj import NOJ
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint,\
    CircleBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEP
from py_wake.examples.data.iea37._iea37 import IEA37Site
from scipy.interpolate.interpolate import RegularGridInterpolator
from py_wake.site._site import UniformSite


def tests_smart_start():
    xs_ref = [1.6, 14.1, 1.6, 7.9, 14.1, 7.9, 19.9, 19.9, 7.8,
              5.8, 14.2, 5.8, 1.5, 16.2, 16.2, 1.6, 3.7, 14.1, 7.9, 3.7]
    ys_ref = [1.6, 1.6, 7.9, 1.6, 7.9, 7.9, 1.6, 7.9, 5.8, 7.8, 5.8, 1.5, 5.8, 7.8, 1.5, 3.7, 1.6, 3.7, 3.7, 7.9]

    x = np.arange(0, 20, 0.1)
    y = np.arange(0, 10, 0.1)
    YY, XX = np.meshgrid(y, x)
    val = np.sin(XX) + np.sin(YY)
    N_WT = 20
    min_space = 2.1
    np.random.seed(0)
    xs, ys = smart_start(XX, YY, val, N_WT, min_space)
    npt.assert_array_almost_equal([xs, ys], [xs_ref, ys_ref])

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


def tests_smart_start_no_feasible():
    x = np.arange(0, 20, 0.1)
    y = np.arange(0, 10, 0.1)
    YY, XX = np.meshgrid(y, x)
    val = np.sin(XX) + np.sin(YY)
    N_WT = 20
    min_space = 5.1

    xs, ys = smart_start(XX, YY, val, N_WT, min_space)

    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, val, 100)
        for i in range(N_WT):
            circle = plt.Circle((xs[i], ys[i]), min_space / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(xs[i], ys[i], 'rx')
        plt.axis('equal')
        plt.show()
        print(xs)
    assert np.isnan(xs).sum() == 12


def test_smart_start_aep_map():
    site = IEA37Site(16)
    n_wt = 4
    np.random.seed(1)
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

    tf.smart_start(XX, YY, aep.get_aep4smart_start(wd=wd_lst, ws=ws_lst), radius=40)
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


def test_smart_start_aep_map_large_radius():
    site = IEA37Site(16)
    n_wt = 4
    np.random.seed(0)
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
        constraints=[SpacingConstraint(160), CircleBoundaryConstraint((0, 0), 2000)]
    )
    x = np.arange(-2000, 2000, 100)
    y = np.arange(-2000, 2000, 100)
    XX, YY = np.meshgrid(x, y)

    tf.smart_start(XX, YY, aep.get_aep4smart_start(wd=wd_lst, ws=ws_lst), radius=40)
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
    npt.assert_almost_equal(aep_1wt * n_wt, tf['AEP'], 3)
