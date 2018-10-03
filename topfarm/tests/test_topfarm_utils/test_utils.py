'''
Created on 17. jul. 2018

@author: mmpe
'''
from topfarm.utils import smart_start
import numpy as np
from topfarm.tests import npt


def tests_smart_start():
    xs_ref = [1.6, 14.1, 1.6, 7.9, 14.1, 7.9,
              19.9, 19.9, 5.8,
              7.8, 14.2, 1.5, 5.8,
              16.2, 16.2, 1.6, 3.7, 14.1, 3.7, 7.9]
    ys_ref = [1.6, 1.6, 7.9, 1.6, 7.9, 7.9, 1.6, 7.9, 7.8,
              5.8, 5.8, 5.8, 1.5,
              7.8, 1.5, 3.7, 1.6, 3.7, 7.9, 3.7]

    x = np.arange(0, 20, 0.1)
    y = np.arange(0, 10, 0.1)
    YY, XX = np.meshgrid(y, x)
    val = np.sin(XX) + np.sin(YY)
    N_WT = 20
    min_space = 2.1

    xs, ys = smart_start(XX, YY, val, N_WT, min_space)
    npt.assert_array_almost_equal([xs, ys], [xs_ref, ys_ref])

    if 0:
        import matplotlib.pyplot as plt
        plt.contourf(XX, YY, val, 100)
        for i in range(N_WT):
            circle = plt.Circle((xs[i], ys[i]), min_space / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(xs[i], ys[i], 'rx')
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
