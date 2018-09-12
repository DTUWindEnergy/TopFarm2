'''
Created on 17. jul. 2018

@author: mmpe
'''
from topfarm.utils import smart_start
import numpy as np

x = np.arange(0, 20, 0.1)
y = np.arange(0, 10, 0.1)
YY, XX = np.meshgrid(y, x)
val = np.sin(XX) + np.sin(YY)
N_WT = 20
min_space = 2.1
xs_ref = [1.6,
          14.100000000000001,
          1.6,
          7.9,
          14.100000000000001,
          7.9,
          19.900000000000002,
          19.900000000000002,
          5.800000000000001,
          7.800000000000001,
          14.200000000000001,
          1.5,
          5.800000000000001,
          16.2,
          16.2,
          1.6,
          3.7,
          14.100000000000001,
          3.7,
          7.9]
ys_ref = [1.6,
          1.6,
          7.9,
          1.6,
          7.9,
          7.9,
          1.6,
          7.9,
          7.800000000000001,
          5.800000000000001,
          5.800000000000001,
          5.800000000000001,
          1.5,
          7.800000000000001,
          1.5,
          3.7,
          1.6,
          3.7,
          7.9,
          3.7]


def tests_smart_start():
    xs, ys = smart_start(XX, YY, val, N_WT, min_space)
    return np.testing.assert_allclose(np.array([xs, ys]),
                                      np.array([xs_ref, ys_ref]))
