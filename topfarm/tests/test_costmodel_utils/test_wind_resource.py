import numpy as np
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.tests import npt


def test_WindResource():
    f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348, 0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
    A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921, 9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
    k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703, 2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
    ti = np.zeros_like(f) + .1
    wr = WindResource(f, A, k, ti)

    wdir, ws, ti, weight = wr([0], [0], [4, 5])
    npt.assert_array_almost_equal(weight, np.array([[0.071381703], [0.088361194]]) * 0.035972 * (12 / 360))
