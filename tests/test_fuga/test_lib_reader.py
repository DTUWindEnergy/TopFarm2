'''
Created on 25. apr. 2018

@author: mmpe
'''
import os
import unittest
from topfarm.cost_models.fuga.lib_reader import read_lib
import numpy as np
from topfarm.cost_models.fuga import py_fuga


class Test(unittest.TestCase):

    def test_lib_reader(self):
        f, A, k = read_lib(os.path.dirname(py_fuga.__file__) + "/Colonel/LUT/Farms/Horns Rev 1/hornsrev2.lib")
        np.testing.assert_array_almost_equal(f, [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064349,
                                                 0.086432, 0.117705, 0.151576, 0.147379, 0.100121, 0.05166])
        np.testing.assert_array_almost_equal(A, [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
                                                 9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803])
        np.testing.assert_array_almost_equal(k, [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
                                                 2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_lib_reader']
    unittest.main()
