'''
Created on 16. apr. 2018

@author: mmpe
'''
import unittest
from cost_models.fuga.py_fuga import PyFuga
import numpy as np


class Test(unittest.TestCase):

    def testAEP(self):
        pyFuga = PyFuga()
        np.testing.assert_array_almost_equal(pyFuga.get_aep([0, 0], [0, 1000]), (19.580917067710224, 19.643449032894644, 0.5590697809801386))
        np.testing.assert_array_almost_equal(pyFuga.get_aep([0, 1000], [0, 0]), (19.5418066445245, 19.643449032894644, 0.5579531092916332))
        np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients([0, 0], [0, 100]), [[0.00259933, -0.00259933],
                                                                                          [-0.0087804,  0.0087804],
                                                                                          [0.,  0.]])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testAEP']
    unittest.main()
