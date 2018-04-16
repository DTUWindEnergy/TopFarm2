'''
Created on 16. apr. 2018

@author: mmpe
'''
import unittest
from cost_models.fuga.py_fuga import PyFuga
import numpy as np


class Test(unittest.TestCase):

    def testAEP(self):
        path = r'C:\mmpe\programming\pascal\Fuga\Colonel/'
        pyFuga = PyFuga(path + "FugaLib/FugaLib.dll", path + "LUT/Farms/", "Horns Rev 1", path + "LUT/",
                        (0, 0, 70), 0.0001, 400, 0, 'Horns Rev 1\hornsrev0.lib')

        np.testing.assert_array_almost_equal(pyFuga.get_aep([0, 0], [0, 200]), [14.044704, 16.753474, 0.401041, 0.838316])
        np.testing.assert_array_almost_equal(pyFuga.get_aep([0, 200], [0, 0]), [16.714122, 16.753474, 0.477265, 0.997651])
        np.testing.assert_array_almost_equal(pyFuga.get_aep([0, 200], [0, 200]), [17.072517, 16.753474, 0.487499, 1.019043])
        np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients([0, 200], [0, 200]), [[0.002905, -0.002905],
                                                                                            [-0.001673, 0.001673],
                                                                                            [0., 0.]])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testAEP']
    unittest.main()
