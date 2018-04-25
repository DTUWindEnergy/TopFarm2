'''
Created on 16. apr. 2018

@author: mmpe
'''
from threading import Thread
import threading
import time
import unittest

import numpy as np
from topfarm.cost_models.fuga.pascal_dll import PascalDLL
from topfarm.cost_models.fuga.py_fuga import PyFuga
import os
from topfarm.cost_models.fuga import py_fuga


fuga_path = os.path.abspath(os.path.dirname(py_fuga.__file__)) + '/Colonel/'


def test_parallel(id):
    pyFuga = PyFuga(farm_name='Horns Rev 1',
                    turbine_model_path=fuga_path + 'LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
                    tb_x=[423974, 424033], tb_y=[6151447, 6150889],
                    mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
                    farms_dir=fuga_path + 'LUT/Farms/', wind_atlas_path='Horns Rev 1\hornsrev0.lib')
    print(pyFuga.stdout_filename, id)
    for i in range(1):
        print(threading.current_thread(), id)
        np.testing.assert_array_almost_equal(pyFuga.get_aep([0, 0], [0, 200]), [14.044704, 16.753474, 0.401041, 0.838316])
        time.sleep(1)


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()

    def get_fuga(self, tb_x=[423974, 424033], tb_y=[6151447, 6150889]):
        return PyFuga(farm_name='Horns Rev 1',
                      turbine_model_path=fuga_path + 'LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
                      tb_x=tb_x, tb_y=tb_y,
                      mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
                      farms_dir=fuga_path + 'LUT/Farms/', wind_atlas_path='Horns Rev 1/hornsrev_north_only.lib', climate_interpolation=False)

    def testCheckVersion(self):
        lib = PascalDLL(fuga_path + "FugaLib/FugaLib.%s" % ('so', 'dll')[os.name == 'nt'])
        self.assertRaisesRegex(Exception, "This version of FugaLib supports interface version ", lib.CheckInterfaceVersion, 1)
        # PyFuga(fuga_path + "FugaLib/FugaLib.dll", fuga_path + "LUT/Farms/", "Horns Rev 1", fuga_path + "LUT/",
        #                (0, 0, 70), 0.0001, 400, 0, 'Horns Rev 1\hornsrev0.lib')

    def testSetup(self):
        pyFuga = self.get_fuga()
        self.assertEqual(pyFuga.get_no_tubines(), 2)
        self.assertIn("Loading", pyFuga.log)

        # check that new setup resets number of turbines
        pyFuga = self.get_fuga()
        self.assertEqual(pyFuga.get_no_tubines(), 2)
        pyFuga.cleanup()

    def testAEP_one_tb(self):
        pyFuga = self.get_fuga([0], [0])
        np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0], [0]]).T), [7.44121, 7.44121, 0.424962, 1.])

    def testAEP(self):
        pyFuga = self.get_fuga()

        np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 200], [0, 0]]).T), [14.848055, 14.882419, 0.423981, 0.997691])
        np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 200], [0, 0]]).T), 0)
        np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 0], [0, 200]]).T), [12.110134, 14.882419, 0.3458, 0.813721])
        np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 0], [0, 200]]).T), [[-0.001792, 0.001792],
                                                                                                        [-0.008116, 0.008116],
                                                                                                        [0., 0.]])
        np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 200], [0, 200]]).T), [14.846827, 14.882419, 0.423946, 0.997608])
        np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 200], [0, 200]]).T), [[-5.165553e-06, 5.165553e-06],
                                                                                                          [1.599768e-06, -1.599768e-06],
                                                                                                          [0.000000e+00, 0.000000e+00]])
        pyFuga.cleanup()

#     def test_parallel(self):
#         from multiprocessing import Pool
#
#         with Pool(5) as p:
#             print(p.map(test_parallel, [1, 2]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testAEP']
    unittest.main()
