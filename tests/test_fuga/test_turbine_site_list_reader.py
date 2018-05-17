'''
Created on 14. maj 2018

@author: mmpe
'''
import unittest

import numpy as np
from tests.test_files import tfp
from tests.test_fuga.test_pyfuga import fuga_path
from topfarm.cost_models.fuga.turbine_site_list_reader import read_turbine_site_list,\
    read_MR_turbine_site_list


class TestTurbineSiteListReader(unittest.TestCase):

    def testRead(self):
        model, ids, pos = read_turbine_site_list(tfp + "fuga_files/TurbineSiteList.txt")
        self.assertEqual(model, "Vestas_V80_(2_MW_offshore)[h=67.00]")
        self.assertEqual(ids[-1], "98")
        np.testing.assert_array_equal(pos[-1], [429431, 6147543, 67])

    def testMRRead(self):
        farm_name, nacelle_models, pos = read_MR_turbine_site_list(tfp + "fuga_files/4xV52SiteList.txt")
        self.assertEqual(farm_name, "4xV52Farm")
        name, (z, arm) = nacelle_models[0]
        self.assertEqual(name, "V52-0.85MWLower[h=50.00]")
        self.assertEqual(z, 50)
        self.assertEqual(arm, 30)
        np.testing.assert_array_equal(pos[0], [-8879, 2367])
        np.testing.assert_array_equal(pos[-1], [-3553., -2250.])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
