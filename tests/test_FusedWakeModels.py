"""Tests for TOPFARM
"""
import os
import unittest

import numpy as np
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource
from tests.test_files import testfilepath


class TestFusedWakeModels(unittest.TestCase):  # unittest version

    def test_GCL(self):
        # f, A, k = read_lib(fuga_path + 'LUT/Farms/Horns Rev 1\hornsrev_north_only_pm45.lib')
        f = [1.0, 0.0, 0.0, 0.0]
        A = [9.176929, 9.782334, 9.531809, 9.909545]
        k = [2.392578, 2.447266, 2.412109, 2.591797]

        wr = WindResource(f, A, k, ti=np.zeros_like(f) + .1)
        wm = FusedWakeGCLWakeModel(testfilepath + "wind_farms/3tb.yml")
        aep_calc = AEPCalculator(wr, wm)
        init_pos = wm.windFarm.pos.T
        self.assertEqual(aep_calc(init_pos), 19.85973533524627)
        self.assertEqual(aep_calc(np.array([[-500, 0, 500], [0, 0, 0]]).T), 22.31788007605505)


if __name__ == "__main__":
    unittest.main()
