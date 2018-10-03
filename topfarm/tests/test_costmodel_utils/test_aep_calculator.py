
import numpy as np
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.tests.test_files import testfilepath
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
import warnings
from topfarm.tests import npt


def test_AEPCalculator():
    f = [1, 0, 0, 0]
    A = [9.176929, 10, 10, 10]
    k = [2.392578, 2, 2, 2]
    wr = WindResource(np.array(f), A, k, ti=np.zeros_like(f) + .1)
    wf_3tb = testfilepath + "wind_farms/3tb.yml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = FusedWakeGCLWakeModel(wf_3tb)
        aep_calc = AEPCalculator(wr, wm)
        npt.assert_almost_equal(aep_calc([-1600, 0, 1600], [0, 0, 0]), 22.3178800761)
