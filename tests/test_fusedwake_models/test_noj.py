import pytest

import numpy as np
from tests.test_files import tfp
from topfarm._topfarm import TopFarm
from topfarm.cost_models.fused_wake_wrappers import FusedWakeNOJWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource


@pytest.fixture()
def aep_calc():
    # f, A, k = read_lib(fuga_path + 'LUT/Farms/Horns Rev 1\hornsrev_north_only_pm45.lib')
    f = [1.0, 0.0, 0.0, 0.0]
    A = [9.176929, 9.782334, 9.531809, 9.909545]
    k = [2.392578, 2.447266, 2.412109, 2.591797]
    wr = WindResource(f, A, k, ti=np.zeros_like(f) + .1)
    wm = FusedWakeNOJWakeModel(tfp + "wind_farms/3tb.yml")
    return AEPCalculator(wr, wm)


def test_GCL(aep_calc):
    init_pos = aep_calc.wake_model.windFarm.pos.T
    assert aep_calc(init_pos) == 18.90684500124578
    assert aep_calc(np.array([[-500, 0, 500], [0, 0, 0]]).T) == 22.31788007605505


def test_GCL_Topfarm(aep_calc):
    init_pos = aep_calc.wake_model.windFarm.pos.T
    tf = TopFarm(init_pos, aep_calc.get_TopFarm_cost_component(), 160, init_pos, boundary_type='square')
    tf.evaluate()
    assert tf.get_cost() == -18.90684500124578
