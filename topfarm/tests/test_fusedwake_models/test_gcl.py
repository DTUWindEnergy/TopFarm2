import numpy as np
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.tests.test_files import tfp

import pytest
import warnings
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm._topfarm import TopFarmProblem


@pytest.fixture()
def aep_calc():
    # f, A, k = read_lib(fuga_path + 'LUT/Farms/Horns Rev 1\hornsrev_north_only_pm45.lib')
    f = [1.0, 0.0, 0.0, 0.0]
    A = [9.176929, 9.782334, 9.531809, 9.909545]
    k = [2.392578, 2.447266, 2.412109, 2.591797]
    wr = WindResource(f, A, k, ti=np.zeros_like(f) + .1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = FusedWakeGCLWakeModel(tfp + "wind_farms/3tb.yml")
    return AEPCalculator(wr, wm)


def test_input_shape_must_be_equal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = FusedWakeGCLWakeModel(tfp + "wind_farms/3tb.yml")
        with pytest.raises(AssertionError, message="Shape of no_wake_wdir, no_wake_wsp and no_wake_ti must equal"):
            wm(wm.windFarm.pos.T, no_wake_wdir=[[270]], no_wake_wsp=[[8, 9]], no_wake_ti=0.1)


def test_GCL(aep_calc):
    init_pos = aep_calc.wake_model.windFarm.pos
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert aep_calc(init_pos[:, 0], init_pos[:, 1]) == 19.85973533524627  # tb aligned north-south -> wake
        assert aep_calc([-500, 0, 500], [0, 0, 0]) == 22.31788007605505  # tb aligned West-East -> no wake


def test_GCL_Topfarm(aep_calc):
    init_pos = aep_calc.wake_model.windFarm.pos
    with warnings.catch_warnings():  # suppress "warning, make sure that this position array is oriented in ndarray([n_wt, 2]) or ndarray([n_wt, 3])"
        warnings.simplefilter("ignore")
        tf = TopFarmProblem(
            dict(zip('xy', init_pos.T)),
            aep_calc.get_TopFarm_cost_component(),
            constraints=[SpacingConstraint(160),
                         XYBoundaryConstraint(init_pos, 'square')])
        tf.evaluate()
    assert tf.cost == -19.85973533524627
