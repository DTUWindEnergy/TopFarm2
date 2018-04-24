'''
Created on 19/04/2018

@author: Mads
'''
import os

from fusedwake import fusedwake
import numpy as np
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent


class AEPCalculator(object):

    def __init__(self, wind_resource, wake_model, wdir=np.arange(360), wsp=np.arange(3, 25)):
        """
        wind_resource: f(turbine_positions, wdir, wsp) -> WS[nWT,nWdir,nWsp], TI[nWT,nWdir,nWsp), Weight[nWdir,nWsp]
        wake_model: f(turbine_positions, WS[nWT,nWdir,nWsp], TI[nWT,nWdir,nWsp) -> power[nWdir,nWsp]
        """
        self.wind_resource = wind_resource
        self.wake_model = wake_model
        self.wdir = wdir
        self.wsp = wsp

    def __call__(self, turbine_positions):
        no_wake_WD, no_wake_WS, no_wake_TI, weight = self.wind_resource(turbine_positions, self.wdir, self.wsp)
        power_GW = self.wake_model(turbine_positions, no_wake_WD, no_wake_WS, no_wake_TI) / 1e9
        return np.sum(power_GW * weight) * 24 * 365

    def get_TopFarm_cost_component(self):
        n_wt = self.wake_model.windFarm.nWT
        return AEPCostModelComponent(n_wt, lambda *args: self(*args))


if __name__ == '__main__':
    f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348, 0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
    A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921, 9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
    k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703, 2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
    wr = WindResource(f / 100, A, k, ti=np.zeros_like(f) + .1)
    hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
    hornsrev_yml_2tb = "../../example_data/hornsrev_2tb.yml"
    wm = FusedWakeGCLWakeModel(hornsrev_yml)
    aep_calc = AEPCalculator(wr, wm)

    print(aep_calc(wm.wf.pos))
