'''
Created on 19/04/2018

@author: Mads
'''
import os

from fusedwake import fusedwake
import numpy as np
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.wind_resource import WindResource


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
        power_GW = self.wake_model(turbine_positions, no_wake_WD, no_wake_WS, no_wake_TI)/1e9
        return np.sum(power_GW * weight) * 24 * 365
    
    
if __name__ == '__main__':
    f = np.array("3.597152 3.948682 5.167395 7.000154 8.364547 6.43485 8.643194 11.77051 15.15757 14.73792 10.01205 5.165975".split(),dtype=np.float)
    A = np.array("9.176929  9.782334 9.531809 9.909545 10.04269 9.593921 9.584007 10.51499 11.39895 11.68746 11.63732 10.08803".split(),dtype=np.float)
    k = np.array("2.392578 2.447266 2.412109 2.591797 2.755859 2.595703 2.583984 2.548828 2.470703 2.607422 2.626953 2.326172".split(),dtype=np.float)
  
    
    wr = WindResource(f/100, A, k, ti=np.zeros_like(f) + .1)
    hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
    hornsrev_yml_2tb ="../../example_data/hornsrev_2tb.yml" 
    wm = FusedWakeGCLWakeModel(hornsrev_yml_2tb)
    aep_calc = AEPCalculator(wr,wm)
    print(aep_calc(wm.wf.pos))
