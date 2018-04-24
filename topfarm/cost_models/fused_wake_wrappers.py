'''
Created on 20. apr. 2018

@author: mmpe
'''
from fusedwake.WindFarm import WindFarm
from fusedwake.gcl.interface import GCL
import numpy as np
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent


class FusedWakeGCLWakeModel(object):

    def __init__(self, yml):
        """Description

        Parameters
        ----------
        yml: str, optional
            A WindIO `yml` file containing the description of the farm
        """
        self.windFarm = WindFarm(yml=yml)
        self.gcl = GCL(WF=self.windFarm, version='fort_gcl')

    def __call__(self, turbine_positions, no_wake_wdir, no_wake_wsp, no_wake_ti):
        self.gcl.update_position(turbine_positions.T)

        WD, WS, TI = [np.mean(np.atleast_3d(v), 0) for v in [no_wake_wdir, no_wake_wsp, no_wake_ti]]
        self.gcl(WS=WS.flatten(), WD=WD.flatten(), TI=TI.flatten())
        p = self.gcl.p_wt
        p = p.reshape(WD.shape + (self.windFarm.nWT,))
        return p.sum(2)  # sum over all turbines



if __name__ == '__main__':
    from fusedwake import fusedwake
    import os

    hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
    wm = FusedWakeGCLWakeModel(hornsrev_yml)
    tb_pos = wm.windFarm.pos

    print(wm(tb_pos, no_wake_wdir=270, no_wake_wsp=8, no_wake_ti=0.1).shape)

    WS_cases = np.arange(4, 12)
    WD_cases = np.arange(0, 360, 10)
    WS_ms, WD_ms = np.meshgrid(WS_cases, WD_cases)
    p = wm(tb_pos, WD_ms, WS_ms, np.zeros_like(WS_ms) + .1)
    print(p.shape)
    print(p)
