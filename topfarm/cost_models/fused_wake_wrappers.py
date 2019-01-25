'''
Created on 20. apr. 2018

@author: mmpe
'''
import pytest


import numpy as np


class FusedWakeModel(object):

    def __init__(self, yml, version=None, **kwargs):
        """Description

        Parameters
        ----------
        yml: str, optional
            A WindIO `yml` file containing the description of the farm
        """
        from fusedwake.WindFarm import WindFarm
        self.windFarm = WindFarm(yml=yml)
        self.version = version or self.version
        self.wake_model = self.wake_model_cls(WF=self.windFarm, version=self.version, **kwargs)

    def __call__(self, turbine_positions, no_wake_wdir, no_wake_wsp, no_wake_ti):
        self.wake_model.update_position(turbine_positions)
        WD, WS, TI = (np.atleast_2d(v) for v in [no_wake_wdir, no_wake_wsp, no_wake_ti])
        assert WD.shape == WS.shape == TI.shape, "Shape of no_wake_wdir, no_wake_wsp and no_wake_ti must equal: %s != %s != %s" % (
            WD.shape, WS.shape, TI.shape)
        if len(WD.shape) == 3:
            WD, WS, TI = [np.mean(v, 0) for v in [WD, WS, TI]]
        self.run_wake_model(WS, WD, TI)
        p = self.wake_model.p_wt
        p = p.reshape(WD.shape + (self.windFarm.nWT,))
        return p.sum(2)  # sum over all turbines

    def run_wake_model(self, WS, WD, TI):
        self.wake_model(WS=WS.flatten(), WD=WD.flatten(), TI=TI.flatten())


class FusedWakeGCLWakeModel(FusedWakeModel):
    from fusedwake.gcl.interface import GCL
    wake_model_cls = GCL
    version = 'fort_gcl'


class FusedWakeNOJWakeModel(FusedWakeModel):
    from fusedwake.noj.interface import NOJ
    wake_model_cls = NOJ
    version = 'fort_noj'


def main():
    if __name__ == '__main__':
        try:
            import fusedwake
        except ModuleNotFoundError:
            return
        import os
        import matplotlib.pyplot as plt
        hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
        noj, _ = wake_models = [FusedWakeNOJWakeModel(hornsrev_yml, K=.1), FusedWakeGCLWakeModel(hornsrev_yml)]
        tb_pos = noj.windFarm.pos
        print(noj(tb_pos.T, no_wake_wdir=270, no_wake_wsp=8, no_wake_ti=0.1))

        for wm, c in zip(wake_models, 'rk'):
            WS_cases = np.arange(11, 12)
            WD_cases = np.arange(0, 360, 1)
            WD_ms, WS_ms = np.meshgrid(WD_cases, WS_cases)
            p = wm(tb_pos.T, WD_ms, WS_ms, np.zeros_like(WS_ms) + .1)
            plt.xlabel("Wdir")
            plt.ylabel("Power")
            plt.plot(p.T, color=c, label=wm.wake_model_cls.__name__)
            plt.legend()
            print(p.sum())
        plt.show()


main()
