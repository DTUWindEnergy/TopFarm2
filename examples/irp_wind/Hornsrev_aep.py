import os
from fusedwake import fusedwake
import numpy as np
from topfarm.cost_models.fuga import py_fuga
from topfarm.cost_models.fuga.py_fuga import PyFuga
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource


def HornsrevAEP_FUSEDWake_GCL():
    # wind parameters from "Horns Rev 1\hornsrev2.lib
    wdir_freq = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348, 0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
    weibull_A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921, 9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
    weibull_k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703, 2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
    ti = np.zeros_like(wdir_freq) + .1
    wr = WindResource(wdir_freq, weibull_A, weibull_k, ti)
    hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
    wm = FusedWakeGCLWakeModel(hornsrev_yml)
    aep_calc = AEPCalculator(wr, wm)
    return aep_calc(wm.windFarm.pos)


def HornsrevAEP_Fuga():
    fuga_path = os.path.abspath(os.path.dirname(py_fuga.__file__)) + '/Colonel/'
    hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
    wm = FusedWakeGCLWakeModel(hornsrev_yml)
    tb_x, tb_y = wm.windFarm.pos
    pyFuga = PyFuga(farm_name='Horns Rev 1',
                    turbine_model_path=fuga_path + 'LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
                    tb_x=tb_x, tb_y=tb_y,
                    mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
                    farms_dir=fuga_path + 'LUT/Farms/', wind_atlas_path='Horns Rev 1\hornsrev2.lib')
    return pyFuga.get_aep(tb_x, tb_x)[0]


if __name__ == '__main__':
    print(HornsrevAEP_FUSEDWake_GCL())
    print(HornsrevAEP_Fuga())
