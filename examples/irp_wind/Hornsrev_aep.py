import os
from fusedwake import fusedwake
import numpy as np
from topfarm.cost_models.fuga import py_fuga
from topfarm.cost_models.fuga.py_fuga import PyFuga
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.cost_models.fuga.lib_reader import read_lib

fuga_path = os.path.abspath(os.path.dirname(py_fuga.__file__)) + '/Colonel/'


def HornsrevAEP_FUSEDWake_GCL():
    # wind parameters from "Horns Rev 1\hornsrev2.lib
    wdir_freq, weibull_A, weibull_k = read_lib(fuga_path + 'LUT/Farms/Horns Rev 1\hornsrev2.lib')
    ti = np.zeros_like(wdir_freq) + .1
    wr = WindResource(wdir_freq, weibull_A, weibull_k, ti)
    hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
    wm = FusedWakeGCLWakeModel(hornsrev_yml)
    aep_calc = AEPCalculator(wr, wm)
    return aep_calc(wm.windFarm.pos.T)


def HornsrevAEP_Fuga():
    hornsrev_yml = os.path.dirname(fusedwake.__file__) + "/../examples/hornsrev.yml"
    wm = FusedWakeGCLWakeModel(hornsrev_yml)
    tb_x, tb_y = wm.windFarm.pos
    pyFuga = PyFuga(farm_name='Horns Rev 1',
                    turbine_model_path=fuga_path + 'LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
                    tb_x=tb_x, tb_y=tb_y,
                    mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
                    farms_dir=fuga_path + 'LUT/Farms/', wind_atlas_path='Horns Rev 1\hornsrev2.lib')
    return pyFuga.get_aep(wm.windFarm.pos.T)[0]


if __name__ == '__main__':
    print(HornsrevAEP_Fuga())
    print(HornsrevAEP_FUSEDWake_GCL())
