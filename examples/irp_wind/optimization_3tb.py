import os
import numpy as np
from topfarm.cost_models.fuga import py_fuga
from topfarm.cost_models.fuga.py_fuga import PyFuga
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.plotting import PlotComp
from topfarm.topfarm import TopFarm


D = 80.0
D2 = 2 * D
initial_position = np.array([(0, D2), (0, 0), (0, -D2)])
boundary = [(-D2, D2), (D2, D2), (D2, -D2), (-D2, -D2)]
minSpacing = 2.0


def optimize_AEP_FusedWake_GCL():
    plot_comp = PlotComp()
    f = [1, 0, 0, 0]
    A = [9.176929, 9.782334, 9.531809, 9.909545]
    k = [2.392578, 2.447266, 2.412109, 2.591797]

    wr = WindResource(f, A, k, ti=np.zeros_like(f) + .1)

    wm = FusedWakeGCLWakeModel(os.path.dirname(__file__) + "/3tb.yml")
    aep_calc = AEPCalculator(wr, wm)
    aep_calc(np.array([[0, 0, 100], [160, 0, -160]]).T)
    init_pos = initial_position.copy()
    init_pos[:, 0] += [-50, 0, 50]
    print(aep_calc(np.array([[-160, 0, 160], [0, 0, 0]]).T))
    tf = TopFarm(init_pos, aep_calc.get_TopFarm_cost_component(), minSpacing * D, boundary=boundary, plot_comp=plot_comp)
    tf.optimize()
    plot_comp.show()


def optimize_AEP_Fuga():
    plot_comp = PlotComp()
    fuga_path = os.path.abspath(os.path.dirname(py_fuga.__file__)) + '/Colonel/'
    pyFuga = PyFuga(farm_name='Horns Rev 1',
                    turbine_model_path=fuga_path + 'LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
                    tb_x=initial_position[:, 0], tb_y=initial_position[:, 1],
                    mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
                    farms_dir=fuga_path + 'LUT/Farms/', wind_atlas_path='Horns Rev 1\hornsrev_north_only.lib')

    init_pos = initial_position.copy()
    init_pos[:, 0] += [-20, 0, 20]
    tf = TopFarm(init_pos, pyFuga.get_TopFarm_cost_component(), minSpacing * D, boundary=boundary, plot_comp=plot_comp)
    tf.optimize()
    plot_comp.show()


if __name__ == '__main__':
    optimize_AEP_Fuga()
    optimize_AEP_FusedWake_GCL()
