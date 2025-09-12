import os
import numpy as np
from topfarm.cost_models.fuga import py_fuga
from topfarm.cost_models.fuga.py_fuga import PyFuga
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.plotting import PlotComp
from topfarm.topfarm import TopFarm
from topfarm.cost_models.fuga.lib_reader import read_lib
import matplotlib.pyplot as plt

D = 80.0
D2 = 2 * D
initial_position = np.array([(0, D2), (0, 0), (0, -D2)])
boundary = [(-D2, D2), (D2, D2), (D2, -D2), (-D2, -D2)]
minSpacing = 2.0
fuga_path = os.path.abspath(os.path.dirname(py_fuga.__file__)) + '/Colonel/'


def optimize_AEP_FusedWake_GCL():
    plot_comp = PlotComp()
    f, A, k = read_lib(fuga_path + 'LUT/Farms/Horns Rev 1\hornsrev_north_only_pm45.lib')
    wr = WindResource(f, A, k, ti=np.zeros_like(f) + .1)
    wm = FusedWakeGCLWakeModel(os.path.dirname(__file__) + "/3tb.yml")
    aep_calc = AEPCalculator(wr, wm)
    init_pos = initial_position.copy()
    init_pos[:, 0] += [-20, 0, 20]
    tf = TopFarm(init_pos, aep_calc.get_TopFarm_cost_component(), minSpacing * D, boundary=boundary, plot_comp=plot_comp,
                             driver_options={'optimizer': 'SLSQP'})
    tf.evaluate()
    print(tf.get_cost())
    tf.optimize()
    print(tf.get_cost())
    save_plot('final_gcl.png', tf, False)
    plot_comp.show()


def optimize_AEP_Fuga():
    plot_comp = PlotComp()

    pyFuga = PyFuga(farm_name='Horns Rev 1',
                    turbine_model_path=fuga_path + 'LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
                    tb_x=initial_position[:, 0], tb_y=initial_position[:, 1],
                    mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
                    farms_dir=fuga_path + 'LUT/Farms/', wind_atlas_path='Horns Rev 1\hornsrev_north_only_pm45.lib', climate_interpolation=False)

    init_pos = initial_position.copy()
    init_pos[:, 0] += [-20, 0, 20]
    tf = TopFarm(init_pos, pyFuga.get_TopFarm_cost_component(), minSpacing * D, boundary=boundary, plot_comp=plot_comp,
                 driver_options={'optimizer': 'SLSQP'})
    print (pyFuga.get_aep_gradients())
    save_plot('initial.png', tf, True)
    tf.evaluate()
    print(tf.get_cost())
    tf.optimize()
    print(tf.get_cost())
    save_plot('final_fuga.png', tf, False)
    plot_comp.show()


def save_plot(filename, tf, initial=False):
    return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3, 3))
    plt.axis('equal')
    plt.xlabel('Distance [m]')
    plt.ylabel('Distance [m]')
    if initial:
        pos = tf.initial_positions
    else:
        pos = tf.turbine_positions
    plt.plot(tf.boundary[:, 0], tf.boundary[:, 1], label='Boundary')
    plt.plot(pos[:, 0], pos[:, 1], '2', ms=15, label='%s turbine positions' % ('Optimized', 'initial')[initial])
    # plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    optimize_AEP_Fuga()
    optimize_AEP_FusedWake_GCL()
