import numpy as np
from numpy import newaxis as na
import time

from topfarm.cost_models.cost_model_wrappers import AEPMaxLoadCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot

from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.examples.data.iea34_130rwt import IEA34_130_1WT_Surrogate 
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.superposition_models import MaxSum
from py_wake.wind_turbines.power_ct_functions import SimpleYawModel

site = LillgrundSite()
x, y = site.initial_position.T
#keeping only every second turbine as lillegrund turbines are approx. half the size of the iea 3.4MW
x = x[::2]
y = y[::2]
# # Wind turbines and wind farm model definition
windTurbines = IEA34_130_1WT_Surrogate() #additional_models=[SimpleYawModel()]

wfm = IEA37SimpleBastankhahGaussian(site, windTurbines,deflectionModel=JimenezWakeDeflection(), turbulenceModel=STF2017TurbulenceModel(addedTurbulenceSuperpositionModel=MaxSum()))

load_signals = ['del_blade_flap', 'del_blade_edge', 'del_tower_bottom_fa',
                'del_tower_bottom_ss', 'del_tower_top_torsion']
wsp = np.asarray([10, 15])
wdir = np.asarray([90])

n_wt = x.size
i = n_wt
k = wsp.size
l = wdir.size

yaw_zero = np.zeros((i, l, k))
maxiter = 10
driver = EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=maxiter)
yaw_min, yaw_max =  - 40, 40
load_fact = 1.02
simulationResult = wfm(x,y,wd=wdir, ws=wsp, yaw=yaw_zero)

nom_loads = simulationResult.loads('OneWT')['LDEL'].values
max_loads = nom_loads * load_fact
step = 1e-2
s = nom_loads.shape[0]
P_ilk = np.broadcast_to(simulationResult.P.values[na], (i, l, k))
lifetime = float(60 * 60 * 24 * 365 * 20)
f1zh = 10.0 ** 7.0
lifetime_on_f1zh = lifetime / f1zh
indices = np.arange(i * l * k).reshape((i, l, k))

def aep_load_func(yaw_ilk):
    simres = wfm(x, y, wd=wdir, ws=wsp, yaw=yaw_ilk)
    aep = simres.aep().sum()
    loads = simres.loads('OneWT')['LDEL'].values
    return aep, loads

def aep_load_gradient(yaw_ilk):
    simres0 = wfm(x, y, wd=wdir, ws=wsp, yaw=yaw_ilk)
    aep0 = simres0.aep()
    DEL0 = simulationResult.loads('OneWT')['DEL'].values
    LDEL0 = simulationResult.loads('OneWT')['LDEL'].values
    d_aep_d_yaw = np.zeros(i*l*k)
    d_load_d_yaw = np.zeros((s * i, i * l * k))
    for n in range(n_wt):
        yaw_step = yaw_ilk.copy()
        yaw_step = yaw_step.reshape(i, l, k)
        yaw_step[n, :, :] += step
        simres_fd = wfm(x, y, wd=wdir, ws=wsp, yaw=yaw_step)
        aep_fd = simres_fd.aep()
        d_aep_d_yaw[n * l * k : (n + 1) * l * k] = (((aep_fd.values - aep0.values) / step).sum((0))).ravel()
        
        DEL_fd = simres_fd.loads('OneWT')['DEL'].values
        for _ls in range(s):
            m = simulationResult.loads('OneWT').m.values[_ls]
            for _wd in range(l):
                for _ws in range(k):
                    DEL_fd_fc = DEL0.copy()
                    DEL_fd_fc[:, :, _wd, _ws] = DEL_fd[:, :, _wd, _ws]
                    DEL_fd_fc = DEL_fd_fc[_ls, :, :, :]
                    f = DEL_fd_fc.mean()
                    LDEL_fd = (((P_ilk * (DEL_fd_fc/f) ** m).sum((1, 2)) * lifetime_on_f1zh) ** (1/m))*f
                    d_load_d_yaw[n_wt * _ls : n_wt * (_ls + 1), indices[n, _wd, _ws]] = (LDEL_fd - LDEL0[_ls]) / step

    return d_aep_d_yaw, d_load_d_yaw

cost_comp = AEPMaxLoadCostModelComponent(input_keys=[('yaw_ilk', np.zeros((i, l, k)))],
                                          n_wt = n_wt,
                                          aep_load_function = aep_load_func,
                                          aep_load_gradient = aep_load_gradient,
                                          max_loads = max_loads, 
                                          objective=True,
                                          output_keys=[('AEP', 0), ('loads', np.zeros((s, i)))]
                                          )
yaw_init = np.zeros((i, l, k))
yaw_30 = np.full_like(yaw_init, 30)
yaw_init_rand = np.random.rand(i, l, k)*80-40
tol = 1e-8
ec = 1e-4
problem = TopFarmProblem(design_vars={'yaw_ilk': (yaw_init, yaw_min, yaw_max)},
                          cost_comp=cost_comp,
                          driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=maxiter, tol=tol),
                          plot_comp=NoPlot(),
                          expected_cost=ec)
tic = time.time()
if 1:
    cost, state, recorder = problem.optimize()

toc = time.time()
print('Optimization took: {:.0f}s'.format(toc-tic))
if 0:
    with open(f'./check_partials_{int(toc)}.txt', 'w') as fid:
                partials = problem.check_partials(out_stream=fid,
                                                      compact_print=True,
                                                      show_only_incorrect=False,
                                                      step=step)
    


