import numpy as np
from numpy import newaxis as na
import time

from topfarm.cost_models.cost_model_wrappers import AEPMaxLoadCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint

from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.examples.data.iea34_130rwt import IEA34_130_1WT_Surrogate 
from py_wake.superposition_models import MaxSum

site = LillgrundSite()
x, y = site.initial_position.T
#keeping only every second turbine as lillegrund turbines are approx. half the size of the iea 3.4MW
x = x[::2]
y = y[::2]
x_init = x
y_init = y
# # Wind turbines and wind farm model definition
windTurbines = IEA34_130_1WT_Surrogate() 
wfm = IEA37SimpleBastankhahGaussian(site, windTurbines, turbulenceModel=STF2017TurbulenceModel(addedTurbulenceSuperpositionModel=MaxSum()))

load_signals = ['del_blade_flap', 'del_blade_edge', 'del_tower_bottom_fa',
                'del_tower_bottom_ss', 'del_tower_top_torsion']
wsp = np.asarray([10, 15])
wdir = np.arange(0,360,45)

n_wt = x.size
i = n_wt
k = wsp.size
l = wdir.size

yaw_zero = np.zeros((i, l, k))
maxiter = 10
yaw_min, yaw_max =  - 40, 40
load_fact = 1.02
simulationResult = wfm(x,y,wd=wdir, ws=wsp, yaw=yaw_zero)

nom_loads = simulationResult.loads('OneWT')['LDEL'].values
max_loads = nom_loads * load_fact
step = 1e-4
s = nom_loads.shape[0]
P_ilk = np.broadcast_to(simulationResult.P.values[na], (i, l, k))
lifetime = float(60 * 60 * 24 * 365 * 20)
f1zh = 10.0 ** 7.0
lifetime_on_f1zh = lifetime / f1zh
indices = np.arange(i * l * k).reshape((i, l, k))

def aep_load_func(x, y):
    simres = wfm(x, y, wd=wdir, ws=wsp)
    aep = simres.aep().sum()
    loads = simres.loads('OneWT')['LDEL'].values
    return aep, loads

tol = 1e-8
ec = 1e-1
min_spacing = 260
xi, xa = x_init.min()-min_spacing, x_init.max()+min_spacing
yi, ya = y_init.min()-min_spacing, y_init.max()+min_spacing
boundary = np.asarray([[xi, ya], [xa, ya], [xa, yi], [xi, yi]])

cost_comp = AEPMaxLoadCostModelComponent(input_keys=[('x', x_init),('y', y_init)],
                                          n_wt = n_wt,
                                          aep_load_function = aep_load_func,
                                          # aep_load_gradient = aep_load_gradient,
                                          max_loads = max_loads, 
                                          objective=True,
                                          step={'x': step, 'y': step},
                                          output_keys=[('AEP', 0), ('loads', np.zeros((s, i)))]
                                          )
problem = TopFarmProblem(design_vars={'x': x_init, 'y': y_init},
                        constraints=[XYBoundaryConstraint(boundary),
                                     SpacingConstraint(min_spacing)],
                          # post_constraints=[(ls, val * load_fact) for ls, val in loads_nom.items()],
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
    with open(f'./check_partials_{int(toc)}_{ec}_{step}.txt', 'w') as fid:
                partials = problem.check_partials(out_stream=fid,
                                                      compact_print=True,
                                                      show_only_incorrect=True,
                                                                  step=step)
    
    
    
