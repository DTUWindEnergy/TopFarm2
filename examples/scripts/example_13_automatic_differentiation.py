import numpy as np

from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp

from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.utils import gradients
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import wt_x, wt_y, LillgrundSite, ct_curve, power_curve
from py_wake.wind_turbines import OneTypeWindTurbines

class SWT2p3_93_65(OneTypeWindTurbines):
    def __init__(self):
        OneTypeWindTurbines.__init__(self, 'SWT2p3_93_65', diameter=92.6, hub_height=65,
                                     ct_func=self._ct, power_func=self._power, power_unit='kW')

    def _ct(self, u):
        return gradients.interp(u, ct_curve[:, 0], ct_curve[:, 1])

    def _power(self, u):
        return gradients.interp(u, power_curve[:, 0], power_curve[:, 1])

n_wt = len(wt_x)
site = LillgrundSite()
wind_turbines = SWT2p3_93_65() 
wf_model = BastankhahGaussian(site, wind_turbines, turbulenceModel=STF2017TurbulenceModel())
constraint_comp = XYBoundaryConstraint(np.asarray([wt_x, wt_y]).T)
cost_comp = PyWakeAEPCostModelComponent(windFarmModel=wf_model,
                                        n_wt=n_wt,
                                        grad_method=autograd)

problem = TopFarmProblem(design_vars={'x': wt_x, 'y': wt_y},
                        constraints=[constraint_comp, SpacingConstraint(min_spacing=wind_turbines.diameter() * 2)],
                        cost_comp=cost_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=5),
                        plot_comp=XYPlotComp())

cost, state, recorder = problem.optimize(disp=True)
