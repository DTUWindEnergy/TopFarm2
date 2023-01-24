import numpy as np

from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp

from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import wt_x, wt_y, LillgrundSite, ct_curve, power_curve
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular


def main():
    if __name__ == '__main__':
        wind_turbines = WindTurbine('SWT2p3_93_65', 92.6, 65, PowerCtTabular(ct_curve[:,0],power_curve[:,1], 'kW', ct_curve[:,1]))
        n_wt = len(wt_x)
        site = LillgrundSite()
        wf_model = BastankhahGaussian(site, wind_turbines)
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

main()
