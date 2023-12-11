import numpy as np

from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponentAdditionalTurbines
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot
from topfarm.tests import npt

from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import wt_x, wt_y, LillgrundSite, SWT2p3_93_65


def test_PyWakeAEPCostModelComponentAdditionalTurbines():
    x2 = np.array([363089.20620581, 362841.19815026])
    y2 = np.array([6154000, 6153854.5244973])
    wind_turbines = SWT2p3_93_65()
    x = wt_x[:4]
    y = wt_y[:4]
    n_wt = len(x)
    site = LillgrundSite()
    wf_model = BastankhahGaussian(site, wind_turbines)
    constraint_comp = XYBoundaryConstraint(np.asarray([x, y]).T)
    cost_comp = PyWakeAEPCostModelComponentAdditionalTurbines(windFarmModel=wf_model,
                                                              n_wt=n_wt,
                                                              add_wt_x=x2,
                                                              add_wt_y=y2,
                                                              grad_method=autograd)
    plot_comp = NoPlot()
    problem = TopFarmProblem(design_vars={'x': x, 'y': y},
                             constraints=[constraint_comp, SpacingConstraint(min_spacing=wind_turbines.diameter() * 2)],
                             cost_comp=cost_comp,
                             driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=5),
                             plot_comp=plot_comp)

    cost, state, recorder = problem.optimize(disp=True)
    npt.assert_almost_equal(cost, -3682.710308568642)
