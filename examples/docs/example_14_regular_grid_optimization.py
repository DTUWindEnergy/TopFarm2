from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake import BastankhahGaussian
from py_wake.utils.gradients import autograd
from topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint

site = Hornsrev1Site()
wt = V80()
D = wt.diameter()
windFarmModel = BastankhahGaussian(site, wt)
n_wt = 16
ec = 0.1
boundary = [[-800,-50], [1200, -50], [1200,2300], [-800, 2300]]

def aep_fun(x, y):
    aep = windFarmModel(x, y).aep().sum()
    return aep

daep = windFarmModel.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'])
aep_comp = CostModelComponent(input_keys=['x', 'y'],
                              n_wt=n_wt,
                              cost_function=aep_fun,
                              cost_gradient_function = daep,
                              output_keys= ("aep", 0),
                              output_unit="GWh",
                              maximize=True,
                              objective=True)

problem = TopFarmProblem(design_vars={'sx': (3*D, 2*D, 15*D), 
                                      'sy': (4*D, 2*D, 15*D), 
                                       'rotation': (50, 0, 90)
                                      },
                         constraints=[XYBoundaryConstraint(boundary),
                                      SpacingConstraint(4*D)],
                        n_wt = n_wt,
                        cost_comp=aep_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=200),
                        plot_comp=XYPlotComp(),
                        expected_cost=ec,
                        additional_input={'stagger': 1*D}
                        )

cost, state, recorder = problem.optimize()



