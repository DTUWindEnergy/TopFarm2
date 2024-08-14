import numpy as np
import matplotlib.pyplot as plt

from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.utils.gradients import autograd
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, HornsrevV80
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver, EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp #AggregatedConstraintsPlotComponent, 
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.recorders import TopFarmListRecorder


def main():
    if __name__ == '__main__':
        plt.close('all')

        site = LillgrundSite()
        site.interp_method = 'linear'
        windTurbines = HornsrevV80() 
        wake_model = BastankhahGaussian(site, windTurbines) 

        x_rows = 5
        y_rows = 5
        spacing = 5
        xu, yu = (x_rows * spacing * windTurbines.diameter(), y_rows * spacing * windTurbines.diameter())
        #x = np.array([np.linspace(0, xu, x_rows) for _ in range(y_rows)]).flatten()
        #y = np.array([np.ones(x_rows) * spacing * windTurbines.diameter() * ii for ii in range(y_rows)]).flatten()
        np.random.seed(1)
        x = np.random.uniform(0, xu, x_rows * y_rows)
        y = np.random.uniform(0, yu, x_rows * y_rows)
        plt.figure()
        plt.plot(x, y, '2k')
        dirs = np.arange(0, 360, 1)
        freqs = site.local_wind(x, y).Sector_frequency_ilk[0, :, 0]
        As = site.local_wind(x, y).Weibull_A_ilk[0, :, 0]
        ks = site.local_wind(x, y).Weibull_k_ilk[0, :, 0]
        samps = 50
        boundary = np.array([(0, 0), (xu, 0), (xu, yu), (0, yu)])
        constraint_comp = XYBoundaryConstraint(boundary, 'rectangle')

        random = True
        # T = 2

        #wd = np.arange(0, 360, 20)
        #ws = np.arange(3, 25, 1)
        def aep_func(x, y, full=False, **kwargs):
            if not random:
                np.random.seed(0)
            idx = np.random.choice(np.arange(dirs.size), samps, p=freqs)
            wd = dirs[idx]
            A = As[idx]
            k = ks[idx]
            if not random:
                np.random.seed(0)
            ws = A * np.random.weibull(k)
            if full:
               wd = np.arange(0, 360, 1)
               ws = np.arange(3, 25, 1)
            return wake_model(x, y, wd=wd, ws=ws, time=not full).aep().sum().values * 1e6

        def aep_func2(x, y, **kwargs):
            wd = np.arange(0, 360, 1)
            ws = np.arange(3, 25, 1)
            return wake_model(x, y, wd=wd, ws=ws).aep().sum().values * 1e6

        def aep_jac(x, y, **kwargs):
            if not random:
                np.random.seed(0)
            idx = np.random.choice(np.arange(dirs.size), samps, p=freqs)
            wd = dirs[idx]
            A = As[idx]
            k = ks[idx]
            if not random:
                np.random.seed(0)
            ws = A * np.random.weibull(k)
            jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, wd=wd, time=True)
            return np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6

        def aep_jac2(x, y, **kwargs):
            wd = np.arange(0, 360, 1)
            ws = np.arange(3, 25, 1)
            jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, wd=wd, time=False)
            return np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6

        n_wt=x.size
        aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func, objective=True, cost_gradient_function=aep_jac, maximize=True)
        aep_comp2 = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func2, objective=True, cost_gradient_function=aep_jac2, maximize=True)
        cost_comps = [aep_comp2, aep_comp]

        def constr_aggr_func(wtSeparationSquared, boundaryDistances, **kwargs):
            separation_constraint = wtSeparationSquared - (2 * windTurbines.diameter()) ** 2
            separation_constraint = separation_constraint[separation_constraint < 0]
            distance_constraint = boundaryDistances
            distance_constraint = distance_constraint[distance_constraint < 0]
            # return np.sum(-1 * np.minimum(wtSeparationSquared - (2 * windTurbines.diameter()) ** 2, 0)) + np.sum(np.minimum(boundaryDistances, 0) ** 2)
            return np.sum(-1 * separation_constraint) + np.sum(distance_constraint ** 2)

        def constr_aggr_grad(wtSeparationSquared, boundaryDistances, **kwargs):
            separation_constraint = wtSeparationSquared - (2 * windTurbines.diameter()) ** 2
            J_separation = np.zeros_like(wtSeparationSquared)
            J_separation[np.where(separation_constraint < 0)] = -1
            J_distance = np.zeros_like(boundaryDistances)
            J_distance[np.where(boundaryDistances < 0)] = 2 * boundaryDistances[np.where(boundaryDistances < 0)]
            return [[J_separation], [J_distance]]

        name = 'sgd_constraint'
        component_args = {'input_keys': [('wtSeparationSquared', np.zeros(int(n_wt * (n_wt - 1) / 2))),
                                         ('boundaryDistances', np.zeros((n_wt, 4)))],
                          'n_wt': n_wt,
                          'cost_function': constr_aggr_func,
                          'cost_gradient_function': constr_aggr_grad,
                          'objective': False,
                          'output_keys': [(name, 0)],
                          'use_constraint_violation': False
                          }
        constraint_args = {'name': name, 'lower': 0}
        from topfarm.constraint_components.constraint_aggregation import ConstraintAggregation

        driver_names = ['SLSQP', 'SGD']
        drivers = [EasyScipyOptimizeDriver(maxiter=200, tol=1e-3), EasySGDDriver(maxiter=400, learning_rate=windTurbines.diameter() / 5, max_time=30 * 60, speedupSGD=True, sgd_thresh=0.5)]
        driver_no = 1
        constraints = [[SpacingConstraint(2 * windTurbines.diameter()), constraint_comp], ConstraintAggregation([SpacingConstraint(2 * windTurbines.diameter()), constraint_comp],
                                                  component_args=component_args, constraint_args=constraint_args)]
        ecs = [1e2, 1]
        tf = TopFarmProblem(
                design_vars={'x':(x,0,xu), 'y':(y,0,yu)}, # setting up our two turbines as design variables
                cost_comp=cost_comps[driver_no], # using dummy cost model
                constraints=constraints[driver_no], # constraint set up for the boundary type provided)
                driver=drivers[driver_no],
                # plot_comp=[XYPlotComp(), AggregatedConstraintsPlotComponent()][driver_no],
                plot_comp=[XYPlotComp(), XYPlotComp()][driver_no],
                expected_cost=ecs[driver_no],
                )
        #tf.driver.learning_rate = windTurbines.diameter() / 5
        if 1:
            cost, state, recorder = tf.optimize()
            recorder.save(f'{driver_names[driver_no]}')

        if 0:
            #plt.figure()
            plt.clf()
            for i in range(2):
                rec = TopFarmListRecorder().load(f'recordings/{driver_names[i]}')
                if driver_names[i] == 'SGD':
                    aep = []
                    for x, y in zip(rec['x'], rec['y']):
                        aep.append(aep_func2(x, y))
                else:
                    aep = rec['Cost']
                    # else:
                    #     aep = 
                plt.plot(rec['timestamp']-rec['timestamp'][0], aep, label=driver_names[i])
            plt.legend()
            plt.savefig('conv')
            plt.clf()
            # plt.plot()

main()
