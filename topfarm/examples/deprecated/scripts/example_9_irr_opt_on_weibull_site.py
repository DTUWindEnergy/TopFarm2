import numpy as np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from topfarm.cost_models.economic_models.turbine_cost import economic_evaluation
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmGroup, TopFarmProblem
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle, RandomSearchDriver

from topfarm.plotting import XYPlotComp, mypause, NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint

from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.site import UniformWeibullSite
from py_wake.site.shear import PowerShear
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from py_wake.tests.check_speed import timeit
from openmdao.core.problem import Problem


def main():
    if __name__ == '__main__':
        def get_site():
            f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348,
                 0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
            A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921,
                 9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
            k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703,
                 2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
            ti = 0.001
            h_ref = 100
            alpha = .1
            site = UniformWeibullSite(f, A, k, ti, shear=PowerShear(h_ref=h_ref, alpha=alpha))
            spacing = 2000
            N = 5
            theta = 76  # deg
            dx = np.tan(np.radians(theta))
            x = np.array([np.linspace(0, (N - 1) * spacing, N) + i * spacing / dx for i in range(N)])
            y = np.array(np.array([N * [i * spacing] for i in range(N)]))
            initial_positions = np.column_stack((x.ravel(), y.ravel()))
            eps = 2000
            delta = 5
            site.boundary = np.array([(0 - delta, 0 - delta),
                                      ((N - 1) * spacing + eps, 0 - delta),
                                      ((N - 1) * spacing * (1 + 1 / dx) + eps * (1 + np.cos(np.radians(theta))),
                                       (N - 1) * spacing + eps * np.sin(np.radians(theta)) - delta),
                                      ((N - 1) * spacing / dx + eps * np.cos(np.radians(theta)), (N - 1) * spacing + eps * np.sin(np.radians(theta)))])
            site.initial_position = initial_positions
            return site


        plot_comp = XYPlotComp()
        site = get_site()
        n_wt = len(site.initial_position)
        windTurbines = DTU10MW()
        min_spacing = 2 * windTurbines.diameter(0)
        windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)
        Drotor_vector = [windTurbines.diameter()] * n_wt
        power_rated_vector = [float(windTurbines.power(20) / 1000)] * n_wt
        hub_height_vector = [windTurbines.hub_height()] * n_wt

        def aep_func(x, y, **_):
            sim_res = windFarmModel(x, y)
            aep = sim_res.aep()
            return aep.sum(['wd', 'ws']).values * 10**6

        def irr_func(aep, **_):
            return economic_evaluation(Drotor_vector, power_rated_vector, hub_height_vector, aep).calculate_irr()

        aep_comp = CostModelComponent(
            input_keys=['x', 'y'],
            n_wt=n_wt,
            cost_function=aep_func,
            output_keys="aep",
            output_unit="GWh",
            objective=False,
            output_vals=np.zeros(n_wt))
        irr_comp = CostModelComponent(
            input_keys=['aep'],
            n_wt=n_wt,
            cost_function=irr_func,
            output_keys="irr",
            output_unit="%",
            objective=True,
            maximize=True)
        group = TopFarmGroup([aep_comp, irr_comp])
        problem = TopFarmProblem(
            design_vars=dict(zip('xy', site.initial_position.T)),
            cost_comp=group,
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=10),
            constraints=[SpacingConstraint(min_spacing),
                         XYBoundaryConstraint(site.boundary), ],
            plot_comp=plot_comp)
        cost, state, recorder = problem.optimize()
        problem.plot_comp.show()


main()
