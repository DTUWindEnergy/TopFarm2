import numpy as np
from openmdao.api import n2
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from topfarm.cost_models.economic_models.turbine_cost import economic_evaluation
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmGroup, TopFarmProblem
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.plotting import XYPlotComp, NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian


def main():
    if __name__ == '__main__':
        try:
            import matplotlib.pyplot as plt
            plt.gcf()
            plot_comp = XYPlotComp()
            plot = True
        except RuntimeError:
            plot_comp = NoPlot()
            plot = False

        n_wt = 16
        site = IEA37Site(n_wt)
        windTurbines = IEA37_WindTurbines()
        windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)
        Drotor_vector = [windTurbines.diameter()] * n_wt
        power_rated_vector = [float(windTurbines.power(20) / 1000)] * n_wt
        hub_height_vector = [windTurbines.hub_height()] * n_wt

        def irr_func(aep, **kwargs):
            my_irr = economic_evaluation(Drotor_vector, power_rated_vector, hub_height_vector, aep).calculate_irr()
            print(my_irr)
            return my_irr

        aep_comp = CostModelComponent(
            input_keys=['x', 'y'],
            n_wt=n_wt,
            cost_function=lambda x, y, **_: windFarmModel(x=x, y=y).aep().sum(['wd', 'ws']) * 10**6,
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
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(600), max_iter=5, max_time=10),
            constraints=[SpacingConstraint(200),
                         CircleBoundaryConstraint([0, 0], 1300.1)],
            plot_comp=plot_comp)
        cost, state, recorder = problem.optimize()
        #        problem.evaluate()
        n2(problem, outfile='example_4_integrated_optimization_aep_and_irr_n2.html', show_browser=False)


main()
