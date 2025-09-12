import numpy as np
from openmdao.api import n2
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from topfarm.cost_models.economic_models.dtu_wind_cm_main import economic_evaluation
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
        power_rated_vector = [float(windTurbines.power(20)) * 1e-6] * n_wt
        hub_height_vector = [windTurbines.hub_height()] * n_wt
        distance_from_shore = 10         # [km]
        energy_price = 0.1              # [Euro/kWh] What we get per kWh
        project_duration = 20            # [years]
        rated_rpm_array = [12] * n_wt    # [rpm]
        water_depth_array = [15] * n_wt  # [m]

        eco_eval = economic_evaluation(distance_from_shore, energy_price, project_duration)

        def irr_func(aep, **kwargs):
            eco_eval.calculate_irr(
                rated_rpm_array,
                Drotor_vector,
                power_rated_vector,
                hub_height_vector,
                water_depth_array,
                aep)
            print(eco_eval.IRR)
            return eco_eval.IRR

        aep_comp = CostModelComponent(
            input_keys=['x', 'y'],
            n_wt=n_wt,
            cost_function=lambda x, y, **_: windFarmModel(x=x, y=y).aep().sum(['wd', 'ws']) * 10**6,
            output_keys="aep",
            output_unit="kWh",
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
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=5, max_time=10),
            constraints=[SpacingConstraint(200),
                         CircleBoundaryConstraint([0, 0], 1300.1)],
            plot_comp=plot_comp)
        cost, state, recorder = problem.optimize()
        #        problem.evaluate()
#        n2(problem, outfile='example_4_integrated_optimization_aep_and_irr_n2.html', show_browser=False)


main()
