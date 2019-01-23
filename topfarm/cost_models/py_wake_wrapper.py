from py_wake.aep_calculator import AEPCalculator
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.cost_model_wrappers import IncomeModelComponent
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from py_wake.wake_models.gaussian import IEA37SimpleBastankhahGaussian


def get_xy_cost_comp(aep_calculator, n_wt):
    return IncomeModelComponent(
        input_keys='xy',
        n_wt=n_wt,
        cost_function=lambda x, y, **kwargs: aep_calculator.calculate_AEP(x_i=x, y_i=y).sum(),
        output_key='AEP',
        output_unit='GWh')


def main():
    if __name__ == '__main__':
        site = IEA37Site(16)
        windTurbines = IEA37_WindTurbines()
        wake_model = IEA37SimpleBastankhahGaussian(windTurbines)
        aep_calc = AEPCalculator(site, windTurbines, wake_model)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', site.initial_position.T)),
            cost_comp=get_xy_cost_comp(aep_calc, 16),
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=5),
            constraints=[CircleBoundaryConstraint([0, 0], 1300.1)],
            plot_comp=XYPlotComp())
        # tf.evaluate()
        tf.optimize()
        tf.plot_comp.show()


main()
