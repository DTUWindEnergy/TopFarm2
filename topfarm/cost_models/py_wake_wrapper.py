from py_wake.aep_calculator import AEPCalculator
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from py_wake.wake_models.gaussian import IEA37SimpleBastankhahGaussian
from topfarm import x_key, y_key, z_key, type_key
import numpy as np
from scipy.interpolate.interpolate import RegularGridInterpolator


class PyWakeAEP(AEPCalculator):
    def get_TopFarm_cost_component(self, n_wt):

        return AEPCostModelComponent(
            input_keys=['x', 'y'],
            n_wt=n_wt,
            cost_function=lambda **kwargs:
                self.calculate_AEP(x_i=kwargs[x_key],
                                   y_i=kwargs[y_key],
                                   h_i=kwargs.get(z_key, None),
                                   type_i=kwargs.get(type_key, None)).sum(),
            output_unit='GWh')

    def get_aep4smart_start(self, ws=[6, 8, 10, 12, 14], wd=np.arange(360)):
        def aep4smart_start(X, Y, wt_x, wt_y):
            x = np.sort(np.unique(X))
            y = np.sort(np.unique(Y))
            X_j, Y_j, aep_map = self.aep_map(x, y, 0, wt_x, wt_y, wd=wd, ws=ws)
#             import matplotlib.pyplot as plt
#             plt.plot(wt_x, wt_y, '2r')
#             plt.show()
#             plt.figure()
#             c = plt.contourf(X_j, Y_j, aep_map, 100)
#             plt.colorbar(c)
#             plt.axis('equal')
            return RegularGridInterpolator((y, x), aep_map)(np.array([Y, X]).T)
        return aep4smart_start


def main():
    if __name__ == '__main__':
        site = IEA37Site(16)
        windTurbines = IEA37_WindTurbines()
        wake_model = IEA37SimpleBastankhahGaussian(windTurbines)
        aep_calc = PyWakeAEP(site, windTurbines, wake_model)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', site.initial_position.T)),
            cost_comp=aep_calc.get_TopFarm_cost_component(16),
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=5),
            constraints=[CircleBoundaryConstraint([0, 0], 1300.1)],
            plot_comp=XYPlotComp())
        # tf.evaluate()
        tf.optimize()
        tf.plot_comp.show()


main()
