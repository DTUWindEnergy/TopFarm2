from py_wake.aep_calculator import AEPCalculator
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines, IEA37Site
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
import topfarm
import numpy as np
from scipy.interpolate.interpolate import RegularGridInterpolator
import warnings
from py_wake.flow_map import HorizontalGrid
import xarray as xr


class PyWakeAEP(AEPCalculator):
    """TOPFARM wrapper for PyWake AEP calculator"""

    def __init__(self, wake_model):
        warnings.warn('PyWakeAEP is deprecated. Use PyWakeAEPCostModelComponent instead', DeprecationWarning)
        AEPCalculator.__init__(self, wake_model)

    def get_TopFarm_cost_component(self, n_wt, wd=None, ws=None):
        """Create topfarm-style cost component

        Parameters
        ----------
        n_wt : int
            Number of wind turbines
        """
        return AEPCostModelComponent(
            input_keys=['x', 'y'],
            n_wt=n_wt,
            cost_function=lambda **kwargs:
                self.calculate_AEP(x_i=kwargs[topfarm.x_key],
                                   y_i=kwargs[topfarm.y_key],
                                   h_i=kwargs.get(topfarm.z_key, None),
                                   type_i=kwargs.get(topfarm.type_key, 0),
                                   wd=wd, ws=ws).sum(),
            output_unit='GWh')

    def get_aep4smart_start(self, ws=[6, 8, 10, 12, 14], wd=np.arange(360)):
        def aep4smart_start(X, Y, wt_x, wt_y):
            x = np.sort(np.unique(X))
            y = np.sort(np.unique(Y))
            X_j, Y_j, aep_map = self.aep_map(x, y, 0, wt_x, wt_y, wd=wd, ws=ws)
#             import matplotlib.pyplot as plt
#             c = plt.contourf(X_j, Y_j, aep_map[:, :, 0], 100)
#             plt.colorbar(c)
#             plt.show()
            return RegularGridInterpolator((y, x), aep_map.values)(np.array([Y, X]).T)
        return aep4smart_start


class PyWakeAEPCostModelComponent(AEPCostModelComponent):
    def __init__(self, windFarmModel, n_wt, wd=None, ws=None, max_eval=None, **kwargs):
        self.windFarmModel = windFarmModel
        AEPCostModelComponent.__init__(self,
                                       input_keys=[topfarm.x_key, topfarm.y_key],
                                       n_wt=n_wt,
                                       cost_function=lambda **kwargs: self.windFarmModel.aep(
                                           x=kwargs[topfarm.x_key],
                                           y=kwargs[topfarm.y_key],
                                           h=kwargs.get(topfarm.z_key, None),
                                           type=kwargs.get(topfarm.type_key, 0),
                                           wd=wd, ws=ws),
                                       output_unit='GWh',
                                       max_eval=max_eval, **kwargs)

    def get_aep4smart_start(self, ws=[6, 8, 10, 12, 14], wd=np.arange(360)):
        def aep4smart_start(X, Y, wt_x, wt_y, type=0):
            sim_res = self.windFarmModel(wt_x, wt_y, type=type, wd=wd, ws=ws)
            x = np.sort(np.unique(X))
            y = np.sort(np.unique(Y))
            aep_map = sim_res.flow_map(HorizontalGrid(x, y)).aep_xy()
            if isinstance(aep_map, xr.DataArray):
                aep_map = aep_map.values[:, :, 0]
            return RegularGridInterpolator((y, x), aep_map)(np.array([Y, X]).T)
        return aep4smart_start


def main():
    if __name__ == '__main__':
        n_wt = 16
        site = IEA37Site(n_wt)
        windTurbines = IEA37_WindTurbines()
        windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', site.initial_position.T)),
            cost_comp=PyWakeAEPCostModelComponent(windFarmModel, n_wt),
            driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=5),
            constraints=[CircleBoundaryConstraint([0, 0], 1300.1)],
            plot_comp=XYPlotComp())
        tf.optimize()
        tf.plot_comp.show()


main()
