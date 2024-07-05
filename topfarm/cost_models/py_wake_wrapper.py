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
from py_wake.flow_map import Points
from py_wake.utils.gradients import autograd


class PyWakeAEPCostModelComponent(AEPCostModelComponent):
    """TOPFARM wrapper for PyWake AEP calculator"""

    def __init__(self, windFarmModel, n_wt, wd=None, ws=None, max_eval=None, grad_method=autograd, n_cpu=1, **kwargs):
        """Initialize wrapper for PyWake AEP calculator

        Parameters
        ----------
        windFarmModel : DeficitModel
            Wake deficit model used
        n_wt : int
            Number of wind turbines
        wd : array_like
            Wind directions to study
        ws : array_like
            Wind speeds to study
        max_eval : int
            Maximum number of function evaluations
        grad_method : function handle
            Selected method to calculate gradients, default is autograd
        """
        self.windFarmModel = windFarmModel
        self.n_cpu = n_cpu

        def aep(**kwargs):
            try:
                return self.windFarmModel.aep(x=kwargs[topfarm.x_key],
                                              y=kwargs[topfarm.y_key],
                                              h=kwargs.get(topfarm.z_key, None),
                                              type=kwargs.get(topfarm.type_key, 0),
                                              wd=wd, ws=ws,
                                              n_cpu=n_cpu)
            except ValueError as e:
                if 'are at the same position' in str(e):
                    return 0

        if grad_method:
            if hasattr(self.windFarmModel, 'dAEPdxy'):
                # for backward compatibility
                dAEPdxy = self.windFarmModel.dAEPdxy(grad_method)
            else:
                def dAEPdxy(**kwargs):
                    return self.windFarmModel.aep_gradients(
                        gradient_method=grad_method, wrt_arg=['x', 'y'], n_cpu=n_cpu, **kwargs)

            def daep(**kwargs):
                return dAEPdxy(x=kwargs[topfarm.x_key],
                               y=kwargs[topfarm.y_key],
                               h=kwargs.get(topfarm.z_key, None),
                               type=kwargs.get(topfarm.type_key, 0),
                               wd=wd, ws=ws)
        else:
            daep = None
        AEPCostModelComponent.__init__(self,
                                       input_keys=[topfarm.x_key, topfarm.y_key],
                                       n_wt=n_wt,
                                       cost_function=aep,
                                       cost_gradient_function=daep,
                                       output_unit='GWh',
                                       max_eval=max_eval, **kwargs)

    def get_aep4smart_start(self, ws=[6, 8, 10, 12, 14], wd=np.arange(360), type=0, **kwargs):
        """Compute AEP with a smart start approach"""
        def aep4smart_start(X, Y, wt_x, wt_y, T=0, wt_t=0):
            H = np.full(X.shape, self.windFarmModel.windTurbines.hub_height())
            if type == 0:
                sim_res = self.windFarmModel(wt_x, wt_y, type=wt_t, wd=wd, ws=ws, n_cpu=self.n_cpu, **kwargs)
                next_type = T
            else:
                type_ = np.atleast_1d(type)
                t = np.zeros_like(wt_x) + type_[:len(wt_x)]
                sim_res = self.windFarmModel(wt_x, wt_y, type=t, wd=wd, ws=ws, n_cpu=self.n_cpu, **kwargs)
                H = np.full(X.shape, self.windFarmModel.windTurbines.hub_height())
                next_type = type_[min(len(type_) - 1, len(wt_x) + 1)]
            return sim_res.aep_map(Points(X, Y, H), type=next_type, n_cpu=self.n_cpu).values
        return aep4smart_start


class PyWakeAEPCostModelComponentAdditionalTurbines(PyWakeAEPCostModelComponent):
    '''PyWake AEP component that allows for including additional turbine positions that are not
    considered design variables but still considered for wake effect. Note that this functionality
    can be limited by your wind farm models ability to predict long distance wakes.'''

    def __init__(self, windFarmModel, n_wt, add_wt_x, add_wt_y, add_wt_type=0, add_wt_h=None,
                 wd=None, ws=None, max_eval=None, grad_method=autograd, n_cpu=1, **kwargs):
        self.x2, self.y2 = add_wt_x, add_wt_y
        self.windFarmModel = windFarmModel
        self.n_cpu = n_cpu

        def aep(**kwargs):
            x = np.concatenate([kwargs[topfarm.x_key], self.x2])
            y = np.concatenate([kwargs[topfarm.y_key], self.y2])
            if add_wt_h is not None:
                h_primary = np.full_like(kwargs[topfarm.x_key], kwargs.get(topfarm.z_key, None))
                h_secondary = np.full_like(add_wt_x)
                h = np.concatenate((h_primary, h_secondary))
            else:
                h = None
            type_primary = np.ones_like(kwargs[topfarm.x_key]) * kwargs.get(topfarm.type_key, 0)
            type_secondary = np.ones_like(add_wt_x) * add_wt_type
            type = np.concatenate([type_primary, type_secondary])
            try:
                return self.windFarmModel(x=x,
                                          y=y,
                                          h=h,
                                          type=type,
                                          wd=wd, ws=ws,
                                          n_cpu=n_cpu).aep().sum(['wd', 'ws']).values[:n_wt].sum()
            except ValueError as e:
                if 'are at the same position' in str(e):
                    return 0

        if grad_method:
            if hasattr(self.windFarmModel, 'dAEPdxy'):
                # for backward compatibility
                dAEPdxy = self.windFarmModel.dAEPdxy(grad_method)
            else:
                def dAEPdxy(**kwargs):
                    return self.windFarmModel.aep_gradients(
                        gradient_method=grad_method, wrt_arg=['x', 'y'], n_cpu=n_cpu, **kwargs)

            def daep(**kwargs):
                x = np.concatenate([kwargs[topfarm.x_key], self.x2])
                y = np.concatenate([kwargs[topfarm.y_key], self.y2])
                if add_wt_h is not None:
                    h_primary = np.full_like(kwargs[topfarm.x_key], kwargs.get(topfarm.z_key, None))
                    h_secondary = np.full_like(add_wt_x)
                    h = np.concatenate((h_primary, h_secondary))
                else:
                    h = None
                type_primary = np.ones_like(kwargs[topfarm.x_key]) * kwargs.get(topfarm.type_key, 0)
                type_secondary = np.ones_like(add_wt_x) * add_wt_type
                type = np.concatenate([type_primary, type_secondary])
                grad = dAEPdxy(x=x,
                               y=y,
                               h=h,
                               type=type,
                               wd=wd, ws=ws)[:, :n_wt]
                return grad
        else:
            daep = None
        AEPCostModelComponent.__init__(self,
                                       input_keys=[topfarm.x_key, topfarm.y_key],
                                       n_wt=n_wt,
                                       cost_function=aep,
                                       cost_gradient_function=daep,
                                       output_unit='GWh',
                                       max_eval=max_eval, **kwargs)


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
