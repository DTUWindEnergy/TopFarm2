# from topfarm.constraint_components.boundary_component import BoundaryComp
# from topfarm.constraint_components.spacing_component import SpacingComp
# from topfarm.plotting import PlotComp
import sys
import time
import warnings

from openmdao.api import ScipyOptimizeDriver
import numpy as np
import topfarm
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.boundary_component import PolygonBoundaryComp
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.drivers import random_search_driver
from topfarm.easy_drivers import EasyRandomSearchDriver


def set_deprecated_keys():
    topfarm.x_key = 'turbineX'
    topfarm.y_key = 'turbineY'
    topfarm.z_key = 'turbineZ'
    topfarm.type_key = 'turbineType'


class TurbineTypeOptimizationProblem(TopFarmProblem):
    def __init__(self, cost_comp, turbineTypes, lower, upper, **kwargs):
        sys.stderr.write("%s is deprecated. Use TopFarmProblem instead\n" % self.__class__.__name__)

        TopFarmProblem.__init__(self,
                                design_vars={topfarm.type_key: (turbineTypes, lower, upper)},
                                cost_comp=cost_comp,
                                **kwargs)
        self.setup()


class TurbineXYZOptimizationProblem(TopFarmProblem):
    def __init__(self, cost_comp, turbineXYZ, boundary_comp, min_spacing=None,
                 driver=ScipyOptimizeDriver(), plot_comp=None, record_id=None, expected_cost=1):
        sys.stderr.write("%s is deprecated. Use TopFarmProblem instead\n" % self.__class__.__name__)
        if plot_comp:
            if plot_comp == "default":
                plot_comp = PlotComp()
        turbineXYZ = np.asarray(turbineXYZ)
        design_vars = {xy: v for xy, v in zip([topfarm.x_key, topfarm.y_key], turbineXYZ.T)}
        constraints = []
        if min_spacing:
            constraints.append(SpacingConstraint(min_spacing))

        if isinstance(boundary_comp, PolygonBoundaryComp):
            constraints.append(XYBoundaryConstraint(boundary_comp.xy_boundary, 'polygon'))
        elif len(boundary_comp.xy_boundary):
            constraints.append(XYBoundaryConstraint(boundary_comp.xy_boundary, boundary_comp.boundary_type))

        if turbineXYZ.shape[1] == 3:
            if len(boundary_comp.z_boundary):
                design_vars[topfarm.z_key] = (turbineXYZ[:, 2], boundary_comp.z_boundary[:, 0], boundary_comp.z_boundary[:, 1])
            else:
                design_vars[topfarm.z_key] = turbineXYZ[:, 2]

        TopFarmProblem.__init__(
            self,
            design_vars=design_vars,
            cost_comp=cost_comp,
            driver=driver,
            constraints=constraints,
            plot_comp=plot_comp,
            record_id=record_id,
            expected_cost=expected_cost)
        self.setup()


class TurbineTypeXYZOptimizationProblem(TurbineTypeOptimizationProblem, TurbineXYZOptimizationProblem):
    def __init__(self, cost_comp, turbineTypes, lower, upper, turbineXYZ, boundary_comp, min_spacing=None,
                 driver=EasyRandomSearchDriver(random_search_driver.RandomizeTurbineTypeAndPosition()), plot_comp=None, record_id=None, expected_cost=1):
        sys.stderr.write("%s is deprecated. Use TopFarmProblem instead\n" % self.__class__.__name__)
        TopFarmProblem.__init__(self, cost_comp, driver, plot_comp, record_id, expected_cost)
        TurbineTypeOptimizationProblem.initialize(self, turbineTypes, lower, upper)
        TurbineXYZOptimizationProblem.initialize(self, turbineXYZ, boundary_comp, min_spacing)
        self.setup(check=True, mode=self.mode)

#     @property
#     def state(self):
#         state = {k: self[k] for k in ['turbineX', 'turbineY', 'turbineZ']}
#         state['turbineType'] = self['turbineType'].astype(int)
#         state.update(TopFarmProblem.state.fget(self))
#         return state


class InitialXYZOptimizationProblem(TurbineXYZOptimizationProblem):
    def __init__(self, cost_comp, turbineXYZ, boundary_comp=None, min_spacing=None,
                 driver=None, plot_comp=None):
        #          if driver is None:
        #              driver = DOEDriver(shuffle_generator(self, 10))
        sys.stderr.write("%s is deprecated. Use TopFarmProblem instead\n" % self.__class__.__name__)
        TurbineXYZOptimizationProblem.__init__(self, cost_comp, turbineXYZ,
                                               boundary_comp, min_spacing,
                                               driver=driver, plot_comp=plot_comp)
        self.setup()

#     def shuffle_positions(self, shuffle_type='rel', n_iter=1000,
#                           step_size=0.1, pad=1.1, offset=5, plot=False,
#                           verbose=False):
#         if shuffle_type is not None:
#             turbines = spos(self.boundary, self.n_wt, self.min_spacing,
#                             self.turbine_positions, shuffle_type, n_iter,
#                             step_size, pad, offset, plot, verbose)
#             self.problem['turbineX'] = turbines.T[0]
#             self.problem['turbineY'] = turbines.T[1]


class TopFarm(TopFarmProblem):
    """Wrapper of TurbineXYZOPtimizationProblem for backward compatibility
    """

    def __init__(self, turbines, cost_comp, min_spacing, boundary,
                 boundary_type='convex_hull', plot_comp=None,
                 driver=ScipyOptimizeDriver(),
                 record_id="Opt_%s" % time.strftime("%Y%m%d_%H%M%S"),
                 expected_cost=1):
        sys.stderr.write("%s is deprecated. Use TopFarmProblem instead\n" % self.__class__.__name__)
        constraints = []
        if min_spacing and len(turbines) > 1:
            constraints.append(SpacingConstraint(min_spacing))
        if boundary is not None:
            constraints.append(XYBoundaryConstraint(boundary, boundary_type))
        TopFarmProblem.__init__(
            self,
            design_vars={k: v for k, v in zip([topfarm.x_key, topfarm.y_key, topfarm.z_key], np.asarray(turbines).T)},
            cost_comp=cost_comp,
            driver=driver,
            constraints=constraints,
            plot_comp=plot_comp,
            record_id=record_id,
            expected_cost=expected_cost)
        self.setup()
