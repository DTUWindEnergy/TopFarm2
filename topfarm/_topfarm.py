from topfarm.constraint_components.boundary_component import BoundaryComp
from topfarm.constraint_components.spacing_component import SpacingComp
from topfarm.plotting import PlotComp
from topfarm.utils import smart_start
import os
import time
import numpy as np
import warnings
from openmdao.drivers.doe_generators import DOEGenerator, ListGenerator
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.core.explicitcomponent import ExplicitComponent

from topfarm.recorders import ListRecorder, NestedTopFarmListRecorder,\
    TopFarmListRecorder
from openmdao.api import Problem, ScipyOptimizeDriver, IndepVarComp


class TopFarmProblem(Problem):
    def __init__(self, cost_comp, driver, plot_comp, record_id, expected_cost):
        Problem.__init__(self)
        if isinstance(cost_comp, TopFarmProblem):
            cost_comp = cost_comp.as_component()
        cost_comp.parent = self
        self.cost_comp = cost_comp
        
        if isinstance(driver, list):
            driver = DOEDriver(ListGenerator(driver))
        elif isinstance(driver, DOEGenerator):
            driver = DOEDriver(generator=driver)
        self.driver = driver

        self.plot_comp = plot_comp

        self.record_id = record_id

        self.indeps = self.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        self.model.add_subsystem('cost_comp', cost_comp, promotes=['*'])
        self.model.add_objective('cost', scaler=1 / abs(expected_cost))

        if plot_comp:
            self.model.add_subsystem('plot_comp', plot_comp, promotes=['*'])
            plot_comp.problem = self

    @property
    def cost(self):
        return self['cost'][0]

    @property
    def state(self):
        if hasattr(self.cost_comp, 'state'):
            return self.cost_comp.state
        else:
            return {}

    def state_array(self, keys):
        return np.array([self[k] for k in keys]).T

    def update_state(self, state):
        for k, v in state.items():
            try:
                c = self[k]  # fail if k not exists
                v = np.array(v)
                if hasattr(c, 'shape') and c.shape != v.shape:
                    v = v.reshape(c.shape)
                self[k] = v
            except KeyError:
                pass

    def evaluate(self, state={}):
        t = time.time()
        self.update_state(state)
        self.recorder = ListRecorder()
        self.driver.add_recorder(self.recorder)
        self.run_model()
        self.driver._rec_mgr._recorders.remove(self.recorder)
        print("Evaluated in\t%.3fs" % (time.time() - t))
        return self.cost, self.state

    def evaluate_gradients(self):
        t = time.time()
        rec = ListRecorder()
        self.driver.add_recorder(rec)
        res = self.compute_totals(['cost'], wrt=['turbineX', 'turbineY'], return_format='dict')
        self.driver._rec_mgr._recorders.remove(rec)
        print("Gradients evaluated in\t%.3fs" % (time.time() - t))
        return res

    def optimize(self, state={}):
        if hasattr(self.cost_comp, 'problem'):
            self.recorder = NestedTopFarmListRecorder(self.cost_comp, self.record_id)
        else:
            self.recorder = TopFarmListRecorder(self.record_id)
        self.update_state(state)
        if len(self.recorder.driver_iteration_lst) > 0:
            try:
                self.update_state({k: self.recorder.get(k)[-1] for k in self.state.keys() if k not in state})
            except ValueError:
                pass  # loaded state does not fit into dimension of current state

        self.driver.add_recorder(self.recorder)
        self.run_driver()
        self.cleanup()
        if self.driver._rec_mgr._recorders != []:  # in openmdao<2.4 cleanup does not delete recorders
            self.driver._rec_mgr._recorders.remove(self.recorder)
        if isinstance(self.driver, DOEDriver):
            costs = self.recorder.get('cost')
            cases = self.recorder.driver_cases
            costs = [cases.get_case(i).outputs['cost'] for i in range(cases.num_cases)]
            best_case_index = int(np.argmin(costs))
            best_case = cases.get_case(best_case_index)
            self.update_state({k: best_case.outputs[k] for k in best_case.outputs})
        return self.cost, self.state, self.recorder

    def check_gradients(self, all=False, tol=1e-3):
        """Check gradient computations"""
        if all:
            comp_name_lst = [comp.pathname for comp in self.model.system_iter()
                             if comp._has_compute_partials]
        else:
            comp_name_lst = [self.cost_comp.pathname]
        print("checking %s" % ", ".join(comp_name_lst))
        res = self.check_partials(includes=comp_name_lst, compact_print=True)
        for comp in comp_name_lst:
            var_pair = list(res[comp].keys())
            worst = var_pair[np.argmax([res[comp][k]['rel error'].forward for k in var_pair])]
            err = res[comp][worst]['rel error'].forward
            if err > tol:
                raise Warning("Mismatch between finite difference derivative of '%s' wrt. '%s' and derivative computed in '%s' is: %f" %
                              (worst[0], worst[1], comp, err))

    def as_component(self):
        return ProblemComponent(self)

    def get_DOE_list(self):
        assert isinstance(self.driver, DOEDriver), 'get_DOE_list only applies to DOEDrivers, and the current driver is: %s' % type(self.driver)
        case_gen = self.driver.options['generator']
        return [c for c in case_gen(self.model.get_design_vars(recurse=True), self.model)]

    def get_DOE_array(self):
        return np.array([[v for k, v in c] for c in self.get_DOE_list()])


class TurbineTypeOptimizationProblem(TopFarmProblem):
    def __init__(self, cost_comp, turbineTypes, lower, upper, driver, plot_comp=None, record_id=None, expected_cost=1):
        TopFarmProblem.__init__(self, cost_comp, driver, plot_comp, record_id, expected_cost)
        self.turbineTypes = turbineTypes
        self.lower = lower
        self.upper = upper

        n_wt = len(turbineTypes)

        lim = np.zeros((n_wt, 2))
        lim[:, 0] += lower
        lim[:, 1] += upper
        assert np.all(lim[:, 0] < lim[:, 1])

        self.model.add_constraint('turbineType', lower=lim[:, 0], upper=lim[:, 1])
        self.indeps.add_output('turbineType', np.array(turbineTypes).astype(np.int))

        self.model.add_design_var('turbineType', lower=lim[:, 0], upper=lim[:, 1])
        if self.plot_comp:
            plot_comp.n_wt = n_wt
        self.setup(check=True, mode='fwd')

    @property
    def state(self):
        state = {'turbineType': np.round(self['turbineType']).astype(np.int)}
        state.update(TopFarmProblem.state.fget(self))
        return state


class TurbineXYZOptimizationProblem(TopFarmProblem):
    def __init__(self, cost_comp, turbineXYZ, boundary_comp, min_spacing=None,
                 driver=ScipyOptimizeDriver(), plot_comp=None, record_id=None, expected_cost=1):
        turbineXYZ = np.array(turbineXYZ)
        self.turbineXYZ = turbineXYZ
        self.n_wt = n_wt = turbineXYZ.shape[0]

        if plot_comp:
            if plot_comp == "default":
                plot_comp = PlotComp()

        TopFarmProblem.__init__(self, cost_comp, driver, plot_comp, record_id, expected_cost)

        turbineXYZ = np.hstack((turbineXYZ, np.zeros((n_wt, 4 - turbineXYZ.shape[1]))))
        self.min_spacing = min_spacing

        spacing_comp = SpacingComp(n_wt, min_spacing)
        self.boundary_comp = boundary_comp

        if self.driver.supports['inequality_constraints']:
            spacing_comp.setup_as_constraints(self)
            boundary_comp.setup_as_constraints(self)
            mode = 'fwd'
        else:
            spacing_comp.setup_as_penalty(self)
            boundary_comp.setup_as_penalty(self)
            mode = 'rev'
        

        do = self.driver.options
        if len(boundary_comp.xy_boundary) > 0:

            ref0_x, ref0_y = self.boundary_comp.xy_boundary.min(0)
            ref_x, ref_y = self.boundary_comp.xy_boundary.max(0)
            if (('optimizer' in do and do['optimizer'] == 'SLSQP') or  # scaling disturbs SLSQP
                    isinstance(driver, DOEDriver)):
                ref0_x, ref0_y, ref_x, ref_y = 0, 0, 1, 1
            vertices = self.boundary_comp.xy_boundary
            self.indeps.add_output('boundary', vertices, units='m')

            if 'optimizer' in do and do['optimizer'] == 'SLSQP':
                # Default +/- sys.float_info.max does not work for SLSQP
                self.model.add_design_var('turbineX', lower=np.nan, upper=np.nan)
                self.model.add_design_var('turbineY', lower=np.nan, upper=np.nan)
            else:
                l, u = [f(vertices[:, 0]) * (ref_x - ref0_x) + ref0_x for f in [np.min, np.max]]
                self.model.add_design_var('turbineX', lower=l, upper=u, ref0=ref0_x, ref=ref_x)
                l, u = [f(vertices[:, 1]) * (ref_y - ref0_y) + ref0_y for f in [np.min, np.max]]
                self.model.add_design_var('turbineY', lower=l, upper=u, ref0=ref0_y, ref=ref_y)

        if len(boundary_comp.z_boundary) > 0:
            ref0_z, ref_z = np.min(self.boundary_comp.z_boundary), np.max(self.boundary_comp.z_boundary)
            if (('optimizer' in do and do['optimizer'] == 'SLSQP') or  # scaling disturbs SLSQP
                    isinstance(driver, DOEDriver)):
                ref0_z, ref_z = 0, 1
            l, u = [self.boundary_comp.z_boundary[:, i] * (ref_z - ref0_z) + ref0_z for i in range(2)]
            self.model.add_design_var('turbineZ', lower=l, upper=u, ref0=ref0_z, ref=ref_z)

        self.indeps.add_output('turbineX', turbineXYZ[:, 0], units='m')
        self.indeps.add_output('turbineY', turbineXYZ[:, 1], units='m')
        self.indeps.add_output('turbineZ', turbineXYZ[:, 2], units='m')

        if plot_comp:
            plot_comp.n_wt = n_wt
            if self.boundary_comp:
                plot_comp.n_vertices = len(self.boundary_comp.xy_boundary)
            else:
                plot_comp.n_vertices = 0

        self.plot_comp = plot_comp
        self.setup(check=True, mode=mode)

    @property
    def turbine_positions(self):
        return self.state_array(['turbineX', 'turbineY', 'turbineZ'])

    @property
    def state(self):
        state = {'turbine%s' % xyz: self['turbine%s' % xyz] for xyz in 'XYZ'}
        state.update(TopFarmProblem.state.fget(self))
        return state

    @property
    def xy_boundary(self):
        xy_b = self.boundary_comp.xy_boundary
        if len(xy_b) > 0:
            return np.r_[xy_b, xy_b[:1]]
        else:
            return xy_b

    @property
    def z_boundary(self):
        return self.boundary_comp.z_boundary

    def smart_start(self, x, y ,val):
        x, y = smart_start(x, y ,val, self.n_wt, self.min_spacing)
        self.update_state({'turbineX': x, 'turbineY': y})
        return x, y


class InitialXYZOptimizationProblem(TurbineXYZOptimizationProblem):
    def __init__(self, cost_comp, turbineXYZ, boundary_comp=None, min_spacing=None,
                 driver=None, plot_comp=None):
        #          if driver is None:
        #              driver = DOEDriver(shuffle_generator(self, 10))
        TurbineXYZOptimizationProblem.__init__(self, cost_comp, turbineXYZ,
                                               boundary_comp, min_spacing,
                                               driver=driver, plot_comp=plot_comp)

#     def shuffle_positions(self, shuffle_type='rel', n_iter=1000,
#                           step_size=0.1, pad=1.1, offset=5, plot=False,
#                           verbose=False):
#         if shuffle_type is not None:
#             turbines = spos(self.boundary, self.n_wt, self.min_spacing,
#                             self.turbine_positions, shuffle_type, n_iter,
#                             step_size, pad, offset, plot, verbose)
#             self.problem['turbineX'] = turbines.T[0]
#             self.problem['turbineY'] = turbines.T[1]


class TopFarm(TurbineXYZOptimizationProblem):
    """Wrapper of TurbineXYZOPtimizationProblem for backward compatibility
    """

    def __init__(self, turbines, cost_comp, min_spacing, boundary,
                 boundary_type='convex_hull', plot_comp=None,
                 driver=ScipyOptimizeDriver(),
                 record_id="Opt_%s" % time.strftime("%Y%m%d_%H%M%S"),
                 expected_cost=1):

        TurbineXYZOptimizationProblem.__init__(self, cost_comp, turbines,
                                               boundary_comp=BoundaryComp(len(turbines), boundary, None, boundary_type),
                                               min_spacing=min_spacing, driver=driver, plot_comp=plot_comp,
                                               record_id=record_id, expected_cost=expected_cost)


class ProblemComponent(ExplicitComponent):
    """class used to wrap a TopFarmProblem as a cost_component"""

    def __init__(self, problem):
        ExplicitComponent.__init__(self)
        self.problem = problem

    def setup(self):
        missing_in_problem = (set([c[0] for c in self.parent.indeps._indep_external]) -
                              set([c[0] for c in self.problem.indeps._indep_external]))

        for name, val, kwargs in self.parent.indeps._indep_external:
            self.add_input(name, val=val, **{k: kwargs[k] for k in ['units']})
            if name in missing_in_problem:
                self.problem.indeps.add_output(name, val, **kwargs)
        self.problem.setup(check=True, mode='fwd')

        self.add_output('cost', val=0.0)

    @property
    def state(self):
        return self.problem.state

    def compute(self, inputs, outputs):
        outputs['cost'] = self.problem.optimize(dict(inputs))[0]


def try_me():
    if __name__ == '__main__':
        from openmdao.drivers.doe_generators import FullFactorialGenerator
        from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
        from topfarm.plotting import NoPlot
        import numpy as np
        from topfarm.easy_drivers import EasyScipyOptimizeDriver
        optimal = [(0, 2, 4, 1), (4, 2, 1, 0)]

        plot_comp = DummyCostPlotComp(optimal)

        cost_comp = DummyCost(
            optimal_state=optimal,
            inputs=['turbineX', 'turbineY', 'turbineZ', 'turbineType'])
        xyz_opt_problem = TurbineXYZOptimizationProblem(
            cost_comp,
            turbineXYZ=[(0, 0, 0), (1, 1, 1)],
            min_spacing=2,
            boundary_comp=BoundaryComp(n_wt=2,
                                       xy_boundary=[(0, 0), (4, 4)],
                                       z_boundary=(0, 4),
                                       xy_boundary_type='square'),
            plot_comp=plot_comp,
            driver=EasyScipyOptimizeDriver(disp=False),
            record_id='test:latest')

        cost, state, recorder = xyz_opt_problem.optimize()
        recorder.save()

        tf = TurbineTypeOptimizationProblem(
            cost_comp=xyz_opt_problem,
            turbineTypes=[0, 0], lower=0, upper=1,
            driver=DOEDriver(FullFactorialGenerator(2)))
        cost, state, recorder = tf.optimize()
        print(state)
        plot_comp.show()


try_me()
