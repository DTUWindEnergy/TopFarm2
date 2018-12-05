
"""TOPFARM OpenMDAO Problem.

This module contains the OpenMDAO problem that can be used for
different types of optimization problems.

Notes
-----

A new TopFarmProblem must be constructed for every optimization problem that
has different design variables. For example, if one problem optimizes a
wind-farm layout but another optimizes turbine types at fixed turbine
positions, these require OpenMDAO problems that are constructed differently.

To get around this, TOPFARM has a base optimization problem,
``TopFarmProblem`` that inherits from the ``Problem`` class in OpenMDAO.
"""
import time
import numpy as np
from openmdao.drivers.doe_generators import DOEGenerator, ListGenerator
from openmdao.drivers.doe_driver import DOEDriver

from topfarm.recorders import ListRecorder, NestedTopFarmListRecorder,\
    TopFarmListRecorder, split_record_id
from openmdao.api import Problem, IndepVarComp
from topfarm.plotting import NoPlot
import warnings
import topfarm
from openmdao.core.explicitcomponent import ExplicitComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.utils import smart_start
from topfarm.constraint_components.spacing import SpacingComp
from topfarm.constraint_components.boundary import BoundaryBaseComp
import copy


class TopFarmProblem(Problem):

    def __init__(self, design_vars, cost_comp, driver=EasyScipyOptimizeDriver(),
                 constraints=[], plot_comp=NoPlot(), record_id=None,
                 expected_cost=1, ext_vars={}):
        """
        Parameters
        ----------
        design_vars : dict or list of key-initial_value-tuples
            Design variables for the problem.
            Ex: {'x': [1,2,3], 'y':([3,2,1],0,1), 'z':([4,5,6],[4,5,4], [6,7,6])}
            Ex: [('x', [1,2,3]), ('y',([3,2,1],0,1)), ('z',([4,5,6],[4,5,4], [6,7,6]))]
            Ex: zip('xy', pos.T)
            The keys (x, y, z) are the names of the design variable.
            The values are either
            - the initial value or
            - a tuple of (initial value, lower bound, upper bound)
        cost_comp : ExplicitComponent or TopFarmProblem
            A cost component in the style of an OpenMDAO v2 ExplicitComponent.
            Pure python cost functions can be wrapped using ``CostModelComponent``
            class in ``topfarm.cost_models.cost_model_wrappers``.
            For nested problems, this is typically a TopFarmProblem
        driver : openmdao Driver, optinal
            Driver used to solve the optimization driver. For an example, see the
            ``EasyScipyOptimizeDriver`` class in ``topfarm.easy_drivers``.
        constraints : list of Constraint-objects
            E.g. XYBoundaryConstraint, SpacingConstraint
        plot_comp : ExplicitComponent, optional
            OpenMDAO ExplicitComponent used to plot the state (during
            optimization).
            For no plotting, pass in the ``topfarm.plotting.NoPlot`` class.
        record_id : string "<record_id>:<case>", optional
            Identifier for the optimization. Allows a user to restart an
            optimization where it left off.
            record_id can be name (saves as recordings/<name>.pkl), abs or relative path
            Case can be:
            - "", "latest", "-1": Continue from latest
            - "best": Continue from best case (minimum cost)
            - "0": Start from scratch (initial position)
            - "4": Start from case number 4
        expected_cost : int or float
            Used to scale the cost. This has influence on some drivers, e.g.
            SLSQP where it affects the step size
        ext_vars : dict or list of key-initial_value tuple
            Used for nested problems to propagate variables from parent problem
            Ex. {'type': [1,2,3]}
            Ex. [('type', [1,2,3])]

        Examples
        --------
        See main() in the bottom of this file
        """
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
        self.load_recorder()

        if not isinstance(design_vars, dict):
            design_vars = dict(design_vars)
        self.design_vars = design_vars
        self.indeps = self.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        for k in [topfarm.x_key, topfarm.y_key, topfarm.type_key]:
            if k in design_vars:
                if isinstance(design_vars[k], tuple):
                    self.n_wt = len(design_vars[k][0])
                else:
                    self.n_wt = len(design_vars[k])
                break
        else:
            self.n_wt = 0

        for constr in constraints:
            if self.driver.supports['inequality_constraints']:
                constr.setup_as_constraint(self)
            else:
                constr.setup_as_penalty(self)
        self.model.constraint_components = [constr.constraintComponent for constr in constraints]

        do = self.driver.options
        for k, v in design_vars.items():
            if isinstance(v, tuple):
                assert len(v) == 3, "Design_vars values must be either value or (value, lower, upper)"
                self.indeps.add_output(k, v[0])

                if ('optimizer' in do and do['optimizer'] == 'COBYLA'):
                    ref0 = np.min(v[1])
                    ref1 = np.max(v[2])
                    l, u = [lu * (ref1 - ref0) + ref0 for lu in [v[1], v[2]]]
                    kwargs = {'ref0': ref0, 'ref': ref1, 'lower': l, 'upper': u}
                else:
                    kwargs = {'lower': v[1], 'upper': v[2]}
            else:
                self.indeps.add_output(k, v)
                kwargs = {}

            if 'optimizer' in do and do['optimizer'] == 'SLSQP':
                # Upper and lower disturbs SLSQP when running with constraints. Add limits as constraints
                self.model.add_constraint(k, kwargs.get('lower', None), kwargs.get('upper', None))
                kwargs = {'lower': np.nan, 'upper': np.nan}  # Default +/- sys.float_info.max does not work for SLSQP
            self.model.add_design_var(k, **kwargs)

        for k, v in ext_vars.items():
            self.indeps.add_output(k, v)
        self.ext_vars = ext_vars

        self.model.add_subsystem('cost_comp', cost_comp, promotes=['*'])
        self.model.add_objective('cost', scaler=1 / abs(expected_cost))

        if plot_comp:
            self.model.add_subsystem('plot_comp', plot_comp, promotes=['*'])
            plot_comp.problem = self
            plot_comp.n_wt = self.n_wt

        self.setup()

    @property
    def cost(self):
        return self['cost'][0]

    @property
    def state(self):
        self.setup()
        state = {k: self[k] for k in self.design_vars}
        state.update({k: self[k] for k in self.ext_vars})
        if hasattr(self.cost_comp, 'state'):
            state.update(self.cost_comp.state)
        if hasattr(self.cost_comp, 'additional_output'):
            state.update({k: self[k] for k, _ in self.cost_comp.additional_output})
        return state

    def state_array(self, keys):
        self.setup()
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

    def load_recorder(self):
        if hasattr(self.cost_comp, 'problem'):
            self.recorder = NestedTopFarmListRecorder(self.cost_comp, self.record_id)
        else:
            self.recorder = TopFarmListRecorder(self.record_id)

    def setup(self):
        if self._setup_status == 0:
            Problem.setup(self, check=True)
        if self._setup_status < 2:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    if len(self.driver._rec_mgr._recorders) == 0:
                        tmp_recorder = ListRecorder()
                        self.driver.add_recorder(tmp_recorder)
                        Problem.final_setup(self)
                        self.driver._rec_mgr._recorders.remove(tmp_recorder)
                    else:
                        Problem.final_setup(self)
                except Warning as w:
                    if str(w).startswith('Inefficient choice of derivative mode'):
                        Problem.setup(self, check=True, mode='fwd')
                    else:
                        raise w

    def evaluate(self, state={}, disp=False):
        """Evaluate the cost model."""
        tmp_recorder = ListRecorder()
        self.driver.add_recorder(tmp_recorder)
        self.setup()
        self.update_state(state)
        t = time.time()
        self.run_model()
        if disp:
            print("Evaluated in\t%.3fs" % (time.time() - t))
        self.driver._rec_mgr._recorders.remove(tmp_recorder)
        return self.cost, copy.deepcopy(self.state)

    def evaluate_gradients(self, disp=False):
        """Evaluate the gradients."""
        self.setup()
        t = time.time()
        rec = ListRecorder()
        self.driver.add_recorder(rec)
        res = self.compute_totals(['cost'], wrt=[topfarm.x_key, topfarm.y_key], return_format='dict')
        self.driver._rec_mgr._recorders.remove(rec)
        if disp:
            print("Gradients evaluated in\t%.3fs" % (time.time() - t))
        return res

    def optimize(self, state={}, disp=False):
        """Run the optimization problem."""
        self.load_recorder()
        self.update_state(state)
        if self.recorder.num_cases > 0:
            try:
                self.update_state({k: self.recorder[k][-1] for k in self.state.keys() if k not in state})
            except (ValueError, KeyError):
                # Restart optimize with n
                self.record_id = split_record_id(self.record_id)[0] + ":0"
                return self.optimize(state, disp)

        self.driver.add_recorder(self.recorder)
        self.setup()
        t = time.time()
        self.run_driver()
        self.cleanup()
        if disp:
            print("Optimized in\t%.3fs" % (time.time() - t))
        if self.driver._rec_mgr._recorders != []:  # in openmdao<2.4 cleanup does not delete recorders
            self.driver._rec_mgr._recorders.remove(self.recorder)
        if isinstance(self.driver, DOEDriver):
            costs = self.recorder.get('cost')
            cases = self.recorder.driver_cases
            costs = [cases.get_case(i).outputs['cost'] for i in range(cases.num_cases)]
            best_case_index = int(np.argmin(costs))
            best_case = cases.get_case(best_case_index)
            self.evaluate({k: best_case.outputs[k] for k in best_case.outputs})
        return self.cost, copy.deepcopy(self.state), self.recorder

    def check_gradients(self, check_all=False, tol=1e-3):
        """Check gradient computations"""
        self.setup()
        if check_all:
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
        self.setup()
        assert isinstance(
            self.driver, DOEDriver), 'get_DOE_list only applies to DOEDrivers, and the current driver is: %s' % type(self.driver)
        case_gen = self.driver.options['generator']
        return [c for c in case_gen(self.model.get_design_vars(recurse=True), self.model)]

    def get_DOE_array(self):
        return np.array([[v for _, v in c] for c in self.get_DOE_list()])

    @property
    def turbine_positions(self):
        return np.array([self[k] for k in [topfarm.x_key, topfarm.y_key]]).T

    def smart_start(self, XX, YY, ZZ):
        min_spacing = [c for c in self.model.constraint_components if isinstance(c, SpacingComp)][0].min_spacing
        X, Y, Z = XX.flatten(), YY.flatten(), ZZ.flatten()
        for comp in self.model.constraint_components:
            if isinstance(comp, BoundaryBaseComp):
                mask = (comp.distances(X, Y).min(1) >= 0)
                X, Y, Z = X[mask], Y[mask], Z[mask]
        x, y = smart_start(X, Y, Z, self.n_wt, min_spacing)
        self.update_state({topfarm.x_key: x, topfarm.y_key: y})
        return x, y


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
        self.problem._setup_status = 0  # redo initial setup
        self.problem.setup()

        self.add_output('cost', val=0.0)
        if hasattr(self.problem.cost_comp, "output_key"):
            self.add_output(self.problem.cost_comp.output_key, val=0.0)

    @property
    def state(self):
        return self.problem.state

    def cost_function(self, **kwargs):
        return self.problem.optimize(kwargs)[0]

    def compute(self, inputs, outputs):
        outputs['cost'] = self.cost_function(**inputs)
        if hasattr(self.problem.cost_comp, "output_key"):
            output_key = self.problem.cost_comp.output_key
            outputs[output_key] = self.problem[output_key]


def main():
    if __name__ == '__main__':
        from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
        from topfarm.constraint_components.spacing import SpacingConstraint
        from topfarm.constraint_components.boundary import XYBoundaryConstraint

        initial = np.array([[6, 0], [6, -8], [1, 1]])  # initial turbine layouts
        optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # optimal turbine layouts
        boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
        desired = np.array([[3, -3], [7, -7], [4, -3]])  # desired turbine layouts

        plot_comp = DummyCostPlotComp(optimal)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', initial.T)),
            cost_comp=DummyCost(optimal_state=desired, inputs=['x', 'y']),
            constraints=[XYBoundaryConstraint(boundary),
                         SpacingConstraint(2)],
            driver=EasyScipyOptimizeDriver(),
            plot_comp=plot_comp
        )
        tf.optimize()
        plot_comp.show()


main()
