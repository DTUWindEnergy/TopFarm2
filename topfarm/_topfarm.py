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

To disable MPI for topfarm add this before importing topfarm
```from openmdao.utils import mpi
mpi.MPI = None```
"""
import time
import numpy as np
import warnings
import copy
from openmdao.api import Problem, IndepVarComp, Group, ParallelGroup,\
    ExplicitComponent, ListGenerator, DOEDriver, SimpleGADriver
from openmdao.drivers.doe_generators import DOEGenerator
from openmdao.utils import mpi
import topfarm
from topfarm.recorders import NestedTopFarmListRecorder,\
    TopFarmListRecorder, split_record_id
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySimpleGADriver, EasyDriverBase
from topfarm.utils import smart_start
from topfarm.constraint_components.spacing import SpacingComp
from topfarm.constraint_components.boundary import BoundaryBaseComp
from topfarm.constraint_components.penalty_component import PenaltyComponent, PostPenaltyComponent
from topfarm.cost_models.aggregated_cost import AggregatedCost
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.post_constraint import PostConstraint


class TopFarmBaseGroup(Group):
    def __init__(self, comps=[], output_key=None, output_unit=''):
        super().__init__()
        self.comps = []
        self.obj_comp = None
        for i, comp in enumerate(comps):
            if hasattr(comp, 'objective') and comp.objective:
                self.output_key = comp.output_key
                self.output_unit = comp.output_unit
                self.cost_factor = comp.cost_factor
                self.obj_comp = comp
            else:
                self.comps.append(comp)


class TopFarmGroup(TopFarmBaseGroup):
    def __init__(self, comps=[], output_key=None, output_unit=''):
        super().__init__(comps, output_key, output_unit)
        for i, comp in enumerate(comps):
            self.add_subsystem('comp_{}'.format(i), comp, promotes=['*'])


class TopFarmParallelGroup(TopFarmBaseGroup):
    def __init__(self, comps=[], output_key=None, output_unit=''):
        super().__init__(comps, output_key, output_unit)
        parallel = ParallelGroup()
        for i, comp in enumerate(self.comps):
            parallel.add_subsystem('comp_{}'.format(i), comp, promotes=['*'])
        self.add_subsystem('parallel', parallel, promotes=['*'])
        self.add_subsystem('objective', self.obj_comp, promotes=['*'])


class TopFarmProblem(Problem):

    def __init__(self, design_vars, cost_comp=None, driver=EasyScipyOptimizeDriver(),
                 constraints=[], plot_comp=NoPlot(), record_id=None,
                 expected_cost=1, ext_vars={}, post_constraints=[], approx_totals=False, additional_recorders=None):
        """Initialize TopFarmProblem

        Parameters
        ----------
        design_vars : dict or list of key-initial_value-tuples
            Design variables for the problem.\n
            Ex: {'x': [1,2,3], 'y':([3,2,1],0,1), 'z':([4,5,6],[4,5,4], [6,7,6])}\n
            Ex: [('x', [1,2,3]), ('y',([3,2,1],0,1)), ('z',([4,5,6],[4,5,4], [6,7,6]))]\n
            Ex: [('x', ([1,2,3],0,3,'m')), ('y',([3,2,1],'m')), ('z',([4,5,6],[4,5,4], [6,7,6]))]\n
            Ex: zip('xy', pos.T)\n
            The keys (x, y, z) are the names of the design variable.\n
            The values are either\n
            - the initial value or\n
            - on of the following tuples:
                (initial value, unit)
                (initial value, lower bound, upper bound)
                (initial value, lower bound, upper bound, unit)
        cost_comp : ExplicitComponent or TopFarmProblem or TopFarmGroup
            Component that provides the cost function. It has to be the style
            of an OpenMDAO v2 ExplicitComponent.
            Pure python cost functions can be wrapped using ``CostModelComponent``
            class in ``topfarm.cost_models.cost_model_wrappers``.\n
            ExplicitComponent are wrapped into a TopFarmGroup.\n
            For nested problems, the cost comp_comp is typically a TopFarmProblem
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
            optimization where it left off.\n
            record_id can be name (saves as recordings/<name>.pkl), abs or relative path
            Case can be:\n
            - "", "latest", "-1": Continue from latest\n
            - "best": Continue from best case (minimum cost)\n
            - "0": Start from scratch (initial position)\n
            - "4": Start from case number 4\n
        expected_cost : int, float or None, optional
            Used to scale the cost, default is 1. This has influence on some drivers, e.g.
            SLSQP where it affects the step size\n
            If None, the value is found by evaluating the cost function
        ext_vars : dict or list of key-initial_value tuple
            Used for nested problems to propagate variables from parent problem\n
            Ex. {'type': [1,2,3]}\n
            Ex. [('type', [1,2,3])]\n
        post_constraints : list of Constraint-objects that needs the cost component to be
            evaluated, unlike (pre-)constraints which are evaluated before the cost component.
            E.g. LoadConstraint
        approx_totals : bool or dict
            If True, approximates the total derivative of the cost_comp group,
            skipping the partial ones. If it is a dictionary, it's elements
            are passed to the approx_totals function of an OpenMDAO Group.
        additional_recorders: list(Recorder) or None
            A list of additional recorders to be added to the problem

        Examples
        --------
        See main() in the bottom of this file
        """
        if mpi.MPI:
            comm = None
        else:
            from openmdao.utils.mpi import FakeComm
            comm = FakeComm()

        self._additional_recorders = additional_recorders

        Problem.__init__(self, comm=comm)
        if cost_comp:
            if isinstance(cost_comp, TopFarmProblem):
                cost_comp = cost_comp.as_component()
            elif isinstance(cost_comp, ExplicitComponent) and (len(post_constraints) > 0):
                cost_comp = TopFarmGroup([cost_comp])
            cost_comp.parent = self
        self.cost_comp = cost_comp

        if isinstance(driver, list):
            driver = DOEDriver(ListGenerator(driver))
        elif isinstance(driver, DOEGenerator):
            driver = DOEDriver(generator=driver)
        self.driver = driver
        self.driver.recording_options['record_desvars'] = True
        self.driver.recording_options['includes'] = ['*']
        self.driver.recording_options['record_inputs'] = True

        self.plot_comp = plot_comp

        self.record_id = record_id
        self.load_recorder()
        if not isinstance(approx_totals, dict) and approx_totals:
            approx_totals = {'method': 'fd'}

        if not isinstance(design_vars, dict):
            design_vars = dict(design_vars)
        self.design_vars = design_vars
        self.indeps = self.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        for k, v in design_vars.items():
            if isinstance(v, tuple):
                if (not isinstance(v[-1], str)) or (not v[-1]):
                    design_vars[k] += (None, )
            else:
                design_vars[k] = (design_vars[k], None)
            v = design_vars[k]
            self.indeps.add_output(k, v[0], units=v[-1])

        for k in [topfarm.x_key, topfarm.y_key, topfarm.type_key]:
            if k in design_vars:
                self.n_wt = len(design_vars[k][0])
                break
        else:
            self.n_wt = 0

        # add external signals before constraints
        for k, v in ext_vars.items():
            self.indeps.add_output(k, v)
        self.ext_vars = ext_vars

        if cost_comp and isinstance(cost_comp, PostConstraint):
            post_constraints = post_constraints + [cost_comp]

        constraints_as_penalty = ((not self.driver.supports['inequality_constraints'] or
                                   isinstance(self.driver, SimpleGADriver) or
                                   isinstance(self.driver, EasySimpleGADriver)) and
                                  len(constraints) + len(post_constraints) > 0)

        if len(constraints) > 0:
            self.model.add_subsystem('pre_constraints', ParallelGroup(), promotes=['*'])
            for constr in constraints:
                if constraints_as_penalty:
                    constr.setup_as_penalty(self)
                else:
                    constr.setup_as_constraint(self)
                    # Use the assembled Jacobian.
                    self.model.pre_constraints.options['assembled_jac_type'] = 'csc'
                    self.model.pre_constraints.linear_solver.assemble_jac = True
            penalty_comp = PenaltyComponent(constraints, constraints_as_penalty)
            self.model.add_subsystem('penalty_comp', penalty_comp, promotes=['*'])
        else:
            if isinstance(self.cost_comp, (CostModelComponent, TopFarmGroup)):
                self.indeps.add_output('penalty', val=0.0)

        self.model.constraint_components = [constr.constraintComponent for constr in constraints]

        for k, v in design_vars.items():
            if isinstance(driver, EasyDriverBase):
                kwargs = driver.get_desvar_kwargs(self.model, k, v)
            else:
                kwargs = EasyDriverBase.get_desvar_kwargs(None, self.model, k, v)
            self.model.add_design_var(k, **kwargs)

        if cost_comp:
            self.model.add_subsystem('cost_comp', cost_comp, promotes=['*'])

            if expected_cost is None:
                expected_cost = self.evaluate()[0]
                self._setup_status = 0
            if isinstance(driver, EasyDriverBase) and driver.supports_expected_cost is False:
                expected_cost = 1
            if isinstance(cost_comp, Group) and approx_totals:
                cost_comp.approx_totals(**approx_totals)
            # Use the assembled Jacobian.
            if 'assembled_jac_type' in self.model.cost_comp.options:
                self.model.cost_comp.options['assembled_jac_type'] = 'dense'
                self.model.cost_comp.linear_solver.assemble_jac = True

        else:
            self.indeps.add_output('cost')

        if len(post_constraints) > 0:
            if constraints_as_penalty:
                penalty_comp = PostPenaltyComponent(post_constraints, constraints_as_penalty)
                self.model.add_subsystem('post_penalty_comp', penalty_comp, promotes=['*'])
            else:
                for constr in post_constraints:
                    if isinstance(constr[-1], dict):
                        self.model.add_constraint(str(constr[0]), **constr[-1])
                    elif len(constr) == 2:  # assuming only name and upper value is specified and value is per turbine
                        if len(constr[1]) == 1:
                            self.model.add_constraint(constr[0], upper=np.full(self.n_wt, constr[1]))
                        else:
                            self.model.add_constraint(constr[0], upper=constr[1])
                    elif len(constr) == 3:  # four arguments are: key, lower, upper ,shape
                        lower = None if constr[1] is None else constr[1]
                        upper = None if constr[2] is None else constr[2]
                        self.model.add_constraint(
                            constr[0], lower=lower, upper=upper)
                    elif len(constr) == 4:  # four arguments are: key, lower, upper ,shape
                        lower = None if constr[1] is None else np.full(constr[3], constr[1])
                        upper = None if constr[2] is None else np.full(constr[3], constr[2])
                        self.model.add_constraint(
                            constr[0], lower=lower, upper=upper)
                    # Use the assembled Jacobian.
#                    self.model.cost_comp.post_constraints.options['assembled_jac_type'] = 'csc'
#                    self.model.cost_comp.post_constraints.linear_solver.assemble_jac = True

        aggr_comp = AggregatedCost(constraints_as_penalty, constraints, post_constraints)
        self.model.add_subsystem('aggr_comp', aggr_comp, promotes=['*'])
        # print(expected_cost)
        self.model.add_objective('aggr_cost', scaler=1 / abs(expected_cost))

        if plot_comp and not isinstance(plot_comp, NoPlot):
            self.model.add_subsystem('plot_comp', plot_comp, promotes=['*'])
            plot_comp.problem = self
            plot_comp.n_wt = self.n_wt

        self.setup()

    @property
    def cost(self):
        return self['aggr_cost'][0]

    def __getitem__(self, name):
        return Problem.__getitem__(self, name).copy()

    @property
    def state(self):
        """Return current state"""
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

    def get_vars_from_recorder(self):
        rec = self.recorder
        x = np.array(rec[topfarm.x_key])
        y = np.array(rec[topfarm.y_key])
        c = np.array(rec[topfarm.cost_key])
        x0 = x[0]
        y0 = y[0]
        cost0 = c[0]
        return {'x0': x0, 'y0': y0, 'cost0': cost0,
                topfarm.x_key: x, topfarm.y_key: y, 'c': c}

    def setup(self):
        if self._setup_status == 0:
            Problem.setup(self, check=True)
        if self._setup_status < 2:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    if len(self.driver._rec_mgr._recorders) == 0:
                        tmp_recorder = TopFarmListRecorder()
                        self.driver.add_recorder(tmp_recorder)
                        Problem.final_setup(self)
                    else:
                        Problem.final_setup(self)
                except Warning as w:
                    if str(w).startswith('Inefficient choice of derivative mode'):
                        Problem.setup(self, check=True, mode='fwd')
                    else:
                        raise w
                finally:
                    try:
                        self.driver._rec_mgr._recorders.remove(tmp_recorder)
                    except Exception:
                        pass

    def evaluate(self, state={}, disp=False):
        """Evaluate the cost model

        Parameters
        ----------
        state : dict, optional
            Initial state\n
            Ex: {'x': [1,2,3], 'y':[3,2,1]}\n
            The current state is used for unspecified variables
        disp : bool, optional
            if True, the time used for the optimization is printed

        Returns
        -------
        Current cost : float
        Current state : dict
        """
        tmp_recorder = TopFarmListRecorder()
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
        rec = TopFarmListRecorder()
        self.driver.add_recorder(rec)
        res = self.compute_totals(['aggr_cost'], wrt=[topfarm.x_key, topfarm.y_key], return_format='dict')
        self.driver._rec_mgr._recorders.remove(rec)
        if disp:
            print("Gradients evaluated in\t%.3fs" % (time.time() - t))
        return res

    def optimize(self, state={}, disp=False):
        """Run the optimization problem

        Parameters
        ----------
        state : dict, optional
            Initial state\n
            Ex: {'x': [1,2,3], 'y':[3,2,1]}\n
            The current state is used to unspecified variables
        disp : bool, optional
            if True, the time used for the optimization is printed

        Returns
        -------
        Optimized cost : float
        state : dict
        recorder : TopFarmListRecorder or NestedTopFarmListRecorder
        """
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
        if self._additional_recorders:
            for r in self._additional_recorders:
                self.driver.add_recorder(r)

#        self.recording_options['includes'] = ['*']
#        self.driver.recording_options['record_desvars'] = True
#        self.driver.recording_options['includes'] = ['*']
#        self.driver.recording_options['record_inputs'] = True
        self.setup()
        t = time.time()
        self.run_driver()
        self.cleanup()
        if disp:
            print("Optimized in\t%.3fs" % (time.time() - t))
        if self.driver._rec_mgr._recorders != []:  # in openmdao<2.4 cleanup does not delete recorders
            self.driver._rec_mgr._recorders.remove(self.recorder)
        if isinstance(self.driver, DOEDriver) or isinstance(self.driver, SimpleGADriver):
            costs = self.recorder['aggr_cost']
            best_case_index = int(np.argmin(costs))
            best_state = {k: self.recorder[k][best_case_index] for k in self.design_vars}
            self.evaluate(best_state)
        return self.cost, copy.deepcopy(self.state), self.recorder

    def check_gradients(self, check_all=False, tol=1e-3):
        """Check gradient computations"""
        self.setup()
        if check_all:
            comp_name_lst = [comp.pathname for comp in self.model.system_iter()
                             if hasattr(comp, '_has_compute_partials') and comp._has_compute_partials]
        else:
            comp_name_lst = [self.cost_comp.pathname]
        print("checking %s" % ", ".join(comp_name_lst))
        res = self.check_partials(includes=comp_name_lst, compact_print=True)
        for comp in comp_name_lst:
            var_pair = [(x, dx) for x, dx in res[comp].keys()
                        if (x not in ['cost_comp_eval'] and
                            not x.startswith('penalty') and
                            not dx.startswith('penalty'))]
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

    def smart_start(self, XX, YY, ZZ=None, radius=None, random_pct=0, plot=False, seed=None):
        if len(XX.shape) == 1:
            XX, YY = np.meshgrid(XX, YY)
        assert XX.shape == YY.shape
        ZZ_is_func = hasattr(ZZ, '__call__')
        spacing_comp_lst = [c for c in self.model.constraint_components if isinstance(c, SpacingComp)]
        if len(spacing_comp_lst) == 1:
            min_spacing = spacing_comp_lst[0].min_spacing
        else:
            min_spacing = 0
        X, Y = XX.flatten(), YY.flatten()
        if not ZZ_is_func:
            if ZZ is None:
                ZZ = np.full(XX.shape, 0)
            Z = ZZ.flatten()
        else:
            Z = ZZ
        for comp in self.model.constraint_components:
            if isinstance(comp, BoundaryBaseComp):
                dist = comp.distances(X, Y)
                if len(dist.shape) == 2:
                    dist = dist.min(1)
                mask = dist >= 0
                X, Y = X[mask], Y[mask]
                if not ZZ_is_func:
                    Z = Z[mask]
        x, y = smart_start(X, Y, Z, self.n_wt, min_spacing, radius, random_pct, plot, seed=seed)
        self.update_state({topfarm.x_key: x, topfarm.y_key: y})
        return x, y


class ProblemComponent(ExplicitComponent):
    """class used to wrap a TopFarmProblem as a cost_component"""

    def __init__(self, problem, additional_inputs=[]):
        ExplicitComponent.__init__(self)
        self.problem = problem
        self.additional_inputs = additional_inputs

    def setup(self):
        missing_in_problem_exceptions = ['penalty']
        missing_in_problem = (set([c[0] for c in self.parent.indeps._indep_external]) -
                              set([c[0] for c in self.problem.indeps._indep_external]))
        self.missing_attrs = []
        for name, val, kwargs in self.parent.indeps._indep_external:
            self.add_input(name, val=val, **{k: kwargs[k] for k in ['units']})
            self.missing_attrs.append(name)
            if name in missing_in_problem:
                if name not in missing_in_problem_exceptions:
                    self.problem.indeps.add_output(name, val, **kwargs)
        self.problem._setup_status = 0  # redo initial setup
        self.problem.setup()

        self.add_output('cost', val=0.0)
        if hasattr(self.problem.cost_comp, "output_key"):
            self.add_output(self.problem.cost_comp.output_key, val=0.0)
        self.comp_lst = [comp for comp in self.problem.model.system_iter()]

    @property
    def state(self):
        return self.problem.state

    def cost_function(self, **kwargs):
        return self.problem.optimize(kwargs)[0]

    def set_input_as_option(self, inputs):
        for comp in self.comp_lst:
            for attr in self.missing_attrs:
                if attr in comp.options:
                    comp.options[attr] = inputs[attr]

    def compute(self, inputs, outputs):
        self.set_input_as_option(inputs)
        outputs['cost'] = self.cost_function(**inputs)
        if hasattr(self.problem.cost_comp, "output_key"):
            output_key = self.problem.cost_comp.output_key
            outputs[output_key] = self.problem[output_key]


def main():
    if __name__ == '__main__':
        from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
        from topfarm.constraint_components.spacing import SpacingConstraint
        from topfarm.constraint_components.boundary import XYBoundaryConstraint
        from openmdao.api import view_model

        initial = np.array([[6, 0], [6, -8], [1, 1]])  # initial turbine layouts
        optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # optimal turbine layouts
        boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
        desired = np.array([[3, -3], [7, -7], [4, -3]])  # desired turbine layouts
        drivers = [EasySimpleGADriver(max_gen=10, pop_size=100, bits={'x': [12] * 3, 'y':[12] * 3}, random_state=1),
                   EasyScipyOptimizeDriver()]
        plot_comp = DummyCostPlotComp(optimal)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', initial.T)),
            cost_comp=DummyCost(optimal_state=desired, inputs=['x', 'y']),
            constraints=[XYBoundaryConstraint(boundary),
                         SpacingConstraint(2)
                         ],
            driver=drivers[1],
            plot_comp=plot_comp
        )
        cost, _, recorder = tf.optimize()
        plot_comp.show()


main()
