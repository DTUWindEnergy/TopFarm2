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
import inspect
from openmdao.api import Problem, IndepVarComp, Group, ParallelGroup, \
    ExplicitComponent, ListGenerator, DOEDriver, SimpleGADriver, OpenMDAOWarning
from openmdao.drivers.doe_generators import DOEGenerator
from openmdao.utils import mpi
from openmdao.core.constants import _SetupStatus
import topfarm
from topfarm.recorders import NestedTopFarmListRecorder, \
    TopFarmListRecorder, split_record_id
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySimpleGADriver, EasyDriverBase, EasySGDDriver
from topfarm.utils import smart_start
from topfarm.constraint_components.spacing import SpacingComp, SpacingTypeComp
from topfarm.constraint_components.boundary import BoundaryBaseComp
# from topfarm.constraint_components.penalty_component import ConstraintViolationComponent, PenaltyComponent #, PostPenaltyComponent
from topfarm.cost_models.topfarm_components import ObjectiveComponent, DummyObjectiveComponent, ConstraintViolationComponent
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
# from topfarm.constraint_components.post_constraint import PostConstraint
from topfarm.constraint_components import Constraint
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


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
                 expected_cost=1, ext_vars={}, approx_totals=False,
                 recorder=None, additional_recorders=None,
                 n_wt=0, grid_layout_comp=None, penalty_comp=None, reports=None, **kwargs):
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
        constraints : list of Constraint-objects or tuples
            Constraint-objects are e.g. XYBoundaryConstraint, SpacingConstraint
            Tuples have the form (variable to constrain, {dict with options passed to the the OpenMDAO method add_constraint})
        penalty_comp : ExplicitComponent, optional
            Component that converts constraints into penalty both for drivers that do and do not support constraints.
            Constraints are automatically converted to penalty for drivers that do not support constraints and the default magnitude of the
            penalty that is added to the objective is the sum of the constraint violations + 10**10.
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
        approx_totals : bool or dict
            If True, approximates the total derivative of the cost_comp group,
            skipping the partial ones. If it is a dictionary, it's elements
            are passed to the approx_totals function of an OpenMDAO Group.
        recorder : Main recorder
        additional_recorders: list(Recorder) or None
            A list of additional recorders to be added to the problem
        n_wt : int
            Number of wind turbines
        grid_layout_comp : ExplicitComponent or TopFarmGroup
            Components that uses at least one of topfarm.grid_x_key or topfarm.grid_y_key as input,
            and provides topfarm.x_key and topfarm.y_key as output.
            Default values for topfarm.grid_x_key or topfarm.grid_y_key are 'sx' and 'sy' respectively.
            These can be overwritten in the same way as e.g. topfarm.x_key. The component is inserted
            before constraint components will enables the use of components relying on x and y situated before
            the main cost component in the workflow.

        Examples
        --------
        See main() in the bottom of this file
        """
        if mpi.MPI:
            comm = None
        else:
            from openmdao.utils.mpi import FakeComm
            comm = FakeComm()

        self.main_recorder = recorder
        self._additional_recorders = additional_recorders
        if not isinstance(constraints, list):
            constraints = [constraints]
        if 'post_constraints' in kwargs:
            warnings.warn("""post_constraints keyword is deprecated. Both Constraint objects and
                          constraint tuples of type (keyword, {constraint options}) can be included in the constriants list.""",
                          DeprecationWarning, stacklevel=2)

            post_constraints = kwargs['post_constraints']
        else:
            post_constraints = [constraint for constraint in constraints if isinstance(constraint, dict) or isinstance(constraint, tuple)]
            constraints = [constraint for constraint in constraints if isinstance(constraint, Constraint)]
        if 'reports' not in inspect.getfullargspec(Problem.__init__).args:
            Problem.__init__(self, comm=comm)
        else:
            Problem.__init__(self, comm=comm, reports=reports)
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
        warnings.filterwarnings('ignore', category=OpenMDAOWarning)
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

        self.n_wt = n_wt
        if not n_wt:
            for k in [topfarm.x_key, topfarm.y_key, topfarm.type_key]:
                if k in design_vars:
                    self.n_wt = len(design_vars[k][0])
                    break
        else:
            self.n_wt = n_wt
        if self.n_wt == 0:
            warnings.warn("Number of turbines is inferred as zero. Please specify number of turbines as 'n_wt' if applicable")

        # add external signals before constraints
        for k, v in ext_vars.items():
            self.indeps.add_output(k, v)
        self.ext_vars = ext_vars

        if grid_layout_comp:
            self.model.add_subsystem('grid_layout_comp', grid_layout_comp, promotes=['*'])

        if cost_comp and hasattr(cost_comp, 'post_constraint'):
            post_constraints = post_constraints + [cost_comp.post_constraint]

        constraints_as_penalty = ((((not self.driver.supports['inequality_constraints'] or
                                     isinstance(self.driver, SimpleGADriver) or
                                     isinstance(self.driver, EasySimpleGADriver)) and
                                    len(constraints) + len(post_constraints) > 0) or
                                   (penalty_comp is not None)) and not
                                  isinstance(self.driver, EasySGDDriver))

        if len(constraints) > 0:
            self.model.add_subsystem('constraint_group', ParallelGroup(), promotes=['*'])
            for constr in constraints:
                if constraints_as_penalty:
                    constr.setup_as_penalty(self)
                else:
                    constr.setup_as_constraint(self)
            constraint_violation_comp = ConstraintViolationComponent(constraints)
            self.model.add_subsystem('constraint_violation_comp', constraint_violation_comp, promotes=['*'])
        else:
            constraint_violation_comp = None
            if isinstance(self.cost_comp, (CostModelComponent, TopFarmGroup)):
                self.indeps.add_output('constraint_violation', val=0.0)

        self.model.constraint_components = [constr.constraintComponent for constr in constraints]

        for k, v in design_vars.items():
            if isinstance(driver, EasyDriverBase):
                kwargs = driver.get_desvar_kwargs(self.model, k, v)
            else:
                kwargs = EasyDriverBase.get_desvar_kwargs(None, self.model, k, v)
            self.model.add_design_var(k, **kwargs)

        if cost_comp:
            self.model.add_subsystem('cost_comp', cost_comp, promotes=['*'])

        if len(post_constraints) > 0:
            if not constraints_as_penalty:
                for constr in post_constraints:
                    if isinstance(constr, Constraint):
                        if 'constraints_group2' not in self.model._subsystems_allprocs:  # and 'post_constraints' not in self.model._static_subsystems_allprocs:
                            self.model.add_subsystem('constraints_group2', ParallelGroup(), promotes=['*'])
                        constr.setup_as_constraint(self, group='constraints_group2')

                    elif isinstance(constr[-1], dict):
                        self.model.add_constraint(str(constr[0]), **constr[-1])
                    else:
                        warnings.warn("""constraint tuples should be of type (keyword, {constraint options}).""",
                                      DeprecationWarning, stacklevel=2)
        if constraints_as_penalty:
            if penalty_comp is None:
                if constraint_violation_comp:
                    constraints_for_aggregation = post_constraints + [constraint_violation_comp]
                else:
                    constraints_for_aggregation = post_constraints
                objective_comp = ObjectiveComponent(constraints_for_aggregation)
            else:
                objective_comp = penalty_comp
        else:
            objective_comp = DummyObjectiveComponent()
        self.model.add_subsystem('objective_comp', objective_comp, promotes=['*'])
        if cost_comp:
            if expected_cost is None:
                expected_cost = self.evaluate()[0]
                if self._metadata:
                    self._metadata['setup_status'] = 0
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

        self.model.add_objective('final_cost', scaler=1 / abs(expected_cost))

        if plot_comp and not isinstance(plot_comp, NoPlot):
            self.model.add_subsystem('plot_comp', plot_comp, promotes=['*'])
            plot_comp.problem = self
            plot_comp.n_wt = self.n_wt

        self.setup()

    # This is needed to avoid an error from interaction between creation of coloring reports and parametrizised pytests in openmdao 3.23-3.26
    def _update_reports(self, driver):
        try:
            from openmdao.utils.reports_system import activate_reports, clear_reports
            if self._driver is not None:
                clear_reports(self._driver)
            driver._set_problem(self)
            activate_reports(self._reports, driver)
        except ModuleNotFoundError:
            pass

    @property
    def cost(self):
        return self['final_cost'][0]

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
        if self.main_recorder:
            self.recorder = self.main_recorder
        elif hasattr(self.cost_comp, 'problem'):
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
        if not self._metadata:
            Problem.setup(self, check=True)
        if self._metadata['setup_status'] == _SetupStatus.PRE_SETUP:
            Problem.setup(self, check=True)
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
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
        res = self.compute_totals(['final_cost'], wrt=[topfarm.x_key, topfarm.y_key], return_format='dict')
        self.driver._rec_mgr._recorders.remove(rec)
        if disp:
            print("Gradients evaluated in\t%.3fs" % (time.time() - t))
        return res

    def optimize(self, state={}, disp=False, recorder_as_list=False):
        """Run the optimization problem

        Parameters
        ----------
        state : dict, optional
            Initial state\n
            Ex: {'x': [1,2,3], 'y':[3,2,1]}\n
            The current state is used to unspecified variables
        disp : bool, optional
            if True, the time used for the optimization is printed
        recorder_as_list : bool, optional
            if True, returns multiprocessing friendly recorder as list of class and attributes
            that can be pickled. Use TopFarmListRecorder().list2recorder to restore TopFarmListRecorder object

        Returns
        -------
        Optimized cost : float
        state : dict
        recorder : TopFarmListRecorder or NestedTopFarmListRecorder or [TopFarmListRecorder.__class__, attributes]
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
            costs = self.recorder['final_cost']
            best_case_index = int(np.argmin(costs))
            best_state = {k: self.recorder[k][best_case_index] for k in self.design_vars}
            self.evaluate(best_state)
        if recorder_as_list:
            return self.cost, copy.deepcopy(self.state), self.recorder.recorder2list()
        else:
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
                            # not x.startswith('penalty') and
                            # not dx.startswith('penalty'))]
                            not (x.startswith('constraint_violation') and
                                 comp != 'constraint_violation_comp')
                            )
                        ]
            worst = var_pair[np.argmax(np.nan_to_num([res[comp][k]['rel error'].forward for k in var_pair]))]
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

    def smart_start(self, XX, YY, ZZ=None, min_space=None, radius=None, random_pct=0, plot=False, seed=None, types=None):
        assert XX.shape == YY.shape
        ZZ_is_func = hasattr(ZZ, '__call__')
        spacing_comp_lst = [c for c in self.model.constraint_components if isinstance(c, SpacingComp)]
        if min_space is not None:
            min_spacing = min_space
        else:
            if len(spacing_comp_lst) == 1:
                min_spacing = spacing_comp_lst[0].min_spacing
            else:
                min_spacing = 0
        if not types:
            min_spacing = np.max(min_spacing)
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
        res = smart_start(X, Y, Z, self.n_wt, min_spacing, radius, random_pct, plot, seed=seed, types=types)
        self.update_state({topfarm.x_key: res[0], topfarm.y_key: res[1]})
        if types:
            self.set_val(topfarm.type_key, res[2])
        return res


class ProblemComponent(ExplicitComponent):
    """class used to wrap a TopFarmProblem as a cost_component"""

    def __init__(self, problem, additional_inputs=[]):
        ExplicitComponent.__init__(self)
        self.problem = problem
        self.additional_inputs = additional_inputs

    def setup(self):
        missing_in_problem_exceptions = ['constraint_violation']
        parent_temp = {}
        parent_temp.update(self.parent.indeps._static_var_rel2meta)
        parent_temp.update(self.parent.indeps._var_rel2meta)
        problem_temp = {}
        problem_temp.update(self.problem.indeps._static_var_rel2meta)
        problem_temp.update(self.problem.indeps._var_rel2meta)
        missing_in_problem = (set(parent_temp) -
                              set(problem_temp))
        indepsargs = ['val', 'units']
        self.missing_attrs = []
        for name, kwargs in parent_temp.items():
            kwargs = {k: v for k, v in kwargs.items() if k in indepsargs}
            self.add_input(name, **kwargs)
            self.missing_attrs.append(name)
            if name in missing_in_problem:
                if name not in missing_in_problem_exceptions:
                    self.problem.indeps.add_output(name, **kwargs)
        self.problem._setup_status = 1  # 1 -- The `setup` method has been called, but vectors not initialized.
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

        initial = np.array([[6, 0], [6, -8], [1, 1]])  # initial turbine layouts
        optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # optimal turbine layouts
        boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
        desired = np.array([[3, -3], [7, -7], [4, -3]])  # desired turbine layouts
        drivers = [EasyScipyOptimizeDriver()]
        plot_comp = DummyCostPlotComp(optimal)
        tf = TopFarmProblem(
            design_vars=dict(zip('xy', initial.T)),
            cost_comp=DummyCost(optimal_state=desired, inputs=['x', 'y']),
            constraints=[XYBoundaryConstraint(boundary),
                         SpacingConstraint(2)
                         ],
            driver=drivers[0],
            plot_comp=plot_comp
        )
        cost, _, recorder = tf.optimize()
        plot_comp.show()


main()
