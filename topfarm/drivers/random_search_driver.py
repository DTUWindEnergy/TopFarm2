from openmdao.core.driver import Driver, RecordingDebugging
from six import iteritems
import numpy as np
from openmdao.core.analysis_error import AnalysisError


class RandomSearchDriver(Driver):
    """
    Driver for a simple genetic algorithm.

    Attributes
    ----------
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    _desvar_idx : dict
        Keeps track of the indices for each desvar, since GeneticAlgorithm seess an array of
        design variables.
    _ga : <GeneticAlgorithm>
        Main genetic algorithm lies here.
    _randomstate : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws.
    """

    def __init__(self, randomize_func, **kwargs):
        """
        Initialize the SimpleGADriver driver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        self.randomize_func = randomize_func
        super().__init__(**kwargs)

        # What we support
        self.supports['integer_design_vars'] = True

        # What we don't support yet
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = False
        self.supports['active_set'] = False

        self._desvar_idx = {}
        self._ga = None

        # random state can be set for predictability during testing
        self._randomstate = None

#         # Support for Parallel models.
#         self._concurrent_pop_size = 0
#         self._concurrent_color = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('max_iter', 200, lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('disp', True,
                             desc='Set to False to prevent printing')

#         self.options.declare('bits', default={}, types=(dict),
#                              desc='Number of bits of resolution. Default is an empty dict, where '
#                              'every unspecified variable is assumed to be integer, and the number '
#                              'of bits is calculated automatically. If you have a continuous var, '
#                              'you should set a bits value as a key in this dictionary.')
#         self.options.declare('elitism', default=True,
#                              desc='If True, replace worst performing point with best from previous'
#                              ' generation each iteration.')
#         self.options.declare('max_gen', default=100,
#                              desc='Number of generations before termination.')
#         self.options.declare('pop_size', default=0,
#                              desc='Number of points in the GA. Set to 0 and it will be computed '
#                              'as four times the number of bits.')
#         self.options.declare('run_parallel', default=False,
#                              desc='Set to True to execute the points in a generation in parallel.')
#         self.options.declare('procs_per_model', default=1, lower=1,
#                              desc='Number of processors to give each model under MPI.')

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        if len(self._objs) > 1:
            msg = 'RandomSearchDriver currently does not support multiple objectives.'
            raise RuntimeError(msg)

        if len(self._cons) > 0:
            msg = 'RandomSearchDriver currently does not support constraints.'
            raise RuntimeError(msg)

#         model_mpi = None
#         comm = self._problem.comm
#         if self._concurrent_pop_size > 0:
#             model_mpi = (self._concurrent_pop_size, self._concurrent_color)
#         elif not self.options['run_parallel']:
#             comm = None

        # self._ga = GeneticAlgorithm(self.objective_callback, comm=comm, model_mpi=model_mpi)

#     def _setup_comm(self, comm):
#         """
#         Perform any driver-specific setup of communicators for the model.
#
#         Here, we generate the model communicators.
#
#         Parameters
#         ----------
#         comm : MPI.Comm or <FakeComm> or None
#             The communicator for the Problem.
#
#         Returns
#         -------
#         MPI.Comm or <FakeComm> or None
#             The communicator for the Problem model.
#         """
#         procs_per_model = self.options['procs_per_model']
#         if MPI and self.options['run_parallel'] and procs_per_model > 1:
#
#             full_size = comm.size
#             size = full_size // procs_per_model
#             if full_size != size * procs_per_model:
#                 raise RuntimeError("The total number of processors is not evenly divisible by the "
#                                    "specified number of processors per model.\n Provide a "
#                                    "number of processors that is a multiple of %d, or "
#                                    "specify a number of processors per model that divides "
#                                    "into %d." % (procs_per_model, full_size))
#             color = comm.rank % size
#             model_comm = comm.Split(color)
#
#             # Everything we need to figure out which case to run.
#             self._concurrent_pop_size = size
#             self._concurrent_color = color
#
#             return model_comm
#
#         self._concurrent_pop_size = 0
#         self._concurrent_color = 0
#         return comm

    def run(self):
        """
        Excute the genetic algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem.model
        ga = self._ga

        # Size design variables.
        desvars = self._designvars
        count = 0
        for name, meta in iteritems(desvars):
            size = meta['size']
            self._desvar_idx[name] = (count, count + size)
            count += size

        lower_bound = np.empty((count, ))
        upper_bound = np.empty((count, ))

        x0 = np.empty(count)
        desvar_vals = self.get_design_var_values()

        # Figure out bounds vectors and initial design vars
        for name, meta in iteritems(desvars):
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta['lower']
            upper_bound[i:j] = meta['upper']
            x0[i:j] = desvar_vals[name]
        max_iter = self.options['max_iter']
        disp = self.options['disp']

        abs2prom = model._var_abs2prom['output']

        # Initial Design Vars
        desvar_vals = self.get_design_var_values()
        i = 0
        for name, meta in iteritems(self._designvars):
            size = meta['size']
            x0[i:i + size] = desvar_vals[name]
            i += size

        obj_value_x0, success = self.objective_callback(x0)
        x1 = x0.copy()
        n_iter = 0

        desvar_info = [(abs2prom[name], *self._desvar_idx[name], meta['lower'], meta['upper']) for name, meta in iteritems(desvars)]
        desvar_dict = {name: (x0[i:j], l, u) for (name, i, j, l, u) in desvar_info}
        while n_iter < max_iter:

            for name, i, j, _, _ in desvar_info:
                desvar_dict[name][0][:] = x0[i:j]
            desvar_dict = self.randomize_func(desvar_dict)
            for name, i, j, _, _ in desvar_info:
                x1[i:j] = desvar_dict[name][0][:]

#             index = np.random.randint(list(desvars.values())[0]['size'])
#             for name, meta in iteritems(desvars):
#                 i, j = self._desvar_idx[name]
#                 lbound, u = meta['lower'], meta['upper']
#                 state.append(('name', x0[i:j], meta['lower'], meta['upper']))
#                 if isinstance(x0[i + index], (int, np.int_)):
#                     x1[i + index] = np.random.randint(lbound, u)
#                 else:
#                     x1[i + index] = np.random.rand() * (u - lbound) + lbound

            obj_value_x1, success = self.objective_callback(x1, record=False)
            if success and obj_value_x1 < obj_value_x0:
                obj_value_x1, success = self.objective_callback(x1, record=True)
                x0 = x1.copy()
                obj_value_x0 = obj_value_x1
                n_iter += 1
                if disp:
                    print(n_iter, obj_value_x1)
            else:
                if obj_value_x1 < 1e10:
                    n_iter += 1
                x1 = x0.copy()
                obj_value_x1 = obj_value_x0

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name in desvars:
            i, j = self._desvar_idx[name]
            val = x1[i:j]
            self.set_design_var(name, val)

        with RecordingDebugging('SimpleGA', self.iter_count, self) as rec:
            model._solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return False

    def objective_callback(self, x, record=True):
        """
        Evaluate problem objective at the requested point.

        Parameters
        ----------
        x : ndarray
            Value of design variables.
        icase : int
            Case number, used for identification when run in parallel.

        Returns
        -------
        float
            Objective value
        bool
            Success flag, True if successful
        int
            Case number, used for identification when run in parallel.
        """
        model = self._problem.model
        success = 1

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # Execute the model
        if record:
            with RecordingDebugging('RandomSearch', self.iter_count, self) as rec:
                self.iter_count += 1
                try:
                    model._solve_nonlinear()
    
                # Tell the optimizer that this is a bad point.
                except AnalysisError:
                    model._clear_iprint()
                    success = 0
    
                for name, val in iteritems(self.get_objective_values()):
                    obj = val
                    break
    
                # Record after getting obj to assure they have
                # been gathered in MPI.
                rec.abs = 0.0
                rec.rel = 0.0
        else:
            try:
                model._solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            for name, val in iteritems(self.get_objective_values()):
                obj = val
                break
        

        # print("Functions calculated")
        # print(x)
        # print(obj)
        return obj, success


class RandomizeTurbinePosition_DirStep():
    def __init__(self, max_move_step):
        self.max_move_step = max_move_step

    def __call__(self, desvar_dict):
        i_wt = np.random.randint(len(desvar_dict['turbineX'][0]))
        step = np.random.rand() * self.max_move_step
        theta = np.random.rand() * np.pi * 2
        for (xy, l, u), dxy in [(desvar_dict['turbineX'], step * np.cos(theta)),
                                (desvar_dict['turbineY'], step * np.sin(theta))]:
            xy[i_wt] = np.maximum(np.minimum(xy[i_wt] + dxy, u), l)
        return desvar_dict


class RandomizeTurbinePosition_Uniform():
    def __call__(self, desvar_dict):
        i_wt = np.random.randint(len(desvar_dict['turbineX'][0]))
        for xy, lbound, ubound in [(desvar_dict['turbineX']),
                                   (desvar_dict['turbineY'])]:
            if hasattr(lbound, 'len'):
                lbound = lbound[i_wt]
            if hasattr(ubound, 'len'):
                ubound = ubound[i_wt]
            v = np.random.rand() * (ubound - lbound) + lbound
            xy[i_wt] = np.maximum(np.minimum(v, ubound), lbound)
        return desvar_dict
