from openmdao.core.driver import Driver, RecordingDebugging
from six import iteritems
import numpy as np
from openmdao.core.analysis_error import AnalysisError
import topfarm
import time


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

        # random state can be set for predictability during testing
        self._randomstate = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('max_iter', 200,
                             desc='Maximum number of iterations (set to None to disable)')
        self.options.declare('max_time', 600,
                             desc='Maximum time in seconds (set to None to disable)')
        self.options.declare('disp', True,
                             desc='Set to False to prevent printing')

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

    def run(self):
        """
        Excute the genetic algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem.model

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
        max_iter = self.options['max_iter'] or 1e20
        max_time = self.options['max_time'] or 1e20
        assert max_iter < 1e20 or max_time < 1e20, "max_iter or max_time must be set"

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

        desvar_info = [(abs2prom[name], *self._desvar_idx[name], lower_bound, upper_bound) for name, meta in iteritems(desvars)]
        desvar_dict = {name: (x0[i:j].copy(), lbound[i:j], ubound[i:j]) for (name, i, j, lbound, ubound) in desvar_info}
        start = time.time()
        while n_iter < max_iter and time.time() - start < max_time:

            for name, i, j, _, _ in desvar_info:
                desvar_dict[name][0][:] = x0[i:j].copy()
            desvar_dict = self.randomize_func(desvar_dict)
            for name, i, j, _, _ in desvar_info:
                x1[i:j] = desvar_dict[name][0][:]

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

        if not success or obj_value_x1 > obj_value_x0:
            obj_value_x1, success = self.objective_callback(x0, record=True)

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


def randomize_turbine_type(desvar_dict, i_wt=None):
    types, lbound, ubound = desvar_dict[topfarm.type_key]
    i_wt = i_wt or np.random.randint(len(types))
    types[i_wt] = np.random.random_integers(lbound[i_wt], ubound[i_wt])
    return desvar_dict


def xy_step_circle(max_step_xy):
    step = np.random.rand() * max(max_step_xy)
    theta = np.random.rand() * np.pi * 2
    return step * np.cos(theta), step * np.sin(theta)


def xy_step_square(max_step_xy):
    return (np.random.rand() * 2 - 1) * max_step_xy[0], (np.random.rand() * 2 - 1) * max_step_xy[1]


def xy_step_normal(max_step_xy):
    return np.random.normal(0, max_step_xy[0] / 2), np.random.normal(0, max_step_xy[1] / 2)


class RandomizeTurbinePosition():
    def __init__(self, max_step=None, xy_step_function=xy_step_circle):
        self.max_step = max_step
        self.xy_step_function = xy_step_function

    def __call__(self, desvar_dict, i_wt=None):
        i_wt = i_wt or np.random.randint(len(desvar_dict[topfarm.x_key][0]))

        max_step_xy = [self.max_step or (desvar_dict[xy][2][i_wt] - desvar_dict[xy][1][i_wt])
                       for xy in [topfarm.x_key, topfarm.y_key]]
        dxy = self.xy_step_function(max_step_xy)
        for (xy, lbound, ubound), dxy_ in zip([desvar_dict[topfarm.x_key], desvar_dict[topfarm.y_key]],
                                              dxy):
            xy[i_wt] = np.maximum(np.minimum(xy[i_wt] + dxy_, ubound[i_wt]), lbound[i_wt])
        return desvar_dict


class RandomizeTurbinePosition_Circle(RandomizeTurbinePosition):
    def __init__(self, max_step=None):
        RandomizeTurbinePosition.__init__(self, max_step=max_step, xy_step_function=xy_step_circle)


class RandomizeTurbinePosition_Square(RandomizeTurbinePosition):
    def __init__(self, max_step=None):
        RandomizeTurbinePosition.__init__(self, max_step=max_step, xy_step_function=xy_step_square)


class RandomizeTurbinePosition_Normal(RandomizeTurbinePosition):
    def __init__(self, max_step=None):
        RandomizeTurbinePosition.__init__(self, max_step=max_step, xy_step_function=xy_step_normal)


class RandomizeTurbineTypeAndPosition(RandomizeTurbinePosition):
    def __init__(self, max_step=None, xy_step_function=xy_step_circle):
        RandomizeTurbinePosition.__init__(self, xy_step_function=xy_step_function, max_step=max_step)

    def __call__(self, desvar_dict, i_wt=None):
        i_wt = i_wt or np.random.randint(len(desvar_dict[topfarm.x_key][0]))
        desvar_dict = randomize_turbine_type(desvar_dict, i_wt=i_wt)
        desvar_dict = RandomizeTurbinePosition.__call__(self, desvar_dict, i_wt=i_wt)
        return desvar_dict
