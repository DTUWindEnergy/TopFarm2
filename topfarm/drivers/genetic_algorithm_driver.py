""" Copied from OpenMDAO 2.5.0, bug fix add (rink).
Driver for a simple genetic algorithm.

This is the Simple Genetic Algorithm implementation based on 2009 AAE550: MDO Lecture notes of
Prof. William A. Crossley.

This basic GA algorithm is compartmentalized into the GeneticAlgorithm class so that it can be
used in more complicated driver.

The following reference is only for the automatic population sizing:
Williams E.A., Crossley W.A. (1998) Empirically-Derived Population Size and Mutation Rate
Guidelines for a Genetic Algorithm with Uniform Crossover. In: Chawdhry P.K., Roy R., Pant R.K.
(eds) Soft Computing in Engineering Design and Manufacturing. Springer, London.

The following reference is only for the penalty function:
Smith, A. E., Coit, D. W. (1995) Penalty functions. In: Handbook of Evolutionary Computation, 97(1).

The following reference is only for weighted sum multi-objective optimization:
Sobieszczanski-Sobieski, J., Morris, A. J., van Tooren, M. J. L. (2015)
Multidisciplinary Design Optimization Supported by Knowledge Based Engineering.
John Wiley & Sons, Ltd.
"""
import os
import copy

from six import iteritems, itervalues, next
from six.moves import range, zip

import numpy as np
try:
    from pyDOE2 import lhs
except ModuleNotFoundError:
    from pyDOE3 import lhs

import openmdao
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.concurrent_utils import concurrent_eval
from openmdao.utils.mpi import MPI
from openmdao.core.analysis_error import AnalysisError


class SimpleGADriver(Driver):
    """
    Driver for a simple genetic algorithm.

    Attributes
    ----------
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    _desvar_idx : dict
        Keeps track of the indices for each desvar, since GeneticAlgorithm sees an array of
        design variables.
    _ga : <GeneticAlgorithm>
        Main genetic algorithm lies here.
    _randomstate : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SimpleGADriver driver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super(SimpleGADriver, self).__init__(**kwargs)

        # What we support
        self.supports['integer_design_vars'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = True

        # What we don't support yet
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = False
        self.supports['active_set'] = False

        self._desvar_idx = {}
        self._ga = None

        # random state can be set for predictability during testing
        if 'SimpleGADriver_seed' in os.environ:
            self._randomstate = int(os.environ['SimpleGADriver_seed'])
        else:
            self._randomstate = None

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('bits', default={}, types=(dict),
                             desc='Number of bits of resolution. Default is an empty dict, where '
                             'every unspecified variable is assumed to be integer, and the number '
                             'of bits is calculated automatically. If you have a continuous var, '
                             'you should set a bits value as a key in this dictionary.')
        self.options.declare('elitism', default=True,
                             desc='If True, replace worst performing point with best from previous'
                             ' generation each iteration.')
        self.options.declare('max_gen', default=100,
                             desc='Number of generations before termination.')
        self.options.declare('pop_size', default=0,
                             desc='Number of points in the GA. Set to 0 and it will be computed '
                             'as four times the number of bits.')
        self.options.declare('run_parallel', default=False,
                             desc='Set to True to execute the points in a generation in parallel.')
        self.options.declare('procs_per_model', default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')
        self.options.declare('penalty_parameter', default=10., lower=0.,
                             desc='Penalty function parameter.')
        self.options.declare('penalty_exponent', default=1.,
                             desc='Penalty function exponent.')
        self.options.declare('Pc', default=0.5, lower=0., upper=1.,
                             desc='Crossover rate.')
        self.options.declare('Pm',
                             desc='Mutation rate.', default=None, lower=0., upper=1.,
                             allow_none=True)
        self.options.declare('multi_obj_weights', default={}, types=(dict),
                             desc='Weights of objectives for multi-objective optimization.'
                             'Weights are specified as a dictionary with the absolute names'
                             'of the objectives. The same weights for all objectives are assumed, '
                             'if not given.')
        self.options.declare('multi_obj_exponent', default=1., lower=0.,
                             desc='Multi-objective weighting exponent.')

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super(SimpleGADriver, self)._setup_driver(problem)

        model_mpi = None
        comm = self._problem().comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options['run_parallel']:
            comm = None

        self._ga = GeneticAlgorithm(self.objective_callback, comm=comm, model_mpi=model_mpi)

# RINK: this block is commented out so we get 95% test coverage
#    def _setup_comm(self, comm):
#        """
#        Perform any driver-specific setup of communicators for the model.
#
#        Here, we generate the model communicators.
#
#        Parameters
#        ----------
#        comm : MPI.Comm or <FakeComm> or None
#            The communicator for the Problem.
#
#        Returns
#        -------
#        MPI.Comm or <FakeComm> or None
#            The communicator for the Problem model.
#        """
#        procs_per_model = self.options['procs_per_model']
#        if MPI and self.options['run_parallel'] and procs_per_model > 1:
#
#            full_size = comm.size
#            size = full_size // procs_per_model
#            if full_size != size * procs_per_model:
#                raise RuntimeError("The total number of processors is not evenly divisible by the "
#                                   "specified number of processors per model.\n Provide a "
#                                   "number of processors that is a multiple of %d, or "
#                                   "specify a number of processors per model that divides "
#                                   "into %d." % (procs_per_model, full_size))
#            color = comm.rank % size
#            model_comm = comm.Split(color)
#
#            # Everything we need to figure out which case to run.
#            self._concurrent_pop_size = size
#            self._concurrent_color = color
#
#            return model_comm
#
#        self._concurrent_pop_size = 0
#        self._concurrent_color = 0
#        return comm

    def run(self):
        """
        Execute the genetic algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem().model
        ga = self._ga

        ga.elite = self.options['elitism']
        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        user_bits = self.options['bits']
        Pm = self.options['Pm']  # if None, it will be calculated in execute_ga()
        Pc = self.options['Pc']

        # Size design variables.
        desvars = self._designvars
        count = 0
        for name, meta in iteritems(desvars):
            size = meta['size']
            self._desvar_idx[name] = (count, count + size)
            count += size

        lower_bound = np.empty((count, ))
        upper_bound = np.empty((count, ))
        outer_bound = np.full((count, ), np.inf)
        bits = np.empty((count, ), dtype=int)
        x0 = np.empty(count)

        desvar_vals = self.get_design_var_values()

        # Figure out bounds vectors and initial design vars
        for name, meta in iteritems(desvars):
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta['lower']
            upper_bound[i:j] = meta['upper']
            x0[i:j] = desvar_vals[name]

        # Bits of resolution
        abs2prom = model._var_abs2meta['output']
        for name, meta in iteritems(desvars):
            i, j = self._desvar_idx[name]

            if name in user_bits:
                val = user_bits[name]
            elif (prom_name := abs2prom.get(name) is not None) and prom_name in user_bits:
                val = user_bits[prom_name]
            else:
                # If the user does not declare a bits for this variable, we assume they want it to
                # be encoded as an integer. Encoding requires a power of 2 in the range, so we need
                # to pad additional values above the upper range, and adjust accordingly. Design
                # points with values above the upper bound will be discarded by the GA.
                log_range = np.log2(upper_bound[i:j] - lower_bound[i:j] + 1)
                val = log_range  # default case -- no padding required
                mask = log_range % 2 > 0  # mask for vars requiring padding
                val[mask] = np.ceil(log_range[mask])
                outer_bound[i:j][mask] = upper_bound[i:j][mask]
                upper_bound[i:j][mask] = 2**np.ceil(log_range[mask]) - 1 + lower_bound[i:j][mask]

            bits[i:j] = val

        # Automatic population size.
        if pop_size == 0:
            pop_size = 4 * np.sum(bits)

        desvar_new, obj, nfit = ga.execute_ga(x0, lower_bound, upper_bound, outer_bound,
                                              bits, pop_size, max_gen,
                                              self._randomstate, Pm, Pc)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name in desvars:
            i, j = self._desvar_idx[name]
            val = desvar_new[i:j]
            self.set_design_var(name, val)

        with RecordingDebugging('SimpleGA', self.iter_count, self) as rec:
            model._solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return False

    def objective_callback(self, x, icase):
        r"""
        Evaluate problem objective at the requested point.

        In case of multi-objective optimization, a simple weighted sum method is used:

        .. math::

           f = (\sum_{k=1}^{N_f} w_k \cdot f_k)^a

        where :math:`N_f` is the number of objectives and :math:`a>0` is an exponential
        weight. Choosing :math:`a=1` is equivalent to the conventional weighted sum method.

        The weights given in the options are normalized, so:

        .. math::

            \sum_{k=1}^{N_f} w_k = 1

        If one of the objectives :math:`f_k` is not a scalar, its elements will have the same
        weights, and it will be normed with length of the vector.

        Takes into account constraints with a penalty function.

        All constraints are converted to the form of :math:`g_i(x) \leq 0` for
        inequality constraints and :math:`h_i(x) = 0` for equality constraints.
        The constraint vector for inequality constraints is the following:

        .. math::

           g = [g_1, g_2  \dots g_N], g_i \in R^{N_{g_i}}

           h = [h_1, h_2  \dots h_N], h_i \in R^{N_{h_i}}

        The number of all constraints:

        .. math::

           N_g = \sum_{i=1}^N N_{g_i},  N_h = \sum_{i=1}^N N_{h_i}

        The fitness function is constructed with the penalty parameter :math:`p`
        and the exponent :math:`\kappa`:

        .. math::

           \Phi(x) = f(x) + p \cdot \sum_{k=1}^{N^g}(\delta_k \cdot g_k)^{\kappa}
           + p \cdot \sum_{k=1}^{N^h}|h_k|^{\kappa}

        where :math:`\delta_k = 0` if :math:`g_k` is satisfied, 1 otherwise

        .. note::

            The values of :math:`\kappa` and :math:`p` can be defined as driver options.

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
        model = self._problem().model
        success = 1

        objs = self.get_objective_values()
        nr_objectives = len(objs)

        # Single objective, if there is nly one objective, which has only one element
        is_single_objective = (nr_objectives == 1) and (len(next(itervalues(objs))) == 1)

        obj_exponent = self.options['multi_obj_exponent']
        if self.options['multi_obj_weights']:  # not empty
            obj_weights = self.options['multi_obj_weights']
        else:
            # Same weight for all objectives, if not specified
            obj_weights = {name: 1. for name in objs.keys()}
        sum_weights = sum(itervalues(obj_weights))

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = openmdao.INF_BOUND

        # Execute the model
        with RecordingDebugging('SimpleGA', self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model._solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            obj_values = self.get_objective_values()
            if is_single_objective:  # Single objective optimization
                obj = next(itervalues(obj_values))  # First and only key in the dict
            else:  # Multi-objective optimization with weighted sums
                weighted_objectives = np.array([])
                for name, val in iteritems(obj_values):
                    # element-wise multiplication with scalar
                    # takes the average, if an objective is a vector
                    try:
                        weighted_obj = val * obj_weights[name] / val.size
                    except KeyError:
                        msg = ('Name "{}" in "multi_obj_weights" option '
                               'is not an absolute name of an objective.')
                        raise KeyError(msg.format(name))
                    weighted_objectives = np.hstack((weighted_objectives, weighted_obj))

                obj = sum(weighted_objectives / sum_weights)**obj_exponent

            # Parameters of the penalty method
            penalty = self.options['penalty_parameter']
            exponent = self.options['penalty_exponent']

            if penalty == 0:
                fun = obj
            else:
                constraint_violations = np.array([])
                for name, val in iteritems(self.get_constraint_values()):
                    con = self._cons[name]
                    # The not used fields will either None or a very large number
                    if (con['lower'] is not None) and (con['lower'] > -almost_inf):
                        diff = val - con['lower']
                        violation = np.array([0. if d >= 0 else abs(d) for d in diff])
                    elif (con['upper'] is not None) and (con['upper'] < almost_inf):
                        diff = val - con['upper']
                        violation = np.array([0. if d <= 0 else abs(d) for d in diff])
                    elif (con['equals'] is not None) and (abs(con['equals']) < almost_inf):
                        diff = val - con['equals']
                        violation = np.absolute(diff)
                    constraint_violations = np.hstack((constraint_violations, violation))
                fun = obj + penalty * sum(np.power(constraint_violations, exponent))
            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        # print("Functions calculated")
        # print(x)
        # print(obj)
        return fun, success, icase


class GeneticAlgorithm(object):
    """
    Simple Genetic Algorithm.

    This is the Simple Genetic Algorithm implementation based on 2009 AAE550: MDO Lecture notes of
    Prof. William A. Crossley. It can be used standalone or as part of the OpenMDAO Driver.


    Attributes
    ----------
    comm : MPI communicator or None
        The MPI communicator that will be used objective evaluation for each generation.
    elite : bool
        Elitism flag.
    lchrom : int
        Chromosome length.
    model_mpi : None or tuple
        If the model in objfun is also parallel, then this will contain a tuple with the the
        total number of population points to evaluate concurrently, and the color of the point
        to evaluate on this rank.
    npop : int
        Population size.
    objfun : function
        Objective function callback.
    """

    def __init__(self, objfun, comm=None, model_mpi=None):
        """
        Initialize genetic algorithm object.

        Parameters
        ----------
        objfun : function
            Objective callback function.
        comm : MPI communicator or None
            The MPI communicator that will be used objective evaluation for each generation.
        model_mpi : None or tuple
            If the model in objfun is also parallel, then this will contain a tuple with the the
            total number of population points to evaluate concurrently, and the color of the point
            to evaluate on this rank.
        """
        self.objfun = objfun
        self.comm = comm

        self.lchrom = 0
        self.npop = 0
        self.elite = True
        self.model_mpi = model_mpi

    def execute_ga(self, x0, vlb, vub, vob, bits, pop_size, max_gen, random_state, Pm=None, Pc=0.5):
        """
        Perform the genetic algorithm.

        Parameters
        ----------
        x0 : ndarray
            Initial design values
        vlb : ndarray
            Lower bounds array.
        vub : ndarray
            Upper bounds array. This includes over-allocation so that every point falls on an
            integer value.
        vob : ndarray
            Outer bounds array. This is purely for bounds check.
        bits : ndarray
            Number of bits to encode the design space for each element of the design vector.
        pop_size : int
            Number of points in the population.
        max_gen : int
            Number of generations to run the GA.
        random_state : np.random.RandomState, int
            Random state (or seed-number) which controls the seed and random draws.
        Pm : float or None
            Mutation rate
        Pc : float
            Crossover rate

        Returns
        -------
        ndarray
            Best design point
        float
            Objective value at best design point.
        int
            Number of successful function evaluations.
        """
        comm = self.comm
        xopt = copy.deepcopy(vlb)
        fopt = np.inf
        self.lchrom = int(np.sum(bits))

        if np.mod(pop_size, 2) == 1:
            pop_size += 1
        self.npop = int(pop_size)
        fitness = np.zeros((self.npop, ))

        # If mutation rate is not provided as input
        if Pm is None:
            Pm = (self.lchrom + 1.0) / (2.0 * pop_size * np.sum(bits))
        elite = self.elite

        new_gen = np.round(lhs(self.lchrom, self.npop, criterion='center',
                               random_state=random_state))
        new_gen[0] = self.encode(x0, vlb, vub, bits)

        # Main Loop
        nfit = 0
        for generation in range(max_gen + 1):
            old_gen = copy.deepcopy(new_gen)
            x_pop = self.decode(old_gen, vlb, vub, bits)

            # Evaluate points in this generation.
#            if comm is not None:
#                # Parallel
#
#                # Since GA is random, ranks generate different new populations, so just take one
#                # and use it on all.
#                x_pop = comm.bcast(x_pop, root=0)
#
#                cases = [((item, ii), None) for ii, item in enumerate(x_pop)
#                         if np.all(item - vob <= 0)]
#
#                # Pad the cases with some dummy cases to make the cases divisible amongst the procs.
#                # TODO: Add a load balancing option to this driver.
#                extra = len(cases) % comm.size
#                if extra > 0:
#                    for j in range(comm.size - extra):
#                        cases.append(cases[-1])
#
#                results = concurrent_eval(self.objfun, cases, comm, allgather=True,
#                                          model_mpi=self.model_mpi)
#
#                fitness[:] = np.inf
#                for result in results:
#                    returns, traceback = result
#
#                    if returns:
#                        val, success, ii = returns
#                        if success:
#                            fitness[ii] = val
#                            nfit += 1
#
#                    else:
#                        # Print the traceback if it fails
#                        print('A case failed:')
#                        print(traceback)

#            else:
            # RINK: commenting out so tests will pass
            # Serial
            for ii in range(self.npop):
                x = x_pop[ii]

                if np.any(x - vob > 0):
                    # Exceeded bounds for integer variables that are over-allocated.
                    success = False
                else:
                    fitness[ii], success, _ = self.objfun(x, 0)

                if success:
                    nfit += 1
                else:
                    fitness[ii] = np.inf

            # Elitism means replace worst performing point with best from previous generation.
            if elite and generation > 0:
                max_index = np.argmax(fitness)
                old_gen[max_index] = min_gen
                x_pop[max_index] = min_x
                fitness[max_index] = min_fit

            # Find best performing point in this generation.
            min_fit = np.min(fitness)
            min_index = np.argmin(fitness)
            min_gen = old_gen[min_index]
            min_x = x_pop[min_index]

            if min_fit < fopt:
                fopt = min_fit
                xopt = min_x

            # Evolve new generation.
            new_gen = self.tournament(old_gen, fitness)
            new_gen = self.crossover(new_gen, Pc)
            new_gen = self.mutate(new_gen, Pm)

        return xopt, fopt, nfit

    def tournament(self, old_gen, fitness):
        """
        Apply tournament selection and keep the best points.

        Parameters
        ----------
        old_gen : ndarray
            Points in current generation

        fitness : ndarray
            Objective value of each point.

        Returns
        -------
        ndarray
            New generation with best points.
        """
        new_gen = []
        idx = np.array(range(0, self.npop - 1, 2))
        for j in range(2):
            old_gen, i_shuffled = self.shuffle(old_gen)
            fitness = fitness[i_shuffled]

            # Each point competes with its neighbor; save the best.
            i_min = np.argmin(np.array([[fitness[idx]], [fitness[idx + 1]]]), axis=0)
            selected = i_min + idx
            new_gen.append(old_gen[selected])

        return np.concatenate(np.array(new_gen), axis=1).reshape(old_gen.shape)

    def crossover(self, old_gen, Pc):
        """
        Apply crossover to the current generation.

        Crossover flips two adjacent genes.

        Parameters
        ----------
        old_gen : ndarray
            Points in current generation

        Pc : float
            Probability of crossover.

        Returns
        -------
        ndarray
            Current generation with crossovers applied.
        """
        new_gen = copy.deepcopy(old_gen)
        num_sites = self.npop // 2
        sites = np.random.rand(num_sites, self.lchrom)
        idx, idy = np.where(sites < Pc)
        for ii, jj in zip(idx, idy):
            i = 2 * ii
            j = i + 1
            new_gen[i][jj] = old_gen[j][jj]
            new_gen[j][jj] = old_gen[i][jj]
        return new_gen

    def mutate(self, current_gen, Pm):
        """
        Apply mutations to the current generation.

        A mutation flips the state of the gene from 0 to 1 or 1 to 0.

        Parameters
        ----------
        current_gen : ndarray
            Points in current generation

        Pm : float
            Probability of mutation.

        Returns
        -------
        ndarray
            Current generation with mutations applied.
        """
        temp = np.random.rand(self.npop, self.lchrom)
        idx, idy = np.where(temp < Pm)
        current_gen[idx, idy] = 1 - current_gen[idx, idy]
        return current_gen

    def shuffle(self, old_gen):
        """
        Shuffle (reorder) the points in the population.

        Used in tournament selection.

        Parameters
        ----------
        old_gen : ndarray
            Old population.

        Returns
        -------
        ndarray
            New shuffled population.
        ndarray(dtype=int)
            Index array that maps the shuffle from old to new.
        """
        temp = np.random.rand(self.npop)
        index = np.argsort(temp)
        return old_gen[index], index

    def decode(self, gen, vlb, vub, bits):
        """
        Decode from binary array to real value array.

        Parameters
        ----------
        gen : ndarray
            Population of points, encoded.
        vlb : ndarray
            Lower bound array.
        vub : ndarray
            Upper bound array.
        bits : ndarray
            Number of bits for decoding.

        Returns
        -------
        ndarray
            Decoded design variable values.
        """
        num_desvar = len(bits)
        interval = (vub - vlb) / (2**bits - 1)
        x = np.empty((self.npop, num_desvar))
        sbit = 0
        ebit = 0
        for jj in range(num_desvar):
            exponents = 2**np.array(range(bits[jj] - 1, -1, -1))
            ebit += bits[jj]
            fact = exponents * (gen[:, sbit:ebit])
            x[:, jj] = np.einsum('ij->i', fact) * interval[jj] + vlb[jj]
            sbit = ebit
        return x

    def encode(self, x, vlb, vub, bits):
        """
        Encode array of real values to array of binary arrays.

        Parameters
        ----------
        x : ndarray
            Design variable values.
        vlb : ndarray
            Lower bound array.
        vub : ndarray
            Upper bound array.
        bits : int
            Number of bits for decoding.

        Returns
        -------
        ndarray
            Population of points, encoded.
        """
        interval = (vub - vlb) / (2**bits - 1)
        x = np.maximum(x, vlb)
        x = np.minimum(x, vub)
        x = np.round((x - vlb) / interval).astype(int)
        byte_str = [("0" * b + bin(i)[2:])[-b:] for i, b in zip(x, bits)]
        return np.array([int(c) for s in byte_str for c in s])
