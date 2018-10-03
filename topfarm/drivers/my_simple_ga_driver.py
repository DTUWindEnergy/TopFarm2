from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver,\
    GeneticAlgorithm

"""
Temporary solution until https://github.com/OpenMDAO/OpenMDAO/pull/756 is merged
"""
import copy

from six import iteritems


import numpy as np
from pyDOE2 import lhs

from openmdao.core.driver import RecordingDebugging
from openmdao.utils.concurrent import concurrent_eval


class MySimpleGADriver(SimpleGADriver):

    def _declare_options(self):
        SimpleGADriver._declare_options(self)
        self.options.declare('Pm', default=None,
                             desc='Probability of mutation.')
        self.options.declare('Pc', default=.5,
                             desc='Probability of cross over.')

    def _setup_driver(self, problem):
        SimpleGADriver._setup_driver(self, problem)
        model_mpi = None
        comm = self._problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options['run_parallel']:
            comm = None

        self._ga = MyGeneticAlgorithm(self.objective_callback, comm=comm, model_mpi=model_mpi)

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

        # Figure out bounds vectors.
        for name, meta in iteritems(desvars):
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta['lower']
            upper_bound[i:j] = meta['upper']

        ga.elite = self.options['elitism']
        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        user_bits = self.options['bits']
        Pm = self.options['Pm']
        Pc = self.options['Pc']

        # Size Problem
        nparam = 0
        for param in iter(self._designvars.values()):
            nparam += param['size']
        x_init = np.empty(nparam)

        # Initial Design Vars
        desvar_vals = self.get_design_var_values()
        i = 0
        for name, meta in iteritems(self._designvars):
            size = meta['size']
            x_init[i:i + size] = desvar_vals[name]
            i += size

        # Bits of resolution
        bits = np.ceil(np.log2(upper_bound - lower_bound + 1)).astype(int)
        prom2abs = model._var_allprocs_prom2abs_list['output']

        for name, val in iteritems(user_bits):
            try:
                i, j = self._desvar_idx[name]
            except KeyError:
                abs_name = prom2abs[name][0]
                i, j = self._desvar_idx[abs_name]

            bits[i:j] = val

        # Automatic population size.
        if pop_size == 0:
            pop_size = 4 * np.sum(bits)

        desvar_new, obj, nfit = ga.execute_ga(x_init, lower_bound, upper_bound,
                                              bits, pop_size, max_gen, Pm, Pc,
                                              self._randomstate)

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


class MyGeneticAlgorithm(GeneticAlgorithm):

    def execute_ga(self, x_init, vlb, vub, bits, pop_size, max_gen, Pm, Pc, random_state):
        """
        Perform the genetic algorithm.

        Parameters
        ----------
        x_init : ndarray
            initial state
        vlb : ndarray
            Lower bounds array.
        vub : ndarray
            Upper bounds array.
        bits : ndarray
            Number of bits to encode the design space for each element of the design vector.
        pop_size : int
            Number of points in the population.
        max_gen : int
            Number of generations to run the GA.
        random_state : np.random.RandomState, int
             Random state (or seed-number) which controls the seed and random draws.

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

        if Pm is None:
            Pm = (self.lchrom + 1.0) / (2.0 * pop_size * np.sum(bits))
        elite = self.elite

        new_gen = np.round(lhs(self.lchrom, self.npop, criterion='center',
                               random_state=random_state))
        new_gen[0] = self.encode(x_init, vlb, vub, bits)

        # Main Loop
        nfit = 0
        for generation in range(max_gen + 1):
            old_gen = copy.deepcopy(new_gen)
            x_pop = self.decode(old_gen, vlb, vub, bits)

            # Evaluate points in this generation.
            if comm is not None:
                # Parallel

                # Since GA is random, ranks generate different new populations, so just take one
                # and use it on all.
                x_pop = comm.bcast(x_pop, root=0)

                cases = [((item, ii), None) for ii, item in enumerate(x_pop)]

                results = concurrent_eval(self.objfun, cases, comm, allgather=True,
                                          model_mpi=self.model_mpi)

                fitness[:] = np.inf
                for result in results:
                    returns, traceback = result

                    if returns:
                        val, success, ii = returns
                        if success:
                            fitness[ii] = val
                            nfit += 1

                    else:
                        # Print the traceback if it fails
                        print('A case failed:')
                        print(traceback)

            else:
                # Serial
                for ii in range(self.npop):
                    x = x_pop[ii]
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
            print("Generation ", generation)

        return xopt, fopt, nfit

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
        gen = np.empty(self.lchrom)
        x = np.maximum(x, vlb)
        x = np.minimum(x, vub)
        x = np.round((x - vlb) / interval).astype(np.int)

        byte_str = [("0" * b + bin(i)[2:])[-b:] for i, b in zip(x, bits)]
        return np.array([int(c) for s in byte_str for c in s])


if __name__ == '__main__':
    array = np.array
    gen = array([array([0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1.,
                        1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0.,
                        1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0.,
                        0., 1., 0.]),
                 array([1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 1.,
                        1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0.,
                        1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1.,
                        0., 1., 0.]),
                 array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.,
                        1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
                        1., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
                        1., 1., 1.]),
                 array([0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0.,
                        0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1.,
                        0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0.,
                        1., 0., 1.]),
                 array([1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0.,
                        0., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1.,
                        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1.,
                        1., 0., 1.]),
                 array([0., 1., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 0.,
                        1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
                        0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
                        0., 0., 0.]),
                 array([1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.,
                        0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
                        0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 1.,
                        1., 0., 0.]),
                 array([1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0.,
                        0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0.,
                        1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 0.,
                        0., 1., 1.])])
    vlb = array([-170.0, -170.0, -170.0, -170.0, -170.0, -170.0])
    vub = array([255.0, 255.0, 255.0, 170.0, 170.0, 170.0])
    bits = array([9, 9, 9, 9, 9, 9])
    x = [array([-69.36399217, 22.12328767, -7.81800391, -66.86888454,
                116.77103718, 76.18395303]),
         array([248.34637965, 191.79060665, -31.93737769, 97.47553816,
                118.76712329, 92.15264188]),
         array([-163.34637965, -27.77886497, -98.47358121, -166.00782779,
                -87.49510763, -128.08219178]),
         array([-160.85127202, 52.8962818, -133.40508806, 78.18003914,
                -108.12133072, -38.92367906]),
         array([188.46379648, 112.77886497, 143.5518591, -66.2035225,
                -126.75146771, -33.60078278]),
         array([34.59882583, 10.47945205, 183.47358121, -105.45988258,
                44.24657534, 42.91585127]),
         array([152.70058708, -111.78082192, 218.40508806, 157.35812133,
                -72.85714286, 125.42074364]),
         array([109.45205479, 89.49119374, 66.2035225, 71.52641879,
                115.44031311, -136.0665362])]

    def objfun(*args, **kwargs):
        return 1

    ga = MyGeneticAlgorithm(objfun)
    ga.npop = 8
    ga.lchrom = int(np.sum(bits))
    np.testing.assert_array_almost_equal(x, ga.decode(gen, vlb, vub, bits))

    np.testing.assert_array_almost_equal(gen, ga.encode(x, vlb, vub, bits))
