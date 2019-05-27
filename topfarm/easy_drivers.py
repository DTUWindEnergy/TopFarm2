from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
# from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver  # version 2.5.0 has bug see Issue #874
from topfarm.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.drivers.random_search_driver import RandomSearchDriver
import sys


class EasyScipyOptimizeDriver(ScipyOptimizeDriver):

    def __init__(self, optimizer='SLSQP', maxiter=200, tol=1e-6, disp=True):
        """
        Parameters
        ----------
        optimizer : {'COBYLA', 'SLSQP'}
            Gradients are only supported by SLSQP
        maxiter : int
            Maximum number of iterations.
        tol : float
            Tolerance for termination. For detailed control, use solver-specific options.
        disp : bool
            Set to False to prevent printing of Scipy convergence messages
        """
        ScipyOptimizeDriver.__init__(self)
        self.options.update({'optimizer': optimizer, 'maxiter': maxiter, 'tol': tol, 'disp': disp})


try:
    class PyOptSparseMissingDriver(object):
        options = {}

    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
#     Not working:
#     capi_return is NULL
#     Call-back cb_slfunc_in_slsqp__user__routines failed.
#
#     class EasyPyOptSparseSLSQP(pyOptSparseDriver):
#         def __init__(self, maxit=200, acc=1e-6):
#             pyOptSparseDriver.__init__(self)
#             self.options.update({'optimizer': 'SLSQP'})
#             self.opt_settings.update({'MAXIT': maxit, 'ACC': acc})

    class EasyPyOptSparseIPOPT(pyOptSparseDriver):
        def __init__(self, max_iter=200):
            pyOptSparseDriver.__init__(self)
            self.options.update({'optimizer': 'IPOPT'})
            self.opt_settings.update({'linear_solver': 'ma27', 'max_iter': max_iter,
                                      'start_with_resto': 'yes',
                                      'expect_infeasible_problem': 'yes'})

    from pyoptsparse.pyIPOPT.pyIPOPT import pyipoptcore
    if pyipoptcore is None:
        setattr(sys.modules[__name__], 'EasyPyOptSparseIPOPT', PyOptSparseMissingDriver)

    class EasyPyOptSparseSNOPT(pyOptSparseDriver):
        def __init__(self, major_iteration_limit=200, major_feasibility_tolerance=1e-6, major_optimality_tolerance=1e-6, difference_interval=1e-6, function_precision=1e-8, print_results=False):
            pyOptSparseDriver.__init__(self)
            self.options.update({'optimizer': 'SNOPT', 'print_results': print_results})
            self.opt_settings.update({
                'Major feasibility tolerance': major_feasibility_tolerance,
                'Major optimality tolerance': major_optimality_tolerance,
                'Difference interval': difference_interval,
                'Hessian full memory': None,
                'Function precision': function_precision,
                'Major iterations limit': major_iteration_limit,
                'Major step limit': 2.0})


except ModuleNotFoundError:
    for n in ['EasyPyOptSparseSLSQP', 'EasyPyOptSparseIPOPT', 'EasyPyOptSparseSNOPT']:
        setattr(sys.modules[__name__], n, PyOptSparseMissingDriver)


class EasySimpleGADriver(SimpleGADriver):
    def __init__(self, max_gen=100, pop_size=25, Pm=None, Pc=.5, elitism=True, bits={}, debug_print=[], run_parallel=False, random_state=None):
        """SimpleGA driver with optional arguments

        Parameters
        ----------
        max_gen : int
            Number of generations before termination.
        pop_size : int
            Number of points in the GA.
        pm : float
            Probability of mutation.
        pc : float
             Probability of cross over.
        elitism : bool, optional
            If True, replace worst performing point with best from previous generation each iteration.
        bits : dict, optional
            Number of bits of resolution. Default is an empty dict, where every
            unspecified variable is assumed to be integer, and the number of
            bits is calculated automatically.
            If you have a continuous var, you should set a bits value as a key
            in this dictionary.
            Ex: {'x':16,'y':16}
        debug_print : list, optional
            List of what type of Driver variables to print at each iteration. Valid items in list are ‘desvars’,’ln_cons’,’nl_cons’,’objs’
        run_parallel : bool
            Set to True to execute the points in a generation in parallel.
        """
        SimpleGADriver.__init__(self, max_gen=max_gen, pop_size=pop_size, Pm=Pm, Pc=Pc, elitism=elitism,
                                bits=bits, debug_print=debug_print, run_parallel=run_parallel)
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        if random_state is not None:
            self._randomstate = random_state


class EasyRandomSearchDriver(RandomSearchDriver):
    def __init__(self, randomize_func, max_iter=100, max_time=600, disp=False):
        """Easy initialization of RandomSearchDriver

        Parameters
        ----------
        randomize_func : f(desvar_dict)
            Function to randomize desired variables of desvar_dict
        max_iter : int, optional
            Maximum iterations
        max_time : int, optional
            Maximum time in seconds
        disp : bool
        """
        RandomSearchDriver.__init__(self, randomize_func=randomize_func,
                                    max_iter=max_iter, max_time=max_time, disp=disp)
