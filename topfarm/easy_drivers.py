from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.drivers.MySimpleGADriver import MySimpleGADriver


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
            self.opt_settings.update({'linear_solver': 'ma27', 'max_iter': max_iter})


except ModuleNotFoundError:
    class PyOptSparseMissingDriver(object):
        options = {}

    EasyPyOptSparseSLSQP = PyOptSparseMissingDriver
    EasyPyOptSparseIPOPT = PyOptSparseMissingDriver



class EasySimpleGADriver(MySimpleGADriver):
    def __init__(self, max_gen=100, pop_size=25, Pm=None, Pc=.5, elitism=True, bits={}, debug_print=[], run_parallel=False):
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
            Number of bits of resolution. Default is an empty dict, where every unspecified variable is assumed to be integer, and the number of bits is calculated automatically. If you have a continuous var, you should set a bits value as a key in this dictionary.
        debug_print : list, optional
            List of what type of Driver variables to print at each iteration. Valid items in list are ‘desvars’,’ln_cons’,’nl_cons’,’objs’
        run_parallel : bool
            Set to True to execute the points in a generation in parallel.
        """
        MySimpleGADriver.__init__(self, max_gen=max_gen, pop_size=pop_size, Pm=Pm, Pc=Pc, elitism=elitism, bits=bits, debug_print=debug_print, run_parallel=run_parallel)
