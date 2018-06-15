from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver


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


# class EasySimpleGADriver(SimpleGADriver):
#     def __init__(self, elitism=True, max_gen=100):
#         """Simple Genetic Algorithm Driver with argument
#
#         Parameters
#         ----------
#         bits : dict
#             Number of bits of resolution. Default is an empty dict, where every unspecified variable is assumed to be integer, and the number of bits is calculated automatically. If you have a continuous var, you should set a bits value as a key in this dictionary.
#             NotImplemented
#         debug_print : list
#             List of what type of Driver variables to print at each iteration. Valid items in list are 'desvars', 'ln_cons', 'nl_cons', 'objs', 'totals'
#             NotImplemented
#         elitism : bool
#             If True, replace worst performing point with best from previous generation each iteration.
#         max_gen : int
#             Number of generations before termination.
#         pop_size :
#             Number of points in the GA. Set to 0 and it will be computed as four times the number of bits.
#             NotImplemented
#         procs_per_model : int
#             Number of processors to give each model under MPI.
#             NotImplemented
#         run_parallel : bool
#             Set to True to execute the points in a generation in parallel.
#             NotImplemented
#         """
#         SimpleGADriver.__init__(self)
#         self.options.update({'elitism': elitism, 'max_gen': max_gen})

#
# class COBYLADriverWrapper(CONMINdriver):
#         # CONMIN-specific Settings
#         self.driver.itmax = 30
#         self.driver.fdch = 0.00001
#         self.driver.fdchm = 0.000001
#         self.driver.ctlmin = 0.01
#         self.driver.delfun = 0.001
#
#         # NEWSUMT-specific Settings
#         #self.driver.itmax = 10
#
#         # COBYLA-specific Settings
#         #self.driver.rhobeg = 1.0
#         #self.driver.rhoend = 1.0e-4
#         #self.driver.maxfun = 1000
#
#         # SLSQP-specific Settings
#         #self.driver.accuracy = 1.0e-6
#         #self.driver.maxiter = 50
#
#         # Genetic-specific Settings
#         #self.driver.population_size = 90
#         #self.driver.crossover_rate = 0.9
#         #self.driver.mutation_rate = 0.02
#         #self.selection_method = 'rank'
