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
