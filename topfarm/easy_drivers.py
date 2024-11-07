from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
# from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver  # version 2.5.0 has bug see Issue #874
from topfarm.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.drivers.random_search_driver import RandomSearchDriver
from topfarm.drivers.stochastic_gradient_descent_driver import SGDDriver
import sys
import numpy as np
import openmdao
import scipy


class EasyDriverBase():
    expected_cost = 1
    max_iter = None

    def get_desvar_kwargs(self, model, desvar_name, desvar_values):
        if len(desvar_values) == 4:
            kwargs = {'lower': desvar_values[1], 'upper': desvar_values[2]}
        else:
            kwargs = {}
        return kwargs

    @property
    def supports_expected_cost(self):
        return True


class EasyScipyOptimizeDriver(ScipyOptimizeDriver, EasyDriverBase):

    def __init__(self, optimizer='SLSQP', maxiter=200, tol=1e-8, disp=True, auto_scale=False, **kwargs):
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
        auto_scale : bool
            Set to true to set ref0 and ref1 to boundaries of the desig variables for the drivers which support it (SLSQP, ).
        """
        ScipyOptimizeDriver.__init__(self)
        self.auto_scale = auto_scale

        if optimizer in ["IPOPT", "SGD"] or optimizer not in ["COBYLA", "SLSQP"]:
            raise RuntimeError(
                f"Optimizer '{optimizer}' is not supported. "
                "For IPOPT use EasyPyOptSparseIPOPT, "
                "for SGD use EasySGDDriver."
            )

#         # TODO: remove
#         def fmt_option(v):
#             if isinstance(v, str):
#                 return v.encode()
#             else:
#                 return v
#         if optimizer == 'IPOPT': # pragma: no cover
#             try:
#                 from cyipopt.scipy_interface import minimize_ipopt
#             except ImportError:
#                 raise ImportError("""Cannot import ipopt wrapper. Please install cyipopt, e.g. via conda
# Windows: conda install -c pycalphad cyipopt
# Linux/OSX: conda install -c conda-forge cyipopt
#                 """)
#             ipopt_options = {k: fmt_option(v) for k, v in kwargs.items()}
#             def minimize_ipopt_wrapper(*args, maxiter=200, disp=True, **kwargs):
#                 from cyipopt.scipy_interface import minimize_ipopt
#                 ipopt_options.update({'max_iter': self.max_iter or maxiter, 'print_level': int(disp)})
#                 return minimize_ipopt(*args, options=ipopt_options, **kwargs)
#             kwargs = {}
#             from openmdao.drivers import scipy_optimizer
#             for lst in [scipy_optimizer._optimizers, scipy_optimizer._gradient_optimizers, scipy_optimizer._bounds_optimizers,
#                         scipy_optimizer._all_optimizers, scipy_optimizer._constraint_optimizers, scipy_optimizer._constraint_grad_optimizers]:
#                 lst.add(minimize_ipopt_wrapper)
#             optimizer = minimize_ipopt_wrapper
#         if optimizer == 'SGD': # pragma: no cover
#             from topfarm.drivers.SGD import SGD
#             from openmdao.drivers import scipy_optimizer
#             from scipy.optimize.optimize import OptimizeResult
#             sgd_options = {k: fmt_option(v) for k, v in kwargs.items()}
#             sgd_opt = SGD(**kwargs)

#             def minimize_sgd_wrapper(*args, maxiter=200, disp=True, **kwargs):
#                 sgd_options.update({'max_iter': self.max_iter or maxiter, 'print_level': int(disp)})
#                 s = sgd_opt.run(*args, options=sgd_options, **kwargs)
#                 return OptimizeResult(x=s, fun=args[0](s), jac=kwargs['jac'](s), nit=int(sgd_opt.T),
#                                       nfev=None, njev=None, status=1,
#                                       message='hello world!', success=1)
#             kwargs = {}
#             for lst in [scipy_optimizer._optimizers, scipy_optimizer._gradient_optimizers, scipy_optimizer._bounds_optimizers,
#                         scipy_optimizer._all_optimizers, scipy_optimizer._constraint_optimizers, scipy_optimizer._constraint_grad_optimizers]:
#                 lst.add(minimize_sgd_wrapper)
#             optimizer = minimize_sgd_wrapper

        self.options.update({'optimizer': optimizer, 'maxiter': self.max_iter or maxiter, 'tol': tol, 'disp': disp})
        if kwargs:
            self.opt_settings.update(kwargs)

    def get_desvar_kwargs(self, model, desvar_name, desvar_values):
        kwargs = super().get_desvar_kwargs(model, desvar_name, desvar_values)
        if self.options['optimizer'] == 'SLSQP':
            if tuple([int(v) for v in scipy.__version__.split(".")]) < (1, 5, 0):
                # Upper and lower disturbs SLSQP when running with constraints. Add limits as constraints
                model.add_constraint(desvar_name, kwargs.get('lower', None), kwargs.get('upper', None))
                kwargs = {'lower': np.nan, 'upper': np.nan}  # Default +/- sys.float_info.max does not work for SLSQP

            ref0 = 0
            ref1 = 1

            if self.auto_scale:
                if len(desvar_values) == 4:
                    ref0 = np.min(desvar_values[1])
                    ref1 = np.max(desvar_values[2])

            kwargs.update({'ref0': ref0, 'ref': ref1})
        elif openmdao.__version__ != '2.6.0' and self.options['optimizer'] == 'COBYLA':
            # COBYLA does not work with ref-setting in openmdao 2.6.0.
            # See issue on Github: https://github.com/OpenMDAO/OpenMDAO/issues/942
            if len(desvar_values) == 4:
                ref0 = np.min(desvar_values[1])
                ref1 = np.max(desvar_values[2])
                # l, u = [lu * (ref1 - ref0) + ref0 for lu in [desvar_values[1], desvar_values[2]]]
                l, u = desvar_values[1], desvar_values[2]
                kwargs = {'ref0': ref0, 'ref': ref1, 'lower': l, 'upper': u}
        return kwargs

    @property
    def supports_expected_cost(self):
        return not (openmdao.__version__ == '2.6.0' and self.options['optimizer'] == 'COBYLA')

    def _get_name(self):
        """Override to add str"""
        return "ScipyOptimize_" + str(self.options['optimizer'])


class EasyIPOPTScipyOptimizeDriver(EasyScipyOptimizeDriver):  # pragma: no cover
    def __init__(self, maxiter=200, tol=1e-8, disp=True,
                 max_cpu_time=1e6,  # : Maximum number of CPU seconds.
                 # A limit on CPU seconds that Ipopt can use to solve one problem. If
                 # during the convergence check this limit is exceeded, Ipopt will
                 # terminate with a corresponding error message. The valid range for this
                 # real option is 0 < max_cpu_time and its default value is 10+06.
                 mu_strategy='monotone',  # : Update strategy for barrier parameter.
                 # Determines which barrier parameter update strategy is to be used. The default value for this string option is "monotone".
                 # Possible values:
                 # - monotone: use the monotone (Fiacco-McCormick) strategy
                 # - adaptive: use the adaptive update strategy
                 acceptable_tol=1e-6,  # : "Acceptable" convergence tolerance (relative).
                 # Determines which (scaled) overall optimality error is considered to be
                 # "acceptable". There are two levels of termination criteria. If the usual
                 # "desired" tolerances (see tol, dual_inf_tol etc) are satisfied at an
                 # iteration, the algorithm immediately terminates with a success message.
                 # On the other hand, if the algorithm encounters "acceptable_iter" many
                 # iterations in a row that are considered "acceptable", it will terminate
                 # before the desired convergence tolerance is met. This is useful in cases
                 # where the algorithm might not be able to achieve the "desired" level of
                 # accuracy. The valid range for this real option is 0 < acceptable_tol and
                 # its default value is 10-06.

                 # All options (https://coin-or.github.io/Ipopt/OPTIONS.html) can be specified via kwargs
                 # The argument type must be correct (str, float or int)
                 **kwargs
                 ):
        raise RuntimeError("Deprecated")
        EasyScipyOptimizeDriver.__init__(self, optimizer='IPOPT', maxiter=self.max_iter or maxiter, tol=tol, disp=disp,
                                         max_cpu_time=float(max_cpu_time),
                                         mu_strategy=mu_strategy,
                                         acceptable_tol=acceptable_tol,
                                         ** kwargs)


class PyOptSparseMissingDriver(object):
    options = {}


try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
    #     Not working:
    #     capi_return is NULL
    #     Call-back cb_slfunc_in_slsqp__user__routines failed.
    #     class EasyPyOptSparseSLSQP(pyOptSparseDriver):
    #         def __init__(self, maxit=200, acc=1e-6):
    #             pyOptSparseDriver.__init__(self)
    #             self.options.update({'optimizer': 'SLSQP'})
    #             self.opt_settings.update({'MAXIT': maxit, 'ACC': acc})

    class EasyPyOptSparseIPOPT(pyOptSparseDriver):
        def __init__(self, max_iter=200):
            pyOptSparseDriver.__init__(self)
            self.options.update({'optimizer': 'IPOPT'})
            self.opt_settings.update(
                {"max_iter": max_iter, "expect_infeasible_problem": "yes"}
            )

    from pyoptsparse.pyIPOPT.pyIPOPT import pyipoptcore
    if pyipoptcore is None:
        setattr(sys.modules[__name__], 'EasyPyOptSparseIPOPT', PyOptSparseMissingDriver)
except (ModuleNotFoundError, Exception) as e:
    for n in ["EasyPyOptSparseSNOPT"]:
        setattr(sys.modules[__name__], n, PyOptSparseMissingDriver)


try:
    # Test if the SNOPT optimizer is available in the installation. This gives an exception even if pyOptSparseDriver is successfully instantiated with 'SNOPT'
    _tmp = __import__('pyoptsparse', globals(), locals(), ['SNOPT'], 0)
    getattr(_tmp, 'SNOPT')()

    class EasyPyOptSparseSNOPT(pyOptSparseDriver, EasyDriverBase):
        def __init__(self, major_iteration_limit=200, major_feasibility_tolerance=1e-6, major_optimality_tolerance=1e-6,
                     difference_interval=1e-6, function_precision=1e-8,
                     Print_file='SNOPT_print.out', Summary_file='SNOPT_summary.out', print_results=False):
            """For information about the arguments see
            https://web.stanford.edu/group/SOL/software/snoptHelp/whgdata/whlstt9.htm#9
            """

            pyOptSparseDriver.__init__(self)
            self.options.update({'optimizer': 'SNOPT', 'print_results': print_results})
            self.opt_settings.update({
                'Major feasibility tolerance': major_feasibility_tolerance,
                'Major optimality tolerance': major_optimality_tolerance,
                'Difference interval': difference_interval,
                'Hessian full memory': None,
                'Function precision': function_precision,
                'Major iterations limit': self.max_iter or major_iteration_limit,
                'Print file': Print_file,
                'Summary file': Summary_file,
                'Major step limit': 2.0})

        def get_desvar_kwargs(self, model, desvar_name, desvar_values):
            kwargs = EasyDriverBase.get_desvar_kwargs(self, model, desvar_name, desvar_values)
            if len(desvar_values) == 4:
                ref0 = np.min(desvar_values[1])
                ref1 = np.max(desvar_values[2])
                l, u = desvar_values[1], desvar_values[2]
                kwargs = {'ref0': ref0, 'ref': ref1, 'lower': l, 'upper': u}
            return kwargs
except (ModuleNotFoundError, Exception) as e:
    for n in ["EasyPyOptSparseSNOPT"]:
        setattr(sys.modules[__name__], n, PyOptSparseMissingDriver)


class EasySimpleGADriver(SimpleGADriver, EasyDriverBase):
    def __init__(self, max_gen=100, pop_size=25, Pm=None, Pc=.5, elitism=True,
                 bits={}, debug_print=[], run_parallel=False, random_state=None):
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
        SimpleGADriver.__init__(self, max_gen=self.max_iter or max_gen, pop_size=pop_size, Pm=Pm, Pc=Pc, elitism=elitism,
                                bits=bits, debug_print=debug_print, run_parallel=run_parallel)
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        if random_state is not None:
            self._randomstate = random_state


class EasyRandomSearchDriver(RandomSearchDriver, EasyDriverBase):
    def __init__(self, randomize_func, max_iter=100, max_time=600, disp=False, run_parallel=False):
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
                                    max_iter=self.max_iter or max_iter, max_time=max_time, disp=disp, run_parallel=run_parallel)


class EasySGDDriver(SGDDriver, EasyDriverBase):
    def __init__(self, maxiter=100, max_time=600, disp=False, run_parallel=False,
                 learning_rate=10, upper=0.1, lower=0, beta1=0.1, beta2=0.2, gamma_min_factor=1e-2,
                 speedupSGD=False, sgd_thresh=0.1):
        """Easy initialization of Stochastic Gradient Descent (SGD) Driver

        Parameters
        ----------
        maxiter : int, optional
            Maximum iterations
        max_time : int
            Maximum evaluation time in seconds
        learning_rate : int, optional
            determines the step size
        gamma_min_factor : int, optional
            initial value for constraint aggregation multiplier
        """
        # self.T = T
        self.learning_rate = learning_rate
        self.upper = upper
        self.lower = lower
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma_min_factor = gamma_min_factor
        self.gamma_min = gamma_min_factor  # * learning_rate
        SGDDriver.__init__(self, maxiter=maxiter, max_time=max_time, disp=disp, run_parallel=run_parallel,
                           speedupSGD=speedupSGD, sgd_thresh=sgd_thresh)
