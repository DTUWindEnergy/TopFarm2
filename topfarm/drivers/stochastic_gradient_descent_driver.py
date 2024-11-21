# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:40:49 2022

@author: mikf
"""

from openmdao.core.driver import Driver, RecordingDebugging
from six import iteritems
import numpy as np
from openmdao.core.analysis_error import AnalysisError
import topfarm
import time
from openmdao.utils.concurrent import concurrent_eval
from openmdao.utils.record_util import create_local_meta


class SGDDriver(Driver):

    def __init__(self, maxiter, **kwargs):
        self.maxiter = maxiter
        super().__init__(**kwargs)

        # What we support
        self.supports['integer_design_vars'] = True
        self.supports['inequality_constraints'] = True

        # What we don't support yet
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = False
        self.supports['active_set'] = False

        self._desvar_idx = {}

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('max_time', 5e10,
                             desc='Maximum time in seconds (set to None to disable)')
        self.options.declare('disp', True,
                             desc='Set to False to prevent printing')
        self.options.declare('run_parallel', False,
                             desc='Flag to run in parallel or not, requires mpi')
        self.options.declare('speedupSGD')
        self.options.declare('sgd_thresh')

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

        comm = self._problem().comm
        if not self.options['run_parallel']:
            comm = None

        if len(self._objs) > 1:
            msg = 'SGDDriver currently does not support multiple objectives.'
            raise RuntimeError(msg)

        self.comm = comm

    def run(self):
        """
        Excute the stochastic gradient descent algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem().model

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
        maxiter = self.maxiter
        max_time = self.options['max_time'] or 1e20
        assert maxiter < 1e20 or max_time < 1e20, "maxiter or max_time must be set"

        disp = self.options['disp']

        abs2prom = model._var_abs2prom['output']
        self.abs2prom = abs2prom

        # Initial Design Vars
        desvar_vals = self.get_design_var_values()
        i = 0
        for name, meta in iteritems(self._designvars):
            size = meta['size']
            x0[i:i + size] = desvar_vals[name]
            i += size

        x1 = x0.copy()
        n_iter = 0

        desvar_info = [(abs2prom[name], *self._desvar_idx[name], lower_bound, upper_bound)
                       for name, meta in iteritems(desvars)]
        desvar_dict = {name: (x0[i:j].copy(), lbound[i:j], ubound[i:j]) for (name, i, j, lbound, ubound) in desvar_info}
        start = time.time()

        self.obj_list = list(self._objs)
        self.con_list = []
        self._dvlist = list(desvar_dict)

        for name, meta in self._cons.items():
            if meta['indices'] is not None:
                meta['size'] = size = meta['indices'].indexed_src_size
            else:
                size = meta['global_size'] if meta['distributed'] else meta['size']
            self.con_list.append(name)
        self.obj_and_con_list = self.obj_list + self.con_list

        start = time.time()
        model._solve_nonlinear()
        jac = self._compute_totals(of=self.obj_and_con_list, wrt=self._dvlist,
                                   return_format='array')
        jac_time = time.time() - start
        max_iters_from_max_time = int(max_time / jac_time)
        self.maxiter = min(self.maxiter, max_iters_from_max_time)
        j = jac[0]
        learning_rate = float(np.copy(self.learning_rate))
        self.alpha0 = np.mean(np.abs(j)) / learning_rate
        self.l0 = float(np.copy(learning_rate))
        alpha = float(np.copy(self.alpha0))

        # set delta
        def multf(t, delta):
            prod = 1
            for ii in range(t):
                prod *= 1 / (1 + delta * ii)
            return learning_rate * prod

        for _ in range(self.maxiter):
            mid = np.mean([self.lower, self.upper])
            etaM = multf(self.maxiter, mid)
            if etaM < self.gamma_min:
                self.upper = mid
            elif etaM > self.gamma_min:
                self.lower = mid
        self.mid = mid

        m = np.zeros(x0.size)
        v = np.zeros(x0.size)
        self.is_converged = False
        while (n_iter < self.maxiter) and (not self.is_converged):
            obj_value_x1, x1, alpha, learning_rate, m, v, success = self.objective_callback(x1, alpha, learning_rate, m, v, record=True)
            n_iter += 1
            if disp:
                print(n_iter, obj_value_x1)
        return False

    def objective_callback(self, x, alpha, learning_rate, m, v, record=False):
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
        model = self._problem().model
        success = 1

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # Execute the model
        # if record:
        with RecordingDebugging('SGD', self.iter_count, self) as rec:
            # self.set_design_var(name, value)
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

            # epsilon (1e-7) to avoid division by zero
            only_cons = self.alpha0 / (alpha + 1e-7) < self.options['sgd_thresh'] and self.options['speedupSGD']

            if only_cons:
                of_list = self.con_list
            else:
                of_list = self.obj_and_con_list

            self._total_jac = None
            outputs = []
            for subsys in model._subsystems_myproc:
                # check if all requested of are accounted for
                if all(self.abs2prom[x] in outputs for x in of_list):
                    subsys.skip_linearize = True
                    # print('skipped: ' + subsys.name)
                else:
                    subsys.skip_linearize = False
                outputs += list(set([b['prom_name'] for a, b in subsys.list_outputs(val=False, prom_name=True, out_stream=None)]))

            # tic = time.time()
            jac = self._compute_totals(of=of_list, wrt=self._dvlist, return_format='array')
            # print(f'time of {of_list}: {time.time() - tic:.5f}')

            if only_cons:
                c = jac[0]
                j = np.zeros_like(c)
                if c.sum() == 0:
                    self.is_converged = True
                    return obj, x, alpha, learning_rate, m, v, success
            else:
                j = jac[0]
                c = jac[1]

            # jacobian is objective gradient + alpha C
            jacobian = j + alpha * c

            # adam
            m = self.beta1 * m + (1 - self.beta1) * jacobian
            v = self.beta2 * v + (1 - self.beta2) * jacobian ** 2
            mhat = m / (1 - self.beta1 ** self.iter_count)
            vhat = v / (1 - self.beta2 ** self.iter_count)

            # update x
            x -= learning_rate * mhat / np.sqrt(vhat)

            # update learning rate and constraint scaler
            learning_rate *= 1. / (1 + self.mid * self.iter_count)
            alpha = self.alpha0 * self.l0 / learning_rate

            # validate s
            if np.any(np.isnan(x)):
                raise Exception("NaN in design Variables")

            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        return obj, x, alpha, learning_rate, m, v, success
