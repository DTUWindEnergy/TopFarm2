from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np


class SpacingComp(ExplicitComponent):
    """
    Calculates inter-turbine spacing for all turbine pairs.
    Code from wake-exchange module
    """

    def __init__(self, n_wt, min_spacing):
        super(SpacingComp, self).__init__()
        self.n_wt = n_wt
        self.min_spacing = min_spacing

    def setup_as_constraints(self, problem):
        if self.min_spacing is not None:
            problem.model.add_subsystem('spacing_comp', self, promotes=['*'])
            zero = np.zeros(int(((self.n_wt - 1.) * self.n_wt / 2.)))
            problem.model.add_constraint('wtSeparationSquared', lower=zero + (self.min_spacing)**2)

    def setup_as_penalty(self, problem, penalty=1e10):
        if self.min_spacing is not None:
            subsystem_order = [ss.name for ss in problem.model._static_subsystems_allprocs]
            problem.model.add_subsystem('spacing_comp', self, promotes=['*'])
            subsystem_order.insert(subsystem_order.index('cost_comp'), 'spacing_comp')
            problem.model.set_order(subsystem_order)

            zero = np.zeros(int(((self.n_wt - 1.) * self.n_wt / 2.)))

            self._cost_comp = problem.cost_comp
            self._org_setup = self._cost_comp.setup
            self._org_compute = self._cost_comp.compute

            def new_setup():
                self._org_setup()
                self._cost_comp.add_input('wtSeparationSquared', val=zero)

            self._cost_comp.setup = new_setup

            def new_compute(inputs, outputs):
                p = -np.minimum(inputs['wtSeparationSquared'] - self.min_spacing**2, 0).sum()
                if p == 0:
                    self._org_compute(inputs, outputs)
                else:
                    outputs['cost'] = penalty + p
            self._cost_comp.compute = new_compute

    def setup(self):

        # set finite difference options (fd used for testing only)
        # self.deriv_options['check_form'] = 'central'
        # self.deriv_options['check_step_size'] = 1.0e-5
        # self.deriv_options['check_step_calc'] = 'relative'

        # Explicitly size input arrays
        self.add_input('turbineX', val=np.zeros(self.n_wt),
                       desc='x coordinates of turbines in wind dir. ref. frame', units='m')
        self.add_input('turbineY', val=np.zeros(self.n_wt),
                       desc='y coordinates of turbines in wind dir. ref. frame', units='m')

        # Explicitly size output array
        self.add_output('wtSeparationSquared', val=np.zeros(int((self.n_wt - 1) * self.n_wt / 2)),
                        desc='spacing of all turbines in the wind farm')

        # self.declare_partials('wtSeparationSquared', ['turbineX', 'turbineY'], method='fd')
        self.declare_partials('wtSeparationSquared', ['turbineX', 'turbineY'])

    def compute(self, inputs, outputs):
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']
        separation_squared = self._compute(turbineX, turbineY)
        outputs['wtSeparationSquared'] = separation_squared

    def _compute(self, turbineX, turbineY):
        n_wt = self.n_wt
        separation_squared = np.zeros(int((n_wt - 1) * n_wt / 2))
        k = 0
        for i in range(0, n_wt):
            for j in range(i + 1, n_wt):
                separation_squared[k] = (turbineX[j] - turbineX[i])**2 + (turbineY[j] - turbineY[i])**2
                k += 1
        return separation_squared

    def compute_partials(self, inputs, J):
        # obtain necessary inputs
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        dSdx, dSdy = self._compute_partials(turbineX, turbineY)
        # populate Jacobian dict
        J['wtSeparationSquared', 'turbineX'] = dSdx
        J['wtSeparationSquared', 'turbineY'] = dSdy

    def _compute_partials(self, turbineX, turbineY):
        # get number of turbines
        n_wt = self.n_wt

        # initialize gradient calculation array
        dSdx = np.zeros((int((n_wt - 1.) * n_wt / 2.), n_wt))  # col: dx_1-dx_n, row: d12, d13,..,d1n, d23..d2n,..
        dSdy = np.zeros((int((n_wt - 1.) * n_wt / 2.), n_wt))

        # set turbine pair counter to zero
        k = 0

        # calculate the gradient of the distance between each pair of turbines w.r.t. turbineX and turbineY
        for i in range(0, n_wt):
            for j in range(i + 1, n_wt):
                # separation wrt Xj
                dSdx[k, j] = 2 * (turbineX[j] - turbineX[i])
                # separation wrt Xi
                dSdx[k, i] = -2 * (turbineX[j] - turbineX[i])
                # separation wrt Yj
                dSdy[k, j] = 2 * (turbineY[j] - turbineY[i])
                # separation wrt Yi
                dSdy[k, i] = -2 * (turbineY[j] - turbineY[i])
                # increment turbine pair counter
                k += 1
        return dSdx, dSdy
