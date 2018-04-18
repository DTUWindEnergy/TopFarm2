from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np

class SpacingComp(ExplicitComponent):
    """
    Calculates inter-turbine spacing for all turbine pairs.
    Code from wake-exchange module
    """

    def __init__(self, nTurbines):

        super(SpacingComp, self).__init__()
        self.nTurbines = nTurbines

    def setup(self):

        # set finite difference options (fd used for testing only)
        #self.deriv_options['check_form'] = 'central'
        #self.deriv_options['check_step_size'] = 1.0e-5
        #self.deriv_options['check_step_calc'] = 'relative'

        # Explicitly size input arrays
        self.add_input('turbineX', val=np.zeros(self.nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame', units='m')
        self.add_input('turbineY', val=np.zeros(self.nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame', units='m')

        # Explicitly size output array
        self.add_output('wtSeparationSquared', val=np.zeros(int((self.nTurbines - 1) * self.nTurbines / 2)),
                        desc='spacing of all turbines in the wind farm')

        #self.declare_partials('wtSeparationSquared', ['turbineX', 'turbineY'], method='fd')
        self.declare_partials('wtSeparationSquared', ['turbineX', 'turbineY'])

    def compute(self, inputs, outputs):
        # print 'in dist const'

        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']
        nTurbines = self.nTurbines
        separation_squared = np.zeros(int((nTurbines - 1) * nTurbines / 2))

        k = 0
        for i in range(0, nTurbines):
            for j in range(i + 1, nTurbines):
                separation_squared[k] = (turbineX[j] - turbineX[i])**2 + (turbineY[j] - turbineY[i])**2
                k += 1
        outputs['wtSeparationSquared'] = separation_squared
#         print("wt sep")
#         print(separation_squared)
#         print()

    def compute_partials(self, inputs, J):

        # obtain necessary inputs
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        # get number of turbines
        nTurbines = self.nTurbines

        # initialize gradient calculation array
        dS = np.zeros((int((nTurbines - 1.) * nTurbines / 2.), 2 * nTurbines))
        dSdx = np.zeros((int((nTurbines - 1.) * nTurbines / 2.),  nTurbines)) #col: dx_1-dx_n, row: d12, d13,..,d1n, d23..d2n,..
        dSdy = np.zeros((int((nTurbines - 1.) * nTurbines / 2.),  nTurbines))

        # set turbine pair counter to zero
        k = 0

        # calculate the gradient of the distance between each pair of turbines w.r.t. turbineX and turbineY
        for i in range(0, nTurbines):
            for j in range(i + 1, nTurbines):
                # separation wrt Xj
                dS[k, j] = 2 * (turbineX[j] - turbineX[i])
                dSdx[k,j]= 2 * (turbineX[j] - turbineX[i])
                # separation wrt Xi
                dS[k, i] = -2 * (turbineX[j] - turbineX[i])
                dSdx[k, i] = -2 * (turbineX[j] - turbineX[i])
                # separation wrt Yj
                dS[k, j + nTurbines] = 2 * (turbineY[j] - turbineY[i])
                dSdy[k, j] = 2 * (turbineY[j] - turbineY[i])
                # separation wrt Yi
                dS[k, i + nTurbines] = -2 * (turbineY[j] - turbineY[i])
                dSdy[k, i] = -2 * (turbineY[j] - turbineY[i])
                # increment turbine pair counter
                k += 1

        # populate Jacobian dict
        J['wtSeparationSquared', 'turbineX'] = dS[:, :nTurbines]
        J['wtSeparationSquared', 'turbineY'] = dS[:, nTurbines:]
