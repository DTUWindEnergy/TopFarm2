import numpy as np
from topfarm.constraint_components import Constraint, ConstraintComponent
import topfarm


class SpacingConstraint(Constraint):
    def __init__(self, min_spacing):
        self.min_spacing = min_spacing

    @property
    def constraintComponent(self):
        return self.spacing_comp

    def setup_as_constraint(self, problem):
        n_wt = problem.n_wt
        self.spacing_comp = SpacingComp(n_wt, self.min_spacing)
        problem.model.add_subsystem('spacing_comp', self.spacing_comp, promotes=['*'])
        zero = np.zeros(int(((n_wt - 1.) * n_wt / 2.)))
        problem.model.add_constraint('wtSeparationSquared', lower=zero + (self.min_spacing)**2)

    def setup_as_penalty(self, problem, penalty=1e10):
        n_wt = problem.n_wt
        zero = np.zeros(int(((n_wt - 1.) * n_wt / 2.)))
        self.spacing_comp = SpacingComp(n_wt, self.min_spacing)

        def setup():
            self._cost_comp.add_input('wtSeparationSquared', val=zero)

        def penalty(inputs):
            return -np.minimum(inputs['wtSeparationSquared'] - self.min_spacing**2, 0).sum()

        self._setup_as_penalty(problem, 'spacing_comp', self.spacing_comp, setup, penalty)


class SpacingComp(ConstraintComponent):
    """
    Calculates inter-turbine spacing for all turbine pairs.
    Code from wake-exchange module
    """

    def __init__(self, n_wt, min_spacing):
        super().__init__()
        self.n_wt = n_wt
        self.min_spacing = min_spacing

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, val=np.zeros(self.n_wt),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_input(topfarm.y_key, val=np.zeros(self.n_wt),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        # Explicitly size output array
        self.add_output('wtSeparationSquared', val=np.zeros(int((self.n_wt - 1) * self.n_wt / 2)),
                        desc='spacing of all turbines in the wind farm')

        self.declare_partials('wtSeparationSquared', [topfarm.x_key, topfarm.y_key])

    def compute(self, inputs, outputs):
        self.x = inputs[topfarm.x_key]
        self.y = inputs[topfarm.y_key]
        separation_squared = self._compute(self.x, self.y)
        outputs['wtSeparationSquared'] = separation_squared

    def _compute(self, x, y):
        n_wt = self.n_wt
        separation_squared = np.zeros(int((n_wt - 1) * n_wt / 2))
        k = 0
        for i in range(0, n_wt):
            for j in range(i + 1, n_wt):
                separation_squared[k] = (x[j] - x[i])**2 + (y[j] - y[i])**2
                k += 1
        return separation_squared

    def compute_partials(self, inputs, J):
        # obtain necessary inputs
        x = inputs[topfarm.x_key]
        y = inputs[topfarm.y_key]

        dSdx, dSdy = self._compute_partials(x, y)
        # populate Jacobian dict
        J['wtSeparationSquared', topfarm.x_key] = dSdx
        J['wtSeparationSquared', topfarm.y_key] = dSdy

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

    def plot(self, ax):
        from matplotlib.pyplot import Circle
        for x, y in zip(self.x, self.y):
            circle = Circle((x, y), self.min_spacing / 2, color='k', ls='--', fill=False)
            ax.add_artist(circle)

    def satisfy(self, state, n_iter=100, step_size=0.1):
        x, y = [state[xy].astype(np.float) for xy in [topfarm.x_key, topfarm.y_key]]
        for _ in range(n_iter):
            dist = self._compute(x, y)
            dx, dy = self._compute_partials(x, y)
            index = int(np.argmin(dist))
            if dist[index] < self.min_spacing**2:
                x += dx[index] * step_size
                y += dy[index] * step_size
            else:
                break
        state.update({topfarm.x_key: x, topfarm.y_key: y})
        return state
