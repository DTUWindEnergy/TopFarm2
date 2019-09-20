import numpy as np
from topfarm.constraint_components import Constraint, ConstraintComponent
import topfarm


class SpacingConstraint(Constraint):
    def __init__(self, min_spacing, units=None):
        """Initialize SpacingConstraint

        Parameters
        ----------
        min_spacing : int or float
            Minimum spacing between turbines [m]
        """
        self.min_spacing = min_spacing
        self.const_id = 'spacing_comp_{}'.format(int(min_spacing))
        self.units = units

    @property
    def constraintComponent(self):
        return self.spacing_comp

    def _setup(self, problem):
        self.n_wt = problem.n_wt
        self.spacing_comp = SpacingComp(self.n_wt, self.min_spacing, self.const_id, self.units)
        problem.model.pre_constraints.add_subsystem(self.const_id, self.spacing_comp,
                                                    promotes=[topfarm.x_key, topfarm.y_key, 'penalty_' + self.const_id, 'wtSeparationSquared'])

    def setup_as_constraint(self, problem):
        self._setup(problem)
        zero = np.zeros(int(((self.n_wt - 1.) * self.n_wt / 2.)))
        problem.model.add_constraint('wtSeparationSquared', lower=zero + (self.min_spacing)**2)

    def setup_as_penalty(self, problem):
        self._setup(problem)


class SpacingComp(ConstraintComponent):
    """
    Calculates inter-turbine spacing for all turbine pairs.

    """

    def __init__(self, n_wt, min_spacing, const_id=None, units=None):
        super().__init__()
        self.n_wt = n_wt
        self.min_spacing = min_spacing
        self.const_id = const_id
        self.veclen = int((n_wt - 1.) * n_wt / 2.)
        self.units = units

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, val=np.zeros(self.n_wt),
                       desc='x coordinates of turbines in wind dir. ref. frame', units=self.units)
        self.add_input(topfarm.y_key, val=np.zeros(self.n_wt),
                       desc='y coordinates of turbines in wind dir. ref. frame', units=self.units)
        self.add_output('penalty_' + self.const_id, val=0.0)
        # Explicitly size output array
        self.add_output('wtSeparationSquared', val=np.zeros(self.veclen),
                        desc='spacing of all turbines in the wind farm')
        # Sparse partial declaration
        cols = np.array([(i, j) for i in range(self.n_wt - 1)
                         for j in range(i + 1, self.n_wt)]).flatten()
        rows = np.repeat(np.arange(self.veclen), 2)

        self.declare_partials('wtSeparationSquared',
                              [topfarm.x_key, topfarm.y_key],
                              rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        self.x = inputs[topfarm.x_key]
        self.y = inputs[topfarm.y_key]
        separation_squared = self._compute(self.x, self.y)
        outputs['wtSeparationSquared'] = separation_squared
        outputs['penalty_' + self.const_id] = -np.minimum(separation_squared - self.min_spacing**2, 0).sum()

    def _compute(self, x, y):
        n_wt = self.n_wt

        # compute distance matrixes
        dX, dY = [np.subtract(*np.meshgrid(xy, xy, indexing='ij')).T
                  for xy in [x, y]]
        dXY2 = dX**2 + dY**2
        # return upper triangle (above diagonal)
        return dXY2[np.triu_indices(n_wt, 1)]

    def compute_partials(self, inputs, J):
        # obtain necessary inputs
        x = inputs[topfarm.x_key]
        y = inputs[topfarm.y_key]

        dSdx, dSdy = self._compute_partials(x, y)
        # populate Jacobian dict
        J['wtSeparationSquared', topfarm.x_key] = dSdx.flatten()
        J['wtSeparationSquared', topfarm.y_key] = dSdy.flatten()

    def _compute_partials(self, x, y):
        # get number of turbines
        n_wt = self.n_wt

        # S = (xi-xj)^2 + (yi-yj)^2 = dx^2 + dy^2
        # dS/dx = [dS/dx_i, dS/dx_j]  = -2dx, 2dx

        # compute distance matrixes
        dX, dY = [np.subtract(*np.meshgrid(xy, xy, indexing='ij')).T
                  for xy in [x, y]]
        # upper triangle -> 1 row per WT pair [(0,1), (0,2),..(n-1,n)]
        dx, dy = dX[np.triu_indices(n_wt, 1)], dY[np.triu_indices(n_wt, 1)]

        dSdx = np.array([-2 * dx, 2 * dx]).T
        dSdy = np.array([-2 * dy, 2 * dy]).T
        return dSdx, dSdy

    def plot(self, ax=None):
        from matplotlib.pyplot import Circle
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()

        def get_xy(xy):
            if not hasattr(self, xy):
                setattr(self, xy, dict(self.list_inputs(out_stream=None))[f'pre_constraints.{self.name}.{xy}']['value'])
            xy = getattr(self, xy)
            return xy if not isinstance(xy, tuple) else xy[0]

        for x, y in zip(get_xy('x'), get_xy('y')):
            circle = Circle((x, y), self.min_spacing / 2, color='k', ls='--', fill=False)
            ax.add_artist(circle)

    def satisfy(self, state, n_iter=100, step_size=0.1):
        x, y = [state[xy].astype(np.float) for xy in [topfarm.x_key, topfarm.y_key]]
        pair_i, pair_j = np.triu_indices(len(x), 1)
        for _ in range(n_iter):
            dist = self._compute(x, y)
            dx, dy = self._compute_partials(x, y)
            index = np.argmin(dist)

            if dist[index] < self.min_spacing**2:
                i, j = pair_i[index], pair_j[index]
                x[i] += dx[index, 0] * step_size
                x[j] += dx[index, 1] * step_size
                y[i] += dy[index, 0] * step_size
                y[j] += dy[index, 1] * step_size
            else:
                break
        state.update({topfarm.x_key: x, topfarm.y_key: y})
        return state
