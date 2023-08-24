import numpy as np
from numpy import newaxis as na
from topfarm.constraint_components import Constraint, ConstraintComponent
import topfarm


class SpacingConstraint(Constraint):
    def __init__(self, min_spacing, units=None, aggregation_function=None, full_aggregation=False, name='spacing_comp'):
        """Initialize SpacingConstraint

        Parameters
        ----------
        min_spacing : int or float
            Minimum spacing between turbines [m]
        aggregation_function : topfarm.utils.AggregationFunction or None
            if None: compute returns all wt-wt spacings (n_wt *(n_wt-1))/2
            if AggregationFunction: compute returns an aggregated (minimum) spacing
        """
        self.min_spacing = min_spacing
        self.aggregation_function = aggregation_function
        self.full_aggregation = full_aggregation
        self.const_id = name
        self.units = units

    @property
    def constraintComponent(self):
        return self.spacing_comp

    def _setup(self, problem):
        self.n_wt = problem.n_wt
        self.spacing_comp = SpacingComp(self.n_wt, self.min_spacing, self.const_id, self.units,
                                        aggregation_function=self.aggregation_function,
                                        full_aggregation=self.full_aggregation)
        problem.model.constraint_group.add_subsystem(self.const_id, self.spacing_comp,
                                                     promotes=[topfarm.x_key, topfarm.y_key, 'wtSeparationSquared'])

    def setup_as_constraint(self, problem):
        self._setup(problem)
        problem.model.add_constraint('wtSeparationSquared', lower=self.min_spacing**2)

    def setup_as_penalty(self, problem):
        self._setup(problem)


class SpacingComp(ConstraintComponent):
    """
    Calculates inter-turbine spacing for all turbine pairs.

    """

    def __init__(self, n_wt, min_spacing, const_id=None, units=None, aggregation_function=None, full_aggregation=False):
        super().__init__()
        self.n_wt = n_wt
        self.min_spacing = min_spacing
        self.const_id = const_id
        if aggregation_function:
            if full_aggregation:
                self.veclen = 1
            else:
                self.veclen = n_wt
        else:
            self.veclen = int((n_wt - 1.) * n_wt / 2.)
        self.units = units
        self.aggregation_function = aggregation_function
        self.full_aggregation = full_aggregation
        self.constraint_key = 'wtSeparationSquared'

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, val=np.zeros(self.n_wt),
                       desc='x coordinates of turbines in wind dir. ref. frame', units=self.units)
        self.add_input(topfarm.y_key, val=np.zeros(self.n_wt),
                       desc='y coordinates of turbines in wind dir. ref. frame', units=self.units)
        # self.add_output('constraint_violation_' + self.const_id, val=0.0)
        # Explicitly size output array
        self.add_output(self.constraint_key, val=np.zeros(self.veclen),
                        desc='spacing of all turbines in the wind farm')

        col_pairs = np.array([(i, j) for i in range(self.n_wt - 1) for j in range(i + 1, self.n_wt)])
        if self.aggregation_function:
            self.declare_partials(self.constraint_key, [topfarm.x_key, topfarm.y_key])

            self.partial_indices = np.array([np.r_[np.where(col_pairs[:, 1] == i)[0], np.where(col_pairs[:, 0] == i)[0]]
                                             for i in range(self.n_wt)]).T
            self.partial_sign = (np.ones((self.n_wt, self.n_wt)) - 2 * np.triu(np.ones((self.n_wt, self.n_wt)), 1))[:-1]
            self.col_pairs = col_pairs
        else:
            # Sparse partial declaration
            cols = col_pairs.flatten()
            rows = np.repeat(np.arange(self.veclen), 2)

            self.declare_partials(self.constraint_key,
                                  [topfarm.x_key, topfarm.y_key],
                                  rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        self.x = inputs[topfarm.x_key]
        self.y = inputs[topfarm.y_key]
        separation_squared = self._compute(self.x, self.y)
        if self.aggregation_function:
            if self.full_aggregation:
                outputs[self.constraint_key] = self.aggregation_function(separation_squared)
            else:
                outputs[self.constraint_key] = self.aggregation_function(
                    separation_squared[self.partial_indices], 0)
                # print(outputs[self.constraint_key])
        else:
            outputs[self.constraint_key] = separation_squared
        # outputs['constraint_violation_' + self.const_id] = -np.minimum(separation_squared - self.min_spacing**2, 0).sum()

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

        # gradient of spacing [(i,j) for i=0..n_wt-1 and j=i+1..n_wt] wrt (wt_x[i], wt_x[j]) and (wt_y[i], wt_y[j])
        dS_dxij, dS_dyij = self._compute_partials(x, y)
        if self.aggregation_function:
            if self.full_aggregation:
                # gradient of aggregated (minimum) spacing wrt. spacing(i,j)
                dSagg_dS = self.aggregation_function.gradient(self._compute(x, y)).flatten()
                # partial_indices extracts the spacing elements that each wt contributes to
                # partial sign gives it the right sign
                # and finally we sum the contributions of each wt
                J[self.constraint_key, topfarm.x_key] = (
                    (dS_dxij[:, 0] * dSagg_dS)[self.partial_indices] * self.partial_sign).sum(0).T
                J[self.constraint_key, topfarm.y_key] = (
                    (dS_dyij[:, 0] * dSagg_dS)[self.partial_indices] * self.partial_sign).sum(0).T
            else:
                # gradient of aggregated (minimum) spacing wrt. spacing(i,j)
                dSdwtx = (dS_dxij[:, 0])[self.partial_indices] * self.partial_sign
                dSdwty = (dS_dyij[:, 0])[self.partial_indices] * self.partial_sign
                S = self._compute(x, y)[self.partial_indices]
                dSagg_dS = self.aggregation_function.gradient(S, 0)

                # partial_indices extracts the spacing elements that each wt contributes to
                # partial sign gives it the right sign
                # and finally we sum the contributions of each wt

                dSagg_dwtx = dSdwtx * dSagg_dS * self.partial_sign
                dSagg_dwty = dSdwty * dSagg_dS * self.partial_sign

                dSagg_dx = np.zeros((self.n_wt, self.n_wt))  # np.diag((dSagg_dwtx).sum(0))
                dSagg_dy = np.zeros((self.n_wt, self.n_wt))  # np.diag((dSagg_dwty).sum(0))
                i = range(self.n_wt)
                for j in range(dSagg_dwtx.shape[0]):
                    ai, bi = self.col_pairs[self.partial_indices][j, i, :].T
                    dSagg_dx[ai, i] += dSagg_dwtx[j, i]
                    dSagg_dx[bi, i] -= dSagg_dwtx[j, i]
                    dSagg_dy[ai, i] += dSagg_dwty[j, i]
                    dSagg_dy[bi, i] -= dSagg_dwty[j, i]

                J[self.constraint_key, topfarm.x_key] = dSagg_dx.T
                J[self.constraint_key, topfarm.y_key] = dSagg_dy.T
        else:
            # populate Jacobian dict
            J[self.constraint_key, topfarm.x_key] = dS_dxij.flatten()
            J[self.constraint_key, topfarm.y_key] = dS_dyij.flatten()

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
                setattr(self, xy, dict(self.list_inputs(out_stream=None))[f'constraint_group.{self.name}.{xy}']['value'])
            xy = getattr(self, xy)
            return xy if not isinstance(xy, tuple) else xy[0]

        for x, y in zip(get_xy('x'), get_xy('y')):
            circle = Circle((x, y), self.min_spacing / 2, color='k', ls='--', fill=False)
            ax.add_artist(circle)

    def satisfy(self, state, n_iter=100, step_size=0.1):
        x, y = [state[xy].astype(float) for xy in [topfarm.x_key, topfarm.y_key]]
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


class SpacingTypeConstraint(SpacingConstraint):
    def __init__(self, min_spacing, units=None, aggregation_function=None, full_aggregation=False, name='spacing_type_comp'):
        """Initialize SpacingConstraint

        Parameters
        ----------
        min_spacing : array_like
            Minimum spacing around turbines [m] (diameter of circle around a turbine that no other turbines can occupy)
        aggregation_function : topfarm.utils.AggregationFunction or None
            if None: compute returns all wt-wt spacings (n_wt *(n_wt-1))/2
            if AggregationFunction: compute returns an aggregated (minimum) spacing
        """
        super().__init__(min_spacing=min_spacing, units=units, aggregation_function=aggregation_function,
                         full_aggregation=full_aggregation, name=name)
        self.min_spacing = np.asarray(min_spacing)

    def _setup(self, problem):
        self.n_wt = problem.n_wt
        self.spacing_comp = SpacingTypeComp(self.n_wt, self.min_spacing, self.const_id, self.units,
                                            aggregation_function=self.aggregation_function,
                                            full_aggregation=self.full_aggregation)
        problem.model.constraint_group.add_subsystem(self.const_id, self.spacing_comp,
                                                     promotes=[topfarm.x_key, topfarm.y_key, topfarm.type_key, 'wtRelativeSeparationSquared'])

    def setup_as_constraint(self, problem):
        self._setup(problem)
        problem.model.add_constraint('wtRelativeSeparationSquared', lower=0)


class SpacingTypeComp(SpacingComp):
    """
    Calculates inter-turbine spacing for all turbine pairs.

    """

    def __init__(self, n_wt, min_spacing, const_id=None, units=None, aggregation_function=None, full_aggregation=False, types=None):
        super().__init__(n_wt=n_wt, min_spacing=min_spacing, const_id=const_id, units=units,
                         aggregation_function=aggregation_function, full_aggregation=full_aggregation)
        self.constraint_key = 'wtRelativeSeparationSquared'
        self.types = types

    def setup(self):
        super().setup()
        self.add_input(topfarm.type_key, val=self.types or np.zeros(self.n_wt),
                       desc='turbine type number')

    def compute(self, inputs, outputs):
        self.x = inputs[topfarm.x_key]
        self.y = inputs[topfarm.y_key]
        self.type = inputs[topfarm.type_key]
        relative_separation_squared = self._compute(self.x, self.y, self.type)
        if self.aggregation_function:
            if self.full_aggregation:
                outputs[self.constraint_key] = self.aggregation_function(relative_separation_squared)
            else:
                outputs[self.constraint_key] = self.aggregation_function(
                    relative_separation_squared[self.partial_indices], 0)
                # print(outputs[self.constraint_key])
        else:
            outputs[self.constraint_key] = relative_separation_squared
        # outputs['constraint_violation_' + self.const_id] = -np.minimum(relative_separation_squared, 0).sum()

    def get_min_eff_spacing(self, t):
        return (self.min_spacing[np.atleast_1d(t).astype(int)][:, na] + self.min_spacing[np.atleast_1d(t).astype(int)][na, :]) / 2

    def _compute(self, x, y, t):
        n_wt = self.n_wt
        # compute distance matrixes
        dX, dY = [np.subtract(*np.meshgrid(xy, xy, indexing='ij')).T
                  for xy in [x, y]]
        dXY2 = dX**2 + dY**2 - self.get_min_eff_spacing(t)**2
        # return upper triangle (above diagonal)
        return dXY2[np.triu_indices(n_wt, 1)]

    def plot(self, ax=None):
        from matplotlib.pyplot import Circle
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()

        def get_xy(xy):
            if not hasattr(self, xy):
                setattr(self, xy, dict(self.list_inputs(out_stream=None))[f'constraint_group.{self.name}.{xy}']['value'])
            xy = getattr(self, xy)
            return xy if not isinstance(xy, tuple) else xy[0]

        for x, y, t in zip(get_xy('x'), get_xy('y'), get_xy('type')):
            circle = Circle((x, y), self.get_min_eff_spacing(t).ravel() / 2, color='k', ls='--', fill=False)
            ax.add_artist(circle)

    def satisfy(self, state, n_iter=100, step_size=0.1):
        x, y, t = [state[xy].astype(float) for xy in [topfarm.x_key, topfarm.y_key, topfarm.type_key]]
        pair_i, pair_j = np.triu_indices(len(x), 1)
        for _ in range(n_iter):
            dist = self._compute(x, y, t)
            dx, dy = self._compute_partials(x, y)
            index = np.argmin(dist)

            if dist[index] < self.get_min_eff_spacing(t)[np.triu_indices(self.n_wt, 1)][index]**2:
                i, j = pair_i[index], pair_j[index]
                x[i] += dx[index, 0] * step_size
                x[j] += dx[index, 1] * step_size
                y[i] += dy[index, 0] * step_size
                y[j] += dy[index, 1] * step_size
            else:
                break
        state.update({topfarm.x_key: x, topfarm.y_key: y})
        return state
