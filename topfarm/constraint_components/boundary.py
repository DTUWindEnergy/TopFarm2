import numpy as np
from numpy import newaxis as na
from scipy.spatial import ConvexHull
from topfarm.constraint_components import Constraint, ConstraintComponent
from topfarm.utils import smooth_max, smooth_max_gradient
import topfarm
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import warnings


class XYBoundaryConstraint(Constraint):
    def __init__(self, boundary, boundary_type='convex_hull', units=None, relaxation=False):
        """Initialize XYBoundaryConstraint

        Parameters
        ----------
        boundary : array_like (n,2) or list of tuples (array_like (n,2), boolean)
            boundary coordinates. If boundary is array_like (n,2) it indicates a single boundary and can be used with
            boundary types: 'convex_hull', 'polygon', 'rectangle','square'. If boundary is list of tuples (array_like (n,2), boolean),
            it is multiple boundaries where the boolean is 1 for inclusion zones and 0 for exclusion zones and can be used with the
            boundary type: 'multi_polygon'.
        boundary_type : 'convex_hull', 'polygon', 'rectangle','square'
            - 'convex_hull' (default): Convex hul around boundary points\n
            - 'polygon': Polygon boundary (may be non convex). Less suitable for gradient-based optimization\n
            - 'rectangle': Smallest axis-aligned rectangle covering the boundary points\n
            - 'square': Smallest axis-aligned square covering the boundary points
            - 'multi_polygon': Mulitple polygon boundaries incl. exclusion zones (may be non convex).\n


        """
        if boundary_type == 'multi_polygon':
            if np.ndim(boundary[0][0]) < 2:
                self.multi_boundary = [(np.asarray(boundary), 1)]
            else:
                self.multi_boundary = [(np.asarray(bound), boolean) for bound, boolean in boundary]
                boundary = boundary[0][0]
        self.boundary = np.asarray(boundary)
        self.boundary_type = boundary_type
        self.const_id = 'xyboundary_comp_{}_{}'.format(boundary_type, int(self.boundary.sum()))
        self.units = units
        self.relaxation = relaxation

    def get_comp(self, n_wt):
        if not hasattr(self, 'boundary_comp'):
            if self.boundary_type == 'polygon':
                self.boundary_comp = PolygonBoundaryComp(n_wt, self.boundary, self.const_id, self.units, self.relaxation)
            elif self.boundary_type == 'multi_polygon':
                self.boundary_comp = MultiPolygonBoundaryComp(n_wt, self.multi_boundary, self.const_id, self.units, self.relaxation)
            else:
                self.boundary_comp = ConvexBoundaryComp(
                    n_wt, self.boundary, self.boundary_type, self.const_id, self.units)
        return self.boundary_comp

    @property
    def constraintComponent(self):
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        if hasattr(self, 'multi_boundary'):
            bound_min = np.vstack([(bound[0]).min(0) for bound in self.multi_boundary]).min(0)
            bound_max = np.vstack([(bound[0]).max(0) for bound in self.multi_boundary]).max(0)
        else:
            bound_min = self.boundary_comp.xy_boundary.min(0)
            bound_max = self.boundary_comp.xy_boundary.max(0)
        for k, l, u in zip([topfarm.x_key, topfarm.y_key], bound_min, bound_max):
            if k in design_vars:
                if len(design_vars[k]) == 4:
                    design_vars[k] = (design_vars[k][0], np.maximum(design_vars[k][1], l),
                                      np.minimum(design_vars[k][2], u), design_vars[k][-1])
                else:
                    design_vars[k] = (design_vars[k][0], l, u, design_vars[k][-1])

    def _setup(self, problem, group='pre_constraints'):
        n_wt = problem.n_wt
        self.boundary_comp = self.get_comp(n_wt)
        self.boundary_comp.problem = problem
        self.set_design_var_limits(problem.design_vars)
        # problem.xy_boundary = np.r_[self.boundary_comp.xy_boundary, self.boundary_comp.xy_boundary[:1]]
        problem.indeps.add_output('xy_boundary', self.boundary_comp.xy_boundary)
        getattr(problem.model, group).add_subsystem('xy_bound_comp', self.boundary_comp, promotes=['*'])

    def setup_as_constraint(self, problem, group='pre_constraints'):
        self._setup(problem, group=group)
        problem.model.add_constraint('boundaryDistances', lower=self.boundary_comp.zeros)

    def setup_as_penalty(self, problem, group='pre_constraints'):
        self._setup(problem, group=group)


class CircleBoundaryConstraint(Constraint):
    def __init__(self, center, radius):
        """Initialize CircleBoundaryConstraint

        Parameters
        ----------
        center : (float, float)
            center position (x,y)
        radius : int or float
            circle radius
        """

        self.center = np.array(center)
        self.radius = radius
        self.const_id = 'circle_boundary_comp_{}_{}'.format(
            '_'.join([str(int(c)) for c in center]), int(radius)).replace('.', '_')

    def get_comp(self, n_wt):
        if not hasattr(self, 'boundary_comp'):
            self.boundary_comp = CircleBoundaryComp(n_wt, self.center, self.radius, self.const_id)
        return self.boundary_comp

    @property
    def constraintComponent(self):
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        for k, l, u in zip([topfarm.x_key, topfarm.y_key],
                           self.center - self.radius,
                           self.center + self.radius):
            if len(design_vars[k]) == 4:
                design_vars[k] = (design_vars[k][0], np.maximum(design_vars[k][1], l),
                                  np.minimum(design_vars[k][2], u), design_vars[k][-1])
            else:
                design_vars[k] = (design_vars[k][0], l, u, design_vars[k][-1])

    def _setup(self, problem):
        n_wt = problem.n_wt
        self.boundary_comp = self.get_comp(n_wt)
        self.set_design_var_limits(problem.design_vars)
        problem.indeps.add_output('xy_boundary', self.boundary_comp.xy_boundary)
        problem.model.pre_constraints.add_subsystem('xy_bound_comp', self.boundary_comp, promotes=['*'])

    def setup_as_constraint(self, problem):
        self._setup(problem)
        problem.model.add_constraint('boundaryDistances', lower=self.boundary_comp.zeros)

    def setup_as_penalty(self, problem, penalty=1e10):
        self._setup(problem)


class BoundaryBaseComp(ConstraintComponent):
    def __init__(self, n_wt, xy_boundary=None, const_id=None, units=None, relaxation=False, **kwargs):
        super().__init__(**kwargs)
        self.n_wt = n_wt
        self.xy_boundary = np.array(xy_boundary)
        self.const_id = const_id
        self.units = units
        self.relaxation = relaxation
        if np.any(self.xy_boundary[0] != self.xy_boundary[-1]):
            self.xy_boundary = np.r_[self.xy_boundary, self.xy_boundary[:1]]

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, np.zeros(self.n_wt),
                       desc='x coordinates of turbines in global ref. frame', units=self.units)
        self.add_input(topfarm.y_key, np.zeros(self.n_wt),
                       desc='y coordinates of turbines in global ref. frame', units=self.units)
        if self.relaxation:
            self.add_input('time', 0)
        self.add_output('penalty_' + self.const_id, val=0.0)
        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', self.zeros,
                        desc="signed perpendicular distances from each turbine to each face CCW; + is inside")
        self.declare_partials('boundaryDistances', [topfarm.x_key, topfarm.y_key])
        if self.relaxation:
            self.declare_partials('boundaryDistances', 'time')

        # self.declare_partials('boundaryDistances', ['boundaryVertices', 'boundaryNormals'], method='fd')

    def compute(self, inputs, outputs):
        # calculate distances from each point to each face
        boundaryDistances = self.distances(x=inputs[topfarm.x_key], y=inputs[topfarm.y_key])
        outputs['boundaryDistances'] = boundaryDistances
        outputs['penalty_' + self.const_id] = -np.minimum(boundaryDistances, 0).sum()

    def compute_partials(self, inputs, partials):
        # return Jacobian dict
        if not self.relaxation:
            dx, dy = self.gradients(**{xy: inputs[k] for xy, k in zip('xy', [topfarm.x_key, topfarm.y_key])})
        else:
            dx, dy, dt = self.gradients(**{xy: inputs[k] for xy, k in zip('xy', [topfarm.x_key, topfarm.y_key])})

        partials['boundaryDistances', topfarm.x_key] = dx
        partials['boundaryDistances', topfarm.y_key] = dy
        if self.relaxation:
            partials['boundaryDistances', 'time'] = dt

    def plot(self, ax):
        """Plot boundary"""
        if isinstance(self, MultiPolygonBoundaryComp):
            colors = ['--k', 'k']
            for bound, io in self.boundaries:
                ax.plot(np.asarray(bound)[:, 0].tolist() + [np.asarray(bound)[0, 0]],
                        np.asarray(bound)[:, 1].tolist() + [np.asarray(bound)[0, 1]], colors[io])
        else:
            ax.plot(self.xy_boundary[:, 0].tolist() + [self.xy_boundary[0, 0]],
                    self.xy_boundary[:, 1].tolist() + [self.xy_boundary[0, 1]], 'k')


class ConvexBoundaryComp(BoundaryBaseComp):
    def __init__(self, n_wt, xy_boundary=None, boundary_type='convex_hull', const_id=None, units=None):
        self.boundary_type = boundary_type
#        self.const_id = const_id
        self.calculate_boundary_and_normals(xy_boundary)
        super().__init__(n_wt, self.xy_boundary, const_id, units)
        self.calculate_gradients()
        self.zeros = np.zeros([self.n_wt, self.nVertices])
#        self.units = units

    def calculate_boundary_and_normals(self, xy_boundary):
        xy_boundary = np.asarray(xy_boundary)
        if self.boundary_type == 'convex_hull':
            # find the points that actually comprise a convex hull
            hull = ConvexHull(list(xy_boundary))

            # keep only xy_vertices that actually comprise a convex hull and arrange in CCW order
            self.xy_boundary = xy_boundary[hull.vertices]
        elif self.boundary_type == 'square':
            min_ = xy_boundary.min(0)
            max_ = xy_boundary.max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_.max() / 2
            self.xy_boundary = np.array([(x_c - r, y_c - r), (x_c + r, y_c - r),
                                         (x_c + r, y_c + r), (x_c - r, y_c + r)])
        elif self.boundary_type == 'rectangle':
            min_ = xy_boundary.min(0)
            max_ = xy_boundary.max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_ / 2
            self.xy_boundary = np.array([(x_c - r[0], y_c - r[1]), (x_c + r[0], y_c - r[1]),
                                         (x_c + r[0], y_c + r[1]), (x_c - r[0], y_c + r[1])])
        else:
            raise NotImplementedError("Boundary type '%s' is not implemented" % self.boundary_type)

        # get the real number of xy_vertices
        self.nVertices = self.xy_boundary.shape[0]

        # initialize normals array
        unit_normals = np.zeros([self.nVertices, 2])

        # determine if point is inside or outside of each face, and distances from each face
        for j in range(0, self.nVertices):

            # calculate the unit normal vector of the current face (taking points CCW)
            if j < self.nVertices - 1:  # all but the set of point that close the shape
                normal = np.array([self.xy_boundary[j + 1, 1] - self.xy_boundary[j, 1],
                                   -(self.xy_boundary[j + 1, 0] - self.xy_boundary[j, 0])])
                unit_normals[j] = normal / np.linalg.norm(normal)
            else:   # the set of points that close the shape
                normal = np.array([self.xy_boundary[0, 1] - self.xy_boundary[j, 1],
                                   -(self.xy_boundary[0, 0] - self.xy_boundary[j, 0])])
                unit_normals[j] = normal / np.linalg.norm(normal)

        self.unit_normals = unit_normals

    def calculate_gradients(self):
        unit_normals = self.unit_normals

        # initialize array to hold distances from each point to each face
        dfaceDistance_dx = np.zeros([self.n_wt * self.nVertices, self.n_wt])
        dfaceDistance_dy = np.zeros([self.n_wt * self.nVertices, self.n_wt])

        for i in range(0, self.n_wt):
            # determine if point is inside or outside of each face, and distances from each face
            for j in range(0, self.nVertices):

                # define the derivative vectors from the point of interest to the first point of the face
                dpa_dx = np.array([-1.0, 0.0])
                dpa_dy = np.array([0.0, -1.0])

                # find perpendicular distances derivatives from point to current surface (vector projection)
                ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j]) * unit_normals[j]
                ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j]) * unit_normals[j]

                # calculate derivatives for the sign of perpendicular distances from point to current face
                dfaceDistance_dx[i * self.nVertices + j, i] = np.vdot(ddistanceVec_dx, unit_normals[j])
                dfaceDistance_dy[i * self.nVertices + j, i] = np.vdot(ddistanceVec_dy, unit_normals[j])

        # return Jacobian dict
        self.dfaceDistance_dx = dfaceDistance_dx
        self.dfaceDistance_dy = dfaceDistance_dy

    def calculate_distance_to_boundary(self, points):
        """
        :param points: points that you want to calculate the distances from to the faces of the convex hull
        :return face_distace: signed perpendicular distances from each point to each face; + is inside
        """

        nPoints = np.array(points).shape[0]
        xy_boundary = self.xy_boundary[:-1]
        nVertices = xy_boundary.shape[0]
        vertices = xy_boundary
        unit_normals = self.unit_normals
        # initialize array to hold distances from each point to each face
        face_distance = np.zeros([nPoints, nVertices])
        from numpy import newaxis as na

        # define the vector from the point of interest to the first point of the face
        PA = (vertices[:, na] - points[na])

        # find perpendicular distances from point to current surface (vector projection)
        dist = np.sum(PA * unit_normals[:, na], 2)
        # calculate the sign of perpendicular distances from point to current face (+ is inside, - is outside)
        d_vec = dist[:, :, na] * unit_normals[:, na]
        face_distance = np.sum(d_vec * unit_normals[:, na], 2)
        return face_distance.T

    def distances(self, x, y):
        return self.calculate_distance_to_boundary(np.array([x, y]).T)

    def gradients(self, x, y):
        return self.dfaceDistance_dx, self.dfaceDistance_dy

    def satisfy(self, state, pad=1.1):
        x, y = [np.asarray(state[xyz], dtype=float) for xyz in [topfarm.x_key, topfarm.y_key]]
        dist = self.distances(x, y)
        dist = np.where(dist < 0, np.minimum(dist, -.01), dist)
        dx, dy = self.gradients(x, y)  # independent of position
        dx = dx[:self.nVertices, 0]
        dy = dy[:self.nVertices, 0]
        for i in np.where(dist.min(1) < 0)[0]:  # loop over turbines that violate edges
            # find smallest movement that where the constraints are satisfied
            d = dist[i]
            v = np.linspace(-np.abs(d.min()), np.abs(d.min()), 100)
            X, Y = np.meshgrid(v, v)
            m = np.ones_like(X)
            for dx_, dy_, d in zip(dx, dy, dist.T):
                m = np.logical_and(m, X * dx_ + Y * dy_ >= -d[i])
            index = np.argmin(X[m]**2 + Y[m]**2)
            x[i] += X[m][index]
            y[i] += Y[m][index]
        state[topfarm.x_key] = x
        state[topfarm.y_key] = y
        return state


class PolygonBoundaryComp(BoundaryBaseComp):
    def __init__(self, n_wt, xy_boundary, const_id=None, units=None, relaxation=False):

        self.nTurbines = n_wt
        self.const_id = const_id
        self.zeros = np.zeros(self.nTurbines)
        self.units = units
        self.boundary_properties = self.get_boundary_properties(xy_boundary)
        BoundaryBaseComp.__init__(self, n_wt, xy_boundary=self.boundary_properties[0], const_id=const_id, units=units, relaxation=relaxation)
        self._cache_input = None
        self._cache_output = None
        self.relaxation = relaxation

    def get_boundary_properties(self, xy_boundary):
        vertices = np.array(xy_boundary)

        def edges_counter_clockwise(vertices):
            if np.any(vertices[0] != vertices[-1]):
                vertices = np.r_[vertices, vertices[:1]]
            x1, y1 = vertices[:-1].T
            x2, y2 = vertices[1:].T
            double_area = np.sum((x1 - x2) * (y1 + y2))  # 2 x Area (+: counterclockwise
            assert double_area != 0, "Area must be non-zero"
            if double_area < 0:  #
                return edges_counter_clockwise(vertices[::-1])
            else:
                return vertices[:-1], x1, y1, x2, y2

        xy_boundary, x1, y1, x2, y2 = edges_counter_clockwise(vertices)
        dx = x2 - x1
        dy = y2 - y1
        length = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
        edge_unit_vec = (np.array([dy, -dx]) / length)
        v = np.hstack((edge_unit_vec, edge_unit_vec[:, :1]))
        xy2_vec = v[:, :-1] + v[:, 1:]
        xy1_vec = np.hstack((xy2_vec[:, -1:], xy2_vec[:, 1:]))
        dEdgeDist_dx = -dy / length
        dEdgeDist_dy = dx / length
        edge_vect_j = np.array([x2 - x1, y2 - y1])
        edge_vect_len_j = np.linalg.norm(edge_vect_j, axis=0)
        return (xy_boundary, x1, y1, x2, y2, dEdgeDist_dx, dEdgeDist_dy,
                edge_unit_vec, dx, xy1_vec, xy2_vec, edge_vect_j, edge_vect_len_j)

    def _calc_distance_and_gradients(self, x, y, boundary_properties=None):
        """
        distances point(x,y) to edge((x1,y1)->(x2,y2))
        +/-: inside/outside
        case (x,y) closest to edge:
            distances = edge_unit_vec dot (x1-x,y1-y)
            ddist_dx = -(y2-y2)/|edge|
            ddist_dy = (x2-x2)/|edge|
        case (x,y) closest to (x1,y1) (and (x2,y2)):
            sign = sign of distances to nearest edge
            distances = sign * (x1-x^2 + y1-y)^2)^.5
            ddist_dx = sign * 2*x-2*x1 / (2 * distances^.5)
            ddist_dy = sign * 2*y-2*y1 / (2 * distances^.5)
        """
        boundary_properties = boundary_properties or self.boundary_properties
        _, x1, y1, x2, y2, dEdgeDist_dx, dEdgeDist_dy, edge_unit_vec, dx, xy1_vec, xy2_vec, edge_vect_j, edge_vect_len_j = boundary_properties
        X, Y = [np.tile(xy, (len(x1), 1)).T for xy in [x, y]]  # dim = (ntb, nEdges)
        X1, Y1, X2, Y2, ddist_dX, ddist_dY = [np.tile(xy, (len(x), 1))
                                              for xy in [x1, y1, x2, y2, dEdgeDist_dx, dEdgeDist_dy]]
        # vector a is the vector from vertex to point
        a = np.array([X - X1, Y - Y1])

        # signed component of a on the edge vector, a_tilde
        a_tilde = np.sum(a * edge_vect_j[:, na, :], axis=0) / edge_vect_len_j[na, :]

        # if a_tilde is negative use vertex 1 and if a_tilde is longer than the edge vector use vertex 2
        use_xy1 = 0 > a_tilde
        use_xy2 = a_tilde > edge_vect_len_j

        # perpendicular distances to edge (dot product)
        d12 = (x1 - X) * edge_unit_vec[0] + (y1 - Y) * edge_unit_vec[1]

        # nearest point on edge
        px = X + d12 * edge_unit_vec[0]
        py = Y + d12 * edge_unit_vec[1]

        # distances to start and end points
        d1 = np.sqrt((x1 - X)**2 + (y1 - Y)**2)
        d2 = np.sqrt((x2 - X)**2 + (y2 - Y)**2)

        px[use_xy1] = X1[use_xy1]
        py[use_xy1] = Y1[use_xy1]
        px[use_xy2] = X2[use_xy2]
        py[use_xy2] = Y2[use_xy2]

        distance = d12.copy()
        v = (px[use_xy1] - X[use_xy1]) * xy1_vec[0, np.where(use_xy1)[1]] + \
            (py[use_xy1] - Y[use_xy1]) * xy1_vec[1, np.where(use_xy1)[1]]
        sign_use_xy1 = np.choose(v >= 0, [-1, 1])
        v = (px[use_xy2] - X[use_xy2]) * xy2_vec[0, np.where(use_xy2)[1]] + \
            (py[use_xy2] - Y[use_xy2]) * xy2_vec[1, np.where(use_xy2)[1]]
        sign_use_xy2 = np.choose(v >= 0, [-1, 1])

        d12[use_xy2]
        d12[:, 1:][use_xy2[:, :-1]]

        distance[use_xy1] = sign_use_xy1 * d1[use_xy1]
        distance[use_xy2] = sign_use_xy2 * d2[use_xy2]

        length = np.sqrt((X1[use_xy1] - X[use_xy1])**2 + (Y1[use_xy1] - Y[use_xy1])**2)
        ddist_dX[use_xy1] = sign_use_xy1 * (2 * X[use_xy1] - 2 * X1[use_xy1]) / (2 * length)
        ddist_dY[use_xy1] = sign_use_xy1 * (2 * Y[use_xy1] - 2 * Y1[use_xy1]) / (2 * length)

        length = np.sqrt((X2[use_xy2] - X[use_xy2])**2 + (Y2[use_xy2] - Y[use_xy2])**2)
        ddist_dX[use_xy2] = sign_use_xy2 * (2 * X[use_xy2] - 2 * X2[use_xy2]) / (2 * length)
        ddist_dY[use_xy2] = sign_use_xy2 * (2 * Y[use_xy2] - 2 * Y2[use_xy2]) / (2 * length)

        return distance, ddist_dX, ddist_dY

    def calc_distance_and_gradients(self, x, y):
        if np.all(np.array([x, y]) == self._cache_input):
            return self._cache_output
        distance, ddist_dX, ddist_dY = self._calc_distance_and_gradients(x, y)
        closest_edge_index = np.argmin(np.abs(distance), 1)
        self._cache_input = np.array([x, y])
        self._cache_output = [np.choose(closest_edge_index, v.T) for v in [distance, ddist_dX, ddist_dY]]
        return self._cache_output

    def distances(self, x, y):
        return self.calc_distance_and_gradients(x, y)[0]

    def gradients(self, x, y):
        _, dx, dy = self.calc_distance_and_gradients(x, y)
        return np.diagflat(dx), np.diagflat(dy)

    def satisfy(self, state, pad=1.1):
        x, y = [np.asarray(state[xy], dtype=float) for xy in [topfarm.x_key, topfarm.y_key]]
        dist = self.distances(x, y)
        dx, dy = map(np.diag, self.gradients(x, y))
        m = dist < 0
        x[m] -= dx[m] * dist[m] * pad
        y[m] -= dy[m] * dist[m] * pad
        state[topfarm.x_key] = x
        state[topfarm.y_key] = y
        return state


class CircleBoundaryComp(PolygonBoundaryComp):
    def __init__(self, n_wt, center, radius, const_id=None, units=None):
        self.center = center
        self.radius = radius
        t = np.linspace(0, 2 * np.pi, 100)
        xy_boundary = self.center + np.array([np.cos(t), np.sin(t)]).T * self.radius
        BoundaryBaseComp.__init__(self, n_wt, xy_boundary, const_id, units)
        self.zeros = np.zeros(self.n_wt)

    def plot(self, ax=None):
        from matplotlib.pyplot import Circle
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        circle = Circle(self.center, self.radius, color='k', fill=False)
        ax.add_artist(circle)

    def distances(self, x, y):
        return self.radius - np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)

    def gradients(self, x, y):
        theta = np.arctan2(y - self.center[1], x - self.center[0])
        dx = -1 * np.ones_like(x)
        dy = -1 * np.ones_like(x)
        dist = self.radius - np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        not_center = dist != self.radius
        dx[not_center], dy[not_center] = -np.cos(theta[not_center]), -np.sin(theta[not_center])
        return np.diagflat(dx), np.diagflat(dy)


class MultiPolygonBoundaryComp(PolygonBoundaryComp):
    def __init__(self, n_wt, xy_multi_boundary, const_id=None, units=None, relaxation=False, method='nearest',
                 simplify_geometry=False):
        '''
        Parameters
        ----------
        n_wt : TYPE
            DESCRIPTION.
        xy_multi_boundary : TYPE
            DESCRIPTION.
        const_id : TYPE, optional
            DESCRIPTION. The default is None.
        units : TYPE, optional
            DESCRIPTION. The default is None.
        method : {'nearest' or 'smooth_min'}, optional
            'nearest' calculate the distance to the nearest edge or point'smooth_min'
            calculates the weighted minimum distance to all edges/points. The default is 'nearest'.
        simplify : float or dict
            if float, simplification tolerance. if dict, shapely.simplify keyword arguments
        Returns
        -------
        None.

        '''
        self.xy_multi_boundary = xy_multi_boundary
        PolygonBoundaryComp.__init__(self, n_wt, xy_boundary=xy_multi_boundary[0][0], const_id=const_id, units=units, relaxation=relaxation)
        self.bounds_poly = [Polygon(x) for x, _ in xy_multi_boundary]
        self.types_bool = [1 if x in ['i', 'include', True, 1, None] else 0 for _, x in xy_multi_boundary]
        self._setup_boundaries()
        self.relaxation = relaxation
        self.method = method
        if simplify_geometry:
            self.simplify(simplify_geometry)

    def simplify(self, simplify_geometry):
        if isinstance(simplify_geometry, dict):
            self.bounds_poly = [rp.simplify(**simplify_geometry) for rp in self.bounds_poly]
        else:
            self.bounds_poly = [rp.simplify(simplify_geometry) for rp in self.bounds_poly]
        self._setup_boundaries()

    def _setup_boundaries(self):
        self.res_poly = self._calc_resulting_polygons(self.bounds_poly)
        self.boundaries = self._poly_to_bound(self.res_poly)
        self.boundary_properties_list = [self.get_boundary_properties(bound) for bound, _ in self.boundaries]
        self.n_edges = np.asarray([len(bound) for bound, _ in self.boundaries])
        self.n_edges_tot = np.sum(self.n_edges)
        self.start_at = np.cumsum(self.n_edges) - self.n_edges
        self.end_at = self.start_at + self.n_edges

    def _poly_to_bound(self, polygons):
        boundaries = []
        for bound in polygons:
            x, y = bound.exterior.xy
            boundaries.append((np.asarray([x, y]).T[:-1, :], 1))
            for interior in bound.interiors:
                x, y = interior.xy
                boundaries.append((np.asarray([x, y]).T[:-1, :], 0))
        return boundaries

    def _calc_resulting_polygons(self, boundary_polygons):
        '''
        Parameters
        ----------
        boundary_polygons : list
            list of shapely polygons as specifed or inferred from user input
        Returns
        -------
        list of merged shapely polygons. Resolves issues arrising if any are overlapping, touching or contained in each other
        '''
        domain = []
        for i in range(len(boundary_polygons)):
            b = boundary_polygons[i]
            if len(domain) == 0:
                if self.types_bool[i]:
                    domain.append(b)
                else:
                    warnings.warn("First boundary should be an inclusion zone or it will be ignored")
                    pass
            else:
                if self.types_bool[i]:
                    temp = []
                    for j, d in enumerate(domain):
                        if d.intersects(b):
                            b = unary_union([d, b])
                        else:
                            if d.contains(b):
                                warnings.warn("Boundary is fully contained preceding polygon and will be ignored")
                                pass
                            elif b.contains(d):
                                b = d
                                warnings.warn("Boundary is fully containing preceding polygon and will override it")
                                pass
                            else:
                                if b.area > 1e-3:
                                    temp.append(d)
                        if j == len(domain) - 1:
                            if b.area > 1e-3:
                                temp.append(b)
                    domain = temp
                else:
                    temp = []
                    for j, d in enumerate(domain):
                        if d.intersects(b):
                            nonoverlap = (d.symmetric_difference(b)).difference(b)
                            if isinstance(nonoverlap, type(Polygon())):
                                temp.append(nonoverlap)
                            elif isinstance(nonoverlap, type(MultiPolygon())):
                                for x in nonoverlap.geoms:
                                    if x.area > 1e-3:
                                        temp.append(x)
                        else:
                            if b.contains(d):
                                warnings.warn("Exclusion boundary fully consumes preceding polygon")
                                pass
                            else:
                                if d.contains(b):
                                    d = Polygon(d.exterior.coords, [b.exterior.coords])
                                if d.area > 1e-3:
                                    temp.append(d)
                    domain = temp
        return domain

    # def _calc_distance_and_gradients(self, x, y, boundary_properties):
    #     '''
    #     x_xn_vect_ij is the vector from edge start point (p1) to x
    #     x_xn_len_ij is the signed length of x_xn_vect_ij which is used to assess x is closer to the edge or either of the end points
    #     overlapping_ij assesses if x is closer to an edge or the end points
    #     Dp_ij is the distance from edge start point to x
    #     Dp_ij_res is the distance from the point to the closest of the edge ends
    #     De_ij is the distance from x to an edge
    #     inside_edge_ij is a boolen array that desribes if the point lies on the correct side of the edge
    #     turns_left indicates if an angle between two consecutive edges is going into the boundary (concave) or out of the boundary (convex).
    #     '''
    #     _, x1, y1, x2, y2, dEdgeDist_dx, dEdgeDist_dy, _, _, _, _, edge_vect_j, edge_vect_len_j = boundary_properties
    #     x = np.asarray(x)
    #     y = np.asarray(y)
    #     shape_ij = (len(x), len(x1))
    #     x_xn_vect_ij = np.array([x[:, na] - x1[na, :], y[:, na] - y1[na, :]])
    #     x_xn_len_ij = np.sum(x_xn_vect_ij * edge_vect_j[:, na, :], axis=0) / edge_vect_len_j[na, :]

    #     D_ij = np.zeros(shape_ij)
    #     dDdx_ij = np.zeros(shape_ij)
    #     dDdy_ij = np.zeros(shape_ij)

    #     before_ij = 0 > x_xn_len_ij
    #     overlapping_ij = (0 <= x_xn_len_ij) & (x_xn_len_ij <= edge_vect_len_j)
    #     after_ij = x_xn_len_ij > edge_vect_len_j
    #     inside_edge_ij = np.cross(edge_vect_j[:, na, :], x_xn_vect_ij, axisa=0, axisb=0) > 0
    #     outside_edge_ij = np.logical_not(inside_edge_ij)
    #     turns_left_ij = np.broadcast_to(np.cross(np.roll(edge_vect_j, 1, axis=1), edge_vect_j, axis=0) > 0, shape_ij)
    #     turns_right_ij = np.logical_not(turns_left_ij)

    #     De_ij = np.abs((x2[na, :] - x1[na, :]) * (y1[na, :] - y[:, na]) - (x1[na, :] - x[:, na]) *
    #                    (y2[na, :] - y1[na, :])) / np.sqrt((x2[na, :] - x1[na, :]) ** 2 + (y2[na, :] - y1[na, :]) ** 2)
    #     Dp_ij = np.sqrt((x[:, na] - x1[na, :]) ** 2 + (y[:, na] - y1[na, :]) ** 2)

    #     D_ij[overlapping_ij] = De_ij[overlapping_ij]
    #     D_ij[before_ij] = Dp_ij[before_ij]
    #     D_ij[after_ij] = np.roll(Dp_ij, -1, axis=1)[after_ij]

    #     dDdx_ij[before_ij] = (x[:, na] - x1[na, :])[before_ij] / Dp_ij[before_ij]
    #     dDdx_ij[after_ij] = np.roll((x[:, na] - x1[na, :]), -1, axis=1)[after_ij] / np.roll(Dp_ij, -1, axis=1)[after_ij]
    #     dDdx_ij[overlapping_ij] = np.broadcast_to(dEdgeDist_dx[na, :], shape_ij)[overlapping_ij]
    #     dDdy_ij[before_ij] = (y[:, na] - y1[na, :])[before_ij] / Dp_ij[before_ij]
    #     dDdy_ij[after_ij] = np.roll((y[:, na] - y1[na, :]), -1, axis=1)[after_ij] / np.roll(Dp_ij, -1, axis=1)[after_ij]
    #     dDdy_ij[overlapping_ij] = np.broadcast_to(dEdgeDist_dy[na, :], shape_ij)[overlapping_ij]

    #     D_ij[outside_edge_ij & overlapping_ij] *= -1
    #     D_ij[before_ij & turns_left_ij] *= -1
    #     D_ij[after_ij & np.roll(turns_left_ij, -1, axis=1)] *= -1
    #     D_ij[before_ij & turns_right_ij & np.roll(outside_edge_ij, 1, axis=1) & outside_edge_ij] *= -1
    #     D_ij[after_ij & np.roll(turns_right_ij, -1, axis=1) &
    #          np.roll(outside_edge_ij, -1, axis=1) & outside_edge_ij] *= -1

    #     dDdx_ij[before_ij & turns_left_ij] *= -1
    #     dDdx_ij[after_ij & np.roll(turns_left_ij, -1, axis=1)] *= -1
    #     dDdx_ij[before_ij & turns_right_ij & np.roll(outside_edge_ij, 1, axis=1) & outside_edge_ij] *= -1
    #     dDdx_ij[after_ij & np.roll(turns_right_ij, -1, axis=1) &
    #             np.roll(outside_edge_ij, -1, axis=1) & outside_edge_ij] *= -1

    #     dDdy_ij[before_ij & turns_left_ij] *= -1
    #     dDdy_ij[after_ij & np.roll(turns_left_ij, -1, axis=1)] *= -1
    #     dDdy_ij[before_ij & turns_right_ij & np.roll(outside_edge_ij, 1, axis=1) & outside_edge_ij] *= -1
    #     dDdy_ij[after_ij & np.roll(turns_right_ij, -1, axis=1) &
    #             np.roll(outside_edge_ij, -1, axis=1) & outside_edge_ij] *= -1

    #     return D_ij, dDdx_ij, dDdy_ij

    def sign(self, Dist_ij):
        return np.sign(Dist_ij[np.arange(Dist_ij.shape[0]), np.argmin(abs(Dist_ij), axis=1)])

    def calc_distance_and_gradients(self, x, y):
        '''
        Parameters
        ----------
        x : 1d array
            Array of x-positions.
        y : 1d array
            Array of y-positions.

        Returns
        -------
        D_ij : 2d array
            Array of point-edge distances. index 'i' is points and index 'j' is total number of edges.
        sign_i : 1d array
            Array of signs of the governing distance.
        dDdk_jk : 2d array
            Jacobian of the distance matrix D_ij with respect to x and y.

        '''
        if np.all(np.array([x, y]) == self._cache_input) & (not self.relaxation):
            return self._cache_output
        Dist_ij = np.zeros((len(x), self.n_edges_tot))
        dDdk_ijk = np.zeros((len(x), self.n_edges_tot, 2))
        for n, (bound, bound_type) in enumerate(self.boundaries):
            sa = self.start_at[n]
            ea = self.end_at[n]
            distance, ddist_dX, ddist_dY = self._calc_distance_and_gradients(x, y, self.boundary_properties_list[n])
            if bound_type == 0:
                distance *= -1
                ddist_dX *= -1
                ddist_dY *= -1
            Dist_ij[:, sa:ea] = distance
            dDdk_ijk[:, sa:ea, 0] = ddist_dX
            dDdk_ijk[:, sa:ea, 1] = ddist_dY

        sign_i = self.sign(Dist_ij)
        self._cache_input = np.array([x, y])
        self._cache_output = [Dist_ij, dDdk_ijk, sign_i]
        return self._cache_output

    def calc_relaxation(self):
        '''
        The tupple relaxation contains a first term for the penalty constant
        and a second term for the n first iterations to apply relaxation.
        '''
        iteration_no = self.problem.cost_comp.n_grad_eval + 1
        return max(0, self.relaxation[0] * (self.relaxation[1] - iteration_no))

    def distances(self, x, y):
        Dist_ij, _, sign_i = self.calc_distance_and_gradients(x, y)
        if self.method == 'smooth_min':
            Dist_i = smooth_max(np.abs(Dist_ij), -np.abs(Dist_ij).max(), axis=1) * sign_i
        elif self.method == 'nearest':
            Dist_i = Dist_ij[np.arange(x.size), np.argmin(np.abs(Dist_ij), axis=1)]
        if self.relaxation:
            Dist_i += self.calc_relaxation()
        return Dist_i

    def gradients(self, x, y):
        '''
        The derivate of the smooth maximum with respect to x and y is calculated with the chain rule:
            dS/dk = dS/dD * dD/dk
            where S is smooth maximum, D is distance to edge and k is the spacial dimension
        '''
        Dist_ij, dDdk_ijk, _ = self.calc_distance_and_gradients(x, y)
        if self.method == 'smooth_min':
            dSdDist_ij = smooth_max_gradient(np.abs(Dist_ij), -np.abs(Dist_ij).max(), axis=1)
            dSdkx_i, dSdky_i = (dSdDist_ij[:, :, na] * dDdk_ijk).sum(axis=1).T
        elif self.method == 'nearest':
            dSdkx_i, dSdky_i = dDdk_ijk[np.arange(x.size), np.argmin(np.abs(Dist_ij), axis=1), :].T

        if self.relaxation:
            # as relaxed distance is relaxation + distance, the gradient with respect to x and y is unchanged
            gradients = np.diagflat(dSdkx_i), np.diagflat(dSdky_i), np.ones(self.n_wt) * self.relaxation[1]
        else:
            gradients = np.diagflat(dSdkx_i), np.diagflat(dSdky_i)
        return gradients


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        plt.close('all')
        i1 = np.array([[2, 17], [6, 23], [16, 23], [26, 15], [19, 0], [14, 4], [4, 4]])
        e1 = np.array([[0, 10], [20, 21], [22, 12], [10, 12], [9, 6], [2, 7]])
        i2 = np.array([[12, 13], [14, 17], [18, 15], [17, 10], [15, 11]])
        e2 = np.array([[5, 17], [5, 18], [8, 19], [8, 18]])
        i3 = np.array([[5, 0], [5, 1], [10, 3], [10, 0]])
        e3 = np.array([[6, -1], [6, 18], [7, 18], [7, -1]])
        e4 = np.array([[15, 9], [15, 11], [20, 11], [20, 9]])
        multi_boundary = [(i1, 'i'), (e1, 'e'), (i2, 'i'), (e2, 'e'), (i3, 'i'), (e3, 'e'), (e4, 'e')]
        N_points = 50
        xs = np.linspace(-1, 30, N_points)
        ys = np.linspace(-1, 30, N_points)
        y_grid, x_grid = np.meshgrid(xs, ys)
        x = x_grid.ravel()
        y = y_grid.ravel()
        n_wt = len(x)
        MPBC = MultiPolygonBoundaryComp(n_wt, multi_boundary)
        distances = MPBC.distances(x, y)
        delta = 1e-9
        distances2 = MPBC.distances(x + delta, y)
        dx_fd = (distances2 - distances) / delta
        dx = np.diag(MPBC.gradients(x + delta / 2, y)[0])

        plt.figure()
        plt.plot(dx_fd, dx, '.')

        plt.figure()
        for n, bound in enumerate(MPBC.boundaries):
            x_bound, y_bound = bound[0].T
            x_bound = np.append(x_bound, x_bound[0])
            y_bound = np.append(y_bound, y_bound[0])
            line, = plt.plot(x_bound, y_bound, label=f'{n}')
            plt.plot(x_bound[0], y_bound[0], color=line.get_color(), marker='o')

        plt.legend()
        plt.grid()
        plt.axis('square')
        plt.contourf(x_grid, y_grid, distances.reshape(N_points, N_points), 50)
        plt.colorbar()

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(
            x.reshape(
                N_points, N_points), y.reshape(
                N_points, N_points), distances.reshape(
                N_points, N_points), 50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if 0:
            for smpl in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                MPBC = MultiPolygonBoundaryComp(n_wt, multi_boundary, simplify_geometry=smpl)
                plt.figure()
                ax = plt.gca()
                MPBC.plot(ax)


main()
