import numpy as np
from scipy.spatial import ConvexHull
from topfarm.constraint_components import Constraint, ConstraintComponent
import topfarm


class XYBoundaryConstraint(Constraint):
    def __init__(self, boundary, boundary_type='convex_hull', units=None):
        """Initialize XYBoundaryConstraint

        Parameters
        ----------
        boundary : array_like (n,2)
            boundary coordinates
        boundary_type : 'convex_hull', 'polygon', 'rectangle','square'
            - 'convex_hull' (default): Convex hul around boundary points\n
            - 'polygon': Polygon boundary (may be non convex). Less suitable for gradient-based optimization\n
            - 'rectangle': Smallest axis-aligned rectangle covering the boundary points\n
            - 'square': Smallest axis-aligned square covering the boundary points

        """
        self.boundary = np.array(boundary)
        self.boundary_type = boundary_type
        self.const_id = 'xyboundary_comp_{}_{}'.format(boundary_type, int(self.boundary.sum()))
        self.units = units

    def get_comp(self, n_wt):
        if not hasattr(self, 'boundary_comp'):
            if self.boundary_type == 'polygon':
                self.boundary_comp = PolygonBoundaryComp(n_wt, self.boundary, self.const_id, self.units)
            else:
                self.boundary_comp = ConvexBoundaryComp(
                    n_wt, self.boundary, self.boundary_type, self.const_id, self.units)
        return self.boundary_comp

    @property
    def constraintComponent(self):
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        for k, l, u in zip([topfarm.x_key, topfarm.y_key],
                           self.boundary_comp.xy_boundary.min(0),
                           self.boundary_comp.xy_boundary.max(0)):
            if k in design_vars:
                if len(design_vars[k]) == 4:
                    design_vars[k] = (design_vars[k][0], np.maximum(design_vars[k][1], l),
                                      np.minimum(design_vars[k][2], u), design_vars[k][-1])
                else:
                    design_vars[k] = (design_vars[k][0], l, u, design_vars[k][-1])

    def _setup(self, problem):
        n_wt = problem.n_wt
        self.boundary_comp = self.get_comp(n_wt)
        self.set_design_var_limits(problem.design_vars)
        # problem.xy_boundary = np.r_[self.boundary_comp.xy_boundary, self.boundary_comp.xy_boundary[:1]]
        problem.indeps.add_output('xy_boundary', self.boundary_comp.xy_boundary)
        problem.model.pre_constraints.add_subsystem('xy_bound_comp', self.boundary_comp, promotes=['*'])

    def setup_as_constraint(self, problem):
        self._setup(problem)
        problem.model.add_constraint('boundaryDistances', lower=self.boundary_comp.zeros)

    def setup_as_penalty(self, problem):
        self._setup(problem)


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
    def __init__(self, n_wt, xy_boundary=None, const_id=None, units=None, **kwargs):
        super().__init__(**kwargs)
        self.n_wt = n_wt
        self.xy_boundary = np.array(xy_boundary)
        self.const_id = const_id
        self.units = units
        if np.any(self.xy_boundary[0] != self.xy_boundary[-1]):
            self.xy_boundary = np.r_[self.xy_boundary, self.xy_boundary[:1]]

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, np.zeros(self.n_wt),
                       desc='x coordinates of turbines in global ref. frame', units=self.units)
        self.add_input(topfarm.y_key, np.zeros(self.n_wt),
                       desc='y coordinates of turbines in global ref. frame', units=self.units)
        self.add_output('penalty_' + self.const_id, val=0.0)
        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', self.zeros,
                        desc="signed perpendicular distances from each turbine to each face CCW; + is inside")

        self.declare_partials('boundaryDistances', [topfarm.x_key, topfarm.y_key])
        # self.declare_partials('boundaryDistances', ['boundaryVertices', 'boundaryNormals'], method='fd')

    def compute(self, inputs, outputs):
        # calculate distances from each point to each face
        boundaryDistances = self.distances(x=inputs[topfarm.x_key], y=inputs[topfarm.y_key])
        outputs['boundaryDistances'] = boundaryDistances
        outputs['penalty_' + self.const_id] = -np.minimum(boundaryDistances, 0).sum()

    def compute_partials(self, inputs, partials):
        # return Jacobian dict
        dx, dy = self.gradients(**{xy: inputs[k] for xy, k in zip('xy', [topfarm.x_key, topfarm.y_key])})

        partials['boundaryDistances', topfarm.x_key] = dx
        partials['boundaryDistances', topfarm.y_key] = dy

    def plot(self, ax):
        """Plot boundary"""
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

        # loop through points and find distances to each face
        for i in range(0, nPoints):

            # determine if point is inside or outside of each face, and distances from each face
            for j in range(0, nVertices):

                # define the vector from the point of interest to the first point of the face
                pa = np.array([vertices[j, 0] - points[i, 0], vertices[j, 1] - points[i, 1]])

                # find perpendicular distances from point to current surface (vector projection)
                d_vec = np.vdot(pa, unit_normals[j]) * unit_normals[j]

                # calculate the sign of perpendicular distances from point to current face (+ is inside, - is outside)
                face_distance[i, j] = np.vdot(d_vec, unit_normals[j])
        # print (face_distance)
        return face_distance

    def distances(self, x, y):
        return self.calculate_distance_to_boundary(np.array([x, y]).T)

    def gradients(self, x, y):
        return self.dfaceDistance_dx, self.dfaceDistance_dy

    def satisfy(self, state, pad=1.1):
        x, y = [np.asarray(state[xyz], dtype=np.float) for xyz in [topfarm.x_key, topfarm.y_key]]
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
    def __init__(self, n_wt, xy_boundary, const_id=None, units=None):

        self.nTurbines = n_wt
        self.const_id = const_id
        self.zeros = np.zeros(self.nTurbines)
        vertices = np.array(xy_boundary)
        self.nVertices = vertices.shape[0]
        self.units = units

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

        xy_boundary, self.x1, self.y1, self.x2, self.y2 = edges_counter_clockwise(vertices)
        BoundaryBaseComp.__init__(self, n_wt, xy_boundary=xy_boundary, const_id=self.const_id, units=self.units)
        self.min_x, self.min_y = np.min([self.x1, self.x2], 0), np.min([self.y1, self.y2], )
        self.max_x, self.max_y = np.max([self.x1, self.x2], 1), np.max([self.y1, self.y2], 0)
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
        self.x2y1 = self.x2 * self.y1
        self.y2x1 = self.y2 * self.x1
        self.length = ((self.y2 - self.y1)**2 + (self.x2 - self.x1)**2)**0.5
        self.edge_unit_vec = (np.array([self.dy, -self.dx]) / self.length)
        v = np.hstack((self.edge_unit_vec, self.edge_unit_vec[:, :1]))
        self.xy2_vec = v[:, :-1] + v[:, 1:]
        self.xy1_vec = np.hstack((self.xy2_vec[:, -1:], self.xy2_vec[:, 1:]))

        self.dEdgeDist_dx = -self.dy / self.length
        self.dEdgeDist_dy = self.dx / self.length
        self._cache_input = None
        self._cache_output = None

    def calc_distance_and_gradients(self, x, y):
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
        if np.all(np.array([x, y]) == self._cache_input):
            return self._cache_output

        X, Y = [np.tile(xy, (len(self.x1), 1)).T for xy in [x, y]]  # dim = (ntb, nEdges)
        X1, Y1, X2, Y2, ddist_dX, ddist_dY = [np.tile(xy, (len(x), 1))
                                              for xy in [self.x1, self.y1, self.x2, self.y2, self.dEdgeDist_dx, self.dEdgeDist_dy]]

        # perpendicular distances to edge (dot product)
        d12 = (self.x1 - X) * self.edge_unit_vec[0] + (self.y1 - Y) * self.edge_unit_vec[1]

        # nearest point on edge
        px = X + d12 * self.edge_unit_vec[0]
        py = Y + d12 * self.edge_unit_vec[1]

        # distances to start and end points
        d1 = np.sqrt((self.x1 - X)**2 + (self.y1 - Y)**2)
        d2 = np.sqrt((self.x2 - X)**2 + (self.y2 - Y)**2)

        # use start or end point if nearest point is outside edge
        use_xy1 = (((self.dx != 0) & (px < self.x1) & (self.x1 < self.x2)) |
                   ((self.dx != 0) & (px > self.x1) & (self.x1 > self.x2)) |
                   ((self.dx == 0) & (py < self.y1) & (self.y1 < self.y2)) |
                   ((self.dx == 0) & (py > self.y1) & (self.y1 > self.y2)))
        use_xy2 = (((self.dx != 0) & (px > self.x2) & (self.x2 > self.x1)) |
                   ((self.dx != 0) & (px < self.x2) & (self.x2 < self.x1)) |
                   ((self.dx == 0) & (py > self.y2) & (self.y2 > self.y1)) |
                   ((self.dx == 0) & (py < self.y2) & (self.y2 < self.y1)))

        px[use_xy1] = X1[use_xy1]
        py[use_xy1] = Y1[use_xy1]
        px[use_xy2] = X2[use_xy2]
        py[use_xy2] = Y2[use_xy2]

        distance = d12.copy()
        v = (px[use_xy1] - X[use_xy1]) * self.xy1_vec[0, np.where(use_xy1)[1]] + \
            (py[use_xy1] - Y[use_xy1]) * self.xy1_vec[1, np.where(use_xy1)[1]]
        sign_use_xy1 = np.choose(v >= 0, [-1, 1])
        v = (px[use_xy2] - X[use_xy2]) * self.xy2_vec[0, np.where(use_xy2)[1]] + \
            (py[use_xy2] - Y[use_xy2]) * self.xy2_vec[1, np.where(use_xy2)[1]]
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
        x, y = [np.asarray(state[xy], dtype=np.float) for xy in [topfarm.x_key, topfarm.y_key]]
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
