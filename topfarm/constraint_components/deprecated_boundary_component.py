import numpy as np
from openmdao.api import ExplicitComponent
from scipy.spatial import ConvexHull
import sys

# ==============================================================================
# This module is deprecated use topfarm.constraint_components.boundary instead
# ==============================================================================


def BoundaryComp(n_wt, xy_boundary, z_boundary=None, xy_boundary_type='convex_hull'):
    if xy_boundary_type == 'polygon':
        return PolygonBoundaryComp(n_wt, xy_boundary, z_boundary)
    else:
        return ConvexBoundaryComp(n_wt, xy_boundary, z_boundary, xy_boundary_type)


class BoundaryBaseComp(ExplicitComponent):
    def __init__(self, n_wt, xy_boundary=None, z_boundary=None, **kwargs):
        sys.stderr.write("%s is deprecated. Use BoundaryConstraint from topfarm.constraint_components.boundary instead\n" % self.__class__.__name__)

        ExplicitComponent.__init__(self, **kwargs)
        self.n_wt = n_wt
        if xy_boundary is None:
            self.xy_boundary = np.zeros((0, 2))
        else:
            self.xy_boundary = np.array(xy_boundary)
        if z_boundary is None:
            z_boundary = []
        if len(z_boundary) > 0:
            z_boundary = np.asarray(z_boundary)
            assert z_boundary.shape[-1] == 2
            if len(z_boundary.shape) == 1:
                z_boundary = np.zeros((self.n_wt, 2)) + [z_boundary]
            assert z_boundary.shape == (self.n_wt, 2)
            assert np.all(z_boundary[:, 0] < z_boundary[:, 1])
        self.z_boundary = z_boundary

#     def setup_as_constraints(self, problem):
#         if len(self.xy_boundary) > 0:
#             problem.model.add_subsystem('xy_bound_comp', self, promotes=['*'])
#             problem.model.add_constraint('boundaryDistances', lower=np.zeros(self.nVertices * self.n_wt))
#         if len(self.z_boundary):
#             problem.model.add_constraint(topfarm.z_key, lower=self.z_boundary[:, 0], upper=self.z_boundary[:, 1])
#
#     def setup_as_penalty(self, problem, penalty=1e10):
#         if len(self.xy_boundary) == 0 and len(self.z_boundary) == 0:
#             return  # no boundary or hub-height constraints
#
#         if len(self.xy_boundary) > 0:
#             subsystem_order = [ss.name for ss in problem.model._static_subsystems_allprocs]
#             problem.model.add_subsystem('xy_bound_comp', self, promotes=['*'])
#             subsystem_order.insert(subsystem_order.index('cost_comp'), 'xy_bound_comp')
#             problem.model.set_order(subsystem_order)
#
#             def xy_boundary_penalty(inputs):
#                 return -np.minimum(inputs['boundaryDistances'], 0).sum()
#         else:
#             def xy_boundary_penalty(inputs):
#                 return 0
#
#         if len(self.z_boundary):
#             def z_boundary_penalty(inputs):
#                 return -np.minimum(inputs[topfarm.z_key] - self.z_boundary[:, 0], 0).sum() + np.maximum(inputs[topfarm.z_key] - self.z_boundary[:, 1], 0).sum()
#         else:
#             def z_boundary_penalty(inputs):
#                 return 0
#
#         self._cost_comp = problem.cost_comp
#         self._org_setup = self._cost_comp.setup
#         self._org_compute = self._cost_comp.compute
#
#         def new_setup():
#             self._org_setup()
#             if len(self.xy_boundary) > 0:
#                 self._cost_comp.add_input('boundaryDistances', val=self.zeros)
#
#         self._cost_comp.setup = new_setup
#
#         def new_compute(inputs, outputs):
#             p = xy_boundary_penalty(inputs) + z_boundary_penalty(inputs)
#             if p == 0:
#                 self._org_compute(inputs, outputs)
#             else:
#                 outputs['cost'] = penalty + p
#         self._cost_comp.compute = new_compute
#         problem._mode = 'rev'


class ConvexBoundaryComp(BoundaryBaseComp):
    def __init__(self, n_wt, xy_boundary=None, z_boundary=None, xy_boundary_type='convex_hull'):
        super().__init__(n_wt, xy_boundary, z_boundary)
        if len(self.xy_boundary):
            self.boundary_type = xy_boundary_type
            self.calculate_boundary_and_normals()
            self.calculate_gradients()
            self.zeros = np.zeros([self.n_wt, self.nVertices])
        else:
            self.zeros = np.zeros([self.n_wt, 0])

    def calculate_boundary_and_normals(self):
        if self.boundary_type == 'convex_hull':
            # find the points that actually comprise a convex hull
            hull = ConvexHull(list(self.xy_boundary))

            # keep only xy_vertices that actually comprise a convex hull and arrange in CCW order
            self.xy_boundary = self.xy_boundary[hull.vertices]
        elif self.boundary_type == 'square':
            min_ = self.xy_boundary.min(0)
            max_ = self.xy_boundary.max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_.max() / 2
            self.xy_boundary = np.array([(x_c - r, y_c - r), (x_c + r, y_c - r), (x_c + r, y_c + r), (x_c - r, y_c + r)])
        elif self.boundary_type == 'rectangle':
            min_ = self.xy_boundary.min(0)
            max_ = self.xy_boundary.max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_ / 2
            self.xy_boundary = np.array([(x_c - r[0], y_c - r[1]), (x_c + r[0], y_c - r[1]), (x_c + r[0], y_c + r[1]), (x_c - r[0], y_c + r[1])])
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
        nVertices = self.xy_boundary.shape[0]
        vertices = self.xy_boundary
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

#     def setup(self):
#
#         # Explicitly size input arrays
#         self.add_input(topfarm.x_key, np.zeros(self.n_wt), units='m',
#                        desc='x coordinates of turbines in global ref. frame')
#         self.add_input(topfarm.y_key, np.zeros(self.n_wt), units='m',
#                        desc='y coordinates of turbines in global ref. frame')
#
#         # Explicitly size output array
#         # (vector with positive elements if turbines outside of hull)
#         self.add_output('boundaryDistances', self.zeros,
#                         desc="signed perpendicular distances from each turbine to each face CCW; + is inside")
#
#         self.declare_partials('boundaryDistances', [topfarm.x_key, topfarm.y_key])
#         # self.declare_partials('boundaryDistances', ['boundaryVertices', 'boundaryNormals'], method='fd')

    def distances(self, turbineX, turbineY):
        return self.calculate_distance_to_boundary(np.array([turbineX, turbineY]).T)

    def gradients(self, turbineX, turbineY):
        return self.dfaceDistance_dx, self.dfaceDistance_dy

#     def compute(self, inputs, outputs):
#         # calculate distances from each point to each face
#         outputs['boundaryDistances'] = self.distances(**inputs)
#
#     def compute_partials(self, inputs, partials):
#         # return Jacobian dict
#         dx, dy = self.gradients(**inputs)
#
#         partials['boundaryDistances', topfarm.x_key] = dx
#         partials['boundaryDistances', topfarm.y_key] = dy

    def move_inside(self, turbineX, turbineY, turbineZ, pad=1.1):
        x, y, z = [np.asarray(xyz, dtype=float) for xyz in [turbineX, turbineY, turbineZ]]
        dist = self.distances(turbineX, turbineY)
        dx, dy = self.gradients(x, y)  # independent of position
        dx = dx[:self.nVertices, 0]
        dy = dy[:self.nVertices, 0]
        for i in np.where(dist.min(1) < 0)[0]:  # loop over turbines that violate edges
            # find smallest movement that where the constraints are satisfied
            d = dist[i]
            v = np.linspace(-np.abs(d.min()), np.abs(d.min()), 100)
            X, Y = np.meshgrid(v, v)
            m = np.ones_like(X)
            for j in range(3):
                m = np.logical_and(m, X * dx[j] + Y * dy[j] >= -dist[i][j])
            index = np.argmin(X[m]**2 + Y[m]**2)
            x[i] += X[m][index]
            y[i] += Y[m][index]
        return x, y, z


class PolygonBoundaryComp(BoundaryBaseComp):
    def __init__(self, n_wt, xy_boundary=None, z_boundary=None, **kwargs):
        BoundaryBaseComp.__init__(self, n_wt, xy_boundary=xy_boundary, z_boundary=z_boundary, **kwargs)

        self.nTurbines = n_wt
        self.zeros = np.zeros(self.nTurbines)
        vertices = self.xy_boundary
        self.nVertices = vertices.shape[0]

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

        self.xy_boundary, self.x1, self.y1, self.x2, self.y2 = edges_counter_clockwise(vertices)
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

#     def setup(self):
#
#         # Explicitly size input arrays
#         self.add_input(topfarm.x_key, np.zeros(self.nTurbines), units='m',
#                        desc='x coordinates of turbines in global ref. frame')
#         self.add_input(topfarm.y_key, np.zeros(self.nTurbines), units='m',
#                        desc='y coordinates of turbines in global ref. frame')
#
#         # Explicitly size output array
#         # (vector with positive elements if turbines outside of hull)
#         self.add_output('boundaryDistances', self.zeros,
#                         desc="signed perpendicular distances from each turbine to each face CCW; + is inside")
#
#         self.declare_partials('boundaryDistances', [topfarm.x_key, topfarm.y_key])
#         # self.declare_partials('boundaryDistances', ['boundaryVertices', 'boundaryNormals'], method='fd')

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
        v = (px[use_xy1] - X[use_xy1]) * self.xy1_vec[0, np.where(use_xy1)[1]] + (py[use_xy1] - Y[use_xy1]) * self.xy1_vec[1, np.where(use_xy1)[1]]
        sign_use_xy1 = np.choose(v >= 0, [-1, 1])
        v = (px[use_xy2] - X[use_xy2]) * self.xy2_vec[0, np.where(use_xy2)[1]] + (py[use_xy2] - Y[use_xy2]) * self.xy2_vec[1, np.where(use_xy2)[1]]
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

    def distances(self, turbineX, turbineY):
        return self.calc_distance_and_gradients(turbineX, turbineY)[0]

    def gradients(self, turbineX, turbineY):
        _, dx, dy = self.calc_distance_and_gradients(turbineX, turbineY)
        return np.diagflat(dx), np.diagflat(dy)

    def move_inside(self, turbineX, turbineY, turbineZ, pad=1.1):
        x, y, z = [np.asarray(xyz, dtype=float) for xyz in [turbineX, turbineY, turbineZ]]
        dist = self.distances(turbineX, turbineY)
        dx, dy = map(np.diag, self.gradients(x, y))
        m = dist < 0
        x[m] -= dx[m] * dist[m] * pad
        y[m] -= dy[m] * dist[m] * pad
        return x, y, z
