import numpy as np
from openmdao.api import Group, IndepVarComp, ExecComp, ExplicitComponent, Problem
from scipy.spatial import ConvexHull


class BoundaryComp(ExplicitComponent):

    def __init__(self, vertices, nTurbines, boundary_type='convex_hull'):
        super(BoundaryComp, self).__init__()
        self.nTurbines = nTurbines
        self.vertices = np.array(vertices)
        self.calculate_boundary_and_normals(vertices, boundary_type)
        self.nVertices = self.vertices.shape[0]
        self.calculate_gradients()

    def calculate_boundary_and_normals(self, vertices, boundary_type):
        if boundary_type == 'convex_hull':
            # find the points that actually comprise a convex hull
            hull = ConvexHull(list(vertices))

            # keep only vertices that actually comprise a convex hull and arrange in CCW order
            vertices = np.array(vertices)[hull.vertices]
        elif boundary_type == 'square':
            min_ = np.array(vertices).min(0)
            max_ = np.array(vertices).max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_.max() / 2
            vertices = np.array([(x_c - r, y_c - r), (x_c + r, y_c - r), (x_c + r, y_c + r), (x_c - r, y_c + r)])
        elif boundary_type == 'rectangle':
            min_ = np.array(vertices).min(0)
            max_ = np.array(vertices).max(0)
            range_ = (max_ - min_)
            x_c, y_c = min_ + range_ / 2
            r = range_ / 2
            vertices = np.array([(x_c - r[0], y_c - r[1]), (x_c + r[0], y_c - r[1]), (x_c + r[0], y_c + r[1]), (x_c - r[0], y_c + r[1])])
        else:
            raise NotImplementedError("Boundary type '%s' is not implemented"%boundary_type)

        # get the real number of vertices
        nVertices = vertices.shape[0]

        # initialize normals array
        unit_normals = np.zeros([nVertices, 2])

        # determine if point is inside or outside of each face, and distance from each face
        for j in range(0, nVertices):

            # calculate the unit normal vector of the current face (taking points CCW)
            if j < nVertices - 1:  # all but the set of point that close the shape
                normal = np.array([vertices[j + 1, 1] - vertices[j, 1],
                                   -(vertices[j + 1, 0] - vertices[j, 0])])
                unit_normals[j] = normal / np.linalg.norm(normal)
            else:   # the set of points that close the shape
                normal = np.array([vertices[0, 1] - vertices[j, 1],
                                   -(vertices[0, 0] - vertices[j, 0])])
                unit_normals[j] = normal / np.linalg.norm(normal)

        self.vertices, self.unit_normals = vertices, unit_normals

    def calculate_gradients(self):
        unit_normals = self.unit_normals

        # initialize array to hold distances from each point to each face
        dfaceDistance_dx = np.zeros([self.nTurbines * self.nVertices, self.nTurbines])
        dfaceDistance_dy = np.zeros([self.nTurbines * self.nVertices, self.nTurbines])

        for i in range(0, self.nTurbines):
            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, self.nVertices):

                # define the derivative vectors from the point of interest to the first point of the face
                dpa_dx = np.array([-1.0, 0.0])
                dpa_dy = np.array([0.0, -1.0])

                # find perpendicular distance derivatives from point to current surface (vector projection)
                ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j]) * unit_normals[j]
                ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j]) * unit_normals[j]

                # calculate derivatives for the sign of perpendicular distance from point to current face
                dfaceDistance_dx[i * self.nVertices + j, i] = np.vdot(ddistanceVec_dx, unit_normals[j])
                dfaceDistance_dy[i * self.nVertices + j, i] = np.vdot(ddistanceVec_dy, unit_normals[j])

        # return Jacobian dict
        self.dfaceDistance_dx = dfaceDistance_dx
        self.dfaceDistance_dy = dfaceDistance_dy

    def calculate_distance_to_boundary(self, points):
        """
        :param points: points that you want to calculate the distance from to the faces of the convex hull
        :return face_distace: signed perpendicular distance from each point to each face; + is inside
        """

        nPoints = np.array(points).shape[0]
        nVertices = self.vertices.shape[0]
        vertices = self.vertices
        unit_normals = self.unit_normals
        # initialize array to hold distances from each point to each face
        face_distance = np.zeros([nPoints, nVertices])

        # loop through points and find distance to each face
        for i in range(0, nPoints):

            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, nVertices):

                # define the vector from the point of interest to the first point of the face
                pa = np.array([vertices[j, 0] - points[i, 0], vertices[j, 1] - points[i, 1]])

                # find perpendicular distance from point to current surface (vector projection)
                d_vec = np.vdot(pa, unit_normals[j]) * unit_normals[j]

                # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
                face_distance[i, j] = np.vdot(d_vec, unit_normals[j])
        #print (face_distance)
        return face_distance

    def setup(self):

        # Explicitly size input arrays
        self.add_input('turbineX', np.zeros(self.nTurbines), units='m',
                       desc='x coordinates of turbines in global ref. frame')
        self.add_input('turbineY', np.zeros(self.nTurbines), units='m',
                       desc='y coordinates of turbines in global ref. frame')

        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', np.zeros([self.nTurbines, self.nVertices]),
                        desc="signed perpendicular distance from each turbine to each face CCW; + is inside")

        self.declare_partials('boundaryDistances', ['turbineX', 'turbineY'])
        #self.declare_partials('boundaryDistances', ['boundaryVertices', 'boundaryNormals'], method='fd')

    def compute(self, inputs, outputs):

        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        # put locations in correct arrangement for calculations
        locations = np.zeros([self.nTurbines, 2])
        for i in range(0, self.nTurbines):
            locations[i] = np.array([turbineX[i], turbineY[i]])

        # print "in comp, locs are: ", locations

        # calculate distance from each point to each face
        outputs['boundaryDistances'] = self.calculate_distance_to_boundary(locations)

    def compute_partials(self, inputs, partials):
        # return Jacobian dict
        partials['boundaryDistances', 'turbineX'] = self.dfaceDistance_dx
        partials['boundaryDistances', 'turbineY'] = self.dfaceDistance_dy


class PolygonBoundaryComp(BoundaryComp):

    def __init__(self, vertices, nTurbines):

        super(BoundaryComp, self).__init__()

        self.nTurbines = nTurbines
        vertices = np.array(vertices)
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

        self.vertices, self.x1, self.y1, self.x2, self.y2 = edges_counter_clockwise(vertices)
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

    def setup(self):

        # Explicitly size input arrays
        self.add_input('turbineX', np.zeros(self.nTurbines), units='m',
                       desc='x coordinates of turbines in global ref. frame')
        self.add_input('turbineY', np.zeros(self.nTurbines), units='m',
                       desc='y coordinates of turbines in global ref. frame')

        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', np.zeros(self.nTurbines),
                        desc="signed perpendicular distance from each turbine to each face CCW; + is inside")

        self.declare_partials('boundaryDistances', ['turbineX', 'turbineY'])
        #self.declare_partials('boundaryDistances', ['boundaryVertices', 'boundaryNormals'], method='fd')

    def calc_distance_and_gradients(self, x, y):
        """
        distance point(x,y) to edge((x1,y1)->(x2,y2))
        +/-: inside/outside 
        case (x,y) closest to edge: 
            distance = edge_unit_vec dot (x1-x,y1-y)
            ddist_dx = -(y2-y2)/|edge|
            ddist_dy = (x2-x2)/|edge|
        case (x,y) closest to (x1,y1) (and (x2,y2)):
            sign = sign of distance to nearest edge
            distance = sign * (x1-x^2 + y1-y)^2)^.5
            ddist_dx = sign * 2*x-2*x1 / (2 * distance^.5)
            ddist_dy = sign * 2*y-2*y1 / (2 * distance^.5)
        """
        if np.all(np.array([x, y]) == self._cache_input):
            return self._cache_output
        
        X, Y = [np.tile(xy, (len(self.x1), 1)).T for xy in [x, y]]  # dim = (ntb, nEdges)
        X1, Y1, X2, Y2, ddist_dX, ddist_dY = [np.tile(xy, (len(x), 1))
                                              for xy in [self.x1, self.y1, self.x2, self.y2, self.dEdgeDist_dx, self.dEdgeDist_dy]]

        # perpendicular distance to edge (dot product)
        d12 = (self.x1 - X) * self.edge_unit_vec[0] + (self.y1 - Y) * self.edge_unit_vec[1]

        # nearest point on edge
        px = X + d12 * self.edge_unit_vec[0]
        py = Y + d12 * self.edge_unit_vec[1]

        # distance to start and end points
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

        self._cache_input = np.array([x,y])
        self._cache_output = [np.choose(closest_edge_index, v.T) for v in [distance, ddist_dX, ddist_dY]]
        return self._cache_output

    def compute(self, inputs, outputs):
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        outputs['boundaryDistances'] = self.calc_distance_and_gradients(turbineX, turbineY)[0]

    def compute_partials(self, inputs, partials):
        turbineX = inputs['turbineX']
        turbineY = inputs['turbineY']

        _, dx, dy = self.calc_distance_and_gradients(turbineX, turbineY)
        partials['boundaryDistances', 'turbineX'] = np.diagflat(dx)
        partials['boundaryDistances', 'turbineY'] = np.diagflat(dy)
