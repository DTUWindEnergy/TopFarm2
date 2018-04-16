import numpy as np
from openmdao.api import Group, IndepVarComp, ExecComp, ExplicitComponent, Problem
from scipy.spatial import ConvexHull
from wetb.utils.timing import print_cum_time
from networkx.algorithms.cuts import boundary_expansion


class BoundaryComp(ExplicitComponent):

    def __init__(self, vertices, nTurbines, boundary_type='convex_hull'):

        super(BoundaryComp, self).__init__()

        self.calculate_boundary_and_normals(vertices, boundary_type)
        self.nTurbines = nTurbines
        self.nVertices = self.vertices.shape[0]

    def calculate_boundary_and_normals(self, vertices, boundary_type):

        if boundary_type == 'convex_hull':
            # find the points that actually comprise a convex hull
            hull = ConvexHull(list(vertices))

            # keep only vertices that actually comprise a convex hull and arrange in CCW order
            vertices = np.array(vertices)[hull.vertices]
        else:
            raise NotImplementedError

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

        self.vertices, self.unit_normals =  vertices, unit_normals

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
        partials['boundaryDistances', 'turbineX'] = dfaceDistance_dx
        partials['boundaryDistances', 'turbineY'] = dfaceDistance_dy
