import numpy as np
from numpy import newaxis as na
from scipy.spatial import ConvexHull
from topfarm.constraint_components import Constraint, ConstraintComponent
from topfarm.utils import smooth_max, smooth_max_gradient
import topfarm
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import warnings
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional


class XYBoundaryConstraint(Constraint):
    def __init__(self, boundary, boundary_type='convex_hull', units=None, relaxation=False, **kwargs):
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
            - 'turbine_specific': Set of multiple polygon boundaries that depend on the wind turbine type. \n


        """
        if boundary_type == 'multi_polygon':
            self.zones = boundary
            self.boundary = np.asarray(self.zones[0].boundary)
        elif boundary_type == 'turbine_specific':
            self.zones = boundary
            assert 'turbines' in list(kwargs)
            self.turbines = kwargs['turbines']
            self.boundary = np.asarray(self.zones[0].boundary)
        else:
            self.boundary = np.asarray(boundary)
        self.boundary_type = boundary_type
        self.const_id = 'xyboundary_comp_{}'.format(boundary_type)
        self.units = units
        self.relaxation = relaxation

    def get_comp(self, n_wt):
        if not hasattr(self, 'boundary_comp'):
            if self.boundary_type == 'polygon':
                self.boundary_comp = PolygonBoundaryComp(
                    n_wt, self.boundary, self.const_id, self.units, self.relaxation)
            elif self.boundary_type == 'multi_polygon':
                self.boundary_comp = MultiPolygonBoundaryComp(n_wt, self.zones, const_id=self.const_id, units=self.units, relaxation=self.relaxation)
            elif self.boundary_type == 'turbine_specific':
                self.boundary_comp = TurbineSpecificBoundaryComp(n_wt, self.turbines, self.zones, const_id=self.const_id, units=self.units, relaxation=self.relaxation)
            else:
                self.boundary_comp = ConvexBoundaryComp(n_wt, self.boundary, self.boundary_type, self.const_id, self.units)
        return self.boundary_comp

    @property
    def constraintComponent(self):
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        if self.boundary_type in ['multi_polygon', 'turbine_specific']:
            bound_min = np.vstack([(bound).min(0) for bound, _ in self.boundary_comp.boundaries]).min(0)
            bound_max = np.vstack([(bound).max(0) for bound, _ in self.boundary_comp.boundaries]).max(0)
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

    def _setup(self, problem, group='constraint_group'):
        n_wt = problem.n_wt
        self.boundary_comp = self.get_comp(n_wt)
        self.boundary_comp.problem = problem
        self.set_design_var_limits(problem.design_vars)
        # problem.xy_boundary = np.r_[self.boundary_comp.xy_boundary, self.boundary_comp.xy_boundary[:1]]
        problem.indeps.add_output('xy_boundary', self.boundary_comp.xy_boundary)
        getattr(problem.model, group).add_subsystem('xy_bound_comp', self.boundary_comp, promotes=['*'])

    def setup_as_constraint(self, problem, group='constraint_group'):
        self._setup(problem, group=group)
        if problem.n_wt == 1:
            lower = 0
        else:
            lower = self.boundary_comp.zeros
        problem.model.add_constraint('boundaryDistances', lower=lower)

    def setup_as_penalty(self, problem, group='constraint_group'):
        self._setup(problem, group=group)


class CircleBoundaryConstraint(XYBoundaryConstraint):
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

    def set_design_var_limits(self, design_vars):
        for k, l, u in zip([topfarm.x_key, topfarm.y_key],
                           self.center - self.radius,
                           self.center + self.radius):
            if len(design_vars[k]) == 4:
                design_vars[k] = (design_vars[k][0], np.maximum(design_vars[k][1], l),
                                  np.minimum(design_vars[k][2], u), design_vars[k][-1])
            else:
                design_vars[k] = (design_vars[k][0], l, u, design_vars[k][-1])


class BoundaryBaseComp(ConstraintComponent):
    def __init__(self, n_wt, xy_boundary=None, const_id=None, units=None, relaxation=False, **kwargs):
        super().__init__(**kwargs)
        self.n_wt = n_wt
        self.xy_boundary = np.array(xy_boundary)
        self.const_id = const_id
        self.units = units
        self.relaxation = relaxation
        if xy_boundary is not None and np.any(self.xy_boundary[0] != self.xy_boundary[-1]):
            self.xy_boundary = np.r_[self.xy_boundary, self.xy_boundary[:1]]

    def setup(self):
        # Explicitly size input arrays
        self.add_input(topfarm.x_key, np.zeros(self.n_wt),
                       desc='x coordinates of turbines in global ref. frame', units=self.units)
        self.add_input(topfarm.y_key, np.zeros(self.n_wt),
                       desc='y coordinates of turbines in global ref. frame', units=self.units)
        if self.relaxation:
            self.add_input('time', 0)
        if hasattr(self, 'types'):
            self.add_input('type', np.zeros(self.n_wt))
        # self.add_output('constraint_violation_' + self.const_id, val=0.0)
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
        args = {x: inputs[x] for x in [topfarm.x_key, topfarm.y_key, topfarm.type_key] if x in inputs}
        boundaryDistances = self.distances(**args)
        outputs['boundaryDistances'] = boundaryDistances
        # outputs['constraint_violation_' + self.const_id] = np.sum(np.minimum(boundaryDistances, 0) ** 2)

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
        if isinstance(self, TurbineSpecificBoundaryComp):
            linestyles = ['--', '-']
            colors = np.array(['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey'] * 10)
            for n, t in enumerate(self.types):
                line, = ax.plot(*self.ts_merged_xy_boundaries[n][0][0][0, :], color=colors[t], linewidth=1, label=f'{self.wind_turbines._names[n]} boundary')
                for bound, io in self.ts_merged_xy_boundaries[n]:
                    ax.plot(np.asarray(bound)[:, 0].tolist() + [np.asarray(bound)[0, 0]],
                            np.asarray(bound)[:, 1].tolist() + [np.asarray(bound)[0, 1]], color=colors[t], linewidth=1, linestyle=linestyles[io])
        elif isinstance(self, MultiPolygonBoundaryComp):
            colors = ['--k', 'k']
            if self.relaxation != 0:
                for bound, io in self.relaxed_polygons():
                    ax.plot(np.asarray(bound)[:, 0].tolist() + [np.asarray(bound)[0, 0]],
                            np.asarray(bound)[:, 1].tolist() + [np.asarray(bound)[0, 1]], c='r', linewidth=1, linestyle='--')
                ax.plot([], c='r', linewidth=1, linestyle='--', label='Relaxed boundaries')
            for bound, io in self.boundaries:
                ax.plot(np.asarray(bound)[:, 0].tolist() + [np.asarray(bound)[0, 0]],
                        np.asarray(bound)[:, 1].tolist() + [np.asarray(bound)[0, 1]], colors[io], linewidth=1)
        else:
            ax.plot(self.xy_boundary[:, 0].tolist() + [self.xy_boundary[0, 0]],
                    self.xy_boundary[:, 1].tolist() + [self.xy_boundary[0, 1]], 'k', linewidth=1)


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


@dataclass
class Boundary(object):
    _vertices: np.ndarray
    design_var_mask: np.ndarray
    is_inclusion: bool = True  # TODO: not implemented
    normals: np.ndarray = None

    @property
    def n_turbines(self):
        return self.design_var_mask.sum()

    @property
    def n_vertices(self):
        if np.all(self.vertices[0] == self.vertices[-1]):
            return self.vertices.shape[0] - 1
        return self.vertices.shape[0]

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

    def __post_init__(self):
        self.__validate()

    def __validate(self):
        self.vertices = np.asarray(self.vertices)
        assert self.vertices.ndim == 2, "Boundary must be a 2D array"
        assert any(
            [x for x in self.vertices.shape if x == 2]
        ), "Boundary must have shape (n, 2) or (2, n)"
        self.vertices = self.vertices.reshape(-1, 2)
        assert self.vertices.shape[0] > 2, "Boundary must have at least 3 vertices"
        assert isinstance(self.is_inclusion, bool), "is_inclusion must be a boolean"
        assert self.design_var_mask.ndim == 1, "design_var_mask must 1 dimensional"
        self.design_var_mask = np.asarray(self.design_var_mask, dtype=bool)


class MultiXYBoundaryConstraint(Constraint):
    def __init__(
        self,
        boundaries: list[Boundary],
        boundary_type="convex_hull",
        units=None,
        relaxation=False,
        **kwargs,
    ):
        if boundary_type != "convex_hull":
            raise NotImplementedError("Only 'convex_hull' type is implemented")
        if not isinstance(boundaries[0], Boundary):
            boundaries = [Boundary(*b) for b in boundaries]
        self.boundaries = boundaries
        self.boundary_type = boundary_type
        self.const_id = f"xyboundary_comp_{boundary_type}"
        self.units = units
        self.relaxation = relaxation

    def get_comp(self, n_wt):
        if hasattr(self, "boundary_comp"):
            return self.boundary_comp
        self.boundary_comp = MultiConvexBoundaryComp(
            n_wt,
            self.boundaries,
            self.const_id,
            self.units,
        )
        return self.boundary_comp

    @property
    def constraintComponent(self):
        assert hasattr(
            self, "boundary_comp"
        ), "Boundary component not initialized, call setup first"
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        _ = design_vars

    def _setup(self, problem, group="constraint_group"):
        n_wt = problem.n_wt
        self.boundary_comp = self.get_comp(n_wt)
        self.boundary_comp.problem = problem
        self.set_design_var_limits(problem.design_vars)
        problem.indeps.add_output("xy_boundary", self.boundary_comp.xy_boundary)
        getattr(problem.model, group).add_subsystem(
            "xy_bound_comp", self.boundary_comp, promotes=["*"]
        )

    def setup_as_constraint(self, problem, group="constraint_group"):
        self._setup(problem, group=group)
        lower = 0 if problem.n_wt == 1 else self.boundary_comp.zeros
        problem.model.add_constraint("boundaryDistances", lower=lower)

    def setup_as_penalty(self, problem, group="constraint_group"):
        self._setup(problem, group=group)


class MultiConvexBoundaryComp(BoundaryBaseComp):
    def __init__(
        self,
        n_wt,
        xy_boundaries: list[Boundary],
        const_id=None,
        units=None,
    ):
        self.xy_boundaries = self.sort_boundaries(xy_boundaries)
        self.xy_boundaries = self.calculate_boundary_and_normals(self.xy_boundaries)
        super().__init__(n_wt, None, const_id, units)
        self.turbine_vertice_prod = 0
        total_n_active_turbines = 0
        for b in self.xy_boundaries:
            if np.any(b.vertices[0] != b.vertices[-1]):
                b.vertices = np.r_[b.vertices, b.vertices[:1]]
            self.turbine_vertice_prod += b.n_turbines * b.n_vertices
            total_n_active_turbines += b.n_turbines
        assert (
            total_n_active_turbines == n_wt
        ), "Number of active turbines in boundaries must match number of total turbines; Check that masks sum up to n_wt i.e. np.concatenate(all_masks).sum() == n_wt."
        self.calculate_gradients()
        self.zeros = np.zeros(self.turbine_vertice_prod)

    def sort_boundaries(self, boundaries):
        def centroid(boundary):  # fmt: skip
            return tuple(np.mean(boundary.vertices, axis=0))
        return sorted(boundaries, key=lambda b: centroid(b))

    def calculate_boundary_and_normals(
        self, xy_boundaries: list[Boundary]
    ) -> list[Boundary]:
        def __compute_normal(boundary_pts, ii, jj):
            """Calculate the unit normal vector of the current face (taking points CCW)"""
            normal = np.array(
                [
                    boundary_pts[ii, 1] - boundary_pts[jj, 1],
                    -(boundary_pts[ii, 0] - boundary_pts[jj, 0]),
                ]
            )
            return normal / np.linalg.norm(normal)

        for boundary in xy_boundaries:
            hull = ConvexHull(list(boundary.vertices))
            # keep only vertices that actually comprise a convex hull and arrange in CCW order
            boundary.vertices = boundary.vertices[hull.vertices]
            # initialize normals array
            unit_normals = np.zeros([boundary.n_vertices, 2])
            # determine if point is inside or outside and distances from each face
            nvtm1 = boundary.n_vertices - 1
            for j in range(0, nvtm1):
                # all but the points that close the shape
                unit_normals[j] = __compute_normal(boundary.vertices, j + 1, j)
            # points that close the shape
            unit_normals[nvtm1] = __compute_normal(boundary.vertices, 0, nvtm1)
            boundary.normals = unit_normals

        return xy_boundaries

    def calculate_gradients(self):
        # this is flawed if the order of boundaries is switched;
        # test with arbitrary number of vertices and arbitrary number
        # of turbines in a boundary; For now it's sorted at the top..
        final_dx = np.zeros([self.turbine_vertice_prod, self.n_wt])
        final_dy = np.zeros([self.turbine_vertice_prod, self.n_wt])
        for bi, boundary in enumerate(self.xy_boundaries):
            assert (
                boundary.design_var_mask is not None
            ), "Design variable mask must be provided"
            assert self.n_wt == len(
                boundary.design_var_mask
            ), "Design variable mask must must be the same length as the number of turbines"

            unit_normals = boundary.normals
            n_vertices = boundary.n_vertices
            n_turbines = boundary.n_turbines
            dfaceDistance_dx = np.zeros([n_turbines * n_vertices, n_turbines])
            dfaceDistance_dy = np.zeros([n_turbines * n_vertices, n_turbines])
            for i in range(0, n_turbines):
                # determine if point is inside or outside of each face, and distances from each face
                for j in range(0, n_vertices):
                    # define the derivative vectors from the point of interest to the first point of the face
                    dpa_dx = np.array([-1.0, 0.0])
                    dpa_dy = np.array([0.0, -1.0])
                    # find perpendicular distances derivatives from point to current surface (vector projection)
                    ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j]) * unit_normals[j]
                    ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j]) * unit_normals[j]
                    # calculate derivatives for the sign of perpendicular distances from point to current face
                    dfaceDistance_dx[i * n_vertices + j, i] = np.vdot(
                        ddistanceVec_dx, unit_normals[j]
                    )
                    dfaceDistance_dy[i * n_vertices + j, i] = np.vdot(
                        ddistanceVec_dy, unit_normals[j]
                    )
            seek = sum([(b.n_vertices * b.n_turbines) for b in self.xy_boundaries[:bi]])
            final_dx[
                seek: seek + (n_turbines * n_vertices), boundary.design_var_mask
            ] = dfaceDistance_dx
            final_dy[
                seek: seek + (n_turbines * n_vertices), boundary.design_var_mask
            ] = dfaceDistance_dy
        dfaceDistance_dx = final_dx
        dfaceDistance_dy = final_dy

        def __wrap_sparse(m):  # fmt: skip
            if m.size < 1e4:
                return m
            from scipy.sparse import csr_matrix  # fmt: skip
            return csr_matrix(m)
        # store Jacobians
        self.dfaceDistance_dx = __wrap_sparse(dfaceDistance_dx)
        self.dfaceDistance_dy = __wrap_sparse(dfaceDistance_dy)

    def distances(self, x, y):
        """
        :param points: points that you want to calculate the distances from to the faces of the convex hull
        :return face_distace: signed perpendicular distances from each point to each face; + is inside
        """
        points = np.array([x, y]).T
        face_distances = np.zeros(self.turbine_vertice_prod)
        for bi, boundary in enumerate(self.xy_boundaries):
            mask = boundary.design_var_mask
            vertices = boundary.vertices[:-1]
            n_vertices = boundary.n_vertices
            PA = vertices[:, na] - points[mask][na]
            dist = np.sum(PA * boundary.normals[:, na], axis=2)
            d_vec = dist[:, :, na] * boundary.normals[:, na]
            seek = sum([(b.n_vertices * b.n_turbines) for b in self.xy_boundaries[:bi]])
            face_distances[seek: seek + (boundary.n_turbines * n_vertices)] = np.sum(
                d_vec * boundary.normals[:, na], axis=2
            ).T.reshape(-1)
        return face_distances

    def gradients(self, x, y):
        return self.dfaceDistance_dx, self.dfaceDistance_dy

    def satisfy(self, state):
        raise NotImplementedError("Not implemented for MultiConvexBoundaryComp")

    def plot(self, ax):
        for b in self.xy_boundaries:
            ax.plot(
                b.vertices[:, 0].tolist(),
                b.vertices[:, 1].tolist(),
                "k",
                linewidth=1,
            )


class PolygonBoundaryComp(BoundaryBaseComp):
    def __init__(self, n_wt, xy_boundary, const_id=None, units=None, relaxation=False):

        self.nTurbines = n_wt
        self.const_id = const_id
        self.zeros = np.zeros(self.nTurbines)
        self.units = units
        self.boundary_properties = self.get_boundary_properties(xy_boundary)
        BoundaryBaseComp.__init__(self, n_wt, xy_boundary=self.boundary_properties[0], const_id=const_id,
                                  units=units, relaxation=relaxation)
        self._cache_input = None
        self._cache_output = None
        self.relaxation = relaxation

    def get_boundary_properties(self, xy_boundary, inclusion_zone=True):
        vertices = np.array(xy_boundary)

        def get_edges(vertices, counter_clockwise):
            if np.any(vertices[0] != vertices[-1]):
                vertices = np.r_[vertices, vertices[:1]]
            x1, y1 = A = vertices[:-1].T
            x2, y2 = B = vertices[1:].T
            double_area = np.sum((x1 - x2) * (y1 + y2))  # 2 x Area (+: counterclockwise
            assert double_area != 0, "Area must be non-zero"
            if (counter_clockwise and double_area < 0) or (not counter_clockwise and double_area > 0):  #
                return get_edges(vertices[::-1], counter_clockwise)
            else:
                return vertices[:-1], A, B

        # inclusion zones are defined counter clockwise (unit-normal vector pointing in) while
        # exclusion zones are defined clockwise (unit-normal vector pointing out)
        xy_boundary, A, B = get_edges(vertices, inclusion_zone)

        dx, dy = AB = B - A
        AB_len = np.linalg.norm(AB, axis=0)
        edge_unit_normal = (np.array([-dy, dx]) / AB_len)

        # A_normal and B_normal are the normal vectors at the nodes A,B (the mean of the adjacent edge normal vectors
        A_normal = (edge_unit_normal + np.roll(edge_unit_normal, 1, 1)) / 2
        B_normal = np.roll(A_normal, -1, 1)

        # import matplotlib.pyplot as plt
        # for (x, y), (dx, dy), (unx, uny) in zip(A.T, AB.T, edge_unit_normal.T):
        #     plt.arrow(x, y, dx, dy, color='k', head_width=.2)
        #     plt.arrow(x, y, unx, uny, color='r', head_width=.2)
        # for (x, y), (nx, ny) in zip(A.T, A_normal.T):
        #     plt.arrow(x, y, nx, ny, color='b', head_width=.2)
        # for (x, y), (nx, ny) in zip(B.T, B_normal.T):
        #     plt.arrow(x, y, nx / 2, ny / 2, color='g', head_width=.2)

        return (xy_boundary, A, B, AB, AB_len, edge_unit_normal, A_normal, B_normal)

    def _calc_distance_and_gradients(self, x, y, boundary_properties=None):
        """
        distances point, P=(x,y) to edge(A->B)
        +/-: inside/outside
        """
        def vec_len(vec):
            return np.linalg.norm(vec, axis=0)

        boundary_properties = boundary_properties or self.boundary_properties[1:]
        A, B, AB, AB_len, edge_unit_normal, A_normal, B_normal = boundary_properties
        """
        A: edge start point
        B: edge end point
        edge_unit_normal: unit vector perpendicular to edge pointing to the good side
        (i.e. inside for inclusion zones and outside for exclusion zones)
        AB: Vector from A to B (edge)
        AB_len: length of AB (edge)
        A_normal: mean of edge unit normal vectors adjacent to A
        B_normal: mean of edge unit normal vectors adjacent to B
        """

        # Add dim to match (2, #P, #Edges), where the first dimension is (x,y)
        P = np.array([x, y])[:, :, na]
        A, B, AB = A[:, na], B[:, na], AB[:, na]
        edge_unit_normal, A_normal, B_normal = edge_unit_normal[:, na], A_normal[:, na], B_normal[:, na]
        AB_len = AB_len[na]

        # ===============================================================================================================
        # Determine if P is closer to A, B or the edge (between A and B)
        # ===============================================================================================================
        AP = P - A  # vector from edge start to point
        BP = P - B  # vector from edge end to point

        # signed component of AP on the edge vector
        a_tilde = np.sum(AP * AB, axis=0) / AB_len

        # a_tilde < 0: closer to A
        # a_tilde > |AB|: closer to B
        # else: closer to edge (between A and B)
        use_A = 0 > a_tilde
        use_B = a_tilde > AB_len

        # ===============================================================================================================
        # Calculate distance from P to closer point on edge
        # ===============================================================================================================

        # Perpendicular distances to edge (AP dot edge_unit_normal product).
        # This is the distance to the edge if not use_A or use_B
        distance = np.sum((AP) * edge_unit_normal, 0)

        # Update distance for points closer to A
        good_side_of_A = (np.sum((AP * A_normal)[:, use_A], 0) > 0)
        sign_use_A = np.where(good_side_of_A, 1, -1)
        distance[use_A] = (vec_len(AP[:, use_A]) * sign_use_A)

        # Update distance for points closer to B
        good_side_of_B = np.sum((BP * B_normal)[:, use_B], 0) > 0
        sign_use_B = np.where(good_side_of_B, 1, -1)
        distance[use_B] = (vec_len(BP[:, use_B]) * sign_use_B)

        # ===============================================================================================================
        # Calculate gradient of distance from P to closer point on edge wrt. x and y
        # ===============================================================================================================

        # Gradient of perpendicular distances to edge.
        # This is the gradient if not use_A or use_B
        ddist_dxy = np.tile(edge_unit_normal, (1, len(x), 1))

        # Update gradient for points closer to A or B
        eps = 1e-7  # avoid division by zero
        ddist_dxy[:, use_A] = sign_use_A * (AP[:, use_A] / (vec_len(AP[:, use_A]) + eps))
        ddist_dxy[:, use_B] = sign_use_B * (BP[:, use_B] / (vec_len(BP[:, use_B]) + eps))
        ddist_dX, ddist_dY = ddist_dxy

        return distance, ddist_dX, ddist_dY

    def calc_distance_and_gradients(self, x, y):
        if not np.shape([x, y]) == np.shape(self._cache_input):
            pass
        elif np.all(np.array([x, y]) == self._cache_input):
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


class Zone(object):
    def __init__(self, boundary, dist2wt, geometry_type, incl, name):
        self.name = name
        self.boundary = boundary
        self.dist2wt = dist2wt
        self.geometry_type = geometry_type
        self.incl = incl


class InclusionZone(Zone):
    def __init__(self, boundary, dist2wt=None, geometry_type='polygon', name=''):
        super().__init__(np.asarray(boundary), dist2wt, geometry_type, incl=1, name=name)


class ExclusionZone(Zone):
    def __init__(self, boundary, dist2wt=None, geometry_type='polygon', name=''):
        super().__init__(np.asarray(boundary), dist2wt, geometry_type, incl=0, name=name)


class MultiPolygonBoundaryComp(PolygonBoundaryComp):
    def __init__(self, n_wt, zones, const_id=None, units=None, relaxation=False, method='nearest',
                 simplify_geometry=False):
        '''
        Parameters
        ----------
        n_wt : TYPE
            DESCRIPTION.
        zones : list
            list of InclusionZone and ExclusionZone objects
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
        self.zones = zones
        self.bounds_poly, xy_boundaries = self.get_xy_boundaries()
        PolygonBoundaryComp.__init__(self, n_wt, xy_boundary=xy_boundaries[0], const_id=const_id, units=units, relaxation=relaxation)
        # self.bounds_poly = [Polygon(x) for x in xy_boundaries]
        self.incl_excls = [x.incl for x in zones]
        self._setup_boundaries(self.bounds_poly, self.incl_excls)
        self.relaxation = relaxation
        self.method = method
        if simplify_geometry:
            self.simplify(simplify_geometry)

    def simplify(self, simplify_geometry):
        bounds = [bi[0] for bi in self.boundaries]
        self.incl_excls = [bi[1] for bi in self.boundaries]
        polygons = [Polygon(b) for b in bounds]
        if isinstance(simplify_geometry, dict):
            self.bounds_poly = [rp.simplify(**simplify_geometry) for rp in polygons]
        else:
            self.bounds_poly = [rp.simplify(simplify_geometry) for rp in polygons]
        self._setup_boundaries(self.bounds_poly, self.incl_excls)

    # def line_to_xy_boundary(self, line, buffer):
    #     return np.asarray(Polygon(LineString(line).buffer(buffer, join_style=2).exterior).exterior.coords)

    def get_xy_boundaries(self):
        polygons = []
        bounds = []
        for z in self.zones:
            if hasattr(z.dist2wt, '__code__'):
                buffer = z.dist2wt(**{k: 100 for k in z.dist2wt.__code__.co_varnames})
            else:
                buffer = 0
            if z.geometry_type == 'line':
                poly = Polygon(LineString(z.boundary).buffer(buffer, join_style=2).exterior)
            elif z.geometry_type == 'polygon':
                poly = Polygon(z.boundary).buffer(buffer, join_style=2)
            polygons.append(poly)
            bounds.append(np.asarray(poly.exterior.coords))
        return polygons, bounds

    def _setup_boundaries(self, bounds_poly, incl_excl):
        self.res_poly = self._calc_resulting_polygons(bounds_poly, incl_excl)
        self.boundaries = self._poly_to_bound(self.res_poly)

        boundary_properties_list_all = list(zip(*[self.get_boundary_properties(bound, incl_excl)[1:]
                                                  for bound, incl_excl in self.boundaries]))

        self.boundary_properties_list_all = [np.concatenate(v, -1)
                                             for v in boundary_properties_list_all]

    def _poly_to_bound(self, polygons):
        boundaries = []
        for bound in polygons:
            x, y = bound.exterior.xy
            boundaries.append((np.asarray([x, y]).T[:-1, :], 1))
            for interior in bound.interiors:
                x, y = interior.xy
                boundaries.append((np.asarray([x, y]).T[:-1, :], 0))
        return boundaries

    def _calc_resulting_polygons(self, boundary_polygons, incl_excls):
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
        for i in tqdm(range(len(boundary_polygons))):
            b = boundary_polygons[i]
            if len(domain) == 0:
                if incl_excls[i]:
                    domain.append(b)
                else:
                    warnings.warn("First boundary should be an inclusion zone or it will be ignored")
                    pass
            else:
                if incl_excls[i]:
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
        if not np.shape([x, y]) == np.shape(self._cache_input):
            pass
        elif np.all(np.array([x, y]) == self._cache_input) & (not self.relaxation):
            return self._cache_output

        Dist_ij, ddist_dX, ddist_dY = self._calc_distance_and_gradients(x, y, self.boundary_properties_list_all)

        dDdk_ijk = np.moveaxis([ddist_dX, ddist_dY], 0, -1)
        sign_i = self.sign(Dist_ij)
        self._cache_input = np.array([x, y])
        self._cache_output = [Dist_ij, dDdk_ijk, sign_i]
        return self._cache_output

    def calc_relaxation(self, iteration_no=None):
        '''
        The tupple relaxation contains a first term for the penalty constant
        and a second term for the n first iterations to apply relaxation.
        '''
        if iteration_no is None:
            iteration_no = self.problem.cost_comp.n_grad_eval + 1
        return max(0, self.relaxation[0] * (self.relaxation[1] - iteration_no))

    def distances(self, x, y):
        Dist_ij, _, sign_i = self.calc_distance_and_gradients(x, y)
        if self.method == 'smooth_min':
            Dist_i = smooth_max(np.abs(Dist_ij), -np.abs(Dist_ij).max(), axis=1) * sign_i
        elif self.method == 'nearest':
            Dist_i = Dist_ij[np.arange(x.size), np.argmin(np.abs(Dist_ij), axis=1)]
        else:
            warning = f'method: {self.method} is not implemented. Available options are smooth_min and nearest.'
            warnings.warn(warning)
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
        if self.relaxation:
            Dist_ij += self.calc_relaxation()
            # dDdt = -self.relaxation[1]
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

    def relaxed_polygons(self, iteration_no=None):
        poly = [Polygon(x.boundary) for x in self.zones]
        booleans = [x.incl for x in self.zones]
        relaxed_poly = []
        for i, p in enumerate(poly):
            if booleans[i] == 0:
                pb = p.buffer(-self.calc_relaxation(iteration_no), join_style=2)
                relaxed_poly.append(pb)
            else:
                pb = p.buffer(self.calc_relaxation(iteration_no), join_style=2)
                relaxed_poly.append(pb)
        merged_poly = self._calc_resulting_polygons(relaxed_poly, booleans)
        return self._poly_to_bound(merged_poly)


class TurbineSpecificBoundaryComp(MultiPolygonBoundaryComp):
    def __init__(self, n_wt, wind_turbines, zones, const_id=None,
                 units=None, relaxation=False, method='nearest', simplify_geometry=False):
        # self.dependencies = [d or {'type': None, 'multiplier': None, 'ref': None} for d in dependencies]
        # self.multi_boundary = xy_boundaries = self.get_xy_boundaries(boundaries, geometry_types, incl_excls)
        self.wind_turbines = wind_turbines
        self.types = wind_turbines.types()
        # self.incl_excls = incl_excls
        self.n_wt = n_wt
        self.zones = zones
        self.ts_polygon_boundaries, ts_xy_boundaries = self.get_ts_boundaries()
        MultiPolygonBoundaryComp.__init__(self, n_wt=n_wt, zones=zones, const_id=const_id, units=units,
                                          relaxation=relaxation, method=method, simplify_geometry=simplify_geometry)
        # self.polygon_boundaries = [Polygon(x) for x, _ in xy_boundaries]
        # self.ts_polygon_boundaries = self.get_ts_polygon_boundaries(self.types)
        self.ts_merged_polygon_boundaries = self.merge_boundaries()
        self.ts_merged_xy_boundaries = self.get_ts_xy_boundaries()
        self.ts_boundary_properties = self.get_ts_boundary_properties()
        self.ts_item_indices = self.get_ts_item_indices()

    def get_ts_boundaries(self):
        polygons = []
        bounds = []
        for t in set(self.types):
            temp1 = []
            temp2 = []
            dist2wt_input = dict(D=self.wind_turbines.diameter(t),
                                 H=self.wind_turbines.hub_height(t))
            for z in self.zones:
                if hasattr(z.dist2wt, '__code__'):
                    buffer = z.dist2wt(**{k: dist2wt_input[k] for k in z.dist2wt.__code__.co_varnames})
                else:
                    buffer = 0
                if z.geometry_type == 'line':
                    poly = Polygon(LineString(z.boundary).buffer(buffer, join_style=2).exterior)
                elif z.geometry_type == 'polygon':
                    poly = Polygon(z.boundary).buffer(buffer, join_style=2)
                bound = np.asarray(poly)
                temp1.append(poly)
                temp2.append(bound)
            polygons.append(temp1)
            bounds.append(temp2)
        return polygons, bounds
        # for wt
        # temp = []
        # for n, (b, t, ie) in enumerate(zip(boundaries, geometry_types, incl_excls)):
        #     if t == 'line':
        #         bound = np.asarray(Polygon(LineString(b).buffer(default_ref, join_style=2).exterior).exterior.coords)
        #         self.dependencies[n]['ref'] = default_ref
        #     elif t == 'polygon':
        #         bound = b
        #     else:
        #         raise NotImplementedError("Geometry type '%s' is not implemented" % b)
        #     temp.append((bound, ie))

    # def get_ts_polygon_boundaries(self, types):
    #     temp = []
    #     for t in set(types):
    #         d = self.wind_turbines.diameter(t)
    #         h = self.wind_turbines.hub_height(t)
    #         temp.append(self.get_ts_polygon_boundary(d, h))
    #     return temp

    def get_ts_xy_boundaries(self):
        return [self._poly_to_bound(b) for b in self.ts_merged_polygon_boundaries]

    # def get_ts_polygon_boundary(self, d=None, h=None):
    #     temp = []
    #     for bound, dep in zip(self.polygon_boundaries, self.dependencies):
    #         ref = dep['ref'] or 0
    #         if dep['type'] == 'D':
    #             ts_polygon_boundary = bound.buffer(dep['multiplier']*d-ref, join_style=2)
    #         elif dep['type'] == 'H':
    #             ts_polygon_boundary = bound.buffer(dep['multiplier']*h-ref, join_style=2)
    #         else:
    #             ts_polygon_boundary = bound
    #         temp.append(ts_polygon_boundary)
    #     return temp

    def merge_boundaries(self):
        return [self._calc_resulting_polygons(bounds, self.incl_excls) for bounds in self.ts_polygon_boundaries]

    def get_ts_boundary_properties(self,):
        return [[self.get_boundary_properties(bound) for bound, _ in bounds] for bounds in self.ts_merged_xy_boundaries]

    def get_ts_item_indices(self):
        temp = []
        for bounds in self.ts_merged_xy_boundaries:
            n_edges = np.asarray([len(bound) for bound, _ in bounds])
            n_edges_tot = np.sum(n_edges)
            start_at = np.cumsum(n_edges) - n_edges
            end_at = start_at + n_edges
            item_indices = [n_edges_tot, start_at, end_at]
            temp.append(item_indices)
        return temp

    def calc_distance_and_gradients(self, x, y, types=None):
        if self._cache_input is None:
            pass
        elif not np.shape([x, y]) == np.shape(self._cache_input[0]) or not np.shape(types) == np.shape(self._cache_input[1]):
            pass
        elif np.all(np.array([x, y]) == self._cache_input[0]) & (not self.relaxation) & np.all(np.asarray([types]) == self._cache_input[1]):
            return self._cache_output
        if types is None:
            types = np.zeros(self.n_wt)
        Dist_i = np.zeros(self.n_wt)
        sign_i = np.zeros(self.n_wt)
        dDdx_i = np.zeros(self.n_wt)
        dDdy_i = np.zeros(self.n_wt)
        for t in set(types):
            t = int(t)
            idx = (types == t)
            n_edges_tot, start_at, end_at = self.ts_item_indices[t]
            Dist_ij = np.zeros((sum(idx), n_edges_tot))
            dDdk_ijk = np.zeros((sum(idx), n_edges_tot, 2))
            for n, (bound, bound_type) in enumerate(self.ts_merged_xy_boundaries[t]):
                sa = start_at[n]
                ea = end_at[n]
                distance, ddist_dX, ddist_dY = self._calc_distance_and_gradients(x[idx], y[idx], self.ts_boundary_properties[t][n][1:])
                if bound_type == 0:
                    distance *= -1
                    ddist_dX *= -1
                    ddist_dY *= -1
                Dist_ij[:, sa:ea] = distance
                dDdk_ijk[:, sa:ea, 0] = ddist_dX
                dDdk_ijk[:, sa:ea, 1] = ddist_dY

            sign_i[idx] = self.sign(Dist_ij)
            Dist_i[idx] = Dist_ij[np.arange(sum(idx)), np.argmin(np.abs(Dist_ij), axis=1)]
            dDdx_i[idx], dDdy_i[idx] = dDdk_ijk[np.arange(sum(idx)), np.argmin(np.abs(Dist_ij), axis=1), :].T
        self._cache_input = (np.array([x, y]), np.asarray(types))
        self._cache_output = [Dist_i, dDdx_i, dDdy_i, sign_i]
        return self._cache_output

    def distances(self, x, y, type=None):
        Dist_i, _, _, _ = self.calc_distance_and_gradients(x, y, types=type)
        if self.relaxation:
            Dist_i += self.calc_relaxation()
        return Dist_i

    def gradients(self, x, y, type=None):
        Dist_i, dDdx_i, dDdy_i, _ = self.calc_distance_and_gradients(x, y, types=type)
        if self.relaxation:
            Dist_i += self.calc_relaxation()
            dDdt = -self.relaxation[0]
        if self.relaxation:
            gradients = np.diagflat(dDdx_i), np.diagflat(dDdy_i), np.ones(self.n_wt) * dDdt
        else:
            gradients = np.diagflat(dDdx_i), np.diagflat(dDdy_i)
        return gradients


class MultiCircleBoundaryConstraint(XYBoundaryConstraint):
    def __init__(self, center, radius, masks):
        """Initialize CircleBoundaryConstraint

        Parameters
        ----------
        center : (float, float)
            center position (x,y)
        radius : int or float
            circle radius
        """

        self.center = np.array(center)
        self.radius = np.array(radius)
        self.masks = np.array(masks)
        assert (
            len(self.center) == len(self.radius) == len(self.masks)
        ), f"Lenght of center, radius and masks must be equal"
        assert len(self.center) > 1
        self.const_id = f"circle_boundary_comp_{id(self)}"

    def get_comp(self, n_wt):
        if not hasattr(self, "boundary_comp"):
            self.boundary_comp = MultiCircleBoundaryComp(
                n_wt, self.center, self.radius, self.masks, self.const_id
            )
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        _ = design_vars
        return


class MultiCircleBoundaryComp(PolygonBoundaryComp):

    def __init__(self, n_wt, center, radius, masks, const_id=None, units=None):
        self.center = center
        self.radius = radius
        self.masks = masks
        xy_boundary = None  # TODO: redundant
        BoundaryBaseComp.__init__(self, n_wt, xy_boundary, const_id, units)
        self.zeros = np.zeros(self.n_wt)

    def plot(self, ax=None):
        from matplotlib.pyplot import Circle
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        for center, radius in zip(self.center, self.radius):
            circle = Circle(center, radius, color="k", fill=False)
            ax.add_artist(circle)

    def distances(self, x, y):
        assert (
            x.shape == y.shape == self.masks[0].shape
        ), f"{x.shape} != {y.shape} != {self.masks[0].shape}"
        distances = np.zeros_like(x)
        for center, radius, mask in zip(self.center, self.radius, self.masks):
            distances += mask * (
                radius - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            )
        return distances

    def gradients(self, x, y):
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        for center, radius, mask in zip(self.center, self.radius, self.masks):
            theta = np.arctan2(y - center[1], x - center[0])
            dx_tmp = -1 * np.ones_like(x)
            dy_tmp = -1 * np.ones_like(x)
            dist = radius - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            not_center = dist != radius
            dx_tmp[not_center], dy_tmp[not_center] = mask[not_center] * -np.cos(
                theta[not_center]
            ), mask[not_center] * -np.sin(theta[not_center])
            dx += dx_tmp
            dy += dy_tmp
        return np.diagflat(dx), np.diagflat(dy)


class MultiWFPolygonBoundaryConstraint(XYBoundaryConstraint):
    def __init__(self, boundaries, turbine_groups):
        """Initialize CircleBoundaryConstraint

        Parameters
        ----------
        center : (float, float)
            center position (x,y)
        radius : int or float
            circle radius
        """
        self.boundaries = boundaries
        self.turbine_groups = turbine_groups
        self.const_id = f"polygon_boundary_comp_{id(self)}"

    def get_comp(self, n_wt):
        if not hasattr(self, "boundary_comp"):
            self.boundary_comp = MultiWFPolygonBoundaryComp(
                n_wt, self.boundaries, self.turbine_groups, const_id=self.const_id
            )
        return self.boundary_comp

    def set_design_var_limits(self, design_vars):
        _ = design_vars
        return


class MultiWFPolygonBoundaryComp(PolygonBoundaryComp):
    def __init__(
        self,
        n_wt: int,
        boundaries: Optional[Dict[int, np.ndarray]],
        turbine_groups: Optional[Dict[int, List[int]]],
        *args,
        **kwargs,
    ):
        """Initialize boundary and group assignments.

        Args:
            num_turbines: Total number of turbines
            boundaries: Dictionary mapping group IDs to boundary coordinates
            turbine_groups: Dictionary mapping group IDs to lists of turbine indices
        """
        if n_wt <= 0:
            raise ValueError("Number of turbines must be positive")

        self.boundaries = {}  # group_id: boundary_coords
        for group_id, boundary_coords in boundaries.items():
            boundary_coords = self.__validate_boundary_coords(boundary_coords)
            # close the boundary if needed
            if not np.all(boundary_coords[0] == boundary_coords[-1]):
                boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])
            self.boundaries[group_id] = boundary_coords

        self.__validate_group_assignments(turbine_groups, n_wt)
        self.turbine_groups = {i: -1 for i in range(n_wt)}  # turbine_idx: group_id
        for group_id, indices in turbine_groups.items():
            if group_id not in self.boundaries:
                raise ValueError(f"No boundary defined for group {group_id}")
            for idx in indices:
                self.turbine_groups[idx] = group_id
        # check that all turbines are assigned to a group
        if -1 in self.turbine_groups.values():
            raise ValueError(f"All turbines must be assigned to a group; All the -1 should be filled\n{self.turbine_groups}")

        super().__init__(
            n_wt=n_wt,
            xy_boundary=np.array(list(boundaries.values())[0]).reshape(-1, 2),
            *args,
            **kwargs,
        )

    def __validate_boundary_coords(self, boundary_coords: np.ndarray) -> None:
        if not isinstance(boundary_coords, (np.ndarray, list)):
            raise TypeError(
                "Boundary coordinates must be a numpy array or a list of lists"
            )
        try:
            boundary_coords = np.array(boundary_coords).reshape(-1, 2)
        except BaseException:
            raise ValueError(
                "Boundary coordinates must be a 2D array with shape (n,2)"
            )
        if boundary_coords.ndim != 2 or boundary_coords.shape[1] != 2:
            raise ValueError("Boundary coordinates must be a 2D array with shape (n,2)")
        if len(boundary_coords) < 3:
            raise ValueError("Boundary must have at least 3 points")
        return boundary_coords

    def __validate_group_assignments(
        self, groups: Dict[int, List[int]], num_turbines: int
    ) -> None:
        if not isinstance(groups, dict):
            raise TypeError("Groups must be a dictionary")
        valid_types = (int, np.integer)
        for group_id, turbine_indices in groups.items():
            if not isinstance(group_id, valid_types) or group_id < 0:
                raise ValueError(
                    f"Invalid group ID: {group_id}; Should be an integer >= 0"
                )
            if not all(
                isinstance(idx, valid_types) and 0 <= idx < num_turbines
                for idx in turbine_indices
            ):
                raise ValueError(
                    f"Invalid turbine indices in group {group_id}; Should be integers in range [0, {num_turbines})"
                )

    def __dist_grad_wrapper(self, x, y, boundary_prop):
        if not np.shape([x, y]) == np.shape(self._cache_input):
            pass
        elif np.all(np.array([x, y]) == self._cache_input):
            return self._cache_output
        distance, ddist_dX, ddist_dY = self._calc_distance_and_gradients(
            x, y, boundary_prop
        )
        closest_edge_index = np.argmin(np.abs(distance), 1)
        self._cache_input = np.array([x, y])
        self._cache_output = [
            np.choose(closest_edge_index, v.T) for v in [distance, ddist_dX, ddist_dY]
        ]
        return self._cache_output

    def __calculate_group_distances_and_gradients(self, x, y):
        """Helper method to calculate distances and gradients for all groups."""
        n = len(x)
        ds = np.zeros(n)
        dx = np.zeros(n)
        dy = np.zeros(n)

        for group_id, boundary in self.boundaries.items():
            group_turbines = [
                i for i in range(n) if self.turbine_groups.get(i) == group_id
            ]
            if not group_turbines:
                continue

            x_group = x[group_turbines]
            y_group = y[group_turbines]

            boundary_properties = self.get_boundary_properties(boundary)[1:]
            distances, dx_group, dy_group = self.__dist_grad_wrapper(
                x_group, y_group, boundary_properties
            )

            ds[group_turbines] = distances
            dx[group_turbines] = dx_group
            dy[group_turbines] = dy_group

        return ds, dx, dy

    def distances(self, x, y):
        ds, _, _ = self.__calculate_group_distances_and_gradients(x, y)
        return ds

    def gradients(self, x, y):
        _, dx, dy = self.__calculate_group_distances_and_gradients(x, y)
        return np.diagflat(dx), np.diagflat(dy)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt  # fmt: skip
        cmap = plt.cm.get_cmap("viridis", len(self.boundaries))
        for ii, (group_id, boundary) in enumerate(self.boundaries.items()):
            ax.plot(*boundary.T, c=cmap(ii), label=f"Group {group_id}", linewidth=1)
        ax.legend()


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from py_wake.wind_turbines import WindTurbines
        from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt

        plt.close('all')
        i1 = np.array([[2, 17], [6, 23], [16, 23], [26, 15], [19, 0], [14, 4], [4, 4]])
        e1 = np.array([[0, 10], [20, 21], [22, 12], [10, 12], [9, 6], [2, 7]])
        i2 = np.array([[12, 13], [14, 17], [18, 15], [17, 10], [15, 11]])
        e2 = np.array([[5, 17], [5, 18], [8, 19], [8, 18]])
        i3 = np.array([[5, 0], [5, 1], [10, 3], [10, 0]])
        e3 = np.array([[6, -1], [6, 18], [7, 18], [7, -1]])
        e4 = np.array([[15, 9], [15, 11], [20, 11], [20, 9]])
        e5 = np.array([[10, 25], [20, 0]])
        zones = [
            InclusionZone(i1, name='i1'),
            InclusionZone(i2, name='i2'),
            InclusionZone(i3, name='i3'),
            ExclusionZone(e1, name='e1'),
            ExclusionZone(e2, name='e2'),
            ExclusionZone(e3, name='e3'),
            ExclusionZone(e4, name='e4'),
            ExclusionZone(e5, name='e5', dist2wt=lambda: 1, geometry_type='line'),
        ]

        N_points = 50
        xs = np.linspace(-1, 30, N_points)
        ys = np.linspace(-1, 30, N_points)
        y_grid, x_grid = np.meshgrid(xs, ys)
        x = x_grid.ravel()
        y = y_grid.ravel()
        n_wt = len(x)
        MPBC = MultiPolygonBoundaryComp(n_wt, zones, method='nearest')
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
        plt.contourf(x_grid, y_grid, distances.reshape(N_points, N_points), np.linspace(-10, 10, 100), cmap='seismic')
        plt.colorbar()

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(
            x.reshape(
                N_points, N_points), y.reshape(
                N_points, N_points), distances.reshape(
                N_points, N_points), np.linspace(-10, 10, 100), cmap='seismic')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if 0:
            for smpl in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                MPBC = MultiPolygonBoundaryComp(n_wt, zones, simplify_geometry=smpl)
                plt.figure()
                ax = plt.gca()
                MPBC.plot(ax)

        wind_turbines = WindTurbines(names=['tb1', 'tb2'],
                                     diameters=[80, 120],
                                     hub_heights=[70, 110],
                                     powerCtFunctions=[
            CubePowerSimpleCt(ws_cutin=3, ws_cutout=25, ws_rated=12,
                              power_rated=2000, power_unit='kW',
                              ct=8 / 9, additional_models=[]),
            CubePowerSimpleCt(ws_cutin=3, ws_cutout=25, ws_rated=12,
                              power_rated=3000, power_unit='kW',
                              ct=8 / 9, additional_models=[])])

        x1 = [0, 3000, 3000, 0]
        y1 = [0, 0, 3000, 3000]
        b1 = np.transpose((x1, y1))

        # Buildings
        x2 = [600, 1400, 1400, 600]
        y2 = [1700, 1700, 2500, 2500]
        b2 = np.transpose((x2, y2))

        # River
        x3 = np.linspace(520, 2420, 16)
        y3 = [0, 133, 266, 400, 500, 600, 700, 733, 866, 1300, 1633,
              2100, 2400, 2533, 2700, 3000]
        b3 = np.transpose((x3, y3))

        # Roads
        x4 = np.linspace(0, 3000, 16)
        y4 = [1095, 1038, 1110, 1006, 1028, 992, 977, 1052, 1076, 1064, 1073,
              1027, 964, 981, 1015, 1058]
        b4 = np.transpose((x4, y4))

        zones = [
            InclusionZone(b1, name='i1'),
            ExclusionZone(b2, dist2wt=lambda H: 4 * H - 360, name='building'),
            ExclusionZone(b3, geometry_type='line', dist2wt=lambda D: 3 * D, name='river'),
            ExclusionZone(b4, geometry_type='line', dist2wt=lambda D, H: max(D * 2, H * 3), name='road'),
        ]
        N_points = 50
        xs = np.linspace(0, 3000, N_points)
        ys = np.linspace(0, 3000, N_points)
        y_grid, x_grid = np.meshgrid(xs, ys)
        x = x_grid.ravel()
        y = y_grid.ravel()
        n_wt = len(x)
        types = np.zeros(n_wt)
        TSBC = TurbineSpecificBoundaryComp(n_wt, wind_turbines, zones)
        distances = TSBC.distances(x, y, type=types)
        delta = 1e-9
        distances2 = TSBC.distances(x + delta, y, type=types)
        dx_fd = (distances2 - distances) / delta
        dx = np.diag(TSBC.gradients(x + delta / 2, y, type=types)[0])

        plt.figure()
        plt.plot(dx_fd, dx, '.')

        plt.figure()
        for ll, t in enumerate(TSBC.types):
            line, = plt.plot(*TSBC.ts_merged_xy_boundaries[ll][0][0][0, :], label=f'type {ll}')
            for n, bound in enumerate(TSBC.ts_merged_xy_boundaries[ll]):
                x_bound, y_bound = bound[0].T
                x_bound = np.append(x_bound, x_bound[0])
                y_bound = np.append(y_bound, y_bound[0])
                plt.plot(x_bound, y_bound, color=line.get_color())

        plt.legend()
        plt.grid()
        plt.axis('square')

        for ll, t in enumerate(TSBC.types):
            plt.figure()
            for n, bound in enumerate(TSBC.ts_merged_xy_boundaries[ll]):
                x_bound, y_bound = bound[0].T
                x_bound = np.append(x_bound, x_bound[0])
                y_bound = np.append(y_bound, y_bound[0])
                plt.plot(x_bound, y_bound, 'b')
            plt.grid()
            plt.title(f'type {ll}')
            plt.axis('square')
            plt.contourf(x_grid, y_grid, TSBC.distances(x, y, type=t * np.ones(n_wt)).reshape(N_points, N_points), 50)
            plt.colorbar()


main()
