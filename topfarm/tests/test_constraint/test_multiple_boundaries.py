from topfarm.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.cost_models.dummy import DummyCost
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
from ..test_files.xy3tb import get_tf, boundary, optimal
import unittest
from topfarm.constraint_components.boundary import (
    MultiConvexBoundaryComp,
    Boundary,
    MultiXYBoundaryConstraint,
    MultiCircleBoundaryComp,
    MultiCircleBoundaryConstraint,
)
import pytest
from topfarm.constraint_components.boundary import (
    PolygonBoundaryComp,
    MultiWFPolygonBoundaryComp,
)
import numpy as np  # fmt: skip
np.random.seed(42)


@pytest.fixture
def setup(request):
    n_wt = request.param if hasattr(request, "param") else 5
    # vertices of boundaries
    v1 = np.array([[0, 0], [1, 1], [0, 1]])
    v2 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]])
    # masks of boundaries
    m1 = np.zeros(n_wt, dtype=bool)
    m1[:2] = True
    m2 = ~m1.copy()
    boundaries: list[Boundary] = [Boundary(v1, m1), Boundary(v2, m2)]
    comp = MultiConvexBoundaryComp(n_wt, boundaries)
    return comp, boundaries, n_wt


@pytest.mark.parametrize("setup", [3, 5, 10], indirect=True)
def test_initialization(setup):
    comp, _, n_wt = setup
    assert comp.n_wt == n_wt
    assert len(comp.xy_boundaries) == 2


def test_calculate_boundary_and_normals(setup):
    comp, boundaries, _ = setup
    boundaries = comp.calculate_boundary_and_normals(boundaries)
    for boundary in boundaries:
        assert boundary.normals is not None


def test_calculate_gradients(setup):
    comp, _, _ = setup
    comp.calculate_gradients()
    assert comp.dfaceDistance_dx is not None
    assert comp.dfaceDistance_dy is not None


@pytest.mark.parametrize("setup", [3, 5, 10], indirect=True)
def test_distances_shape(setup):
    comp, boundaries, n_wt = setup
    x = np.random.rand(n_wt)
    y = np.random.rand(n_wt)
    distances = comp.distances(x, y)
    assert distances is not None
    assert len(distances) == comp.turbine_vertice_prod
    assert distances.size == sum([b.n_vertices * b.n_turbines for b in boundaries])


def test_distances_value():
    def __compute_distances_to_faces(point, boundaries):
        def point_to_line_distance(point, line_start, line_end):
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end
            distance = np.abs(
                (y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1
            ) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            # Determine the sign of the distance
            cross_product = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
            sign = np.sign(cross_product)
            return sign * distance

        distances = []
        for boundary in boundaries:
            for i in range(len(boundary)):
                line_start = boundary[i]
                line_end = boundary[(i + 1) % len(boundary)]
                distance = point_to_line_distance(point, line_start, line_end)
                distances.append(distance)
        return distances

    def __check_dists():
        n_wt = 2
        m1 = np.ones(n_wt, dtype=bool)
        m1[0] = False
        m2 = ~m1
        boundaries: list[Boundary] = [Boundary(v1, m1), Boundary(v2, m2)]
        comp = MultiConvexBoundaryComp(n_wt, boundaries)

        # place both turbines at (0,0)
        x = np.zeros(n_wt)
        y = np.zeros(n_wt)
        distances = comp.distances(x, y)
        boundaries_gt = [b.vertices[:-1] for b in comp.xy_boundaries]
        distances_gt = __compute_distances_to_faces((0, 0), boundaries_gt)
        assert distances.size == sum([b.n_vertices * b.n_turbines for b in boundaries])
        assert np.allclose(distances, distances_gt, atol=1e-6)

    # vertices of boundaries
    v1 = np.array([[0, 0], [1, 1], [0, 1]])
    v2 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]])
    __check_dists()
    v1 = np.array([[0, 0], [2, 2], [1, 2]])
    v2 = np.array([[2, 2], [3, 4], [3, 3], [1, 3]])
    __check_dists()


@pytest.mark.parametrize("setup", [3, 5, 10], indirect=True)
def test_gradients_shape(setup):
    comp, boundaries, n_wt = setup
    x = np.random.rand(n_wt)
    y = np.random.rand(n_wt)
    dx, dy = comp.gradients(x, y)
    assert dx is not None
    assert dy is not None
    n_distances = sum([b.n_vertices * b.n_turbines for b in boundaries])
    assert dx.shape[0] == n_distances
    assert dx.shape[1] == n_wt
    assert dy.shape[0] == n_distances
    assert dy.shape[1] == n_wt


@pytest.mark.parametrize("n_wt", [3, 5, 10])
@pytest.mark.parametrize(
    "n_vertices",
    [
        (3, 4),
        (4, 3),
        (6, 7),
        (5, 3),
    ],
)
def test_order_of_boundaries_does_not_affect_results(n_wt, n_vertices):
    # vertices of boundaries
    v1 = np.random.randn(n_vertices[0], 2)
    v2 = np.random.randn(n_vertices[1], 2)
    # masks of boundaries
    m1 = np.zeros(n_wt, dtype=bool)
    m1[0] = True
    m2 = ~m1.copy()
    boundaries: list[Boundary] = [Boundary(v1, m1), Boundary(v2, m2)]
    comp0 = MultiConvexBoundaryComp(n_wt, boundaries)
    # switch boundaries order
    boundaries: list[Boundary] = [Boundary(v2, m2), Boundary(v1, m1)]
    comp1 = MultiConvexBoundaryComp(n_wt, boundaries)

    assert np.allclose(comp0.dfaceDistance_dx, comp1.dfaceDistance_dx)
    assert np.allclose(comp0.dfaceDistance_dy, comp1.dfaceDistance_dy)
    x_points = np.zeros(n_wt)
    y_points = np.zeros(n_wt)
    assert np.allclose(
        comp0.distances(x_points, y_points), comp1.distances(x_points, y_points)
    )


def test_gradient_is_returned_as_sparse_for_large_matrices():
    n_wt = 1000
    n_vertices = (3, 4)
    # vertices of boundaries
    v1 = np.random.randn(n_vertices[0], 2)
    v2 = np.random.randn(n_vertices[1], 2)
    # masks of boundaries
    m1 = np.zeros(n_wt, dtype=bool)
    m1[0] = True
    m2 = ~m1.copy()
    boundaries: list[Boundary] = [Boundary(v1, m1), Boundary(v2, m2)]
    comp = MultiConvexBoundaryComp(n_wt, boundaries)
    x = np.random.rand(n_wt)
    y = np.random.rand(n_wt)
    dx, dy = comp.gradients(x, y)
    assert dx.nnz > 0
    assert dy.nnz > 0
    import scipy.sparse as sp  # fmt: skip
    assert isinstance(dx, sp.csr_matrix)
    assert isinstance(dy, sp.csr_matrix)


@pytest.mark.parametrize("n_vertices", [(3, 4), (4, 3), (6, 7), (5, 3)])
def test_gradients_and_distances_are_calculated_based_on_masked_boundaries(n_vertices):
    # vertices of boundaries
    v1 = np.random.rand(n_vertices[0], 2)
    v2 = np.random.rand(n_vertices[1], 2)
    # masks of boundaries
    n_wt = 4
    m1 = np.zeros(n_wt, dtype=bool)
    m1[:2] = True
    m2 = ~m1.copy()
    b1m1 = Boundary(v1, m1)
    b1m2 = Boundary(v1, m2)
    b2m1 = Boundary(v2, m1)
    b2m2 = Boundary(v2, m2)
    # switch masks in the boundaries
    comp1 = MultiConvexBoundaryComp(n_wt, [b1m1, b2m2])
    comp0 = MultiConvexBoundaryComp(n_wt, [b1m2, b2m1])
    # the jacobians should reflect the change in mask. i.e. cannot be the same
    assert not np.allclose(comp0.dfaceDistance_dx, comp1.dfaceDistance_dx)
    assert not np.allclose(comp0.dfaceDistance_dy, comp1.dfaceDistance_dy)
    # distances from same point should not change due to the same order of boundaries
    x_points = np.zeros(n_wt)
    y_points = np.zeros(n_wt)
    assert np.allclose(
        comp0.distances(x_points, y_points), comp1.distances(x_points, y_points)
    )


def test_boundary_obj_fails_gracefully_with_less_than_3_vertices():
    n_wt = 5
    n_vertices = 2  # too little
    v2 = np.random.randn(n_vertices, 2)
    m1 = np.zeros(n_wt, dtype=bool)
    with pytest.raises(
        AssertionError,
        match="Boundary must have at least 3 vertices",
    ):
        Boundary(v2, m1)


def test_initialization():
    vertices = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    mask = np.array([True, False, True, True])
    boundary = Boundary(vertices, mask)
    assert np.array_equal(boundary.vertices, vertices)
    assert boundary.is_inclusion is True
    assert boundary.design_var_mask is not None
    assert (boundary.design_var_mask == mask).all()


def test_initialization_invalid_vertices():
    mask = np.array([True, False, True, True])
    with pytest.raises(AssertionError, match="Boundary must be a 2D array"):
        Boundary(np.array([0, 1, 2]), mask)

    for vert in [
        np.random.randn(1, 3),
        np.random.randn(3, 1),
        np.random.randn(4, 3),
        np.random.randn(3, 4),
        np.random.randn(5, 3),
        np.random.randn(3, 5),
    ]:
        msg = "Boundary must have shape \\(n, 2\\) or \\(2, n\\)"
        with pytest.raises(AssertionError, match=msg):
            Boundary(vert, mask)


def test_n_turbines():
    vertices = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    design_var_mask = np.array([True, False, True, True])
    boundary = Boundary(vertices, design_var_mask)
    assert boundary.n_turbines == design_var_mask.sum() == 3

    vertices = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    design_var_mask = np.array([True, False, False, True])
    boundary = Boundary(vertices, design_var_mask)
    assert boundary.n_turbines == np.sum(design_var_mask) == 2


def test_n_vertices():
    # vertices[-1] == vertices[0], so the last vertex is not counted
    vertices = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    mask = np.ones(42, dtype=bool)
    boundary = Boundary(vertices, mask)
    assert boundary.n_vertices == 3

    vertices = np.array([[0, 0], [1, 1], [2, 0]])
    boundary = Boundary(vertices, mask)
    assert boundary.n_vertices == 3

    vertices = np.array([[0, 0], [1, 1], [2, 0], [5, 5]])
    boundary = Boundary(vertices, mask)
    assert boundary.n_vertices == 4


def test_vertices_property_override():
    mask = np.ones(42, dtype=bool)
    vertices = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    boundary = Boundary(vertices, mask)
    new_vertices = np.array([[1, 1], [2, 2], [3, 1], [1, 1]])
    boundary.vertices = new_vertices
    assert np.array_equal(boundary.vertices, new_vertices)


def test_validate():
    vertices = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    mask = np.ones(42, dtype=bool)
    boundary = Boundary(vertices, mask)
    assert boundary.vertices.shape == (4, 2)
    assert boundary.design_var_mask.shape == (42,)
    assert isinstance(boundary.is_inclusion, bool)


def test_design_var_mask_is_set_correctly():
    vertices = np.array([[0, 0], [1, 1], [2, 0], [0, 0]])
    design_var_mask = np.array([True, False, True, True])
    boundary = Boundary(vertices, design_var_mask)
    assert np.array_equal(boundary.design_var_mask, design_var_mask)
    mask = np.ones(42, dtype=bool)
    boundary = Boundary(vertices, mask)
    default_all_true = np.ones(42, dtype=bool)
    assert np.array_equal(boundary.design_var_mask, default_all_true)


def test_plot(setup):
    comp, _, _ = setup
    import matplotlib.pyplot as plt  # fmt: skip
    fig, ax = plt.subplots()
    comp.plot(ax)
    plt.close(fig)


class TestMultiXYBoundaryConstraint(unittest.TestCase):
    def setUp(self):
        self.n_wt = 3
        m1 = np.ones(self.n_wt, dtype=bool)
        m1[: self.n_wt // 2] = False
        assert np.concatenate([m1, ~m1]).sum() == self.n_wt
        self.boundaries = [Boundary(boundary, m1), Boundary(boundary, ~m1)]
        self.constraint = MultiXYBoundaryConstraint(self.boundaries)
        self.tf_problem = get_tf(  # 3 turbines problem
            {"constraints": [self.constraint, SpacingConstraint(2)]}
        )

    def test_initialization(self):
        self.assertEqual(self.constraint.boundaries, self.boundaries)
        self.assertEqual(self.constraint.boundary_type, "convex_hull")
        self.assertEqual(self.constraint.const_id, "xyboundary_comp_convex_hull")
        self.assertIsNone(self.constraint.units)
        self.assertFalse(self.constraint.relaxation)

    def test_get_comp(self):
        comp = self.constraint.get_comp(self.n_wt)
        self.assertIsNotNone(comp)
        self.assertEqual(comp.n_wt, self.n_wt)
        self.assertEqual(comp.xy_boundaries, self.boundaries)

    def test_can_solve(self):
        self.tf_problem.optimize()
        tb_pos = self.tf_problem.turbine_positions
        tol = 1e-6
        np.all(sum((tb_pos[2] - tb_pos[0]) ** 2) > 2**2 - tol)
        assert np.all(tb_pos[1][0] < 6 + tol)
        dec_prec = 4
        np.testing.assert_array_almost_equal(tb_pos[:, :2], optimal, dec_prec)

    def test_gradients(self):
        self.tf_problem.check_gradients(True)


class TestMultiCircleBoundaryConstraint(unittest.TestCase):
    def setUp(self):
        self.centers = [(0, 0), (1, 1)]
        self.radii = [1.0, 2.0]
        self.masks = [np.array([1, 0]), np.array([0, 1])]
        self.constraint = MultiCircleBoundaryConstraint(
            self.centers, self.radii, self.masks
        )

    def test_initialization(self):
        np.testing.assert_array_equal(self.constraint.center, np.array(self.centers))
        np.testing.assert_array_equal(self.constraint.radius, np.array(self.radii))
        np.testing.assert_array_equal(self.constraint.masks, np.array(self.masks))

    def test_get_comp(self):
        n_wt = 2
        comp = self.constraint.get_comp(n_wt)
        self.assertIsInstance(comp, MultiCircleBoundaryComp)

    def test_gradients(self):
        optimal = np.arange(4).reshape(2, 2)
        desvar = dict(zip("xy", optimal.T + 1.5))
        plot_comp = NoPlot()
        tf = TopFarmProblem(
            desvar,
            DummyCost(optimal, "xy"),
            constraints=[self.constraint],
            plot_comp=plot_comp,
            driver=SimpleGADriver(),
        )
        tf.check_gradients(True)


class TestMultiCircleBoundaryComp(unittest.TestCase):
    def setUp(self):
        self.centers = [(0, 0), (1, 1)]
        self.radii = [1.0, 2.0]
        self.masks = [np.array([1, 0]), np.array([0, 1])]
        self.n_wt = 2
        self.comp = MultiCircleBoundaryComp(
            self.n_wt, self.centers, self.radii, self.masks
        )

    def test_initialization(self):
        self.assertEqual(self.comp.n_wt, self.n_wt)
        self.assertEqual(self.comp.center, self.centers)
        self.assertEqual(self.comp.radius, self.radii)
        self.assertEqual(self.comp.masks, self.masks)

    def test_distances(self):
        x = np.array([0.5, 1.5])
        y = np.array([0.5, 1.5])
        distances = self.comp.distances(x, y)
        expected_distances = np.zeros_like(x)
        for center, radius, mask in zip(self.centers, self.radii, self.masks):
            expected_distances += mask * (
                radius - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            )
        np.testing.assert_array_almost_equal(distances, expected_distances)


class TestMultiWFPolygonBoundaryComp(unittest.TestCase):
    def test_single_group_equivalence(self):

        boundary_coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]]  # Close the polygon
        )
        x = np.array([0.25, 0.75, 0.5, -0.2])
        y = np.array([0.25, 0.25, 1.2, 0.5])
        n_wt = len(x)

        single = PolygonBoundaryComp(n_wt, boundary_coords)
        d0 = single.distances(x, y)
        dx0, dy0 = single.gradients(x, y)

        group_id = 1
        multi = MultiWFPolygonBoundaryComp(
            n_wt=n_wt,
            boundaries={group_id: boundary_coords},
            turbine_groups={group_id: np.arange(n_wt)},
        )
        d1 = multi.distances(x, y)
        dx1, dy1 = multi.gradients(x, y)

        np.testing.assert_array_almost_equal(
            d0,
            d1,
            decimal=6,
            err_msg="Distances don't match between single and multi boundary",
        )
        np.testing.assert_array_almost_equal(
            dx0,
            dx1,
            decimal=6,
            err_msg="X gradients don't match between single and multi boundary",
        )
        np.testing.assert_array_almost_equal(
            dy0,
            dy1,
            decimal=6,
            err_msg="Y gradients don't match between single and multi boundary",
        )

    def test_two_group_equivalence(self):
        boundary_coords = {
            1: np.random.rand(4, 2),
            2: np.random.rand(4, 2),
        }
        x = np.random.rand(8)
        y = np.random.rand(8)
        n_wt = len(x)
        hn_wt = n_wt // 2

        single_1b = PolygonBoundaryComp(hn_wt, boundary_coords[1])
        d0_1b = single_1b.distances(x[:hn_wt], y[:hn_wt])
        dx0_1b, dy0_1b = single_1b.gradients(x[:hn_wt], y[:hn_wt])
        single_2b = PolygonBoundaryComp(hn_wt, boundary_coords[2])
        d0_2b = single_2b.distances(x[hn_wt:], y[hn_wt:])
        dx0_2b, dy0_2b = single_2b.gradients(x[hn_wt:], y[hn_wt:])

        multi = MultiWFPolygonBoundaryComp(
            n_wt=n_wt,
            boundaries=boundary_coords,
            turbine_groups={1: np.arange(hn_wt), 2: np.arange(hn_wt, n_wt)},
        )
        d1 = multi.distances(x, y)
        dx1, dy1 = multi.gradients(x, y)

        np.testing.assert_array_almost_equal(
            d0_1b,
            d1[:hn_wt],
            decimal=6,
            err_msg="Distances don't match between single and multi boundary",
        )
        np.testing.assert_array_almost_equal(
            d0_2b,
            d1[hn_wt:],
            decimal=6,
            err_msg="Distances don't match between single and multi boundary",
        )

        np.testing.assert_array_almost_equal(
            dx0_1b,
            dx1[:hn_wt, :hn_wt],
            decimal=6,
            err_msg="X gradients don't match between single and multi boundary",
        )
        np.testing.assert_array_almost_equal(
            dx0_2b,
            dx1[hn_wt:, hn_wt:],
            decimal=6,
            err_msg="X gradients don't match between single and multi boundary",
        )

        np.testing.assert_array_almost_equal(
            dy0_1b,
            dy1[:hn_wt, :hn_wt],
            decimal=6,
            err_msg="Y gradients don't match between single and multi boundary",
        )
        np.testing.assert_array_almost_equal(
            dy0_2b,
            dy1[hn_wt:, hn_wt:],
            decimal=6,
            err_msg="Y gradients don't match between single and multi boundary",
        )

    def test_group_boundary_order_impact_on_result(self):
        boundary_coords = {
            1: np.random.rand(4, 2),
            2: np.random.rand(4, 2),
        }
        x = np.random.rand(8)
        y = np.random.rand(8)
        n_wt = len(x)
        hn_wt = n_wt // 2

        multi = MultiWFPolygonBoundaryComp(
            n_wt=n_wt,
            boundaries=boundary_coords,
            turbine_groups={1: np.arange(hn_wt), 2: np.arange(hn_wt, n_wt)},
        )
        d1 = multi.distances(x, y)

        multi = MultiWFPolygonBoundaryComp(
            n_wt=n_wt,
            boundaries=boundary_coords,
            turbine_groups={2: np.arange(hn_wt, n_wt), 1: np.arange(hn_wt)},
        )
        d2 = multi.distances(x, y)

        multi = MultiWFPolygonBoundaryComp(
            n_wt=n_wt,
            boundaries=boundary_coords,
            turbine_groups={2: np.arange(hn_wt), 1: np.arange(hn_wt, n_wt)},
        )
        d3 = multi.distances(x, y)

        np.testing.assert_array_almost_equal(
            d1,
            d2,
            decimal=6,
            err_msg="Distances don't match between different group order",
        )
        assert not np.allclose(d1, d3)
        assert not np.allclose(d2, d3)

    def setUp(self):
        self.sample_boundary = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.sample_boundaries = {
            1: self.sample_boundary,
            2: self.sample_boundary + 2,
        }  # Shifted copy
        self.sample_turbine_groups = {
            1: [0, 1],  # First two turbines in group 1
            2: [2, 3],  # Last two turbines in group 2
        }

    def test_validate_positive_n_wt(self):
        with pytest.raises(ValueError, match="Number of turbines must be positive"):
            MultiWFPolygonBoundaryComp(
                0, self.sample_boundaries, self.sample_turbine_groups
            )
        with pytest.raises(ValueError, match="Number of turbines must be positive"):
            MultiWFPolygonBoundaryComp(
                -1, self.sample_boundaries, self.sample_turbine_groups
            )

    def test_validate_boundary_coordinates(self):
        # Test invalid type
        invalid_boundaries = {1: 123}  # Not array-like
        with pytest.raises(
            TypeError,
            match="Boundary coordinates must be a numpy array or a list of lists",
        ):
            MultiWFPolygonBoundaryComp(
                2, invalid_boundaries, self.sample_turbine_groups
            )

        # Test invalid shape (1D array)
        invalid_boundaries = {1: np.array([1, 2, 3])}
        with pytest.raises(
            ValueError,
            match="Boundary coordinates must be a 2D array with shape \\(n,2\\)",
        ):
            MultiWFPolygonBoundaryComp(
                2, invalid_boundaries, self.sample_turbine_groups
            )

        # Test too few points
        invalid_boundaries = {1: np.array([[0, 0], [1, 1]])}
        with pytest.raises(ValueError, match="Boundary must have at least 3 points"):
            MultiWFPolygonBoundaryComp(
                2, invalid_boundaries, self.sample_turbine_groups
            )

    def test_validate_turbine_groups(self):
        # Test invalid type
        with pytest.raises(TypeError, match="Groups must be a dictionary"):
            MultiWFPolygonBoundaryComp(2, self.sample_boundaries, "not a dict")

        # Test invalid group ID type
        invalid_groups = {"a": [0, 1]}  # String instead of int
        with pytest.raises(ValueError, match="Invalid group ID"):
            MultiWFPolygonBoundaryComp(2, self.sample_boundaries, invalid_groups)

        # Test negative group ID
        invalid_groups = {-1: [0, 1]}
        with pytest.raises(ValueError, match="Invalid group ID"):
            MultiWFPolygonBoundaryComp(2, self.sample_boundaries, invalid_groups)

        # Test invalid turbine indices
        invalid_groups = {1: [0, 5]}  # Index 5 is out of range for n_wt=2
        with pytest.raises(ValueError, match="Invalid turbine indices"):
            MultiWFPolygonBoundaryComp(2, self.sample_boundaries, invalid_groups)

    def test_boundary_group_correspondence(self):
        # Test when group in turbine_groups has no corresponding boundary
        invalid_groups = {1: [0, 1], 3: [2, 3]}  # Group 3 has no boundary
        with pytest.raises(ValueError, match="No boundary defined for group 3"):
            MultiWFPolygonBoundaryComp(4, self.sample_boundaries, invalid_groups)

    def test_single_group_distance_calculation(self):
        boundaries = {1: self.sample_boundary}
        groups = {1: [0, 1]}
        comp = MultiWFPolygonBoundaryComp(
            n_wt=2, boundaries=boundaries, turbine_groups=groups
        )

        # Test points inside boundary
        x = np.array([0.5, 0.5])
        y = np.array([0.5, 0.5])
        distances = comp.distances(x, y)
        assert np.all(distances > 0)  # Points inside should have positive distance

        # Test points outside boundary
        x = np.array([-1.0, -1.0])
        y = np.array([-1.0, -1.0])
        distances = comp.distances(x, y)
        assert np.all(distances < 0)  # Points outside should have negative distance

    def test_multiple_group_distance_calculation(self):
        comp = MultiWFPolygonBoundaryComp(
            n_wt=4,
            boundaries=self.sample_boundaries,
            turbine_groups=self.sample_turbine_groups,
        )

        # Test points with mixed positions
        x = np.array(
            [0.5, 0.5, 2.5, 2.5]
        )  # First two for group 1, last two for group 2
        y = np.array([0.5, 0.5, 2.5, 2.5])
        distances = comp.distances(x, y)

        # Check that each point gets evaluated against its assigned boundary
        assert len(distances) == 4
        assert np.all(
            distances > 0
        )  # All points are inside their respective boundaries

    def test_gradient_calculation_mf_polygon(self):
        comp = MultiWFPolygonBoundaryComp(
            n_wt=4,
            boundaries=self.sample_boundaries,
            turbine_groups=self.sample_turbine_groups,
        )

        x = np.random.rand(4)
        y = np.random.rand(4)

        # Calculate gradients
        dx, dy = comp.gradients(x, y)

        # Check shapes
        assert dx.shape == (4, 4)
        assert dy.shape == (4, 4)

        # Verify gradients using finite differences
        eps = 1e-6
        x_perturbed = x + eps
        numerical_dx = (comp.distances(x_perturbed, y) - comp.distances(x, y)) / eps

        print(f"GRAD: {dx}; Numerical: {numerical_dx}")

        # Compare analytical vs numerical gradients
        np.testing.assert_allclose(
            np.diag(dx),
            numerical_dx,
            rtol=1e-5,
            err_msg="Analytical gradients don't match numerical gradients",
        )

    def test_group_order_independence(self):
        # Create component with original order
        comp1 = MultiWFPolygonBoundaryComp(
            n_wt=4,
            boundaries=self.sample_boundaries,
            turbine_groups=self.sample_turbine_groups,
        )

        # Create component with reversed group order
        reversed_boundaries = {
            2: self.sample_boundaries[2],
            1: self.sample_boundaries[1],
        }
        reversed_groups = {
            2: self.sample_turbine_groups[2],
            1: self.sample_turbine_groups[1],
        }
        comp2 = MultiWFPolygonBoundaryComp(
            n_wt=4, boundaries=reversed_boundaries, turbine_groups=reversed_groups
        )

        x = np.array([0.5, 0.5, 2.5, 2.5])
        y = np.array([0.5, 0.5, 2.5, 2.5])

        # Results should be the same regardless of group order
        np.testing.assert_allclose(
            comp1.distances(x, y), comp2.distances(x, y), rtol=1e-10
        )

        dx1, dy1 = comp1.gradients(x, y)
        dx2, dy2 = comp2.gradients(x, y)
        np.testing.assert_allclose(dx1, dx2, rtol=1e-10)
        np.testing.assert_allclose(dy1, dy2, rtol=1e-10)
