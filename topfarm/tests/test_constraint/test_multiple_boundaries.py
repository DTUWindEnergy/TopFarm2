from topfarm.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.cost_models.dummy import DummyCost
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot
import unittest
from topfarm.constraint_components.boundary import (
    MultiConvexBoundaryComp,
    Boundary,
    MultiCircleBoundaryComp,
    PolygonBoundaryComp,
    MultiWFPolygonBoundaryComp,
    BoundaryType,
    MultiWFBoundaryConstraint,
    InclusionZone,
    ExclusionZone,
    MultiPolygonBoundaryComp,
)
import pytest
import numpy as np  # fmt: skip
np.random.seed(42)


@pytest.fixture
def setup(request):
    n_wt = request.param if hasattr(request, "param") else 5
    # vertices of boundaries
    v1 = np.array([[0, 0], [1, 1], [0, 1]])
    v2 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]])
    comp = MultiConvexBoundaryComp(n_wt, [v1, v2], [[0, 1], np.arange(2, n_wt)])
    return comp, n_wt


@pytest.mark.parametrize("setup", [3, 5, 10], indirect=True)
def test_initialization(setup):
    comp, n_wt = setup
    assert comp.n_wt == n_wt
    assert len(comp.xy_boundaries) == 2


def test_calculate_boundary_and_normals(setup):
    comp, _ = setup
    boundaries = comp.calculate_boundary_and_normals(comp.xy_boundaries)
    for boundary in boundaries:
        assert boundary.normals is not None


def test_calculate_gradients(setup):
    comp, _ = setup
    comp.calculate_gradients()
    assert comp.dfaceDistance_dx is not None
    assert comp.dfaceDistance_dy is not None


@pytest.mark.parametrize("setup", [3, 5, 10], indirect=True)
def test_distances_shape(setup):
    comp, n_wt = setup
    x = np.random.rand(n_wt)
    y = np.random.rand(n_wt)
    distances = comp.distances(x, y)
    assert distances is not None
    assert len(distances) == comp.turbine_vertice_prod
    assert distances.size == sum(
        [b.n_vertices * b.n_turbines for b in comp.xy_boundaries]
    )


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
        comp = MultiConvexBoundaryComp(n_wt, [v1, v2], [[1], [0]])

        # place both turbines at (0,0)
        x = np.zeros(n_wt)
        y = np.zeros(n_wt)
        distances = comp.distances(x, y)
        boundaries_gt = [b.vertices[:-1] for b in comp.xy_boundaries]
        distances_gt = __compute_distances_to_faces((0, 0), boundaries_gt)
        assert distances.size == sum(
            [b.n_vertices * b.n_turbines for b in comp.xy_boundaries]
        )
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
    comp, n_wt = setup
    x = np.random.rand(n_wt)
    y = np.random.rand(n_wt)
    dx, dy = comp.gradients(x, y)
    assert dx is not None
    assert dy is not None
    n_distances = sum([b.n_vertices * b.n_turbines for b in comp.xy_boundaries])
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
    comp0 = MultiConvexBoundaryComp(n_wt, [v1, v2], [[0], np.arange(1, n_wt)])
    # switch boundaries order
    comp1 = MultiConvexBoundaryComp(n_wt, [v2, v1], [np.arange(1, n_wt), [0]])
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
    comp = MultiConvexBoundaryComp(n_wt, [v1, v2], [[0], np.arange(1, n_wt)])
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
    n_wt = 4
    comp1 = MultiConvexBoundaryComp(n_wt, [v1, v2], [[0, 1], [2, 3]])
    comp0 = MultiConvexBoundaryComp(n_wt, [v1, v2], [[2, 3], [0, 1]])
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
    comp, _ = setup
    import matplotlib.pyplot as plt  # fmt: skip
    fig, ax = plt.subplots()
    comp.plot(ax)
    plt.close(fig)


class TestMultiCircleBoundaryConstraint(unittest.TestCase):
    def setUp(self):
        self.centers = [(0, 0), (1, 1)]
        self.radii = [1.0, 2.0]
        self.masks = [np.array([1, 0]), np.array([0, 1])]
        self.constraint = MultiWFBoundaryConstraint(
            geometry=[(c, r) for c, r in zip(self.centers, self.radii)],
            wt_groups=[np.arange(1), np.arange(1, 2)],
            boundtype=BoundaryType.CIRCLE,
        )

    def test_initialization(self):
        n_wt = 2
        comp = self.constraint.get_comp(n_wt)
        np.testing.assert_array_equal(comp.center, np.array(self.centers))
        np.testing.assert_array_equal(comp.radius, np.array(self.radii))
        np.testing.assert_array_equal(comp.masks, np.array(self.masks))

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
            self.n_wt, [(c, r) for c, r in zip(self.centers, self.radii)], [[0], [1]]
        )

    def test_initialization(self):
        self.assertEqual(self.comp.n_wt, self.n_wt)
        self.assertEqual(self.comp.center, self.centers)
        self.assertEqual(self.comp.radius, self.radii)

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

        multi = MultiWFPolygonBoundaryComp(
            n_wt,
            [boundary_coords],
            [np.arange(n_wt)],
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
        boundary_coords = [
            np.random.rand(4, 2),
            np.random.rand(4, 2),
        ]
        x = np.random.rand(8)
        y = np.random.rand(8)
        n_wt = len(x)
        hn_wt = n_wt // 2

        single_1b = PolygonBoundaryComp(hn_wt, boundary_coords[0])
        d0_1b = single_1b.distances(x[:hn_wt], y[:hn_wt])
        dx0_1b, dy0_1b = single_1b.gradients(x[:hn_wt], y[:hn_wt])
        single_2b = PolygonBoundaryComp(hn_wt, boundary_coords[1])
        d0_2b = single_2b.distances(x[hn_wt:], y[hn_wt:])
        dx0_2b, dy0_2b = single_2b.gradients(x[hn_wt:], y[hn_wt:])

        multi = MultiWFPolygonBoundaryComp(
            n_wt,
            boundary_coords,
            [np.arange(hn_wt), np.arange(hn_wt, n_wt)],
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
        boundary_coords = [
            np.random.rand(4, 2),
            np.random.rand(4, 2),
        ]
        x = np.random.rand(8)
        y = np.random.rand(8)
        n_wt = len(x)
        hn_wt = n_wt // 2

        multi = MultiWFPolygonBoundaryComp(
            n_wt,
            boundary_coords,
            [np.arange(hn_wt), np.arange(hn_wt, n_wt)],
        )
        d1 = multi.distances(x, y)

        multi = MultiWFPolygonBoundaryComp(
            n_wt,
            reversed(boundary_coords),
            [np.arange(hn_wt, n_wt), np.arange(hn_wt)],
        )
        d2 = multi.distances(x, y)
        np.testing.assert_array_almost_equal(
            d1,
            d2,
            decimal=6,
            err_msg="Distances don't match between different group order",
        )

        multi = MultiWFPolygonBoundaryComp(
            n_wt,
            reversed(boundary_coords),
            [np.arange(hn_wt), np.arange(hn_wt, n_wt)],
        )
        d3 = multi.distances(x, y)

        assert not np.allclose(d1, d3)
        assert not np.allclose(d2, d3)

    def setUp(self):
        self.sample_boundary = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.sample_boundaries = [
            self.sample_boundary,
            self.sample_boundary + 2,
        ]
        self.sample_turbine_groups = [
            np.arange(2),
            np.arange(2, 4),
        ]

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

    def test_single_group_distance_calculation(self):
        comp = MultiWFPolygonBoundaryComp(
            2, [self.sample_boundary], [[0, 1]]
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
            4,
            self.sample_boundaries,
            self.sample_turbine_groups,
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
            4,
            self.sample_boundaries,
            self.sample_turbine_groups,
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
            4,
            self.sample_boundaries,
            self.sample_turbine_groups,
        )

        # Create component with reversed group order
        comp2 = MultiWFPolygonBoundaryComp(
            4, reversed(self.sample_boundaries), reversed(self.sample_turbine_groups)
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

    def test_polygon_gradient_with_large_number_of_vertices(self):
        N = 100
        boundary_coords = np.array(
            [[np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)] for i in range(N)]
        )
        x = np.array([0.25, 0.75, 0.5, -0.2])
        y = np.array([0.25, 0.25, 1.2, 0.5])
        n_wt = len(x)
        multi = MultiWFPolygonBoundaryComp(
            n_wt,
            [boundary_coords],
            [np.arange(n_wt)],
        )
        # should not fail
        _ = multi.gradients(x, y)


@pytest.fixture
def sample_zones():
    # Create sample inclusion and exclusion zones
    inclusion = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    exclusion = np.array([[3, 3], [7, 3], [7, 7], [3, 7]])

    zones = [
        InclusionZone(inclusion, name="inclusion"),
        ExclusionZone(exclusion, name="exclusion"),
    ]
    return zones


def test_multipolygon_boundary_comp_basic_init(sample_zones):
    # Test basic initialization
    comp = MultiPolygonBoundaryComp(
        n_wt=5, zones=sample_zones, const_id="test", units="m"
    )

    assert comp.n_wt == 5
    assert len(comp.zones) == 2
    assert comp.const_id == "test"
    assert comp.units == "m"
    assert comp.method == "nearest"
    assert not comp.relaxation


def test_multipolygon_boundary_comp_with_simplify_float(sample_zones):
    # Test initialization with float simplification
    comp = MultiPolygonBoundaryComp(n_wt=5, zones=sample_zones, simplify_geometry=0.1)

    # Verify simplification was applied
    assert hasattr(comp, "bounds_poly")
    assert len(comp.bounds_poly) > 0


def test_multipolygon_boundary_comp_with_simplify_dict(sample_zones):
    # Test initialization with dict simplification
    simplify_params = {"tolerance": 0.1, "preserve_topology": True}
    comp = MultiPolygonBoundaryComp(
        n_wt=5, zones=sample_zones, simplify_geometry=simplify_params
    )

    # Verify simplification was applied
    assert hasattr(comp, "bounds_poly")
    assert len(comp.bounds_poly) > 0


def test_multipolygon_boundary_comp_method_options(sample_zones):
    # Test different method options
    comp_nearest = MultiPolygonBoundaryComp(
        n_wt=5, zones=sample_zones, method="nearest"
    )
    assert comp_nearest.method == "nearest"

    comp_smooth = MultiPolygonBoundaryComp(
        n_wt=5, zones=sample_zones, method="smooth_min"
    )
    assert comp_smooth.method == "smooth_min"


def test_multipolygon_boundary_comp_with_relaxation(sample_zones):
    # Test initialization with relaxation
    comp = MultiPolygonBoundaryComp(
        n_wt=5,
        zones=sample_zones,
        relaxation=(0.1, 10),  # Example relaxation parameters
    )
    assert comp.relaxation == (0.1, 10)


def test_simplify_method_float(sample_zones):
    comp = MultiPolygonBoundaryComp(n_wt=5, zones=sample_zones)
    initial_boundaries = len(comp.boundaries)

    # Apply float simplification
    comp.simplify(0.1)

    # Verify boundaries were simplified but structure maintained
    assert len(comp.boundaries) == initial_boundaries
    assert all(isinstance(b[0], np.ndarray) for b in comp.boundaries)


def test_simplify_method_dict(sample_zones):
    comp = MultiPolygonBoundaryComp(n_wt=5, zones=sample_zones)
    initial_boundaries = len(comp.boundaries)

    # Apply dict simplification
    simplify_params = {"tolerance": 0.1, "preserve_topology": True}
    comp.simplify(simplify_params)

    # Verify boundaries were simplified but structure maintained
    assert len(comp.boundaries) == initial_boundaries
    assert all(isinstance(b[0], np.ndarray) for b in comp.boundaries)


def test_simplify_preserves_validity(sample_zones):
    comp = MultiPolygonBoundaryComp(n_wt=5, zones=sample_zones)

    # Apply simplification
    comp.simplify(0.1)

    # Verify that boundaries remain valid
    assert hasattr(comp, "bounds_poly")
    assert hasattr(comp, "boundaries")
    assert len(comp.boundaries) > 0
    assert all(isinstance(b[0], np.ndarray) for b in comp.boundaries)
    assert all(
        len(b[0]) >= 3 for b in comp.boundaries
    )  # Minimum 3 points for valid polygon


def test_simplify_maintains_types(sample_zones):
    comp = MultiPolygonBoundaryComp(n_wt=5, zones=sample_zones)
    original_incl_excls = comp.incl_excls.copy()

    # Apply simplification
    comp.simplify(0.1)

    # Verify inclusion/exclusion flags maintained
    assert np.array_equal(comp.incl_excls, original_incl_excls)


class TestMultiWFCircleValidation(unittest.TestCase):

    def test_validate_input_happy_path(self):
        # Setup valid test inputs
        geometry = [((0, 0), 1), ((1, 1), 2)]  # List of (center, radius) tuples
        wt_groups = [[0, 1], [2, 3]]  # List of lists with turbine indices
        n_wt = 4
        comp = MultiCircleBoundaryComp(n_wt, geometry, wt_groups)
        assert len(comp.center) == 2
        assert len(comp.radius) == 2
        assert len(comp.masks) == 2

    def test_validate_input_mismatched_lengths(self):
        # Setup test inputs with mismatched lengths
        geometry = [((0, 0), 1)]  # Only one boundary
        wt_groups = [[0, 1], [2, 3]]  # But two groups
        n_wt = 4

        with pytest.raises(AssertionError):
            MultiCircleBoundaryComp(n_wt, geometry, wt_groups)

    def test_validate_input_invalid_geometry_format(self):
        # Test cases with invalid geometry format
        invalid_cases = [
            ([[0, 1]], [[0, 1]]),  # geometry not a tuple
            ([(0, 0)], [[0, 1]]),  # missing radius
            ([(0, 0, 1)], [[0, 1]]),  # not a tuple of (center, radius)
            ([((0,), 1)], [[0, 1]]),  # center not 2D
        ]

        n_wt = 2
        for geometry, wt_groups in invalid_cases:
            with pytest.raises(AssertionError):
                MultiCircleBoundaryComp(n_wt, geometry, wt_groups)

    def test_validate_input_invalid_center_radius_format(self):
        # Test cases with invalid center/radius format
        invalid_cases = [
            ([((0, 0, 0), 1)], [[0, 1]]),  # center is 3D
            ([((0,), 1)], [[0, 1]]),  # center is 1D
            ([((0, 0), (1, 1))], [[0, 1]]),  # radius is 2D
            ([((0, 0), [1])], [[0, 1]]),  # radius is list instead of scalar
        ]

        n_wt = 2
        for geometry, wt_groups in invalid_cases:
            with pytest.raises(AssertionError):
                MultiCircleBoundaryComp(n_wt, geometry, wt_groups)
