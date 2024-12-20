import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint, \
    PolygonBoundaryComp, InclusionZone, ExclusionZone, MultiPolygonBoundaryComp
from topfarm._topfarm import TopFarmProblem
from topfarm.tests.utils import __assert_equal_unordered
import unittest
from shapely import Polygon


def get_tf(initial, optimal, boundary, plot_comp=NoPlot(), boundary_type='polygon'):
    initial, optimal = map(np.array, [initial, optimal])
    return TopFarmProblem(
        {'x': initial[:, 0], 'y': initial[:, 1]},
        DummyCost(optimal),
        constraints=[XYBoundaryConstraint(boundary, boundary_type)],
        driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False),
        plot_comp=plot_comp)


def testPolygon():
    boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
    b = PolygonBoundaryComp(0, boundary)
    np.testing.assert_array_equal(b.xy_boundary[:, :2], [[0, 0],
                                                         [1, 1],
                                                         [2, 0],
                                                         [2, 2],
                                                         [0, 2],
                                                         [0, 0]])


def testPolygonConcave():
    optimal = [(1.5, 1.3), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()
    initial = [(-0, .1), (4, 1.5)][::-1]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
    plot_comp.show()


def testPolygonTwoRegionsStartInWrong():
    optimal = [(1, 1), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()
    initial = [(3.5, 1.5), (0.5, 1.5)]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    plot_comp.show()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)


def testMultiPolygon():
    optimal = [(1.75, 1.3), (4, 1)]
    boundary = [InclusionZone([(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]),
                InclusionZone([(3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5)]),
                ExclusionZone([(0.5, 0.5), (1.75, 0.5), (1.75, 1.5), (0.5, 1.5)]),
                ExclusionZone([(0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)]),
                ]
    xy_bound_ref_ = np.array([[0., 0.],
                              [5., 0.],
                              [5., 2.],
                              [3., 2.],
                              [3., 1.],
                              [2., 1.],
                              [2., 2.],
                              [0., 2.],
                              [0., 0.]])

    bound_dist_ref = np.array([0, 1])
    plot_comp = NoPlot()
    initial = np.asarray([(-0, .1), (4, 1.5)][::-1])
    tf = get_tf(initial, optimal, boundary, plot_comp, boundary_type='multi_polygon')
    tf.evaluate()
    cost, state, recorder = tf.optimize()
    np.testing.assert_array_almost_equal(recorder['xy_boundary'][-1], xy_bound_ref_, 4)
    np.testing.assert_array_almost_equal(recorder['boundaryDistances'][-1], bound_dist_ref, 4)
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)


def test_calculate_distance_to_boundary():
    import matplotlib.pyplot as plt
    boundary = np.array([(0, 0), (10, 0), (20, 10), (20, 20), (0, 20)])
    points = np.array([(2, 10), (10, 21), (14, 6)])

    boundary_constr = XYBoundaryConstraint(boundary, 'convex_hull').get_comp(10)
    import numpy.testing as npt
    if 0:
        plt.plot(*boundary.T, )
        plt.plot(*points.T, '.')
        plt.axis('equal')
        plt.grid()
        plt.show()
    npt.assert_array_almost_equal(boundary_constr.calculate_distance_to_boundary(points),
                                  [[18., 10., 2., 10., 12.72792206],
                                   [10., -1., 10., 21., 14.8492424],
                                   [6., 14., 14., 6., 1.41421356]])


def testDistanceRelaxation():
    boundary = [InclusionZone([(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]),
                InclusionZone([(3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5)]),
                ExclusionZone([(0.5, 0.5), (1.75, 0.5), (1.75, 1.5), (0.5, 1.5)]),
                ExclusionZone([(0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)]),
                ]
    initial = [(-0, .1), (4, 1.5)][::-1]
    optimal = [(1.75, 1.3), (4, 1)]
    initial, optimal = map(np.array, [initial, optimal])
    plot_comp = NoPlot()
    tf = TopFarmProblem({'x': initial[:, 0], 'y': initial[:, 1]}, DummyCost(optimal, inputs=['x', 'y']),
                        constraints=[XYBoundaryConstraint(boundary, 'multi_polygon', relaxation=(0.9, 4))],
                        plot_comp=plot_comp, driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False))
    tf.evaluate()
    cost, state, recorder = tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
    relaxation = tf.model.constraint_components[0].calc_relaxation() \
        + tf.model.constraint_components[0].relaxation[0]
    assert tf.cost_comp.n_grad_eval <= 10
    assert tf.model.constraint_group.xy_bound_comp == tf.model.constraint_components[0]
    # distances in the 2 lower corners should be the same
    assert tf.model.constraint_components[0].distances(np.array([0]), np.array([0])) \
        == tf.model.constraint_components[0].distances(np.array([5]), np.array([0]))
    # gradients with respect of iteration number should be the same at every point
    assert tf.model.constraint_components[0].gradients(np.array([3]), np.array([5]))[2][0] \
        == tf.model.constraint_components[0].gradients(np.array([1.5]), np.array([8]))[2][1]


def testDistanceRelaxationPolygons():
    zones = [InclusionZone([(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]),
             InclusionZone([(3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5)]),
             ExclusionZone([(0.5, 0.5), (1.75, 0.5), (1.75, 1.5), (0.5, 1.5)]),
             ExclusionZone([(0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)]),
             ]
    MPBC = MultiPolygonBoundaryComp(1, zones, relaxation=(0.1, 10))
    (rp1, rp2) = MPBC.relaxed_polygons(7)
    bp1 = MPBC.get_boundary_properties(*rp1)
    bp2 = MPBC.get_boundary_properties(*rp2)
    # order does not matter
    __assert_equal_unordered(bp1[0], np.array([[2.3, 1.3],
                                               [2.7, 1.3],
                                               [2.7, 2.3],
                                               [5.3, 2.3],
                                               [5.3, -0.3],
                                               [-0.3, -0.3],
                                               [-0.3, 2.3],
                                               [2.3, 2.3]]))
    __assert_equal_unordered(bp2[0], np.array([[1.45, 1.2],
                                               [0.8, 1.2],
                                               [0.8, 0.8],
                                               [1.45, 0.8]]))


def testChangingNumberOfTurbines():
    zones = [InclusionZone([(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]),
             InclusionZone([(3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5)]),
             ExclusionZone([(0.5, 0.5), (1.75, 0.5), (1.75, 1.5), (0.5, 1.5)]),
             ExclusionZone([(0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)]),
             ]

    MPBC = MultiPolygonBoundaryComp(1, zones)

    xs, ys = np.linspace(0, 5, 5), np.linspace(0, 2, 5)
    XS, YS = np.meshgrid(xs, ys)
    X, Y = XS.ravel(), YS.ravel()
    _, _, sign = MPBC.calc_distance_and_gradients(X, Y)
    sign_ref = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0])
    np.testing.assert_allclose(sign, sign_ref)

    xs2, ys2 = np.linspace(0, 5, 2), np.linspace(0, 2, 1)
    XS2, YS2 = np.meshgrid(xs2, ys2)
    X2, Y2 = XS2.ravel(), YS2.ravel()
    _, _, sign2 = MPBC.calc_distance_and_gradients(X2, Y2)
    sign_ref2 = np.array([0, 0])
    np.testing.assert_allclose(sign2, sign_ref2)


class TestMultiPolygonBoundaryCompResultingPolygons(unittest.TestCase):
    def test_single_inclusion(self):
        """Test basic case of single inclusion polygon"""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        polygons = [square]
        incl_excls = [True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 1
        assert result[0].equals(square)

    def test_multiple_non_overlapping(self):
        """Test multiple non-overlapping inclusion polygons"""
        square1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        square2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        polygons = [square1, square2]
        incl_excls = [True, True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 2

    def test_overlapping_inclusions(self):
        """Test overlapping inclusion polygons"""
        square1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        square2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        polygons = [square1, square2]
        incl_excls = [True, True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 1
        assert result[0].area == 7  # Area of union

        d = Polygon([(0.1, 0.1), (1.1, 0.1), (1.1, 1.1), (0.1, 1.1), (0.1, 0.1)])
        b = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
        polygons = [d, b]
        incl_excls = [True, True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 1
        assert result[0].area == 4  # Area of union

    def test_inclusion_with_hole(self):
        """Test inclusion with hole (exclusion inside)"""
        outer = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        inner = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        polygons = [outer, inner]
        incl_excls = [True, False]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 1
        assert result[0].area == outer.area - inner.area

    def test_multiple_exclusions(self):
        """Test multiple exclusion areas"""
        main = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        excl1 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        excl2 = Polygon([(3, 3), (3.5, 3), (3.5, 3.5), (3, 3.5)])
        polygons = [main, excl1, excl2]
        incl_excls = [True, False, False]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 1
        expected_area = main.area - excl1.area - excl2.area
        assert abs(result[0].area - expected_area) < 1e-10

    def test_tiny_polygons(self):
        """Test handling of tiny polygons (area < 1e-3)"""
        main = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        tiny = Polygon([(0.1, 0.1), (0.11, 0.1), (0.11, 0.11), (0.1, 0.11)])
        polygons = [main, tiny]
        incl_excls = [True, True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 1  # Tiny polygon should be filtered out
        assert result[0].equals(main)

    def test_empty_input(self):
        """Test empty input handling"""
        result = MultiPolygonBoundaryComp._calc_resulting_polygons([], [])
        assert len(result) == 0

    def test_complex_case(self):
        """Test complex case with multiple inclusions/exclusions"""
        # Create complex arrangement of polygons
        main = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        incl1 = Polygon([(12, 0), (15, 0), (15, 5), (12, 5)])
        excl1 = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
        excl2 = Polygon([(13, 1), (14, 1), (14, 2), (13, 2)])

        polygons = [main, incl1, excl1, excl2]
        incl_excls = [True, True, False, False]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) > 0
        total_area = sum(p.area for p in result)
        expected_area = main.area + incl1.area - excl1.area - excl2.area
        assert abs(total_area - expected_area) < 1e-10

    def test_touching_polygons(self):
        """Test handling of touching polygons"""
        square1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        square2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # Touches square1
        polygons = [square1, square2]
        incl_excls = [True, True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(polygons, incl_excls)

        assert len(result) == 1  # Should merge into single polygon
        assert result[0].area == square1.area + square2.area

    def test_single_inclusion(self):
        # Create a simple square inclusion zone
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        boundary_polygons = [square]
        incl_excls = [True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(
            boundary_polygons, incl_excls
        )

        assert len(result) == 1
        assert result[0].equals(square)

    def test_multiple_inclusions_not_overlapping(self):
        # Two separate squares
        square1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        square2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        boundary_polygons = [square1, square2]
        incl_excls = [True, True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(
            boundary_polygons, incl_excls
        )

        assert len(result) == 2
        assert result[0].equals(square1)
        assert result[1].equals(square2)

    def test_multiple_inclusions_overlapping(self):
        # Two overlapping squares
        square1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        square2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        boundary_polygons = [square1, square2]
        incl_excls = [True, True]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(
            boundary_polygons, incl_excls
        )

        # Should merge into single polygon
        assert len(result) == 1
        expected_area = 7  # Total area minus overlap
        assert abs(result[0].area - expected_area) < 1e-10

    def test_exclusion_in_inclusion(self):
        # Square with hole
        outer = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        inner = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        boundary_polygons = [outer, inner]
        incl_excls = [True, False]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(
            boundary_polygons, incl_excls
        )

        assert len(result) == 1
        expected_area = outer.area - inner.area
        assert abs(result[0].area - expected_area) < 1e-10

    def test_multiple_exclusions(self):
        # Square with two holes
        outer = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
        hole1 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        hole2 = Polygon([(2, 2), (2.5, 2), (2.5, 2.5), (2, 2.5)])
        boundary_polygons = [outer, hole1, hole2]
        incl_excls = [True, False, False]

        result = MultiPolygonBoundaryComp._calc_resulting_polygons(
            boundary_polygons, incl_excls
        )

        assert len(result) == 1
        expected_area = outer.area - hole1.area - hole2.area
        assert abs(result[0].area - expected_area) < 1e-10
