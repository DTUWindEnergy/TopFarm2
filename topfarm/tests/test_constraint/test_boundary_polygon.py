import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint, \
    PolygonBoundaryComp, InclusionZone, ExclusionZone, MultiPolygonBoundaryComp
from topfarm._topfarm import TopFarmProblem
from topfarm.tests.utils import __assert_equal_unordered
import pytest
from shapely import Polygon


@pytest.fixture
def multi_polygon_zones():
    return [
        InclusionZone([(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]),
        InclusionZone([(3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5)]),
        ExclusionZone([(0.5, 0.5), (1.75, 0.5), (1.75, 1.5), (0.5, 1.5)]),
        ExclusionZone([(0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)]),
    ]


def get_tf(initial, optimal, boundary, plot_comp=NoPlot(), boundary_type='polygon'):
    initial, optimal = map(np.array, [initial, optimal])
    return TopFarmProblem(
        {'x': initial[:, 0], 'y': initial[:, 1]},
        DummyCost(optimal),
        constraints=[XYBoundaryConstraint(boundary, boundary_type)],
        driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False),
        plot_comp=plot_comp)


def test_polygon():
    boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
    b = PolygonBoundaryComp(0, boundary)
    np.testing.assert_array_equal(b.xy_boundary[:, :2], [[0, 0],
                                                         [1, 1],
                                                         [2, 0],
                                                         [2, 2],
                                                         [0, 2],
                                                         [0, 0]])


def test_polygon_concave():
    optimal = [(1.5, 1.3), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()
    initial = [(-0, .1), (4, 1.5)][::-1]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
    plot_comp.show()


def test_polygon_two_regions_start_in_wrong():
    optimal = [(1, 1), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()
    initial = [(3.5, 1.5), (0.5, 1.5)]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    plot_comp.show()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)


def test_multi_polygon():
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


def test_distance_relaxation(multi_polygon_zones):
    boundary = multi_polygon_zones
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


def test_distance_relaxation_polygons(multi_polygon_zones):
    MPBC = MultiPolygonBoundaryComp(1, multi_polygon_zones, relaxation=(0.1, 10))
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


def test_changing_number_of_turbines(multi_polygon_zones):
    MPBC = MultiPolygonBoundaryComp(1, multi_polygon_zones)

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


def test_resulting_polygons_single_inclusion():
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([square], [True])
    assert len(result) == 1
    assert result[0].equals(square)


def test_resulting_polygons_multiple_non_overlapping():
    square1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    square2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([square1, square2], [True, True])
    assert len(result) == 2


def test_resulting_polygons_overlapping_inclusions():
    square1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    square2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([square1, square2], [True, True])
    assert len(result) == 1
    assert result[0].area == 7

    d = Polygon([(0.1, 0.1), (1.1, 0.1), (1.1, 1.1), (0.1, 1.1), (0.1, 0.1)])
    b = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([d, b], [True, True])
    assert len(result) == 1
    assert result[0].area == 4


def test_resulting_polygons_inclusion_with_hole():
    outer = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    inner = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([outer, inner], [True, False])
    assert len(result) == 1
    assert result[0].area == outer.area - inner.area


def test_resulting_polygons_multiple_exclusions():
    main = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    excl1 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    excl2 = Polygon([(3, 3), (3.5, 3), (3.5, 3.5), (3, 3.5)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([main, excl1, excl2], [True, False, False])
    assert len(result) == 1
    assert abs(result[0].area - (main.area - excl1.area - excl2.area)) < 1e-10


def test_resulting_polygons_tiny_polygons():
    main = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    tiny = Polygon([(0.1, 0.1), (0.11, 0.1), (0.11, 0.11), (0.1, 0.11)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([main, tiny], [True, True])
    assert len(result) == 1
    assert result[0].equals(main)


def test_resulting_polygons_empty_input():
    assert len(MultiPolygonBoundaryComp._calc_resulting_polygons([], [])) == 0


def test_resulting_polygons_complex_case():
    main = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    incl1 = Polygon([(12, 0), (15, 0), (15, 5), (12, 5)])
    excl1 = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    excl2 = Polygon([(13, 1), (14, 1), (14, 2), (13, 2)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([main, incl1, excl1, excl2], [True, True, False, False])
    assert len(result) > 0
    assert abs(sum(p.area for p in result) - (main.area + incl1.area - excl1.area - excl2.area)) < 1e-10


def test_resulting_polygons_touching_polygons():
    square1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    square2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    result = MultiPolygonBoundaryComp._calc_resulting_polygons([square1, square2], [True, True])
    assert len(result) == 1
    assert result[0].area == square1.area + square2.area
