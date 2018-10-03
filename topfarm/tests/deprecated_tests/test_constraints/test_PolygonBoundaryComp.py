import numpy as np
from topfarm.constraint_components.boundary_component import PolygonBoundaryComp
import pytest
from topfarm.tests import npt


@pytest.mark.parametrize('boundary', [
    [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)],
    [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2), (0, 0)],  # StartEqEnd
    [(0, 0), (0, 2), (2, 2), (2, 0), (1, 1)],  # Clockwise
    [(0, 0), (0, 2), (2, 2), (2, 0), (1, 1), (0, 0)]  # StartEqEndClockwise
])
def testPolygon(boundary):
    pbc = PolygonBoundaryComp(1, boundary)
    np.testing.assert_array_equal(pbc.xy_boundary, [[0, 0],
                                                    [1, 1],
                                                    [2, 0],
                                                    [2, 2],
                                                    [0, 2]])


def check(boundary, points, distances):
    pbc = PolygonBoundaryComp(1, boundary)
    d, dx, dy = pbc.calc_distance_and_gradients(points[:, 0], points[:, 1])
    np.testing.assert_array_almost_equal(d, distances)
    eps = 1e-7
    d1, _, _ = pbc.calc_distance_and_gradients(points[:, 0] + eps, points[:, 1])
    np.testing.assert_array_almost_equal((d1 - d) / eps, dx)
    d2, _, _ = pbc.calc_distance_and_gradients(points[:, 0], points[:, 1] + eps)
    np.testing.assert_array_almost_equal((d2 - d) / eps, dy)


def test_calc_distance_edge():
    boundary = np.array([(0, 0), (1, 0), (2, 1), (0, 2), (0, 0)])
    points = np.array([(0.5, .2), (1, .5), (.5, 1.5), (.2, 1)])
    check(boundary, points, [0.2, np.sqrt(2 * .25**2), .5 * np.sin(np.arctan(.5)), 0.2])


def test_calc_distance_edge_outside():
    boundary = np.array([(0, 0), (1, 0), (2, 1), (0, 2), (0, 0)])
    points = np.array([(0.5, -.2), (1.5, 0), (.5, 2), (-.2, 1)])
    check(boundary, points, [-0.2, -np.sqrt(2 * .25**2), -.5 * np.sin(np.arctan(.5)), -0.2])


def test_calc_distance_point_vertical():
    boundary = np.array([(0, 0), (1, 1), (2, 0), (2, 2), (0, 2), (0, 0)])
    points = np.array([(.8, 1), (.8, 1.2), (1, 1.2), (1.1, 1.2), (1.2, 1.2), (1.2, 1)])
    check(boundary, points, [np.sqrt(.2**2 / 2), np.sqrt(2 * .2**2), .2,
                             np.sqrt(.1**2 + .2**2), np.sqrt(2 * .2**2), np.sqrt(.2**2 / 2)])


def test_calc_distance_point_vertical_outside():
    boundary = np.array([(0, 0), (1, 1), (2, 0), (0, 0)])
    points = np.array([(.8, 1), (.8, 1.2), (1, 1.2), (1.1, 1.2), (1.2, 1.2), (1.2, 1)])

    check(boundary, points, [-np.sqrt(.2**2 / 2), -np.sqrt(2 * .2**2), -.2,
                             -np.sqrt(.1**2 + .2**2), -np.sqrt(2 * .2**2), -np.sqrt(.2**2 / 2)])


def test_calc_distance_point_horizontal():
    boundary = np.array([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (0, 0)])
    points = np.array([(1, .8), (.8, .8), (.8, 1), (.8, 1.1), (.8, 1.2), (1, 1.2)])
    check(boundary, points, [np.sqrt(.2**2 / 2), np.sqrt(2 * .2**2), .2,
                             np.sqrt(.1**2 + .2**2), np.sqrt(2 * .2**2), np.sqrt(.2**2 / 2)])


def testPolygon_Line():
    boundary = [(0, 0), (0, 2)]
    with pytest.raises(AssertionError, match="Area must be non-zero"):
        PolygonBoundaryComp(1, boundary)


def test_calc_distance_U_shape():
    boundary = np.array([(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)])
    points = np.array([(-.1, 1.5), (.1, 1.5), (.9, 1.5), (1.1, 1.5), (1.5, 1.5), (1.9, 1.5), (2.1, 1.5), (2.9, 1.5), (3.1, 1.5)])
    check(boundary, points, [-.1, .1, .1, -.1, -.5, -.1, .1, .1, -.1])


def test_calc_distance_V_shape():
    boundary = np.array([(0, 0), (1, 2), (2, 0), (2, 2), (1, 4), (0, 2)])
    points = np.array([(.8, 2), (.8, 2.2), (1, 2.2), (1.2, 2.2), (1.2, 2), (.8, 4), (.8, 4.2), (1, 4.2), (1.2, 4.2), (1.2, 4)])
    v1 = np.sqrt(.2**2 * 4 / 5)
    v2 = np.sqrt(2 * .2**2)
    check(boundary, points, [v1, v2, .2, v2, v1, -v1, -v2, -.2, -v2, -v1])


def test_move_inside():
    pbc = PolygonBoundaryComp(1, [(0, 0), (10, 0), (10, 10)])
    x, y, z = pbc.move_inside([3, 3, 3], [0, 5, 10], [])
    npt.assert_array_less(y, x)
