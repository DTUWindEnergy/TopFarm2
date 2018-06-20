import unittest

import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm import TopFarm
from topfarm.constraint_components.boundary_component import PolygonBoundaryComp


class TestPolygonBoundaryComp(unittest.TestCase):

    def testPolygon(self):
        boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
        pbc = PolygonBoundaryComp(boundary, 1)
        np.testing.assert_array_equal(pbc.vertices, [[0, 0],
                                                     [1, 1],
                                                     [2, 0],
                                                     [2, 2],
                                                     [0, 2]])

    def testPolygon_StartEqEnd(self):
        boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2), (0, 0)]
        pbc = PolygonBoundaryComp(boundary, 1)
        np.testing.assert_array_equal(pbc.vertices, [[0, 0],
                                                     [1, 1],
                                                     [2, 0],
                                                     [2, 2],
                                                     [0, 2]])

    def testPolygon_Clockwise(self):
        boundary = [(0, 0), (0, 2), (2, 2), (2, 0), (1, 1)]
        pbc = PolygonBoundaryComp(boundary, 1)
        np.testing.assert_array_equal(pbc.vertices, [[0, 0],
                                                     [1, 1],
                                                     [2, 0],
                                                     [2, 2],
                                                     [0, 2]])

    def testPolygon_StartEqEndClockwise(self):
        boundary = [(0, 0), (0, 2), (2, 2), (2, 0), (1, 1), (0, 0)]
        pbc = PolygonBoundaryComp(boundary, 1)
        np.testing.assert_array_equal(pbc.vertices, [[0, 0],
                                                     [1, 1],
                                                     [2, 0],
                                                     [2, 2],
                                                     [0, 2]])

    def check(self, boundary, points, distances):
        pbc = PolygonBoundaryComp(boundary, 1)
        d, dx, dy = pbc.calc_distance_and_gradients(points[:, 0], points[:, 1])
        np.testing.assert_array_almost_equal(d, distances)
        eps = 1e-7
        d1, _, _ = pbc.calc_distance_and_gradients(points[:, 0] + eps, points[:, 1])
        np.testing.assert_array_almost_equal((d1 - d) / eps, dx)
        d2, _, _ = pbc.calc_distance_and_gradients(points[:, 0], points[:, 1] + eps)
        np.testing.assert_array_almost_equal((d2 - d) / eps, dy)

    def test_calc_distance_edge(self):
        boundary = np.array([(0, 0), (1, 0), (2, 1), (0, 2), (0, 0)])
        points = np.array([(0.5, .2), (1, .5), (.5, 1.5), (.2, 1)])
        self.check(boundary, points, [0.2, np.sqrt(2 * .25**2), .5 * np.sin(np.arctan(.5)), 0.2])

    def test_calc_distance_edge_outside(self):
        boundary = np.array([(0, 0), (1, 0), (2, 1), (0, 2), (0, 0)])
        points = np.array([(0.5, -.2), (1.5, 0), (.5, 2), (-.2, 1)])
        self.check(boundary, points, [-0.2, -np.sqrt(2 * .25**2), -.5 * np.sin(np.arctan(.5)), -0.2])

    def test_calc_distance_point_vertical(self):
        boundary = np.array([(0, 0), (1, 1), (2, 0), (2, 2), (0, 2), (0, 0)])
        points = np.array([(.8, 1), (.8, 1.2), (1, 1.2), (1.1, 1.2), (1.2, 1.2), (1.2, 1)])
        self.check(boundary, points, [np.sqrt(.2**2 / 2), np.sqrt(2 * .2**2), .2,
                                      np.sqrt(.1**2 + .2**2), np.sqrt(2 * .2**2), np.sqrt(.2**2 / 2)])

    def test_calc_distance_point_vertical_outside(self):
        boundary = np.array([(0, 0), (1, 1), (2, 0), (0, 0)])
        points = np.array([(.8, 1), (.8, 1.2), (1, 1.2), (1.1, 1.2), (1.2, 1.2), (1.2, 1)])

        self.check(boundary, points, [-np.sqrt(.2**2 / 2), -np.sqrt(2 * .2**2), -.2,
                                      -np.sqrt(.1**2 + .2**2), -np.sqrt(2 * .2**2), -np.sqrt(.2**2 / 2)])

    def test_calc_distance_point_horizontal(self):
        boundary = np.array([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (0, 0)])
        points = np.array([(1, .8), (.8, .8), (.8, 1), (.8, 1.1), (.8, 1.2), (1, 1.2)])
        self.check(boundary, points, [np.sqrt(.2**2 / 2), np.sqrt(2 * .2**2), .2,
                                      np.sqrt(.1**2 + .2**2), np.sqrt(2 * .2**2), np.sqrt(.2**2 / 2)])

    def testPolygon_Line(self):
        boundary = [(0, 0), (0, 2)]
        self.assertRaisesRegex(AssertionError, "Area must be non-zero", PolygonBoundaryComp, boundary, 1)

    def test_calc_distance_U_shape(self):
        boundary = np.array([(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)])
        points = np.array([(-.1, 1.5), (.1, 1.5), (.9, 1.5), (1.1, 1.5), (1.5, 1.5), (1.9, 1.5), (2.1, 1.5), (2.9, 1.5), (3.1, 1.5)])
        self.check(boundary, points, [-.1, .1, .1, -.1, -.5, -.1, .1, .1, -.1])

    def test_calc_distance_V_shape(self):
        boundary = np.array([(0, 0), (1, 2), (2, 0), (2, 2), (1, 4), (0, 2)])
        points = np.array([(.8, 2), (.8, 2.2), (1, 2.2), (1.2, 2.2), (1.2, 2), (.8, 4), (.8, 4.2), (1, 4.2), (1.2, 4.2), (1.2, 4)])
        v1 = np.sqrt(.2**2 * 4 / 5)
        v2 = np.sqrt(2 * .2**2)
        self.check(boundary, points, [v1, v2, .2, v2, v1, -v1, -v2, -.2, -v2, -v1])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
