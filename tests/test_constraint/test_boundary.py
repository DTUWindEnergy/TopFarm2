import unittest

import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm import TopFarm


class TestBoundary(unittest.TestCase):

    def testSquare(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 3)]
        tf = TopFarm(optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='square')
        np.testing.assert_array_equal(tf.boundary, [[-1, 0],
                                                    [2, 0],
                                                    [2, 3],
                                                    [-1, 3],
                                                    [-1, 0]])

    def testRectangle(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 3)]
        tf = TopFarm(optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='rectangle')
        np.testing.assert_array_equal(tf.boundary, [[0, 0],
                                                    [1, 0],
                                                    [1, 3],
                                                    [0, 3],
                                                    [0, 0]])

    def testConvexHull(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
        tf = TopFarm(optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='convex_hull')
        np.testing.assert_array_equal(tf.boundary, [[0, 0],
                                                    [2, 0],
                                                    [2, 2],
                                                    [0, 2],
                                                    [0, 0]])

    def testNotImplemented(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
        self.assertRaisesRegex(NotImplementedError, "Boundary type 'Something' is not implemented", TopFarm, optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='Something')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
