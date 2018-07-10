import unittest

import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm import TopFarm
import pytest


class TestBoundary(unittest.TestCase):

    def testSquare(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 3)]
        tf = TopFarm(optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='square', record_id=None)
        np.testing.assert_array_equal(tf.boundary, [[-1, 0],
                                                    [2, 0],
                                                    [2, 3],
                                                    [-1, 3],
                                                    [-1, 0]])

    def testRectangle(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 3)]
        tf = TopFarm(optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='rectangle', record_id=None)
        np.testing.assert_array_equal(tf.boundary, [[0, 0],
                                                    [1, 0],
                                                    [1, 3],
                                                    [0, 3],
                                                    [0, 0]])

    def testConvexHull(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
        tf = TopFarm(optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='convex_hull', record_id=None)
        np.testing.assert_array_equal(tf.boundary, [[0, 0],
                                                    [2, 0],
                                                    [2, 2],
                                                    [0, 2],
                                                    [0, 0]])

    def testNotImplemented(self):
        optimal = [(0, 0)]
        boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
        self.assertRaisesRegex(NotImplementedError, "Boundary type 'Something' is not implemented", TopFarm,
                               optimal, DummyCost(optimal), 2, boundary=boundary, boundary_type='Something', record_id=None)


xy, z = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], (70, 90)


@pytest.mark.parametrize('b,xy,z', [(None, None, None),
                                    [(None, None), None, None],
                                    (xy, xy, None),
                                    ([xy], xy, None),
                                    ([xy, None], xy, None),
                                    ([xy, z], xy, z),
                                    ([None, z], None, z)])
def test_boundary_xy_z_input(b, xy, z):
    optimal = [(0, 0)]
    tf = TopFarm(optimal, DummyCost(optimal), 2, boundary=b, record_id=None)
    if xy is None:
        assert tf.boundary_comp is None
    else:
        np.testing.assert_array_equal(tf.boundary, xy)
    if z is None:
        assert 'turbineZ' not in tf.problem.model._static_design_vars.keys()
    else:
        assert tf.problem.model._static_design_vars['turbineZ']['lower'] == [70]
        assert tf.problem.model._static_design_vars['turbineZ']['upper'] == [90]


def test_boundary_xy_z_input_TypeError():
    optimal = [(0, 0)]
    with pytest.raises(TypeError, match="Boundary must be one of"):
        TopFarm(optimal, DummyCost(optimal), 2, boundary=[xy, z, z], record_id=None)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
