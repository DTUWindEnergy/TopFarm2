import numpy as np
from topfarm import TurbineXYZOptimizationProblem
from topfarm.constraint_components.boundary_component import BoundaryComp, \
    ConvexBoundaryComp
from topfarm.tests import npt
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.cost_models.dummy import DummyCost
import pytest


def test_z_boundary():
    optimal = [(0, 0, 0), (0, 0, 0)]
    tf = TurbineXYZOptimizationProblem(
        DummyCost(optimal, ['z']),
        optimal,
        BoundaryComp(2, None, [70, 90]),
        driver=SimpleGADriver())
    tf.setup()
    desvars = tf.driver._designvars
    np.testing.assert_array_equal(desvars['indeps.z']['lower'], [70, 70])
    np.testing.assert_array_equal(desvars['indeps.z']['upper'], [90, 90])


def test_boundary_component():
    with pytest.raises(NotImplementedError, match="Boundary type 'missing' is not implemented"):
        ConvexBoundaryComp(3, [(0, 0), (0, 1)], xy_boundary_type='missing')

# def test_xyz_boundary():
#     optimal = [(0, 0, 0)]
#     boundary = [(0, 0), (1, 3)]
#     tf = TurbineXYZOptimizationProblem(
#         DummyCost(optimal, ['x', 'y', 'z']),
#         optimal,
#         BoundaryComp(1, boundary, [70, 90], xy_boundary_type='rectangle'),
#         driver=SimpleGADriver())
#
#     np.testing.assert_array_equal(tf.xy_boundary, [[0, 0],
#                                                    [1, 0],
#                                                    [1, 3],
#                                                    [0, 3],
#                                                    [0, 0]])
#     desvars = tf.driver._designvars
#     np.testing.assert_array_equal(desvars['indeps.z']['lower'], [70])
#     np.testing.assert_array_equal(desvars['indeps.z']['upper'], [90])


def test_move_inside():
    pbc = ConvexBoundaryComp(1, [(0, 0), (10, 0), (10, 10)])
    x0, y0 = [3, 3, 3, 12, 12, 12], [3, 5, 10, 8, 10, 12]
    x, y, z = pbc.move_inside(x0, y0, [])
#     import matplotlib.pyplot as plt
#     b = np.r_[pbc.xy_boundary, pbc.xy_boundary[:1]]
#     plt.plot(b[:, 0], b[:, 1], 'k')
#     for x0_, x_, y0_, y_ in zip(x0, x, y0, y):
#         plt.plot([x0_, x_], [y0_, y_], '.-')
#     plt.show()
    eps = 1e-10
    npt.assert_array_less(y, x + eps)
    npt.assert_array_less(x, 10 + eps)
    npt.assert_array_less(y, 10 + eps)
    npt.assert_array_less(-x, 0 + eps)
    npt.assert_array_less(-y, 0 + eps)
