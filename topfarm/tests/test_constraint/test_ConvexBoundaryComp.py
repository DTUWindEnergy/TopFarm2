import numpy as np
import pytest
from topfarm.tests import npt
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint, \
    ConvexBoundaryComp
from topfarm._topfarm import TopFarmProblem
from topfarm.cost_models.dummy import DummyCost


def testSquare():
    boundary = [(0, 0), (1, 3)]
    b = ConvexBoundaryComp(0, boundary, 'square')
    np.testing.assert_array_equal(b.xy_boundary, [[-1, 0],
                                                  [2, 0],
                                                  [2, 3],
                                                  [-1, 3],
                                                  [-1, 0]])


def testRectangle():
    boundary = [(0, 0), (1, 3)]
    b = ConvexBoundaryComp(0, boundary, 'rectangle')
    np.testing.assert_array_equal(b.xy_boundary, [[0, 0],
                                                  [1, 0],
                                                  [1, 3],
                                                  [0, 3],
                                                  [0, 0]])


def testConvexHull():
    boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
    b = ConvexBoundaryComp(0, boundary, 'convex_hull')
    np.testing.assert_array_equal(b.xy_boundary, [[0, 0],
                                                  [2, 0],
                                                  [2, 2],
                                                  [0, 2],
                                                  [0, 0]])


def testNotImplemented():
    boundary = [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
    with pytest.raises(NotImplementedError, match="Boundary type 'Something' is not implemented"):
        ConvexBoundaryComp(0, boundary, boundary_type='Something')


def test_z_boundary():
    optimal = np.array([(0, 0, 0)]).T
    tf = TopFarmProblem(
        {'z': (optimal, 70, 90)},
        DummyCost(optimal, 'z'),
        driver=pyOptSparseDriver(**{"optimizer": "NSGA2"})
    )

    desvars = tf.driver._designvars
    print(desvars)
    np.testing.assert_array_equal(desvars['indeps.z']['lower'], [70, 70])
    np.testing.assert_array_equal(desvars['indeps.z']['upper'], [90, 90])


def test_xyz_boundary():
    optimal = np.array([(0, 0, 0)])
    boundary = [(0, 0), (1, 3)]
    desvar = dict(zip('xy', optimal.T))
    desvar['z'] = (optimal[:, 2], 70, 90)
    b = XYBoundaryConstraint(boundary, boundary_type='rectangle')
    tf = TopFarmProblem(
        desvar,
        DummyCost(optimal, 'xyz'),
        constraints=[b],
        driver=pyOptSparseDriver(**{"optimizer": "NSGA2"}))

    np.testing.assert_array_equal(b.constraintComponent.xy_boundary, [[0, 0],
                                                                      [1, 0],
                                                                      [1, 3],
                                                                      [0, 3],
                                                                      [0, 0]])
    desvars = tf.driver._designvars
    np.testing.assert_array_equal(desvars['indeps.z']['lower'], [70])
    np.testing.assert_array_equal(desvars['indeps.z']['upper'], [90])


def test_move_inside():
    pbc = ConvexBoundaryComp(1, [(0, 0), (9, 0), (10, 1), (10, 10)])
    x0, y0 = [3, 3, 3, 12, 12, 12], [3, 5, 10, 8, 10, 12]
    state = pbc.satisfy({'x': x0, 'y': y0})
    x, y = state['x'], state['y']
    if 0:
        import matplotlib.pyplot as plt
        b = np.r_[pbc.xy_boundary, pbc.xy_boundary[:1]]
        plt.plot(b[:, 0], b[:, 1], 'k')
        for x0_, x_, y0_, y_ in zip(x0, x, y0, y):
            plt.plot([x0_, x_], [y0_, y_], '.-')
        plt.show()
    eps = 1e-10
    npt.assert_array_less(y, x + eps)
    npt.assert_array_less(x, 10 + eps)
    npt.assert_array_less(y, 10 + eps)
    npt.assert_array_less(-x, 0 + eps)
    npt.assert_array_less(-y, 0 + eps)
