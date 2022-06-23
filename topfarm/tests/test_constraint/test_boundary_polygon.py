import numpy as np
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp

from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint,\
    PolygonBoundaryComp
from topfarm._topfarm import TopFarmProblem


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
    plot_comp = NoPlot()  # DummyCostPlotComp(optimal)
    initial = [(-0, .1), (4, 1.5)][::-1]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
    plot_comp.show()


def testPolygonTwoRegionsStartInWrong():
    optimal = [(1, 1), (4, 1)]
    boundary = [(0, 0), (5, 0), (5, 2), (3, 2), (3, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    plot_comp = NoPlot()  # DummyCostPlotComp(optimal, delay=.1)
    initial = [(3.5, 1.5), (0.5, 1.5)]
    tf = get_tf(initial, optimal, boundary, plot_comp)
    tf.optimize()
    plot_comp.show()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)


def testMultiPolygon():
    optimal = [(1.75, 1.3), (4, 1)]
    boundary = [([(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)], 1),
                ([(3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5)], 1),
                ([(0.5, 0.5), (1.75, 0.5), (1.75, 1.5), (0.5, 1.5)], 0),
                ([(0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)], 0),
                ]
    xy_bound_ref_ = np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                               [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                               [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                               [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                              [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                               [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]).T
    bound_dist_ref = np.array([[5.00000000e-01, -0.00000000e+00],
                               [5.00000000e-01, -0.00000000e+00],
                               [4.44089210e-16, 8.88178420e-16],
                               [-2.65765765e-01, 4.57207206e-01],
                               [-4.25507089e-02, 6.16166344e-01],
                               [0.00000000e+00, 9.87427871e-01],
                               [0.00000000e+00, 9.91504055e-01],
                               [7.95692918e-04, 9.99406616e-01],
                               [0.00000000e+00, 9.99843063e-01],
                               [0.00000000e+00, 9.99997428e-01]])
    plot_comp = NoPlot()  # DummyCostPlotComp(optimal)
    initial = [(-0, .1), (4, 1.5)][::-1]
    tf = get_tf(initial, optimal, boundary, plot_comp, boundary_type='multi_polygon')
    tf.evaluate()
    cost, state, recorder = tf.optimize()
    np.testing.assert_array_almost_equal(recorder['xy_boundary'], xy_bound_ref_, 4)
    np.testing.assert_array_almost_equal(recorder['boundaryDistances'], bound_dist_ref, 4)
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
    boundary = [([(0, 0), (5, 0), (5, 2), (3, 2), (3, 1), (2, 1), (2, 2), (0, 2), (0, 0)], 1),
                ([(3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (3.5, 1.5)], 1),
                ([(0.5, 0.5), (1.75, 0.5), (1.75, 1.5), (0.5, 1.5)], 0),
                ([(0.75, 0.75), (1.25, 0.75), (1.25, 1.25), (0.75, 1.25)], 0),
                ]
    initial = [(-0, .1), (4, 1.5)][::-1]
    optimal = [(1.75, 1.3), (4, 1)]
    initial, optimal = map(np.array, [initial, optimal])
    plot_comp = NoPlot()
    tf = TopFarmProblem({'x': initial[:, 0], 'y': initial[:, 1]}, DummyCost(optimal, inputs=['x', 'y']),
                        constraints=[XYBoundaryConstraint(boundary, 'multi_polygon', relaxation=(0.9, 10))],
                        plot_comp=plot_comp, driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False))
    tf.evaluate()
    tf.optimize()
    np.testing.assert_array_almost_equal(tf.turbine_positions[:, :2], optimal, 4)
    relaxation = tf.model.constraint_components[0].calc_relaxation() \
        + tf.model.constraint_components[0].relaxation[0]
    assert tf.cost_comp.n_grad_eval == 7
    assert tf.model.pre_constraints.xy_bound_comp == tf.model.constraint_components[0]
    assert tf.model.constraint_components[0].relaxation[1] - tf.cost_comp.n_grad_eval == 3
    # distances in the 2 lower corners should be the same
    assert tf.model.constraint_components[0].distances(np.array([0]), np.array([0])) \
        == tf.model.constraint_components[0].distances(np.array([5]), np.array([0]))
    # gradients with respect of iteration number should be the same at every point
    assert tf.model.constraint_components[0].gradients(np.array([3]), np.array([5]))[2][0] \
        == tf.model.constraint_components[0].gradients(np.array([1.5]), np.array([8]))[2][1]
