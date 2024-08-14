from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import FullFactorialGenerator, \
    ListGenerator, UniformGenerator
import pytest
from topfarm import InitialXYZOptimizationProblem
import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt, uta
from topfarm.constraint_components.constrained_generator import ConstrainedXYZGenerator
from topfarm.constraint_components.boundary_component import BoundaryComp


@pytest.fixture
def get_tf():
    def get_InitialXYZOptimizationProblem(driver,
                                          min_spacing=None,
                                          turbineXYZ=[[0, 0, 0],
                                                      [2, 2, 2]],
                                          xy_boundary=[(10, 6), (11, 8)],
                                          xy_boundary_type='rectangle',
                                          z_boundary=[3, 4]):
        cost_comp = DummyCost([(1, 0, 4),
                               (0, 1, 3)], ['x', 'y', 'z'])
        return InitialXYZOptimizationProblem(
            cost_comp, turbineXYZ,
            BoundaryComp(len(turbineXYZ), xy_boundary, z_boundary, xy_boundary_type),
            min_spacing,
            driver)

    return get_InitialXYZOptimizationProblem


def test_list_driver(get_tf):
    xyz = [[[1, 2], [3, 4], [5, 6]],
           [[4, 3], [6, 5], [2, 1]]]
    lst = [[('x', [1, 2]), ('y', [3, 4]), ('z', [5, 6])],
           [('x', [4, 3]), ('y', [6, 5]), ('z', [2, 1])]]

    tf = get_tf(driver=lst)  # pure list
    npt.assert_array_equal(tf.get_DOE_array(), xyz)

    tf = get_tf(driver=ListGenerator(lst))  # list generator
    npt.assert_array_equal(tf.get_DOE_array(), xyz)

    tf = get_tf(driver=DOEDriver(ListGenerator(lst)))  # DOEDriver
    npt.assert_array_equal(tf.get_DOE_array(), xyz)


def test_with_uniform_generator(get_tf):
    tf = get_tf(driver=DOEDriver(UniformGenerator(10)))
    arr = tf.get_DOE_array()
    uta.assertGreaterEqual(arr[:, 0].min(), 10)  # x
    uta.assertLessEqual(arr[:, 0].max(), 11)  # x
    uta.assertGreaterEqual(arr[:, 1].min(), 6)  # y
    uta.assertLessEqual(arr[:, 1].max(), 8)  # y
    uta.assertGreaterEqual(arr[:, 2].min(), 3)  # z
    uta.assertLessEqual(arr[:, 2].max(), 4)  # z
    cost, _, recorder = tf.optimize()
    npt.assert_equal(cost, np.min(recorder.get('cost')))


def test_with_constrained_generator_convex_boundary(get_tf):
    tf = get_tf(xy_boundary=[(0, 0), (10, 0), (10, 10)], xy_boundary_type='convex_hull',
                driver=ConstrainedXYZGenerator(UniformGenerator(10, 0)))
    arr = tf.get_DOE_array()
    x, y, z = [arr[:, i] for i in range(3)]
    uta.assertGreaterEqual(x.min(), 0)  # x
    uta.assertGreaterEqual(y.min(), 0)  # y
    npt.assert_array_less(y, x)


def test_with_constrained_generator_polygon(get_tf):
    tf = get_tf(xy_boundary=[(0, 0), (10, 0), (10, 10)],
                xy_boundary_type='polygon',
                driver=ConstrainedXYZGenerator(UniformGenerator(10, 0)))
    arr = tf.get_DOE_array()
    x, y = [arr[:, i] for i in range(2)]
    uta.assertGreaterEqual(x.min(), 0)  # x
    uta.assertGreaterEqual(y.min(), 0)  # y
    npt.assert_array_less(y, x)


def test_with_constrained_generator_spacing(get_tf):
    lst = [[('x', [1, 1]), ('y', [3, 4]), ('z', [5, 6])],
           [('x', [2, 2]), ('y', [6, 5]), ('z', [2, 1])]]

    tf = get_tf(xy_boundary=[(0, 0), (10, 10)],
                xy_boundary_type='rectangle',
                min_spacing=2,
                driver=ConstrainedXYZGenerator(lst))
    arr = tf.get_DOE_array()

    x, y, z = [arr[:, i] for i in range(3)]
    assert all(np.sqrt(np.diff(x)**2 + np.diff(y)**2) >= 2)
