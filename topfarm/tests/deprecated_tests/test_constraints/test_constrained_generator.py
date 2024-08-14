from openmdao.drivers.doe_generators import UniformGenerator
import pytest
import numpy as np
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt, uta
from topfarm.constraint_components.constrained_generator import ConstrainedXYZGenerator, \
    ConstrainedDiscardXYZGenerator
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm._topfarm import TopFarmProblem


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
        turbineXYZ = np.array(turbineXYZ)
        desvar = dict(zip('xy', turbineXYZ.T))
        desvar['z'] = (turbineXYZ[:, 2], z_boundary[0], z_boundary[1])
        constraints = [XYBoundaryConstraint(xy_boundary, xy_boundary_type)]
        if min_spacing:
            constraints.append(SpacingConstraint(min_spacing))

        return TopFarmProblem(
            desvar,
            cost_comp,
            constraints=constraints,
            driver=driver)

    return get_InitialXYZOptimizationProblem


@pytest.mark.parametrize('driver', [ConstrainedXYZGenerator,
                                    ConstrainedDiscardXYZGenerator])
def test_with_convex_boundary(get_tf, driver):
    tf = get_tf(xy_boundary=[(0, 0), (10, 0), (10, 10)], xy_boundary_type='convex_hull',
                driver=driver(UniformGenerator(10, 0)))
    arr = tf.get_DOE_array()
    x, y = [arr[:, i] for i in range(2)]
    uta.assertGreaterEqual(x.min(), 0)  # x
    uta.assertGreaterEqual(y.min(), 0)  # y
    npt.assert_array_less(y, x)


@pytest.mark.parametrize('driver', [ConstrainedXYZGenerator,
                                    ConstrainedDiscardXYZGenerator])
def test_with_constrained_generator_polygon(get_tf, driver):
    tf = get_tf(xy_boundary=[(0, 0), (10, 0), (10, 10)],
                xy_boundary_type='polygon',
                driver=driver(UniformGenerator(10, 0)))
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

    x, y = [arr[:, i] for i in range(2)]
    assert all(np.sqrt(np.diff(x)**2 + np.diff(y)**2) >= 2)


@pytest.mark.parametrize('driver', [ConstrainedXYZGenerator,
                                    ConstrainedDiscardXYZGenerator])
def test_with_convex_boundary_and_constrain(get_tf, driver):
    tf = get_tf(xy_boundary=[(0, 0), (10, 0), (10, 10)], xy_boundary_type='convex_hull',
                min_spacing=2, driver=driver(UniformGenerator(10, 0)))
    arr = tf.get_DOE_array()
    x, y = [arr[:, i] for i in range(2)]
    uta.assertGreaterEqual(x.min(), 0)  # x
    uta.assertGreaterEqual(y.min(), 0)  # y
    npt.assert_array_less(y, x)
    assert all(np.sqrt(np.diff(x)**2 + np.diff(y)**2) >= 2)
