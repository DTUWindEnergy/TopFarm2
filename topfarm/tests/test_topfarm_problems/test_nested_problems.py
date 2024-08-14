from openmdao.drivers.doe_generators import FullFactorialGenerator, \
    ListGenerator
from openmdao.drivers.doe_driver import DOEDriver
from topfarm.cost_models.dummy import DummyCost
import numpy as np
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.tests import npt
from topfarm import TopFarmProblem
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint


optimal = np.array([(0, 2, 4, 1), (4, 2, 1, 0)])


def test_2level_turbineType_and_XYZ_optimization():
    design_vars = {k: v for k, v in zip('xy', optimal.T)}
    design_vars['z'] = (optimal[:, 2], 0, 4)
    xyz_problem = TopFarmProblem(
        design_vars,
        cost_comp=DummyCost(optimal, ['x', 'y', 'z', 'type']),
        constraints=[SpacingConstraint(2), XYBoundaryConstraint([(0, 0), (4, 4)], 'square')],
        driver=EasyScipyOptimizeDriver(disp=False))
    tf = TopFarmProblem(
        {'type': ([0, 0], 0, 1)},
        cost_comp=xyz_problem,
        driver=DOEDriver(FullFactorialGenerator(2)))
    cost = tf.optimize()[0]
    assert cost == 0


def test_3level_type_multistart_XYZ_optimization():
    design_vars = {k: v for k, v in zip('xy', optimal.T)}
    design_vars['z'] = (optimal[:, 2], 0, 4)
    xyz_problem = TopFarmProblem(
        design_vars,
        cost_comp=DummyCost(optimal, ['x', 'y', 'z', 'type']),
        constraints=[SpacingConstraint(2), XYBoundaryConstraint([(0, 0), (4, 4)], 'square')],
        driver=EasyScipyOptimizeDriver(disp=False))

    initial_xyz_problem = TopFarmProblem(
        design_vars={k: v for k, v in zip('xyz', optimal.T)},
        cost_comp=xyz_problem,
        driver=DOEDriver(ListGenerator([[('x', [0, 4]), ('y', [2, 2]), ('z', [4, 1])]])))

    tf = TopFarmProblem(
        {'type': ([0, 0], 0, 1)},
        cost_comp=initial_xyz_problem,
        driver=DOEDriver(FullFactorialGenerator(2)))

    cost, _, recorder = tf.optimize()
    best_index = np.argmin(recorder.get('cost'))
    initial_xyz_recorder = recorder['recorder'][best_index]
    xyz_recorder = initial_xyz_recorder.get('recorder')[0]
    npt.assert_almost_equal(xyz_recorder['cost'][-1], cost)
