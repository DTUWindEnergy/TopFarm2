import pytest
from topfarm.cost_models.dummy import DummyCostPlotComp, DummyCost
from topfarm import TurbineXYZOptimizationProblem
import numpy as np
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary_component import BoundaryComp


optimal = np.array([(5, 4, 3),
                    (3, 2, 1)])


@pytest.fixture
def tf():
    def get_TurbineXYZOptimizationProblem(
            cost_comp=DummyCost(optimal, ['x', 'y', 'z']),
            turbineXYZ=[[0, 0, 0],
                        [2, 2, 2]],
            xy_boundary=[(0, 0), (5, 5)],
            z_boundary=[1, 4],
            xy_boundary_type='square',
            plot_comp=None):
        return TurbineXYZOptimizationProblem(
            cost_comp=cost_comp,
            turbineXYZ=turbineXYZ,
            boundary_comp=BoundaryComp(len(turbineXYZ), xy_boundary, z_boundary, xy_boundary_type),
            plot_comp=plot_comp,
            driver=EasyScipyOptimizeDriver(disp=False))
    return get_TurbineXYZOptimizationProblem


def test_evaluate(tf):
    cost, state = tf().evaluate()
    assert cost == 52
    np.testing.assert_array_equal(state['x'], [0, 2])


def test_optimize(tf):
    #     plot_comp = DummyCostPlotComp(optimal)
    plot_comp = NoPlot()
    tf = tf(plot_comp=plot_comp)
    cost = tf.optimize()[0]
    plot_comp.show()
    assert cost < 1e6
    np.testing.assert_array_almost_equal(tf.turbine_positions, optimal[:, :2], 3)
