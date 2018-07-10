from topfarm import TopFarm
import numpy as np
import pytest
from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasyPyOptSparseIPOPT


initial = [[6, 0], [6, -8], [1, 1]]  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # desired turbine layouts
boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
desired = [[3, -3], [7, -7], [4, -3]]  # desired turbine layouts


@pytest.fixture
def topfarm_generator():
    def _topfarm_obj(driver):
        # from topfarm.cost_models.dummy import DummyCostPlotComp
        # plot_comp = DummyCostPlotComp(desired)
        plot_comp = NoPlot()
        return TopFarm(initial, DummyCost(desired), 2, plot_comp=plot_comp, boundary=boundary, driver=driver, record_id=None)
    return _topfarm_obj


@pytest.mark.parametrize('driver,tol', [(EasyScipyOptimizeDriver(), 1e-4),
                                        (EasyScipyOptimizeDriver(tol=1e-3), 1e-2),
                                        (EasyScipyOptimizeDriver(maxiter=13), 1e-1),
                                        (EasyScipyOptimizeDriver(optimizer='COBYLA', tol=1e-3, disp=False), 1e-2),
                                        (EasyPyOptSparseIPOPT(), 1e-4),
                                        ][:])
def test_optimizers(driver, tol, topfarm_generator):
    if driver.__class__.__name__ == "PyOptSparseMissingDriver":
        pytest.xfail("reason")
    tf = topfarm_generator(driver)
    tf.evaluate()
    print(driver.__class__.__name__)
    tf.optimize()
    tb_pos = tf.turbine_positions
    # tf.plot_comp.show()

    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0] < 6 + tol  # check within border
    np.testing.assert_array_almost_equal(tb_pos, optimal, -int(np.log10(tol)))
