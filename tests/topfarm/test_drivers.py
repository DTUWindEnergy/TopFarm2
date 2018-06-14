'''
Created on 17. maj 2018

@author: mmpe
'''
from topfarm import TopFarm

import numpy as np
import pytest
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.plotting import NoPlot
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySimpleGADriver,\
    EasyPyOptSparseSLSQP, EasyPyOptSparseIPOPT


initial = [[6, 0], [6, -8], [1, 1]]  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # desired turbine layouts
boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
desired = [[3, -3], [7, -7], [4, -3]]  # desired turbine layouts


@pytest.fixture
def topfarm_generator():
    def _topfarm_obj(driver):
        plot_comp = DummyCostPlotComp(desired)
        #plot_comp = NoPlot()
        return TopFarm(initial, DummyCost(desired), 2, plot_comp=plot_comp, boundary=boundary, driver=driver)
    return _topfarm_obj




#         # CONMIN-specific Settings
#         self.driver.itmax = 30
#         self.driver.fdch = 0.00001
#         self.driver.fdchm = 0.000001
#         self.driver.ctlmin = 0.01
#         self.driver.delfun = 0.001
# 
#         # NEWSUMT-specific Settings
#         #self.driver.itmax = 10
# 
#         # COBYLA-specific Settings
#         #self.driver.rhobeg = 1.0
#         #self.driver.rhoend = 1.0e-4
#         #self.driver.maxfun = 1000
# 
#         # SLSQP-specific Settings
#         #self.driver.accuracy = 1.0e-6
#         #self.driver.maxiter = 50
# 
#         # Genetic-specific Settings
#         #self.driver.population_size = 90
#         #self.driver.crossover_rate = 0.9
#         #self.driver.mutation_rate = 0.02
#         #self.selection_method = 'rank'


@pytest.mark.parametrize('driver,tol',[(EasyScipyOptimizeDriver(), 1e-4),
                                       (EasyScipyOptimizeDriver(tol=1e-3), 1e-2),
                                       (EasyScipyOptimizeDriver(maxiter=13), 1e-1),
                                       (EasyScipyOptimizeDriver(optimizer='COBYLA', tol=1e-3), 1e-2),
                                       (EasyPyOptSparseSLSQP(),1e-4),
                                       (EasyPyOptSparseIPOPT(),1e-4),
                                       #(EasySimpleGADriver(), 1e-4)
                                       ][-1:])
def test_optimizers(driver, tol, topfarm_generator):
    if isinstance(driver, str):
        pytest.xfail("reason")
    tf = topfarm_generator(driver)
    tf.optimize()
    tb_pos = tf.turbine_positions
    #tf.plot_comp.show()

    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing
    assert tb_pos[1][0]< 6 + tol  # check within border
    np.testing.assert_array_almost_equal(tb_pos, optimal, -int(np.log10(tol)))
    #print (tb_pos - optimal)
