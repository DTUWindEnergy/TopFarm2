"""Example: optimizing a layout with constraints

This example uses a dummy cost function to optimize a simple wind turbine
layout that is subject to constraints. The optimization pushes the wind turbine
locations to specified locations in the farm.
"""
import os

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np

from topfarm import TopFarm
from topfarm.cost_models.dummy import DummyCost
from topfarm._topfarm import TurbineTypeOptimizationProblem
from openmdao.drivers.doe_generators import FullFactorialGenerator
from topfarm.plotting import TurbineTypePlotComponent, NoPlot


# ------------------------ INPUTS ------------------------

# define the conditions for the wind farm
positions = np.array([[0, 0], [6, 6]])  # initial turbine pos
optimal_types = np.array([[2], [6]])  # optimal layout


#===============================================================================
# Setup the problem and plotting
#===============================================================================

try:
    import matplotlib.pyplot as plt
    plt.gcf()
    plot_comp = TurbineTypePlotComponent(turbine_type_names=["Turbine %d" % i for i in range(10)], delay=0.1)
    plot = True
except RuntimeError:
    plot_comp = NoPlot()
    plot = False

# create the wind farm
tf = TurbineTypeOptimizationProblem(
    cost_comp=DummyCost(optimal_types, ['turbineType']),
    turbineTypes=[0, 0],
    lower=0, upper=9,
    plot_comp=plot_comp,
    driver=FullFactorialGenerator(10))


#===============================================================================
# #  Run the optimization
#===============================================================================
state = {'turbineX': positions[:, 0], 'turbineY': positions[:, 1]}
cost, state, recorder = tf.optimize(state)

#===============================================================================
# plot and prin the the final, optimal types
#===============================================================================
print(state['turbineType'])
tf.evaluate(state)


# save the figure
if plot:
    plt.savefig(os.path.basename(__file__).replace('.py', '.png'))
