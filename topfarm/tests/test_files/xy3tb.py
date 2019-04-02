import numpy as np

from topfarm.cost_models.dummy import DummyCost
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm._topfarm import TopFarmProblem
from topfarm.plotting import NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.capacity import CapacityConstraint, CapacityComp
import topfarm
from topfarm.constraint_components.boundary import XYBoundaryConstraint
initial = np.array([[6, 0], [6, -8], [1, 1]])  # initial turbine layouts
optimal = np.array([[2.5, -3], [6, -7], [4.5, -3]])  # optimal turbine layouts
boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
desired = np.array([[3, -3], [7, -7], [4, -3]])  # desired turbine layouts
desvars = {topfarm.x_key: initial[:, 0], topfarm.y_key: initial[:, 1]}
capacit = {"max_capacity": 5000, "rated_power_array": [3000, 1000, 500]}


def get_tf(**kwargs):
    k = {'cost_comp': DummyCost(desired[:, :2], [topfarm.x_key, topfarm.y_key]),
         'design_vars': {topfarm.x_key: initial[:, 0], topfarm.y_key: initial[:, 1]},
         'driver': EasyScipyOptimizeDriver(disp=True, tol=1e-8),
         'plot_comp': NoPlot(),
         'constraints': [SpacingConstraint(2), XYBoundaryConstraint(boundary), CapacityConstraint(**capacit)]}

    k.update(kwargs)
    return TopFarmProblem(**k)
