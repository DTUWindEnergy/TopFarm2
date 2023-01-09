# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:11:37 2023

@author: mikf
"""

import numpy as np
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.plotting import XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from py_wake.examples.data.iea37._iea37 import IEA37Site
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt

n_wt = 16
site = IEA37Site(n_wt)
windTurbines = WindTurbines(names=['T1', 'T2', 'T3'],
                                    diameters=[110, 130, 150],
                                    hub_heights=[110, 130, 150],
                                    powerCtFunctions = [CubePowerSimpleCt(power_rated=200 * 110 ** 2, power_unit='W'),
                                                       CubePowerSimpleCt(power_rated=200 * 130 ** 2, power_unit='W'),
                                                       CubePowerSimpleCt(power_rated=200 * 150 ** 2, power_unit='W')],)
windFarmModel = IEA37SimpleBastankhahGaussian(site, windTurbines)
types = 5 * [0] + 6 * [1] + 5 * [2]
tf = TopFarmProblem(
    design_vars=dict(zip('xy', site.initial_position.T)),
    cost_comp=PyWakeAEPCostModelComponent(windFarmModel, n_wt, additional_input=[('type', types)], grad_method=None),
    driver=EasyScipyOptimizeDriver(maxiter=50),
    constraints=[CircleBoundaryConstraint([0, 0], 1300.1),
                 SpacingConstraint(windTurbines.diameter(0) * 4)],
    plot_comp=XYPlotComp())
x = np.linspace(0,2600,101)
y = np.linspace(0,2600,101)
YY, XX = np.meshgrid(y, x)
idx = (XX - 1300) ** 2 + (YY - 1300) ** 2 <= 1300 ** 2
XX = XX[idx].ravel() - 1300
YY = YY[idx].ravel() - 1300

cost1, state1 = tf.evaluate(dict(zip('xy', site.initial_position.T)))
cost2, state2, recorder2 = tf.optimize()


tf.smart_start(XX, YY, tf.cost_comp.get_aep4smart_start(type=types))
cost3, state3 = tf.evaluate()

cost4, state4, recorder4 = tf.optimize()

costs = [cost1, cost2, cost3, cost4]
strings = ['initial', 'initial + 50 iter. optimization', 'smart start', 'smart start + 50 iter. optimization']
for s, c in zip(strings, costs):
    print(f'{s:35}: {abs(c):.1f}')
    