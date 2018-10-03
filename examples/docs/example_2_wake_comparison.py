"""Example: optimization with different wake models

This example uses a dummy cost function to optimize a simple wind turbine
layout that is subject to constraints. The optimization pushes the wind turbine
locations to specified locations in the farm.
"""
import os
import warnings

from matplotlib.patches import Polygon

import matplotlib.pyplot as plt
import numpy as np
from topfarm._topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.fused_wake_wrappers import FusedWakeGCLWakeModel, \
    FusedWakeNOJWakeModel
from topfarm.cost_models.utils.aep_calculator import AEPCalculator
from topfarm.cost_models.utils.wind_resource import WindResource
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.tests import test_files


def main():
    if __name__ == '__main__':
        # ------------------------ INPUTS ------------------------

        # paths to input files
        test_files_dir = os.path.dirname(test_files.__file__) + "/"  # file locations
        wf_path = test_files_dir + 'wind_farms/3tb.yml'  # path to wind farm

        # ------------------------ DEFINE WIND RESOURCE ------------------------
        # wind resource info (wdir frequency, weibull a and k)
        f = [3.597152, 3.948682, 5.167395, 7.000154, 8.364547, 6.43485, 8.643194,
             11.77051, 15.15757, 14.73792, 10.01205, 5.165975]
        a = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921, 9.584007,
             10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
        k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703, 2.583984,
             2.548828, 2.470703, 2.607422, 2.626953, 2.326172]

        wind_res = WindResource(f, a, k, np.zeros_like(k))

        # ------------------------ setup problem ____---------------------------
        rot_diam = 80.0  # rotor diameter [m]
        init_pos = np.array([(0, 2 * rot_diam), (0, 0),
                             (0, -2 * rot_diam)])  # initial turbine positions
        b = 2 * rot_diam + 10  # boundary size
        boundary = [(-b, -b), (-b, b), (b, b), (b, -b)]  # corners of wind farm boundary
        min_spacing = 2.0 * rot_diam  # minimum spacing between turbines [m]

        # ------------------------ OPTIMIZATION ------------------------

        def get_tf(wake_model):
            return TopFarmProblem(
                design_vars=dict(zip('xy', init_pos.T)),
                cost_comp=AEPCalculator(wind_res, wake_model,
                                        wdir=np.arange(0, 360, 12)
                                        ).get_TopFarm_cost_component(),
                constraints=[SpacingConstraint(min_spacing),
                             XYBoundaryConstraint(boundary)],
                driver=EasyScipyOptimizeDriver())

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')  # temporarily disable fusedwake warnings

            # GCL: define the wake model and optimization problem
            wake_mod_gcl = FusedWakeGCLWakeModel(wf_path)
            tf_gcl = get_tf(wake_mod_gcl)

            # NOJ: define the wake model and optimization problem
            wake_mod_noj = FusedWakeNOJWakeModel(wf_path)
            tf_noj = get_tf(wake_mod_noj)

            # run the optimization
            cost_gcl, state_gcl, recorder_gcl = tf_gcl.optimize()
            cost_noj, state_noj, recorder_noj = tf_noj.optimize()

            # ------------------------ POST-PROCESS ------------------------

            # get the optimized locations
            opt_gcl = tf_gcl.turbine_positions
            opt_noj = tf_noj.turbine_positions

            # create the array of costs for easier printing
            costs = np.diag([cost_gcl, cost_noj])
            costs[0, 1] = tf_noj.evaluate(state_gcl)[0]  # noj cost of gcl locs
            costs[1, 0] = tf_gcl.evaluate(state_noj)[0]  # gcl cost of noj locs

        # ------------------------ PRINT STATS ------------------------

        aep_diffs = 200 * (costs[:, 0] - costs[:, 1]) / (costs[:, 0] + costs[:, 1])
        loc_diffs = 200 * (costs[0, :] - costs[1, :]) / (costs[0, :] + costs[1, :])

        print('\nComparison of cost models vs. optimized locations:')
        print('\nCost    |    GCL_aep      NOJ_aep')
        print('---------------------------------')
        print(f'GCL_loc |{costs[0,0]:11.2f} {costs[0,1]:11.2f}' +
              f'   ({aep_diffs[0]:.2f}%)')
        print(f'NOJ_loc |{costs[1,0]:11.2f} {costs[1,1]:11.2f}' +
              f'   ({aep_diffs[1]:.2f}%)')
        print(f'             ({loc_diffs[0]:.2f}%)     ({loc_diffs[1]:.2f}%)')

        # ------------------------ PLOT (if possible) ------------------------

        try:

            # initialize the figure and axes
            fig = plt.figure(1, figsize=(7, 5))
            plt.clf()
            ax = plt.axes()

            # plot the boundary and desired locations
            ax.add_patch(Polygon(boundary, fill=False,
                                 label='Boundary'))  # boundary
            ax.plot(init_pos[:, 0], init_pos[:, 1], 'xk',
                    label='Initial')
            ax.plot(opt_gcl[:, 0], opt_gcl[:, 1], 'o',
                    label='GCL')
            ax.plot(opt_noj[:, 0], opt_noj[:, 1], '^',
                    label='NOJ')

            # make a few adjustments to the plot
            ax.autoscale_view()  # autoscale the boundary
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=4, mode='expand', borderaxespad=0.)  # add a legend
            plt.tight_layout()  # zoom the plot in
            plt.axis('off')  # remove the axis

            # save the png
            folder, file = os.path.split(__file__)
            fig.savefig(folder + "/figures/" + file.replace('.py', '.png'))


        except RuntimeError:
            pass


main()
