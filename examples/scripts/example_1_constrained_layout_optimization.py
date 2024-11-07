"""Example: optimizing a layout with constraints

This example uses a dummy cost function to optimize a simple wind turbine
layout that is subject to constraints. The optimization pushes the wind turbine
locations to specified locations in the farm.
"""
import os

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np

from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot


def main():
    if __name__ == '__main__':
        # ------------------------ INPUTS ------------------------

        # define the conditions for the wind farm
        boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
        initial = np.array([[6, 0], [6, -8], [1, 1], [-1, -8]])  # initial turbine pos
        desired = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine pos
        optimal = np.array([[2.5, -3], [6, -7], [4.5, -3], [3, -7]])  # optimal layout
        min_spacing = 2  # min distance between turbines

        try:
            import matplotlib.pyplot as plt
            plt.gcf()
            plot_comp = DummyCostPlotComp(desired)
            plot = True
        except RuntimeError:
            plot_comp = NoPlot()
            plot = False
        # ------------------------ OPTIMIZATION ------------------------

        # create the wind farm and run the optimization

        tf = TopFarmProblem(
            design_vars={'x': initial[:, 0], 'y': initial[:, 1]},
            cost_comp=DummyCost(desired, ['x', 'y']),
            constraints=[XYBoundaryConstraint(boundary),
                         SpacingConstraint(min_spacing)],
            plot_comp=plot_comp,
            driver=EasyScipyOptimizeDriver()
        )
        cost, state, recorder = tf.optimize()
        tf.plot_comp.show()

        # final position
        final_x, final_y = state['x'], state['y']

        # get the positions tried during optimization from the recorder
        rec_x, rec_y = recorder['x'], recorder['y']

        # get the final, optimal positions
        optimized = tf.turbine_positions

        # ------------------------ PLOT (if possible) ------------------------

        if plot:

            # initialize the figure and axes
            fig = plt.figure(1, figsize=(7, 5))
            plt.clf()
            ax = plt.axes()

            # plot the boundary and desired locations
            ax.add_patch(Polygon(boundary, closed=True, fill=False,
                                 label='Boundary'))  # boundary
            plt.plot(desired[:, 0], desired[:, 1], 'ok', mfc='None', ms=10,
                     label='Desired')  # desired positions

            # plot the history of each turbine
            for i_turb in range(rec_x.shape[1]):
                l, = plt.plot(rec_x[0, i_turb], rec_y[0, i_turb],
                              'x', ms=8, label=f'Turbine {i_turb+1}')  # initial
                plt.plot(rec_x[:, i_turb], rec_y[:, i_turb],
                         c=l.get_color())  # tested values
                plt.plot(rec_x[-1, i_turb], rec_y[-1, i_turb],
                         'o', ms=8, c=l.get_color())  # final

            # make a few adjustments to the plot
            ax.autoscale_view()  # autoscale the boundary
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=4, mode='expand', borderaxespad=0.)  # add a legend
            plt.tight_layout()  # zoom the plot in
            plt.axis('off')  # remove the axis

            # save the png
            folder, file = os.path.split(__file__)
            os.makedirs(folder + "/figures/", exist_ok=True)
            fig.savefig(folder + "/figures/" + file.replace('.py', '.png'))

#        except RuntimeError:
#            pass


main()
