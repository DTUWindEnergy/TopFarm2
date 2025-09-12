"""Example: optimizing a layout with constraints

This example uses a dummy cost function to optimize turbine types.
"""
import numpy as np
import os
from topfarm.cost_models.dummy import DummyCost
from topfarm._topfarm import TopFarmProblem
from openmdao.drivers.doe_generators import FullFactorialGenerator
from topfarm.plotting import TurbineTypePlotComponent, NoPlot


def main():
    if __name__ == '__main__':
        # ------------------------ INPUTS ------------------------

        # define the conditions for the wind farm
        positions = np.array([[0, 0], [6, 6]])  # initial turbine pos
        optimal_types = np.array([[2], [6]])  # optimal layout

        # ===============================================================================
        # Setup the problem and plotting
        # ===============================================================================

        try:
            import matplotlib.pyplot as plt
            plt.gcf()
            plot_comp = TurbineTypePlotComponent(
                turbine_type_names=["Turbine %d" % i for i in range(5)],
                plot_initial=False,
                delay=0.1, legendloc=0)
            plot = True
        except RuntimeError:
            plot_comp = NoPlot()
            plot = False

        # create the wind farm
        tf = TopFarmProblem(
            design_vars={'type': ([0, 0], 0, 4)},
            cost_comp=DummyCost(optimal_types, ['type']),
            plot_comp=plot_comp,
            driver=FullFactorialGenerator(5),
            ext_vars={'x': positions[:, 0], 'y': positions[:, 1]},
        )

        # ===============================================================================
        # #  Run the optimization
        # ===============================================================================
        state = {}
        cost, state, recorder = tf.optimize(state)

        # ===============================================================================
        # plot and prin the the final, optimal types
        # ===============================================================================
        print(state['type'])
        tf.evaluate(state)

        # save the figure
        if plot:
            folder, file = os.path.split(__file__)
            os.makedirs(folder + "/figures/", exist_ok=True)
            plt.savefig(folder + "/figures/" + file.replace('.py', '.png'))
            plt.show()


main()
