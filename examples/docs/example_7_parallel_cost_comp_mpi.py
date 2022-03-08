import numpy as np
from topfarm._topfarm import TopFarmProblem, TopFarmGroup, TopFarmParallelGroup
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot
import time
from openmdao.api import n2



from openmdao.api import Problem, IndepVarComp, ParallelGroup, ExecComp, Group

def main():
    if __name__ == '__main__':
        # define the conditions for the wind farm
        boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]  # turbine boundaries
        initial = np.array([[6, 0], [6, -8], [1, 1], [-1, -8]])  # initial turbine pos
        desired = np.array([[3, -3], [7, -7], [4, -3], [3, -7]])  # desired turbine pos
        optimal = np.array([[2.5, -3], [6, -7], [4.5, -3], [3, -7]])  # optimal layout
        min_spacing = 2  # min distance between turbines

        # ------------------------ OPTIMIZATION ------------------------

        # create the wind farm and run the optimization

        def wt_cost(i, x, y):
            time.sleep(0.01)
            return (desired[i, 0] - x[i])**2 + (desired[i, 1] - y[i])**2

        n_wt = len(initial)
        comps = [CostModelComponent('xy', 4,
                                    cost_function=lambda x, y, i=i:wt_cost(i, x, y),
                                    objective=False,
                                    output_keys='cost%d' % i) for i in range(n_wt)]

        def sum_map(**kwargs):

            return np.sum([kwargs['cost%d' % i] for i in range(n_wt)])

        comps.append(CostModelComponent(['cost%d' % i for i in range(n_wt)], 1,
                                        cost_function=sum_map,
                                        objective=True))
        cost_comp = TopFarmParallelGroup(comps)

        tf = TopFarmProblem(
            design_vars={'x': initial[:, 0], 'y': initial[:, 1]},
            cost_comp=cost_comp,
            constraints=[XYBoundaryConstraint(boundary),
                         SpacingConstraint(min_spacing)],
#            plot_comp=DummyCostPlotComp(desired),
            plot_comp=NoPlot(),
            driver=EasyScipyOptimizeDriver()
        )
#        n2(tf)
        #print(tf.evaluate({'x': desired[:, 0], 'y': desired[:, 1]}))
        print(tf.evaluate({'x': optimal[:, 0], 'y': optimal[:, 1]}, disp=False))
        #print(tf.evaluate({'x': initial[:, 0], 'y': initial[:, 1]}))
#        tic = time.time()
        cost, state, recorder = tf.optimize()
#        toc = time.time()
#        print('optimized in {:.3f}s '.format(toc-tic))
        tf.plot_comp.show()


main()
N_PROCS = 2
