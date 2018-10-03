import os
import numpy as np
import multiprocessing
from topfarm.constraint_components.boundary_component import BoundaryComp
from topfarm.cost_models.dummy import DummyCost
from openmdao.drivers.doe_generators import UniformGenerator
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm._topfarm import TopFarmProblem
import threading


class ParallelRunner():

    def __init__(self, processes=None):
        """
        Parameters
        ----------
        processes : int or None, optional
            Number of processes passed to multiprocessing.Pool
        """
        self.pool = multiprocessing.Pool(processes)

    def __call__(self, state_lst, seq_runner):
        """Run in parallel

        Parameters
        ----------
        state_lst : list
            List of states (as returned by TopFarmProblem.get_DOE_list()
        seq_runner : function
            function for sequential run. Interface must be:
            def f(lst):
                tf = TopfarmProblem(...)
                return tf.optimize()

        Returns
        -------
        best : (cost, state, recorder)
            best result
        results : [(cost1, state1, recorder1),...]
            all results
        """

        indexes = np.round(np.linspace(0, len(state_lst), self.pool._processes + 1)).astype(np.int)
        seq_lst = [state_lst[i1:i2] for i1, i2 in zip(indexes[:-1], indexes[1:])]

        results = self.pool.map(seq_runner, seq_lst)
        best = results[np.argmin([r[0] for r in results])]
        return best, results


# ===============================================================================
# Example functions
# ===============================================================================
def get_InitialXYZOptimizationProblem(driver):
    optimal = [(1, 0, 4),
               (0, 1, 3)]
    return TopFarmProblem(
        design_vars={'x': [0, 2], 'y': [0, 2], 'z': ([0, 2], 3, 4)},
        cost_comp=DummyCost(optimal, ['x', 'y', 'z']),
        constraints=[XYBoundaryConstraint([(10, 6), (11, 8)], 'rectangle')],
        driver=driver)


def seq_runner_example(lst):
    print("%d cases executed by thread: %s" % (len(lst), threading.get_ident()))
    return get_InitialXYZOptimizationProblem(lst).optimize()
# ===============================================================================
#
# ===============================================================================


def main():
    if __name__ == '__main__':

        lst = get_InitialXYZOptimizationProblem(driver=UniformGenerator(200)).get_DOE_list()
        print("Current thread: %s" % threading.get_ident())
        print("\nRun on two processors")
        # run in parallel
        par_runner = ParallelRunner(2)
        (cost, state, recorder), results = par_runner(lst, seq_runner_example)
        print("Minimum cost: %.2f" % cost)

        print("\nRun on one processor")
        # run sequential
        cost, state, recorder = seq_runner_example(lst)
        print("Minimum cost: %.2f" % cost)


main()
