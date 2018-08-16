import os
import numpy as np
import multiprocessing
from topfarm._topfarm import InitialXYZOptimizationProblem
from topfarm.constraint_components.boundary_component import BoundaryComp
from topfarm.cost_models.dummy import DummyCost
from openmdao.drivers.doe_generators import UniformGenerator


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

        indexes = np.round(np.linspace(0, len(state_lst), self.pool._processes)).astype(np.int)
        seq_lst = [state_lst[i1:i2] for i1, i2 in zip(indexes[:-1], indexes[1:])]

        results = self.pool.map(seq_runner, seq_lst)
        best = results[np.argmin([r[0] for r in results])]
        return best, results


def get_InitialXYZOptimizationProblem(driver):
    return InitialXYZOptimizationProblem(
        cost_comp=DummyCost([(1, 0, 4),
                             (0, 1, 3)]),
        min_spacing=None,
        turbineXYZ=[[0, 0, 0],
                    [2, 2, 2]],
        boundary_comp=BoundaryComp(n_wt=2,
                                   xy_boundary=[(10, 6), (11, 8)],
                                   xy_boundary_type='rectangle',
                                   z_boundary=[3, 4]),
        driver=driver)


def seq_runner_example(lst):
    return get_InitialXYZOptimizationProblem(lst).optimize()


def try_me():
    if __name__ == '__main__':
        lst = get_InitialXYZOptimizationProblem(driver=UniformGenerator(200)).get_DOE_list()

        # run in parallel
        par_runner = ParallelRunner()
        (cost, state, recorder), results = par_runner(lst, seq_runner_example)
        print(cost)

        # run sequential
        cost, state, recorder = seq_runner_example(lst)
        print(cost)


try_me()
