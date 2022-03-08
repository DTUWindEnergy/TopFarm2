import numpy as np
import multiprocessing
import threading
from topfarm.tests.test_files import xy3tb
from topfarm.easy_drivers import EasyScipyOptimizeDriver


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
                return tf.optimize(recorder_as_list=True)

        Returns
        -------
        best : (cost, state, recorder)
            best result
        results : [(cost1, state1, recorder1),...]
            all results
        """

        indexes = np.round(np.linspace(0, len(state_lst), self.pool._processes + 1)).astype(int)
        seq_lst = [state_lst[i1:i2] for i1, i2 in zip(indexes[:-1], indexes[1:])]

        results = self.pool.map(seq_runner, seq_lst)
        results = [r for r_seq in results for r in r_seq]
        best = results[np.argmin([r[0] for r in results])]
        return best, results


# ===============================================================================
# Example functions
# ===============================================================================
def get_topfarm_problem(id, plot=False):
    # setup topfarm problem with max 3 slsqp iterations
    tf = xy3tb.get_tf(plot=plot, driver=EasyScipyOptimizeDriver(maxiter=3, disp=False))
    # Shuffle position via smartstart, XX and YY is posible starting point coordinates
    tf.smart_start(XX=np.linspace(0, 6, 10), YY=np.linspace(-10, 0, 10), random_pct=100)
    return tf


def seq_runner_example(id_lst):
    print("%d cases executed by thread: %s" % (len(id_lst), threading.current_thread().ident))
    # optimize for all elements in lst
    return [get_topfarm_problem(id).optimize(recorder_as_list=True) for id in id_lst]
# ===============================================================================
#
# ===============================================================================


def main():
    if __name__ == '__main__':
        # make a list of 4 ids. Could as well be for sets of inputs
        id_lst = np.arange(4)

        print("Current thread: %s" % threading.get_ident())

        # optimize the 4 cases sequential
        print("\nRun on one processor")
        results = seq_runner_example(id_lst)
        for i, r in zip(id_lst, results):
            print("Id %d: %.2f" % (i, r[0]))

        # optimize the four cases in parallel
        print("\nRun on two processors")
        par_runner = ParallelRunner(2)
        (best_cost, best_state, best_recorder), results = par_runner(id_lst, seq_runner_example)
        for i, r in zip(id_lst, results):
            print("Id %d: %.2f" % (i, r[0]))
        print("Minimum cost: %.2f" % best_cost)


main()
