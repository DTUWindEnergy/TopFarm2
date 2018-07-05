import os
import numpy as np
import multiprocessing


def run(N, single_starter, processes=None):
    """Run N single starts in parallel

    Parameters
    ----------
    N : int
        Number of single starts
    single_starter : function
        function for single start. Interface must be:
        def f(id):
            tf = Topfarm(...)
            return tf.optimize
    processes : int or None, optional
        Number of processes passed to multiprocessing.Pool

    Returns
    -------
    best : (cost, turbine_positions)
        best result
    results : [(cost1, turbine_positions1),...]
        all results
    """
    pool = multiprocessing.Pool(processes)
    results = pool.map(single_starter, range(N))
    best = results[np.argmin([r[0] for r in results])]
    return best, results


def single_start_example(id, maxiter=5, plot_comp=None):
    from topfarm._topfarm import TopFarm
    from topfarm.cost_models.fused_wake_wrappers import FusedWakeNOJWakeModel
    from topfarm.cost_models.utils.aep_calculator import AEPCalculator
    from topfarm.cost_models.utils.wind_resource import WindResource
    from topfarm.easy_drivers import EasyScipyOptimizeDriver
    from topfarm.tests.test_files import tfp

    D = 80.0
    D2 = 2 * D
    init_pos = np.array([(0, D2), (0, 0), (0, -D2)])
    boundary = [(-D2, D2), (D2, D2), (D2, -D2), (-D2, -D2)]
    minSpacing = 2.0
    f, A, k = [1, 0, 0, 0], [10, 10, 10, 10], [2, 2, 2, 2]
    wr = WindResource(f, A, k, ti=np.zeros_like(f) + .1)
    wm = FusedWakeNOJWakeModel(tfp + "wind_farms/3tb.yml")
    aep_calc = AEPCalculator(wr, wm)
    driver = EasyScipyOptimizeDriver(maxiter=maxiter, disp=False)
    tf = TopFarm(init_pos, aep_calc.get_TopFarm_cost_component(), minSpacing * D,
                 boundary=boundary, plot_comp=plot_comp, driver=driver)
    tf.shuffle_positions('abs')
    return tf.optimize()


def try_me():
    if __name__ == '__main__':
        print(run(4, single_start_example)[0], 2)
        print(single_start_example(0, 10))


try_me()
