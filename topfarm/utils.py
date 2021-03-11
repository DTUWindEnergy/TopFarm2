import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na
import multiprocessing
import threading
import time


def smart_start(XX, YY, ZZ, N_WT, min_space, radius=None, random_pct=0, plot=False, seed=None):
    """Selects the a number of gridpoints (N_WT) in the grid defined by x and y,
    where ZZ has the maximum value, while chosen points spacing (min_space)
    is respected.

    Parameters
    ----------
    XX : array_like
        x coordinates
    YY : array_like
        y coordinates
    ZZ : array_like
        Values at (XX, YY) of the desired variable in the
        grid points. This could be e.g. the AEP or wind speed.
    N_WT : integer
        number of wind turbines
    min_space: float
        minimum space between turbines
    random_pct : float
        select by random position of the <random_pct> best points
    plot : boolean
        if True, each step is plotted in new figure

    Returns
    -------
    Positions where the aep or wsp is highest, while
    respecting the minimum spacing requirement.

    Notes
    -----
    XX, YY and ZZ can be 1D or 2D, but must have same size
    """
    assert 0 <= random_pct <= 100
    ZZ_is_func = hasattr(ZZ, '__call__')
    if ZZ_is_func:
        arr = np.array([XX.flatten(), YY.flatten()])
    else:
        arr = np.array([XX.flatten(), YY.flatten(), ZZ.flatten()])

    # set radius to None(faster) if if grid resolution > radius
    if radius is not None and (np.diff(np.sort(np.unique(arr[0]))).min() > radius and
                               (np.diff(np.sort(np.unique(arr[1]))).min() > radius)):
        radius = None
    xs, ys = [], []
    if seed is None:
        seed = np.uint32(int((time.time() - int(time.time())) * 1e8 +
                             multiprocessing.current_process().ident + threading.get_ident()) % (2**31))
        np.random.seed(seed)
        seed = np.random.randint(0, 2**31)
    np.random.seed(seed)
    for i in range(N_WT):
        if arr.shape[1] == 0:
            raise Exception('No feasible positions for wt %d' % i)

        if ZZ_is_func:
            z = ZZ(arr[0], arr[1], xs, ys)
        else:
            z = arr[2]

        if radius is not None:
            # average over the rotor area, i.e. all points within one radius from the point
            x, y = arr
            z = np.array([np.mean(z[ind]) for ind in np.hypot((x - x[:, na]), (y - y[:, na])) < radius])

        if random_pct == 0:
            # pick one of the optimal points
            next_ind = np.random.choice(np.where(z == z.max())[0])
        else:
            # pick one of the <random_pct> best points
            n_random = int(np.round(random_pct / 100 * len(z)))
            next_ind = np.random.choice(np.argsort(z)[-(n_random):])

        x0 = arr[0][next_ind]
        y0 = arr[1][next_ind]
        xs.append(x0)
        ys.append(y0)

        if plot:
            plt.figure()
            c = plt.scatter(arr[0], arr[1], c=z)
            plt.colorbar(c)
            plt.plot(xs, ys, '2k', ms=10)
            plt.plot(xs[-1], ys[-1], '2r', ms=10)
            plt.axis('equal')
            plt.show()

        # Remove all point within min_space from the newly added wt
        index = np.where((arr[0] - x0)**2 + (arr[1] - y0)**2 >= min_space**2)[0]
        arr = arr[:, index]

    return xs, ys


def main():
    if __name__ == '__main__':
        N_WT = 30
        min_space = 2.1

        x = np.arange(0, 20, 0.1)
        y = np.arange(0, 10, 0.1)
        YY, XX = np.meshgrid(y, x)
        val = np.sin(XX) + np.sin(YY)
        min_space = 2.1
        xs, ys = smart_start(XX, YY, val, N_WT, min_space)
        c = plt.contourf(XX, YY, val, 100)
        plt.colorbar(c)
        for i in range(N_WT):
            circle = plt.Circle((xs[i], ys[i]), min_space / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(xs[i], ys[i], 'rx')
        plt.axis('equal')
        plt.show()


main()
