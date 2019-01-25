import numpy as np


def smart_start(XX, YY, ZZ, N_WT, min_space, radius=None):
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

    Returns
    -------
    Positions where the aep or wsp is highest, while
    respecting the minimum spacing requirement.

    Notes
    -----
    XX, YY and ZZ can be 1D or 2D, but must have same size
    """
    ZZ_is_func = hasattr(ZZ, '__call__')
    if ZZ_is_func:
        arr = np.array([XX.flatten(), YY.flatten()])
    else:
        arr = np.array([XX.flatten(), YY.flatten(), ZZ.flatten()])
    if radius is not None and (np.diff(np.sort(np.unique(arr[0]))).min() > radius and
                               (np.diff(np.sort(np.unique(arr[1]))).min() > radius)):
        radius = None

    xs, ys = [], []
    for _ in range(N_WT):
        try:
            if ZZ_is_func:
                z = ZZ(arr[0], arr[1], xs, ys)
                ind = np.where(z == z.max())[0]
                if radius is None or len(ind) == len(z):
                    max_ind = ind[np.random.randint(len(ind))]
                else:
                    x, y = arr
                    z_rotor = np.array([np.mean(z[(x[i] - x)**2 + (y[i] - y)**2 < radius**2]) for i in ind])
                    ind = ind[np.where(z_rotor == z_rotor.max())[0]]
                    max_ind = ind[np.random.randint(len(ind))]
            else:
                z = arr[2]
                ind = np.where(z == z.max())[0]
                max_ind = ind[np.random.randint(len(ind))]

            x0 = arr[0][max_ind]
            y0 = arr[1][max_ind]
            xs.append(x0)
            ys.append(y0)
            index = np.where((arr[0] - x0)**2 + (arr[1] - y0)**2 >= min_space**2)[0]
            arr = arr[:, index]
            import os

        except ValueError:
            xs.append(np.nan)
            ys.append(np.nan)
            print('Could not respect the spacing constraint')
    return xs, ys


def main():
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
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
