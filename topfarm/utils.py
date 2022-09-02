import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na
import multiprocessing
import threading
import time
from tqdm import tqdm
from openmdao.core.explicitcomponent import ExplicitComponent
import topfarm


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
    for i in tqdm(range(N_WT), desc='Smartstart'):
        if arr.shape[1] == 0:
            raise Exception('No feasible positions for wt %d' % i)

        if ZZ_is_func:
            if random_pct < 100:
                z = ZZ(arr[0], arr[1], xs, ys)
            else:
                z = np.zeros_like(arr[0])
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


def smooth_max(X, alpha, axis=0):
    '''
    Returns the smooth maximum of a matrix for positive values of alpha and smoth minimum for negative values of alpha
    Parameters
    ----------
    X : ndarray
        Matrix of which the smooth maximum is calculated.
    alpha : float
        smoothness parameter.
    axis : int, optional
        Axis along which the smooth maximum is calculated. The default is 0.

    Returns
    -------
    ndarray
        Matrix of smooth maximum values.

    '''
    return (X * np.exp(alpha * X)).sum(axis=axis) / np.exp(alpha * X).sum(axis=axis)


def smooth_max_gradient(X, alpha, axis=0):
    '''
    Parameters
    ----------
    X : ndarray
        Matrix of which the smooth maximum derivative is calculated.
    alpha : float
        smoothness parameter.
    axis : int, optional
        Axis along which the smooth maximum is calculated. The default is 0. The default is 0.

    Returns
    -------
    ndarray
        Matrix of smooth maximum derivatives.

    '''
    return np.exp(alpha * X) / np.expand_dims(np.exp(alpha * X).sum(axis=axis), axis) * \
        (1 + alpha * (X - np.expand_dims(smooth_max(X, alpha, axis=axis), axis)))


def regular_generic_layout(n_wt, sx, sy, stagger, rotation, ratio=1.0):
    '''
    Parameters
    ----------
    n_wt : int
        number of wind turbines
    sx : float
        spacing (in turbine diameters or meters) between turbines in x direction
    sy : float
        spacing (in turbine diameters or meters) between turbines in y direction
    stagger : float
        stagger (in turbine diameters or meters) distance every other turbine column
    rotation : float
        rotational angle of the grid in degrees
    ratio : float
        ratio between number of columns and number of rows (1.0)

    Returns
    -------
    xy : array
        2D array of x- and y-coordinates (in turbine diameters or meters)

     '''

    if ratio > 1:
        n_row = int(np.round((n_wt * ratio) ** 0.5))
        n_col = int(n_wt / n_row)
    else:
        n_col = int(np.round((n_wt * ratio) ** 0.5))
        n_row = int(n_wt / n_col)
    rest = n_wt - n_col * n_row
    theta = np.radians(float(rotation))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    x_grid = np.linspace(0, n_col * float(sx), n_col, endpoint=False)
    y_grid = np.linspace(0, n_row * float(sy), n_row, endpoint=False)
    xm, ym = np.meshgrid(x_grid, y_grid)
    ym[:, 1::2] += stagger
    x, y = xm.ravel(), ym.ravel()
    x = np.hstack((x, x_grid[:rest]))
    y = np.hstack((y, (y[-n_col:-(n_col - rest)] + float(sy))))
    return np.matmul(R, np.array([x, y]))


def regular_generic_layout_gradients(n_wt, sx, sy, stagger, rotation, ratio=1.0):
    '''
    Parameters
    ----------
    n_wt : int
        number of wind turbines
    sx : float
        spacing (in turbine diameters or meters) between turbines in x direction
    sy : float
        spacing (in turbine diameters or meters) between turbines in y direction
    stagger : float
        stagger (in turbine diameters or meters) distance every other turbine column
    rotation : float
        rotational angle of the grid in degrees
    ratio : float
        ratio between number of columns and number of rows (1.0)

    Returns
    -------
    dx_dsx, dy_dsx, dx_dsy, dy_dsy, dx_dr, dy_dr : tuple
        tuple of gradients of x and y with respect to x-spacing, y-spacing and grid rotation

     '''

    if ratio > 1:
        n_row = int(np.round((n_wt * ratio) ** 0.5))
        n_col = int(n_wt / n_row)
    else:
        n_col = int(np.round((n_wt * ratio) ** 0.5))
        n_row = int(n_wt / n_col)
    rest = n_wt - n_col * n_row
    theta = np.radians(float(rotation))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    x_grid = np.linspace(0, n_col, n_col, endpoint=False)
    y_grid = np.zeros(n_row)
    xm, ym = np.meshgrid(x_grid, y_grid)
    x, y = xm.ravel(), ym.ravel()
    x = np.hstack((x, x_grid[:rest]))
    y = np.hstack((y, (y[-n_col:-(n_col - rest)])))
    dxy_dsx = np.matmul(R, np.array([x, y]))
    dx_dsx = dxy_dsx[0, :]
    dy_dsx = dxy_dsx[1, :]

    x_grid = np.zeros(n_col)
    y_grid = np.linspace(0, n_row, n_row, endpoint=False)
    xm, ym = np.meshgrid(x_grid, y_grid)
    x, y = xm.ravel(), ym.ravel()
    x = np.hstack((x, x_grid[:rest]))
    y = np.hstack((y, (y[-n_col:-(n_col - rest)] + 1)))
    dxy_dsy = np.matmul(R, np.array([x, y]))
    dx_dsy = dxy_dsy[0, :]
    dy_dsy = dxy_dsy[1, :]

    x_grid = np.linspace(0, n_col * float(sx), n_col, endpoint=False)
    y_grid = np.linspace(0, n_row * float(sy), n_row, endpoint=False)
    xm, ym = np.meshgrid(x_grid, y_grid)
    ym[:, 1::2] += stagger
    x, y = xm.ravel(), ym.ravel()
    x = np.hstack((x, x_grid[:rest]))
    y = np.hstack((y, (y[-n_col:-(n_col - rest)] + float(sy))))
    dRdr = np.array([[-np.sin(theta), -np.cos(theta)],
                     [np.cos(theta), -np.sin(theta)]])
    dx_dr, dy_dr = np.matmul(dRdr, np.array([x, y])) * np.pi / 180

    return dx_dsx, dy_dsx, dx_dsy, dy_dsy, dx_dr, dy_dr


class RegularGridComponent(ExplicitComponent):
    def __init__(self, design_vars, n_wt, **additional_input):
        super().__init__()
        self.design_vars = design_vars
        self.n_wt = n_wt
        default_vals = {'n_wt': n_wt,
                        'sx': None,
                        'sy': None,
                        'stagger': 0,
                        'rotation': 0,
                        'ratio': 1.0}
        additional_vals = {k: v for k, v in additional_input.items() if k in default_vals}
        self.default_dict = {**default_vals, **additional_vals}

    def setup(self):
        for k, v in self.design_vars.items():
            if k in self.default_dict:
                self.add_input(k, v[0], units=v[-1])
                if k in ['sx', 'sy', 'rotation']:
                    self.declare_partials(topfarm.x_key, k, method='exact')
                    self.declare_partials(topfarm.y_key, k, method='exact')
                else:
                    self.declare_partials(topfarm.x_key, k, method='fd')
                    self.declare_partials(topfarm.y_key, k, method='fd')
        self.add_output(topfarm.x_key, val=np.zeros(self.n_wt))
        self.add_output(topfarm.y_key, val=np.zeros(self.n_wt))

    def compute(self, inputs, outputs):
        x, y = regular_generic_layout(**{**self.default_dict, **dict(inputs)})
        outputs[topfarm.x_key] = x
        outputs[topfarm.y_key] = y

    def compute_partials(self, inputs, J):
        dx_dsx, dy_dsx, dx_dsy, dy_dsy, dx_dr, dy_dr = regular_generic_layout_gradients(**{**self.default_dict, **dict(inputs)})
        J[topfarm.x_key, 'sx'] = dx_dsx
        J[topfarm.x_key, 'sy'] = dx_dsy
        J[topfarm.y_key, 'sx'] = dy_dsx
        J[topfarm.y_key, 'sy'] = dy_dsy
        J[topfarm.x_key, 'rotation'] = dx_dr
        J[topfarm.y_key, 'rotation'] = dy_dr


def main():
    if __name__ == '__main__':
        plt.close('all')
        N_WT = 30
        min_space = 2.1

        x = np.arange(0, 20, 0.1)
        y = np.arange(0, 10, 0.1)
        YY, XX = np.meshgrid(y, x)
        val = np.sin(XX) + np.sin(YY)
        min_space = 2.1
        xs, ys = smart_start(XX, YY, val, N_WT, min_space)
        plt.figure()
        c = plt.contourf(XX, YY, val, 100)
        plt.colorbar(c)
        for i in range(N_WT):
            circle = plt.Circle((xs[i], ys[i]), min_space / 2, color='b', fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.plot(xs[i], ys[i], 'rx')
        plt.axis('equal')
        plt.show()

        plt.figure()
        xy = regular_generic_layout(n_wt=100, sx=5, sy=4, stagger=2, rotation=35, ratio=1.25)

        dsx = 0.1
        dxyx = regular_generic_layout(n_wt=100, sx=5 + dsx, sy=4, stagger=2, rotation=35, ratio=1.25)
        dxy_dsx = (dxyx - xy) / dsx
        dx_dsx, dy_dsx = dxy_dsx

        dsy = 0.1
        dxyy = regular_generic_layout(n_wt=100, sx=5, sy=4 + dsy, stagger=2, rotation=35, ratio=1.25)
        dxy_dsy = (dxyy - xy) / dsy
        dx_dsy, dy_dsy = dxy_dsy

        dR = 0.1
        dxyR = regular_generic_layout(n_wt=100, sx=5, sy=4, stagger=2, rotation=35 + dR, ratio=1.25)
        dxy_dR = (dxyR - xy) / dR
        dx_dR, dy_dR = dxy_dR

        x, y = xy
        plt.plot(x, y, '.')
        plt.axis('equal')

        dx_dsxg, dy_dsxg, dx_dsyg, dy_dsyg, dx_dRg, dy_dRg = regular_generic_layout_gradients(n_wt=100, sx=5, sy=4, stagger=2, rotation=35, ratio=1.25)
        plt.figure()
        plt.plot(dx_dsx.ravel(), dx_dsxg.ravel(), '.')
        plt.figure()
        plt.plot(dy_dsx.ravel(), dy_dsxg.ravel(), '.')
        plt.figure()
        plt.plot(dx_dsy.ravel(), dx_dsyg.ravel(), '.')
        plt.figure()
        plt.plot(dy_dsy.ravel(), dy_dsyg.ravel(), '.')

        plt.figure()
        plt.plot(dx_dR.ravel(), dx_dRg.ravel(), '.')
        plt.figure()
        plt.plot(dy_dR.ravel(), dy_dRg.ravel(), '.')


main()
