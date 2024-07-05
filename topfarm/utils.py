import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na
import multiprocessing
import threading
import time
from tqdm import tqdm
# from openmdao.core.explicitcomponent import ExplicitComponent
# import topfarm
from abc import abstractmethod, ABC
import gc
from scipy.special import gamma
from scipy.optimize import fsolve
import pandas as pd


def smart_start(XX, YY, ZZ, N_WT, min_space, radius=None, random_pct=0, plot=False, seed=None, types=None):
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
    types : array_like of integers or None
        list of turbine type numbers e.g. types = [0, 1, 2, 3] for 4 different types. ZZ, min_space and optionally
        also radius should also have type dimension. if ZZ is callable it should be callable with type argument.

    Returns
    -------
    Positions where the aep or wsp is highest, while
    respecting the minimum spacing requirement.

    Notes
    -----
    XX, YY and ZZ can be 1D or 2D, but must have same size. If multiple turbine types
    ZZ must have the shape (n_types, shape_like_XX)
    """
    assert 0 <= random_pct <= 100
    ZZ_is_func = hasattr(ZZ, '__call__')
    if types:
        n_types = len(types)
        n_points = XX.size
        arr = np.zeros((3, n_points, n_types))
        arr[0, :, :] = XX.ravel()[:, np.newaxis]
        arr[1, :, :] = YY.ravel()[:, np.newaxis]
        if ZZ_is_func:
            arr[2, :, :] = np.zeros_like(arr[0])
        else:
            arr[2, :, :] = ZZ.reshape(n_types, -1).T
    else:
        if ZZ_is_func:
            arr = np.array([XX.flatten(), YY.flatten()])
        else:
            arr = np.array([XX.flatten(), YY.flatten(), ZZ.flatten()])

    # set radius to None(faster) if if grid resolution > radius
    if radius is not None and (np.diff(np.sort(np.unique(arr[0]))).min() > radius and
                               (np.diff(np.sort(np.unique(arr[1]))).min() > radius)):
        radius = None
    xs, ys = [], []
    if types:
        type_i = []
    if seed is None:
        seed = np.uint32(int((time.time() - int(time.time())) * 1e8 +
                             multiprocessing.current_process().ident + threading.get_ident()) % (2**31))
        np.random.seed(seed)
        seed = np.random.randint(0, 2**31)
    np.random.seed(seed)
    mask = np.full(arr.shape[1:], True)
    for i in tqdm(range(N_WT), desc='Smartstart'):
        idx = np.where(mask < 1)
        if arr.shape[1] == 0:
            raise Exception('No feasible positions for wt %d' % i)

        if ZZ_is_func:
            if random_pct < 100:
                if types:
                    z = np.asarray([ZZ(arr[0, :, tt], arr[1, :, tt], xs, ys, tt, type_i) for tt in types]).T
                    z[idx] = -np.inf
                    arr[2] = z
                else:
                    z = ZZ(arr[0], arr[1], xs, ys)
            else:
                z = np.zeros_like(arr[0])
                if types:
                    arr[2] = z
        else:
            z = arr[2]
            if types:
                z[idx] = -np.inf

        if radius is not None:
            # average over the rotor area, i.e. all points within one radius from the point
            x, y = arr
            z = np.array([np.mean(z[ind]) for ind in np.hypot((x - x[:, na]), (y - y[:, na])) < radius])

        # pick one of the <random_pct> best points
        n_random = np.maximum(1, int(np.round(random_pct / 100 * z.size)))
        min_z = np.sort(z.ravel())[-(n_random)]
        if types:
            choises = np.argwhere(z >= min_z)
            arg_choise = np.random.choice(np.arange(choises.shape[0]))
            next_ind, t = choises[arg_choise]
            next_ind = (next_ind, 0)
            type_i.append(t)
        else:
            next_ind = np.random.choice(np.argwhere(z >= min_z)[:, 0])

        x0 = arr[0][next_ind]
        y0 = arr[1][next_ind]
        xs.append(x0)
        ys.append(y0)

        if plot:
            # if types:
            #     for typ in types:
            #         plt.figure()
            #         c = plt.scatter(arr[0, :, typ], arr[1, :, typ], c=arr[2, :, typ])
            #         plt.colorbar(c)
            #         plt.plot(xs, ys, '2k', ms=10)
            #         plt.plot(xs[-1], ys[-1], '2r', ms=10)
            #         plt.axis('equal')
            #         plt.show()
            # else:
            plt.figure()
            c = plt.scatter(arr[0], arr[1], c=z)
            plt.colorbar(c)
            plt.plot(xs, ys, '2k', ms=10)
            plt.plot(xs[-1], ys[-1], '2r', ms=10)
            plt.axis('equal')
            plt.show()

        # Remove all point within min_space from the newly added wt
        if types:
            eff_min_space = (min_space + min_space[t]) / 2
            mask = np.logical_and(mask, ((arr[0, :, 0] - x0) ** 2 + (arr[1, :, 0] - y0) ** 2)[:, na] >= eff_min_space[na, :] ** 2)
            idx = mask.sum((1)) > 0
            arr = arr[:, idx, :]
            mask = mask[idx, :]
        else:
            index = np.where((arr[0] - x0)**2 + (arr[1] - y0)**2 >= min_space**2)[0]
            arr = arr[:, index]
        gc.collect()

    print(
        f"{len(XX.flatten())} possible points, {N_WT} wt, {len(XX)/N_WT:.1f} points pr wt, {arr.shape[1]}({arr.shape[1]/len(XX.flatten())*100:.0f}%) unused points")
    if types:
        return xs, ys, type_i
    else:
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


def gauss(X):
    return np.exp(-X ** 2)


def gauss_inv(X):
    return np.sqrt(-np.log(X))


def gauss_prime(X):
    return -2 * X * gauss(X)


def gauss_inv_prime(X):
    return -1 / (2 * X * gauss_inv(X))


def smooth_zero(X, alpha=1, axis=0):
    '''
    Function that from an array approximates the value cloest to zero. A differentiable alternative
    to min(abs(X)).
    Parameters
    ----------
    X : ndarray
        Matrix of which the smooth zero is calculated.
    alpha : float
        smoothness parameter.
    axis : int, optional
        Axis along which the smooth zero is calculated. The default is 0.

    Returns
    -------
    ndarray
        Matrix of smooth zero.

    '''
    sign = np.sign(X[np.expand_dims(np.argmin(np.abs(X)), axis)])
    return sign * gauss_inv(smooth_max(gauss(X), alpha, axis))


def smooth_zero_gradient(X, alpha=1, axis=0):
    '''
    Gradients of the smooth_zero function.
    Parameters
    ----------
    X : ndarray
        Matrix of which the smooth zero derivative is calculated.
    alpha : float
        smoothness parameter.
    axis : int, optional
        Axis along which the smooth zero is calculated. The default is 0.

    Returns
    -------
    ndarray
        Matrix of smooth zero derivatives.

    '''
    sign = np.sign(X[np.expand_dims(np.argmin(np.abs(X)), axis)])
    return sign * gauss_prime(X) * smooth_max_gradient(gauss(X), alpha, axis) * gauss_inv_prime(smooth_max(gauss(X), alpha, axis))


class AggregationFunction(ABC):
    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, x, axis=-1):
        """compute function value over axis"""

    @abstractmethod
    def gradient(self, x, axis=-1):
        """compute gradients of aggregated value wrt x over axis"""


class StrictMax(AggregationFunction):
    """Normal max with discontinous gradient"""

    def __call__(self, x, axis=-1):
        return np.max(x, axis)

    def gradient(self, x, axis=-1):
        return x == self(np.atleast_2d(x), axis)


class StrictMin(StrictMax):
    """Normal min with discontinous gradient"""

    def __call__(self, x, axis=-1):
        return np.min(x, axis)


def SoftMax(x, alpha, axis=-1):
    """gradient of LogSumExpMax

    https://en.wikipedia.org/wiki/Softmax_function"""
    if alpha > 0:
        x_max = np.max(np.atleast_2d(x), axis)
    else:
        x_max = np.min(np.atleast_2d(x), axis)
    xn = x - np.expand_dims(x_max, axis)
    return np.exp(alpha * xn) / np.sum(np.exp(alpha * xn), axis)


class SmoothMax(AggregationFunction):
    def __init__(self, base=1):
        """
        https://en.wikipedia.org/wiki/Smooth_maximum

        Underpredict the maximum of similar values

        Parameters
        ----------
        base : float
            smoothing factor ]0;inf[. higher number gives more smooth transition
            For two numbers, (a,b), the smoothing approximately starts when |a-b| < 4*base
        """
        self.alpha = 1 / base

    def __str__(self):
        return f"{self.__class__.__name__}({1/self.alpha})"

    def __call__(self, x, axis=-1):
        if self.alpha > 0:
            x_max = np.max(np.atleast_2d(x), axis)
        else:
            x_max = np.min(np.atleast_2d(x), axis)
        xn = x - np.expand_dims(x_max, axis)
        return np.sum(x * np.exp(self.alpha * xn), axis) / np.sum(np.exp(self.alpha * xn), axis)

    def gradient(self, x, axis=-1):
        x = np.asarray(x)
        a = self.alpha
        if a > 0:
            x_max = np.max(np.atleast_2d(x), axis)
        else:
            x_max = np.min(np.atleast_2d(x), axis)
        xn = x - np.expand_dims(x_max, axis)
        return (np.exp(a * xn) /
                np.expand_dims(np.sum(np.exp(a * xn), axis), axis) * (1 + a * (x - np.expand_dims(self(x, axis), axis))))


class SmoothMin(SmoothMax):
    def __init__(self, base=1):
        SmoothMax.__init__(self, base=-base)

    def __str__(self):
        return f"{self.__class__.__name__}({-1/self.alpha})"


class LogSumExpMax(SmoothMax):
    """LogSumExp

    Overpredict the maximum of similar values

    Parameters
    ----------
    base : float
        smoothing factor ]0;inf[. higher number gives more smooth transition
        For two numbers, (a,b), the smoothing approximately starts when |a-b| < 4*base
    """

    def __str__(self):
        return f"{self.__class__.__name__}({1/self.alpha})"

    def __call__(self, x, axis=-1):
        # factor used to reduce numerical errors in power
        if self.alpha > 0:
            x_max = np.max(np.atleast_2d(x), axis)
        else:
            x_max = np.min(np.atleast_2d(x), axis)
        xn = x - np.expand_dims(x_max, axis)
        alpha = self.alpha
        return x_max + 1 / alpha * np.log(np.sum(np.exp(alpha * (xn)), axis))

    def gradient(self, x, axis=-1):
        return SoftMax(x, self.alpha, axis)


class LogSumExpMin(LogSumExpMax):
    def __init__(self, base=1):
        LogSumExpMax.__init__(self, base=-base)

    def __str__(self):
        return f"{self.__class__.__name__}({-1/self.alpha})"


def regular_generic_layout(n_wt, sx, sy, stagger, rotation, x0=0, y0=0, ratio=1.0):
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
    xy_grid = np.matmul(R, np.array([x, y]))
    return xy_grid + np.asarray([x0, y0])[:, na]


def regular_generic_layout_gradients(n_wt, sx, sy, stagger, rotation, x0=0, y0=0, ratio=1.0):
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

    return [dx_dsx, dy_dsx, dx_dsy, dy_dsy, dx_dr, dy_dr]


def downsample_ts(ws, wd, timestamps, start=None, end=None, freq='D'):
    if start is None:
        start = timestamps[0]
    if end is None:
        end = timestamps[-1]
    df = pd.DataFrame({'WS': ws, 'WD': wd}, index=timestamps)
    df['WD_R'] = 180 - df.WD
    df['WD_a'] = np.cos(np.radians(df.WD_R))
    df['WD_b'] = np.sin(np.radians(df.WD_R))
    bins = pd.date_range(start=start, end=end, freq=freq)
    df2 = df.groupby(pd.cut(df.index, bins=bins, labels=bins.astype(str).values[:-1]), observed=False).mean()
    df2['WD'] = 180 - np.degrees(np.arctan2(df2.WD_b, df2.WD_a))
    df2 = df2[['WS', 'WD']]
    df2.index = pd.to_datetime(df2.index)
    return df2


def fit_sectorwise_weib(ws, wd, n_sectors=12):
    sector_width = 360 / n_sectors
    bins = np.linspace(0, 360, n_sectors + 1) - sector_width / 2
    wd[wd > 360 - sector_width / 2] = wd[wd > 360 - sector_width / 2] - 360
    df = pd.DataFrame({'WS': ws, 'WD': wd})
    df['sector'] = pd.cut(df.WD, bins=bins, labels=np.arange(n_sectors))
    As, ks, ps = [], [], []
    for sector in range(n_sectors):
        A, k = fit_weib(df.WS[df.sector == sector])
        p = df.WD[df.sector == sector].size / df.WD.size
        As.append(A)
        ks.append(k)
        ps.append(p)
    return As, ks, ps


def fit_weib(u):
    u_bar = np.mean(u)
    u3_bar = np.mean(u**3)

    def moments(p):
        A, k = p
        u_est = A * gamma(1 / k + 1)
        u3_est = A**3 * gamma(3 / k + 1)
        return (u_est - u_bar, u3_est - u3_bar)

    A, k = fsolve(moments, (u_bar, 2.0))
    return A, k


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
        xy = regular_generic_layout(n_wt=100, sx=5, sy=4, stagger=2, rotation=35, ratio=1.25, x0=20, y0=40)

        dsx = 0.1
        dxyx = regular_generic_layout(n_wt=100, sx=5 + dsx, sy=4, stagger=2, rotation=35, ratio=1.25, x0=20, y0=40)
        dxy_dsx = (dxyx - xy) / dsx
        dx_dsx, dy_dsx = dxy_dsx

        dsy = 0.1
        dxyy = regular_generic_layout(n_wt=100, sx=5, sy=4 + dsy, stagger=2, rotation=35, ratio=1.25, x0=20, y0=40)
        dxy_dsy = (dxyy - xy) / dsy
        dx_dsy, dy_dsy = dxy_dsy

        dR = 0.1
        dxyR = regular_generic_layout(n_wt=100, sx=5, sy=4, stagger=2, rotation=35 + dR, ratio=1.25, x0=20, y0=40)
        dxy_dR = (dxyR - xy) / dR
        dx_dR, dy_dR = dxy_dR

        x, y = xy
        plt.plot(x, y, '.')
        plt.axis('equal')

        dx_dsxg, dy_dsxg, dx_dsyg, dy_dsyg, dx_dRg, dy_dRg = regular_generic_layout_gradients(
            n_wt=100, sx=5, sy=4, stagger=2, rotation=35, ratio=1.25, x0=20, y0=40)
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
