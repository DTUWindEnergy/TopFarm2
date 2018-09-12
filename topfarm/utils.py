from topfarm.constraint_components.boundary_component import \
    PolygonBoundaryComp
from topfarm.constraint_components.spacing_component import SpacingComp
import os
import numpy as np
from openmdao.api import CaseReader


def pos_from_case(case_recorder_filename):
    '''
    Input: a recorded optimization case
    Returns the positions at the last step of that optimization
    '''
    if not os.path.exists(case_recorder_filename):
        string = 'The specified optimization recording does not exists: '
        string += case_recorder_filename
        raise Warning(string)
    cr = CaseReader(case_recorder_filename)
    driver_case = cr.driver_cases.get_case(-1)
    desvars = driver_case.get_desvars()
    x = np.array(desvars['turbineX'])
    y = np.array(desvars['turbineY'])
    turbines = np.column_stack((x, y))
    return turbines


def latest_id(case_recorder_dir):
    '''
    Input: path to the directory of recorded optimizations
    Returns the absolute path to the most recent recording in that directory
    '''
    files = os.listdir(case_recorder_dir)
    files = [x for x in files if x.startswith('cases_') and x.endswith('.sql')]
    if len(files) == 0:
        string = 'No recorded files found in the specified directory: '
        string += case_recorder_dir + '\n' + 9 * ' '
        string += 'Start a new optimization or specify another directory '
        string += 'for resumed optimization'
        raise Warning(string)
    latest = max(files)
    latest = os.path.join(case_recorder_dir, latest)
    return latest


def shuffle_positions(boundary, n_wt, min_space, init_pos, shuffle_type='abs',
                      n_iter=1000, step_size=0.1, pad=1.1, offset=0.5,
                      plot=True, verbose=True):
    '''
    Input:
        boundary:   list of tuples, e.g.: [(0, 0), (6, 1), (7, -11), (-1, -10)]
        n_wt:       number of wind turbines
        min_space:  the minimum spacing between turbines
        init_pos:   inital positions that suffeling is based off of if the
                    shuffle_type requires it
        shuffle_type:
                    'abs': absolute random positions that respect the boundary,
                    and aims to respect the minimum spacing
                    'rel': moves each turbine with an offset compared to the
                    initial positions - not implemented yet.
        n_iter:     number of iterations allowed to try and satisfy the minimum
                    spacing constraint
        step_size:  the multiplier on the spacing gradient that the turbines
                    are moved in each step
        pad:        the multiplier on the boundary gradient
        plot:       plot the generated random layout
        verbose:    print helpful text to the console
    Returns an array of xy coordinates of the wind turbines
    '''
    if shuffle_type == 'abs':
        def _random(b):
            return np.random.rand(n_wt) * (max(b) - min(b)) + min(b)
        x, y = map(_random, np.array(boundary).T)
        turbines = _shuffle_positions_abs(x, y, boundary, n_wt, n_iter,
                                          step_size, min_space, pad, plot,
                                          verbose)
    elif shuffle_type == 'rel':
        turbines = _shuffle_postions_rel(init_pos, offset, boundary, n_wt,
                                         plot)
    return turbines


def _shuffle_postions_rel(init_pos, offset, boundary, n_wt, plot):
    turbineX = init_pos[:, 0]
    turbineY = init_pos[:, 1]
    ba = np.array(boundary).T
    turbines = np.array(init_pos) + np.random.rand(n_wt, 2) * 2 * offset - offset
    if plot:
        plt.figure()
        plt.cla()
        plt.plot(ba[0], ba[1])
        plt.plot(turbineX, turbineY, '.')
        plt.plot(turbines.T[0], turbines.T[1], 'o')
    return turbines


def _shuffle_positions_abs(turbineX, turbineY, boundary, n_wt, n_iter,
                           step_size, min_space, pad, plot, verbose):
    boundary_comp = PolygonBoundaryComp(boundary, n_wt)
    spacing_comp = SpacingComp(n_wt=n_wt)
    min_space2 = min_space**2
    ba = np.array(boundary).T
    if plot:
        plt.figure()
        plt.cla()
        plt.plot(ba[0], ba[1])
        plt.plot(turbineX, turbineY, '.')
    for j in range(n_iter):
        dist = spacing_comp._compute(turbineX, turbineY)
        dx, dy = spacing_comp._compute_partials(turbineX, turbineY)
        index = np.argmin(dist)
        if dist[index] < min_space2 or j == 0:
            turbineX += dx[index] * step_size
            turbineY += dy[index] * step_size
            turbineX, turbineY = _move_inside_boundary(n_wt, turbineX,
                                                       turbineY, boundary_comp,
                                                       pad)
        else:
            if verbose:
                print('Obtained required spacing after {} iterations'.format(
                    j))
            break
    if plot:
        plt.plot(turbineX, turbineY, 'o')
    dist = spacing_comp._compute(turbineX, turbineY)
    index = np.argmin(dist)
    if verbose:
        print('Spacing obtained: {}'.format(dist[index]**0.5))
        print('Spacing required: {}'.format(min_space))
    turbines = np.array([turbineX, turbineY]).T
    return turbines


def _move_inside_boundary(n_wt, turbineX, turbineY, boundary_comp, pad):
    for i in range(0, n_wt):
        dng = boundary_comp.calc_distance_and_gradients(turbineX, turbineY)
        dist = dng[0][i]
        if dist < 0:
            dx = dng[1][i]
            dy = dng[2][i]
            turbineX[i] -= dx * dist * pad
            turbineY[i] -= dy * dist * pad
    return turbineX, turbineY


def smart_start(x, y, val, N_WT, min_space):
    '''
    Selects the a number of gridpoints (N_WT) in the grid defined by x and y,
    where val has the maximum value, while chosen points spacing (min_space)
    is respected.
    Input:
        x and y: arrays (nD) with coordinates
        val: array (nD), the corresponding value of the desired variable in the
        grid points. This could be e.g. the AEP or wind speed.
        N_WT: integer, number of wind turbines
        min_space: float, minimum space between turbines
    Output:
        Positions where the aep or wsp is highest, while
        respecting the minimum spacing requirement.
    '''
    arr = np.array([x.flatten(), y.flatten(), val.flatten()])
    xs, ys = [], []
    for i in range(N_WT):
        try:
            max_ind = np.argmax(arr[2])
            x0 = arr[0][max_ind]
            y0 = arr[1][max_ind]
            xs.append(x0)
            ys.append(y0)
            index = np.where((arr[0] - x0)**2 + (arr[1] - y0)**2 >= min_space**2)[0]
            arr = arr[:, index]
        except ValueError:
            xs.append(np.nan)
            ys.append(np.nan)
            print('Could not respect the spacing constraint')
    return xs, ys


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.clf()
    boundary = [(0, 0), (6, 1), (7, -11), (-1, -10), (0, 0)]
    n_wt = 20
    init_pos = np.column_stack((np.random.randint(0, 6, (n_wt)),
                                np.random.randint(-10, 0, (n_wt))))
    min_space = 2.1
    turbines = shuffle_positions(boundary, n_wt, min_space, init_pos,
                                 shuffle_type='rel')
    print(turbines)

    x = np.arange(0, 20, 0.1)
    y = np.arange(0, 10, 0.1)
    YY, XX = np.meshgrid(y, x)
    val = np.sin(XX) + np.sin(YY)
    N_WT = 30
    min_space = 2.1
    xs, ys = smart_start(XX, YY, val, N_WT, min_space)
    plt.figure(1)
    c = plt.contourf(XX, YY, val, 100)
    plt.colorbar(c)
    for i in range(N_WT):
        circle = plt.Circle((xs[i], ys[i]), min_space / 2, color='b', fill=False)
        plt.gcf().gca().add_artist(circle)
        plt.plot(xs[i], ys[i], 'rx')
    plt.axis('equal')
    plt.show()
