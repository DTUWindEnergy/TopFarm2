from topfarm.constraint_components.boundary_component import \
    PolygonBoundaryComp
from topfarm.constraint_components.spacing_component import SpacingComp
import os
from copy import copy
import numpy as np
from openmdao.api import CaseReader
import matplotlib.pyplot as plt


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
    case_list = cr.driver_cases.list_cases()
    case_len = len(case_list)
    case_arg = 'rank0:SLSQP|{:d}'.format(case_len-1)
    case = cr.driver_cases.get_case(case_arg)
    x = np.array(case.desvars['turbineX'])
    y = np.array(case.desvars['turbineY'])
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
        string += case_recorder_dir + '\n' + 9*' '
        string += 'Start a new optimization or specify another directory '
        string += 'for resumed optimization'
        raise Warning(string)
    latest = max(files)
    latest = os.path.join(case_recorder_dir, latest)
    return latest


def random_positions(boundary, n_wt, n_iter, step_size, min_space,
                     pad=1.1, plot=False, verbose=True):
    '''
    Input:
        boundary:   list of tuples, e.g.: [(0, 0), (6, 1), (7, -11), (-1, -10)]
        n_wt:       number of wind turbines
        n_iter:     number of iterations allowed to try and satisfy the minimum
                    spacing constraint
        step_size:  the multiplier on the spacing gradient that the turbines
                    are moved in each step
        min_space:  the minimum spacing between turbines
        pad:        the multiplier on the boundary gradient
        plot:       plot the generated random layout
        verbose:    print helpful text to the console
    Returns an array of xy coordinates of the wind turbines
    '''
    x, y = map(_random, np.array(boundary).T)
    turbines = _random_positions(x, y, boundary, n_wt, n_iter, step_size,
                                 min_space, pad, plot, verbose)
    return turbines


def _random_positions(x, y, boundary, n_wt, n_iter, step_size, min_space,
                      pad, plot, verbose):
    boundary_comp = PolygonBoundaryComp(boundary, n_wt)
    spacing_comp = SpacingComp(nTurbines=n_wt)
    turbineX = copy(x)
    turbineY = copy(y)
    min_space2 = min_space**2
    ba = np.array(boundary).T
    bx = np.append(ba[0], ba[0][0])
    by = np.append(ba[1], ba[1][0])
    if plot:
        plt.cla()
        plt.plot(bx, by)
        plt.plot(x, y, '.')
    for j in range(n_iter):
        dist = spacing_comp._compute(turbineX, turbineY)
        dx, dy = spacing_comp._compute_partials(turbineX, turbineY)
        index = np.argmin(dist)
        if dist[index] < min_space2:
            turbineX += dx[index]*step_size
            turbineY += dy[index]*step_size
            turbineX, turbineY = _contain(n_wt, turbineX, turbineY,
                                          boundary_comp, pad)
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


def _random(b):
    return np.random.rand(n_wt) * (max(b) - min(b)) + min(b)


def _contain(n_wt, turbineX, turbineY, boundary_comp, pad):
    for i in range(0, n_wt):
        dng = boundary_comp.calc_distance_and_gradients(turbineX, turbineY)
        dist = dng[0][i]
        if dist < 0:
            dx = dng[1][i]
            dy = dng[2][i]
            turbineX[i] -= dx*dist*pad
            turbineY[i] -= dy*dist*pad
    return turbineX, turbineY


if __name__ == '__main__':
    this_dir = os.getcwd()
    crf = r"tests\test_files\recordings\cases_20180621_111710.sql"
    case_recorder_filename = crf
    path = os.path.join(this_dir, crf)
    turbines = pos_from_case(path)
    print(turbines)

    case_recorder_dir = r'tests\test_files\recordings'
    latest_id = latest_id(case_recorder_dir)
    print(latest_id)

    boundary = [(0, 0), (6, 1), (7, -11), (-1, -10)]
    n_wt = 20
    n_iter = 1000
    step_size = 0.1
    min_space = 2.1
    pad = 1.01
    plot = True
    verbose = True
    turbines = random_positions(boundary, n_wt, n_iter, step_size, min_space,
                                pad, plot, verbose)
    print(turbines)
