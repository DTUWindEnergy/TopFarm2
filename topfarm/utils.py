import os
from copy import copy
import numpy as np
from openmdao.api import CaseReader
import matplotlib.pyplot as plt
from topfarm.constraint_components.boundary_component import \
    PolygonBoundaryComp
from topfarm.constraint_components.spacing_component import SpacingComp


def pos_from_case(case_recorder_filename):
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
                     plot=False, verbose=True):
    ba = np.array(boundary).T
    x_max = max(ba[0])
    x_min = min(ba[0])
    y_max = max(ba[1])
    y_min = min(ba[1])
    x = np.random.rand(n_wt)
    y = np.random.rand(n_wt)
    xm = (x_max+x_min)/2
    ym = (y_max+y_min)/2
    xa = (x_max-xm)
    ya = (y_max-ym)
    x *= 2*xa
    y *= 2*ya
    x += xm-xa
    y += ym-ya

    boundary_comp = PolygonBoundaryComp(boundary, n_wt)
    spacing_comp = SpacingComp(nTurbines=n_wt)
    turbineX = copy(x)
    turbineY = copy(y)
    inputs = {}
    outputs = {}
    J = {}
    inputs['turbineX'] = turbineX
    inputs['turbineY'] = turbineY
    min_space2 = min_space**2
    bx = np.append(ba[0], ba[0][0])
    by = np.append(ba[1], ba[1][0])
    if plot:
        plt.cla()
        plt.plot(bx, by)
        plt.plot(x, y, '.')
    for j in range(n_iter):
        spacing_comp.compute(inputs, outputs)
        spacing_comp.compute_partials(inputs, J)
        dx = J['wtSeparationSquared', 'turbineX']
        dy = J['wtSeparationSquared', 'turbineY']
        dist = outputs['wtSeparationSquared']
        index = np.argmin(dist)
        if dist[index] < min_space2:
            turbineX += dx[index]*step_size
            turbineY += dy[index]*step_size
            turbineX, turbineY = _contain(n_wt, turbineX, turbineY,
                                          boundary_comp)
        else:
            if verbose:
                print('Obtained required spacing after {} iterations'.format(
                        j))
            break
    if plot:
        plt.plot(turbineX, turbineY, 'o')
    spacing_comp.compute(inputs, outputs)
    dist = outputs['wtSeparationSquared']
    index = np.argmin(dist)
    if verbose:
        print('Spacing obtained: {}'.format(dist[index]**0.5))
        print('Spacing required: {}'.format(min_space))
    turbines = np.array([turbineX, turbineY]).T
    return turbines


def _contain(n_wt, turbineX, turbineY, boundary_comp):
    for i in range(0, n_wt):
        dng = boundary_comp.calc_distance_and_gradients(turbineX,
                                                        turbineY)
        dist = dng[0][i]
        if dist < 0:
            dx = dng[1][i]
            dy = dng[2][i]
            turbineX[i] -= dx*dist*1.1
            turbineY[i] -= dy*dist*1.1
            dng = boundary_comp.calc_distance_and_gradients(turbineX,
                                                            turbineY)
            dist = dng[0][i]
    return turbineX, turbineY


if __name__ == '__main__':
    crf = r"C:\Sandbox\Git\TopFarm2\topfarm\cases_20180621_111710.sql"
    case_recorder_filename = crf
    turbines = pos_from_case(case_recorder_filename)
    print(turbines)

    case_recorder_dir = r'C:\Sandbox\Git\TopFarm2\topfarm'
    latest_id = latest_id(case_recorder_dir)
    print(latest_id)

    boundary = [(0, 0), (6, 1), (7, -11), (-1, -10)]
    n_wt = 20
    n_iter = 1000
    step_size = 0.1
    min_space = 2.1
    plot = True
    verbose = True
    turbines = random_positions(boundary, n_wt, n_iter, step_size, min_space,
                                plot, verbose)
    print(turbines)
