import os
import numpy as np
from openmdao.api import CaseReader


def pos_from_case(case_recorder_filename):
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
    latest = os.path.join(case_recorder_dir,latest)
    return latest
    
if __name__ == '__main__':
    crf = r"C:\Sandbox\Git\TopFarm2\topfarm\cases_20180621_104446.sql"
    case_recorder_filename = crf
    turbines = pos_from_case(case_recorder_filename)
    print(turbines)

    case_recorder_dir = r'C:\Sandbox\Git\TopFarm2\topfarm'
    latest_id = latest_id(case_recorder_dir)
    print(latest_id)
