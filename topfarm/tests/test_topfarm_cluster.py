# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:33:46 2024

@author: mikf
"""

from topfarm.examples.energy_island import EnergyIsland
from topfarm.tests.test_files import testfilepath
import numpy as np
import os
import pandas as pd


def run_wind_farm_cluster():
    TFC = EnergyIsland()
    TFC.x_target = TFC.x_target[:6]
    TFC.y_target = TFC.y_target[:6]
    RPs = np.arange(10, 16).astype(int)
    n_wt_list = (100 / RPs).astype(int)
    wt_types = [5, 2, 4, 3, 1, 3, 2, 2, 2, 2]
    n_wts = n_wt_list[wt_types].tolist()
    seeds_ss = 10 * [0]
    construction_days = [0, 2956, 2681, 2251, 3012, 774, 324, 84, 1071, 639]
    df = TFC.run(wt_types,
                 n_wts,
                 construction_days,
                 seeds_ss,)
    return df


def save_df():
    df = run_wind_farm_cluster()
    file_path = os.path.join(testfilepath, 'wind_farm_cluster_res.csv')
    df.to_csv(file_path, sep=';')


def load_df():
    file_path = os.path.join(testfilepath, 'wind_farm_cluster_res.csv')
    df = pd.read_csv(file_path, sep=';', index_col=0, parse_dates=True)
    return df


# def test_wind_farm_cluster():
    # array = run_wind_farm_cluster().to_numpy()
    # array_ref = load_df().to_numpy()
    # np.testing.assert_allclose(array, array_ref)

# save_df()
# test_wind_farm_cluster()
