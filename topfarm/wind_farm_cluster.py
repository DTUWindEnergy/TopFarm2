# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:21:50 2024

@author: mikf
"""
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

from py_wake.wind_turbines.power_ct_functions import SimpleYawModel, PowerCtTabular
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.utils.gradients import autograd
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_farm_models.wind_farm_model import SimulationResult

from topfarm._topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint


class TopFarmCluster:
    def __init__(self,
                 wind_farm_boundaries,
                 target_indicator,
                 x_target,
                 y_target,
                 wt_list,
                 wfm,
                 n_points_x,
                 n_points_y,
                 site,
                 ws_sim,
                 wd_sim,
                 time_stamps,
                 random_pct=50,
                 ws_ss=[10],
                 wd_ss=np.arange(0, 360, 30),
                 fn_prefix_ss='ss_states/ss_state',
                 fn_prefix_sim='sim/sim_res',
                 wf_model_args={'TI': 0.1},
                 ):
        '''
        Parameters
        ----------
        wind_farm_boundaries : list of lists
            Wind farm boudnary coordinates.
        target_indicator : int
            index of the wind farm of which the power with under the influence of the surrounding wind farms is calculated.
        x_target : list
            x-coordinates of target wind farm.
        y_target : list
            y-coordinate of target wind farm.
        wt_list : list
            list of PyWake WindTurbine objects for all turbines in the cluster.
        wfm : WindFarmModel
            PyWake WindFarmModel.
        n_points_x : int
            number of points in x-direction for the grid for smart start.
        n_points_y : int
            number of points in y-direction for the grid for smart start.
        site : Site
            PyWake Site.
        ws_sim : list
            time series of wind speeds.
        wd_sim : list
            time series of wind directions.
        time_stamps : list
            times series time stamps
        random_pct : float, optional
            random percentage for smart start. The default is 50.
        ws_ss : list, optional
            wind speeds that are considered for smart start. The default is [10].
        wd_ss : array, optional
            wind directions that are considered for smart start. The default is np.arange(0, 360, 30).
        fn_prefix_ss : str, optional
            prefix to the path where smart start layouts are stored if enabled. The default is 'ss_states/ss_state'.
        fn_prefix_sim : str, optional
            prefex to the path where the PyWake simulation results are stored if enabled. The default is 'sim/sim_res'.
        wf_model_args : dict, optional
            extra parameters that should be passed to the PyWake WindFarmModel call. The default is {'TI': 0.1}.

        '''
        self.wind_farm_boundaries = wind_farm_boundaries
        self.target_indicator = target_indicator
        self.x_target = x_target
        self.y_target = y_target
        self.wt_list = wt_list
        self.wfm = wfm
        self.n_points_x = n_points_x
        self.n_points_y = n_points_y
        self.site = site
        self.random_pct = random_pct
        self.ws_ss = ws_ss
        self.wd_ss = wd_ss
        self.ws_sim = ws_sim
        self.wd_sim = wd_sim
        self.time_stamps = time_stamps
        self.fn_prefix_ss = fn_prefix_ss
        self.fn_prefix_sim = fn_prefix_sim
        self.wf_model_args = wf_model_args

        self.n_wfs = len(wind_farm_boundaries)

    def run(self,
            wt_types,
            n_wts,
            construction_days,
            seeds_ss,
            save_ss=False,
            load_ss=False,
            save_sim=False,
            load_sim=False,
            ):
        '''
        Parameters
        ----------
        wt_types : list of int
            list of wind turbine types for each farm in the cluster where the number is the index in wt_list
        n_wts : list of int
            list of the number of wind turbines in each farm in the cluster.
        construction_days : list of int
            list of the index in time_stamps at which each farm is starting operation.
        seeds_ss : list of int
            list of smart start seed numbers
        save_ss : bool, optional
            save smart start layouts. The default is True.
        load_ss : bool, optional
            load smart start layouts if the files exist. The default is False.
        save_sim : bool, optional
            save PyWake SimulationResults. The default is True.
        load_sim : bool, optional
            load SimulationREsults if the files exist. The default is False.

        Returns
        -------
        df : DataFrame
            Dataframe containing the power of the target wind farm with and without own and neighbouring wakes as well as internal and external wake loss.

        '''
        wt_list = self.wt_list.copy()
        wts = np.array(wt_list)[wt_types].tolist()

        x_neighbours = []
        y_neighbours = []
        df = pd.DataFrame({'WS': self.ws_sim, 'WD': self.wd_sim}, index=self.time_stamps)

        wt_ref = wts[self.target_indicator]
        u = wt_ref.powerCtFunction.ws_tab
        p, _ = wt_ref.powerCtFunction.power_ct_tab
        ct = np.zeros_like(p)
        powerCtFunction = PowerCtTabular(u, p, 'w', ct, ws_cutin=None, ws_cutout=None,
                                         power_idle=0, ct_idle=None, method='linear',
                                         additional_models=[SimpleYawModel()])
        wt_ref_no_wake = WindTurbine('WT_ref_no_wake', wt_ref.diameter(0), wt_ref.hub_height(0), powerCtFunction)
        wt_list.append(wt_ref_no_wake)
        n_wt_target = n_wts[self.target_indicator]

        wt_types_neighbours = wt_types.copy()
        wt_types_neighbours.pop(self.target_indicator)

        n_wts_neighbours = n_wts.copy()
        n_wts_neighbours.pop(self.target_indicator)

        construction_day_neighbours = construction_days.copy()
        construction_day_neighbours.pop(self.target_indicator)

        for n in range(self.n_wfs):
            if not n == self.target_indicator:
                wt = wts[n]
                n_wt = n_wts[n]
                bound = self.wind_farm_boundaries[n]
                seed_ss = seeds_ss[n]
                wf_model = self.wfm(self.site, wt)
                file_name_ss = f'{self.fn_prefix_ss}_{n}.pkl'
                if load_ss and os.path.exists(file_name_ss):
                    with open(file_name_ss, 'rb') as f:
                        state = pickle.load(f)
                else:
                    x_init = (np.random.random(n_wt) - 0.5) * 1000 + np.mean(bound[0])
                    y_init = (np.random.random(n_wt) - 0.5) * 1000 + np.mean(bound[1])
                    problem = TopFarmProblem(design_vars={'x': x_init, 'y': y_init},
                                             cost_comp=PyWakeAEPCostModelComponent(wf_model, n_wt, grad_method=autograd,),
                                             constraints=[XYBoundaryConstraint(np.asarray(bound).T, boundary_type='polygon'),
                                                          SpacingConstraint(wt.diameter())],
                                             plot_comp=XYPlotComp())
                    xs = np.linspace(np.min(bound[0]), np.max(bound[0]), self.n_points_x)
                    ys = np.linspace(np.min(bound[1]), np.max(bound[1]), self.n_points_y)
                    YY, XX = np.meshgrid(ys, xs)
                    problem.smart_start(XX,
                                        YY,
                                        problem.cost_comp.get_aep4smart_start(ws=self.ws_ss,
                                                                              wd=self.wd_ss,
                                                                              **self.wf_model_args),
                                        random_pct=self.random_pct,
                                        seed=seed_ss)
                    state = problem.state
                    if save_ss:
                        dir_path = os.path.dirname(file_name_ss)
                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)
                        with open(file_name_ss, 'wb') as f:
                            pickle.dump(state, f)

                x_neighbours.append(list(state['x']))
                y_neighbours.append(list(state['y']))

        sequence = np.argsort(construction_day_neighbours)
        construction_day_sort = np.sort(construction_day_neighbours)
        wf_model = self.wfm(self.site, WindTurbines.from_WindTurbine_lst(wt_list))
        powers = []
        powers_no_wake = []
        for m in tqdm(range(self.n_wfs), desc='FarmFlow'):
            file_name_sim = f'{self.fn_prefix_sim}_{m}.nc'
            file_name_sim_no_wake = f'{self.fn_prefix_sim}_no_wake_{m}.nc'
            if load_sim and os.path.exists(file_name_sim) and os.path.exists(file_name_sim_no_wake):
                sim_res_time = SimulationResult.load(file_name_sim, wf_model)
                sim_res_time_no_wake = SimulationResult.load(file_name_sim_no_wake, wf_model)
            else:
                if m == 0:
                    ts_part = df.iloc[:construction_day_sort[0]]
                elif m == self.n_wfs - 1:
                    ts_part = df.iloc[construction_day_sort[self.n_wfs - 2]:]
                else:
                    ts_part = df.iloc[construction_day_sort[m - 1]:construction_day_sort[m]]

                if ts_part.size > 0:
                    active_farms = sequence[:m]
                    x = np.asarray(self.x_target)
                    y = np.asarray(self.y_target)
                    types_active = (wt_types[self.target_indicator] * np.ones(n_wt_target)).astype(int)
                    types_active_no_wake = ((len(wt_list) - 1) * np.ones(n_wt_target)).astype(int)
                    for af in active_farms:
                        x = np.concatenate([x, x_neighbours[af]])
                        y = np.concatenate([y, y_neighbours[af]])
                        types_active = np.concatenate([types_active, wt_types_neighbours[af] * np.ones(np.asarray(n_wts_neighbours)[af])])
                        types_active_no_wake = np.concatenate([types_active_no_wake, wt_types_neighbours[af] * np.ones(np.asarray(n_wts_neighbours)[af])])
                    sim_res_time = wf_model(x, y,  # wind turbine positions
                                            type=types_active,
                                            wd=ts_part.WD,  # Wind direction time series
                                            ws=ts_part.WS,  # Wind speed time series
                                            time=ts_part.index,  # time stamps
                                            **self.wf_model_args
                                            )
                    sim_res_time_no_wake = wf_model(x, y,  # wind turbine positions
                                                    type=types_active_no_wake,
                                                    wd=ts_part.WD,  # Wind direction time series
                                                    ws=ts_part.WS,  # Wind speed time series
                                                    time=ts_part.index,  # time stamps
                                                    **self.wf_model_args
                                                    )
                    if save_sim:
                        dir_path = os.path.dirname(file_name_sim)
                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)
                        sim_res_time.save(file_name_sim)
                        sim_res_time_no_wake.save(file_name_sim_no_wake)

            wake_power = sim_res_time.Power[:n_wt_target].sum('wt').values
            no_wake_power = sim_res_time_no_wake.Power[:n_wt_target].sum('wt').values

            powers.append(wake_power)
            powers_no_wake.append(no_wake_power)

        sim_res_no_neighbours = wf_model(self.x_target, self.y_target,  # wind turbine positions
                                         type=(wt_types[self.target_indicator] * np.ones(n_wt_target)).astype(int),
                                         wd=df.WD,  # Wind direction time series
                                         ws=df.WS,  # Wind speed time series
                                         time=df.index,  # time stamps
                                         **self.wf_model_args)
        sim_res_no_neighbours_no_wakes = wf_model(self.x_target, self.y_target,  # wind turbine positions
                                                  type=((len(wt_list) - 1) * np.ones(n_wt_target)).astype(int),
                                                  wd=df.WD,  # Wind direction time series
                                                  ws=df.WS,  # Wind speed time series
                                                  time=df.index,  # time stamps
                                                  **self.wf_model_args)

        df['power'] = np.hstack(powers)
        df['power_no_wake'] = np.hstack(powers_no_wake)
        df['power_no_neighbours'] = sim_res_no_neighbours.Power.sum('wt').values
        df['power_no_neighbours_no_wake'] = sim_res_no_neighbours_no_wakes.Power.sum('wt').values
        df['total_wake_loss'] = (df.power_no_neighbours_no_wake - df.power) / df.power_no_neighbours_no_wake * 100
        df['internal_wake_loss'] = (df.power_no_wake - df.power) / df.power_no_neighbours_no_wake * 100
        df['external_wake_loss'] = df.total_wake_loss - df.internal_wake_loss
        self.df = df
        return df
