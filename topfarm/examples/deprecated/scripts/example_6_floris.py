from topfarm.tests.test_files.example_data.floris.NREL5MW import NREL5MWREF, ct_curve, power_curve, cp_curve_spline, cp_curve, Amalia1Site
from py_wake.deficit_models.noj import NOJ
import matplotlib.pyplot as plt
import numpy as np
from topfarm import TopFarmGroup, TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Square
from topfarm.plotting import XYPlotComp, NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
#from topfarm.recorders import TopFarmSqliteRecorder
import topfarm
import time


def main():
    if __name__ == '__main__':
        if not 'plantenergy' in topfarm.plugins:
            pass
        else:
            try:

                from plantenergy.GeneralWindFarmGroups import AEPGroup
                from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
                try:
                    import matplotlib.pyplot as plt
                    plt.gcf()
                    plot_comp = XYPlotComp()
                    plot = True
                except RuntimeError:
                    plot_comp = NoPlot()
                    plot = False
    #            plot_comp = NoPlot()

                def setup_prob(differentiable):

                    #####################################
                    ## Setup Floris run with gradients ##
                    #####################################
                    topfarm.x_key = 'turbineX'
                    topfarm.y_key = 'turbineY'
                    turbineX = np.array([1164.7, 947.2, 1682.4, 1464.9, 1982.6, 2200.1])
                    turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
                    f = np.array([3.597152, 3.948682, 5.167395, 7.000154, 8.364547, 6.43485,
                                  8.643194, 11.77051, 15.15757, 14.73792, 10.01205, 5.165975])
                    wind_speed = 8
                    site = Amalia1Site(f, mean_wsp=wind_speed)
                    site.initial_position = np.array([turbineX, turbineY]).T
                    wt = NREL5MWREF()
                    wake_model = NOJ(site, wt)
                    aep_calculator = AEPCalculator(wake_model)
                    n_wt = len(turbineX)
                    differentiable = differentiable
                    wake_model_options = {'nSamples': 0,
                                          'nRotorPoints': 1,
                                          'use_ct_curve': True,
                                          'ct_curve': ct_curve,
                                          'interp_type': 1,
                                          'differentiable': differentiable,
                                          'use_rotor_components': False}

                    aep_comp = AEPGroup(n_wt, differentiable=differentiable, use_rotor_components=False, wake_model=floris_wrapper,
                                        params_IdepVar_func=add_floris_params_IndepVarComps,
                                        wake_model_options=wake_model_options, datasize=len(power_curve), nDirections=len(f), cp_points=len(power_curve))  # , cp_curve_spline=None)

                    def cost_func(AEP, **kwargs):
                        return AEP
                    cost_comp = CostModelComponent(input_keys=[('AEP', [0])], n_wt=n_wt, cost_function=cost_func,
                                                   output_keys="aep", output_unit="kWh", objective=True, maximize=True, input_units=['kW*h'])
                    group = TopFarmGroup([aep_comp, cost_comp])
                    boundary = np.array([(900, 1000), (2300, 1000), (2300, 2100), (900, 2100)])  # turbine boundaries
                    prob = TopFarmProblem(
                        design_vars={'turbineX': (turbineX, 'm'), 'turbineY': (turbineY, 'm')},
                        cost_comp=group,
                        driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Square(), max_iter=500),
                        #                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP',tol=10**-12),
                        #                        driver=EasyScipyOptimizeDriver(optimizer='COBYLA'),
                        constraints=[SpacingConstraint(200, units='m'),
                                     XYBoundaryConstraint(boundary, units='m')],
                        plot_comp=plot_comp,
                        expected_cost=-100e2,
                    )
                    turbineZ = np.array([90.0, 100.0, 90.0, 80.0, 70.0, 90.0])
                    air_density = 1.1716    # kg/m^3
                    rotorDiameter = np.zeros(n_wt)
                    hubHeight = np.zeros(n_wt)
                    axialInduction = np.zeros(n_wt)
                    generatorEfficiency = np.zeros(n_wt)
                    yaw = np.zeros(n_wt)
                    for turbI in range(0, n_wt):
                        rotorDiameter[turbI] = wt.diameter()       # m
                        hubHeight[turbI] = wt.hub_height()            # m
                        axialInduction[turbI] = 1.0 / 3.0
                        generatorEfficiency[turbI] = 1.0  # 0.944
                        yaw[turbI] = 0.     # deg.
                    prob['turbineX'] = turbineX
                    prob['turbineY'] = turbineY
                    prob['hubHeight'] = turbineZ
                    prob['yaw0'] = yaw
                    prob['rotorDiameter'] = rotorDiameter
                    prob['hubHeight'] = hubHeight
                    prob['axialInduction'] = axialInduction
                    prob['generatorEfficiency'] = generatorEfficiency
                    prob['windSpeeds'] = np.ones(len(f)) * wind_speed
                    prob['air_density'] = air_density

                    prob['windDirections'] = np.arange(0, 360, 360 / len(f))
                    prob['windFrequencies'] = f / 100
                    # turns off cosine spread (just needs to be very large)
                    prob['model_params:cos_spread'] = 1E12
                    prob['model_params:shearExp'] = 0.25
                    prob['model_params:z_ref'] = 80.
                    prob['model_params:z0'] = 0.
                    prob['rated_power'] = np.ones(n_wt) * 5000.
                    prob['cut_in_speed'] = np.ones(n_wt) * 3
                    prob['cp_curve_wind_speed'] = cp_curve[:, 0]
                    prob['cp_curve_cp'] = cp_curve[:, 1]
                    prob['rated_wind_speed'] = np.ones(n_wt) * 11.4
                    prob['cut_out_speed'] = np.ones(n_wt) * 25.0
                    # if 0:
                    # prob.check_partials(compact_print=True,includes='*direction_group0*')
                    # else:
                    return prob

                differentiable = True
                prob = setup_prob(differentiable)
    #            n2(prob)
                cost_init, state_init = prob.evaluate()
                tic = time.time()
                cost, state, recorder = prob.optimize()
                toc = time.time()
                print('FLORIS calculation with differentiable = {0} took {1} sec.'.format(differentiable, toc - tic))
                print(prob[topfarm.x_key])

    #            ########################################
    #            ## Setup Floris run without gradients ##
    #            ########################################
    #
    #            differentiable = False
    #            prob2 = setup_prob(differentiable)
    #            cost_init, state_init = prob2.evaluate()
    #            tic = time.time()
    #            cost, state, recorder = prob2.optimize()
    #            toc = time.time()
    #            print('FLORIS calculation with differentiable = {0} took {1} sec.'.format(differentiable, toc-tic))
    #
    #
    #
    #            ########################################
    #            ## Setup Pywake run without gradients ##
    #            ########################################
    #
    #
    #
    #            class PyWakeAEP(AEPCalculator):
    #                """TOPFARM wrapper for PyWake AEP calculator"""
    #
    #                def get_TopFarm_cost_component(self, n_wt, wd=None, ws=None):
    #                    """Create topfarm-style cost component
    #
    #                    Parameters
    #                    ----------
    #                    n_wt : int
    #                        Number of wind turbines
    #                    """
    #                    return AEPCostModelComponent(
    #                        input_keys=['turbineX', 'turbineY'],
    #                        n_wt=n_wt,
    #                        cost_function=lambda **kwargs:
    #                            self.calculate_AEP(x_i=kwargs[topfarm.x_key],
    #                                               y_i=kwargs[topfarm.y_key],
    #                                               h_i=kwargs.get(topfarm.z_key, None),
    #                                               type_i=kwargs.get(topfarm.type_key, None),
    #                                               wd=wd, ws=ws).sum(),
    #                        output_unit='GWh')
    #            aep_calc = PyWakeAEP(site, wt, wake_model)
    #            tf = TopFarmProblem(
    #                    design_vars={'turbineX': turbineX, 'turbineY': turbineY},
    #                cost_comp=aep_calc.get_TopFarm_cost_component(len(turbineX)),
    #                    driver=EasyScipyOptimizeDriver(optimizer='SLSQP'),
    #                    constraints=[SpacingConstraint(200),
    #                                 XYBoundaryConstraint(boundary)],
    #                    plot_comp=plot_comp)
    #            cost_init_pw, state_init_pw = tf.evaluate()
    #            cost_pw, state_pw, recorder_pw = tf.optimize()
    #
    #
    #            prob['turbineX'] = state_pw['turbineX']
    #            prob['turbineY'] = state_pw['turbineY']
    #            cost_pw_fl, state_pw_fl = prob.evaluate()
    #            print('\n***Optimized by PyWake:***')
    #            print('AEP FLORISSE initial:    ', float(-cost_init/10**6))
    #            print('AEP FLORISSE optimized:  ', float(-cost_pw_fl)/10**6)
    #            print('AEP PyWake initial:    ', float(-cost_init_pw))
    #            print('AEP PyWake optimized:  ', float(-cost_pw))
    #
    #            print('***Optimized by FLORISSE:***')
    #            print('AEP FLORISSE initial:    ', float(-cost_init/10**6))
    #            print('AEP FLORISSE optimized:  ', float(-cost)/10**6)
    #            print('AEP Pywake initial:     ', aep_calculator.calculate_AEP(turbineX, turbineY).sum())
    #            print('AEP Pywake optimized:   ', aep_calculator.calculate_AEP(state['turbineX'], state['turbineY']).sum())
            finally:
                topfarm.x_key = 'x'
                topfarm.y_key = 'y'


main()
