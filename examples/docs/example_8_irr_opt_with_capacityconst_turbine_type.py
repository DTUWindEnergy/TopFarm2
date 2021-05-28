import numpy as np
from py_wake.aep_calculator import AEPCalculator
from topfarm.cost_models.economic_models.turbine_cost import economic_evaluation
from topfarm import TopFarmGroup, TopFarmProblem
from py_wake.examples.data import hornsrev1
from topfarm.easy_drivers import EasyRandomSearchDriver, EasySimpleGADriver
from topfarm.drivers.random_search_driver import RandomizeAllUniform
from topfarm.plotting import TurbineTypePlotComponent, NoPlot
from py_wake.wind_turbines import WindTurbines
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
# from py_wake.wind_turbines import cube_power, dummy_thrust
from py_wake.deficit_models.noj import NOJ
from topfarm.constraint_components.capacity import CapacityConstraint
from py_wake.site._site import UniformWeibullSite
import topfarm

import time

def cube_power(ws_cut_in=3, ws_cut_out=25, ws_rated=12, power_rated=5000):
    def power_func(ws):
        ws = np.asarray(ws)
        power = np.zeros_like(ws, dtype=np.float)
        m = (ws >= ws_cut_in) & (ws < ws_rated)
        power[m] = power_rated * ((ws[m] - ws_cut_in) / (ws_rated - ws_cut_in))**3
        power[(ws >= ws_rated) & (ws <= ws_cut_out)] = power_rated
        return power
    return power_func


def dummy_thrust(ws_cut_in=3, ws_cut_out=25, ws_rated=12, ct_rated=8 / 9):
    # temporary thrust curve fix
    def ct_func(ws):
        ws = np.asarray(ws)
        ct = np.zeros_like(ws, dtype=np.float)
        if ct_rated > 0:
            # ct = np.ones_like(ct)*ct_rated
            m = (ws >= ws_cut_in) & (ws < ws_rated)
            ct[m] = ct_rated
            idx = (ws >= ws_rated) & (ws <= ws_cut_out)
            # second order polynomial fit for above rated
            ct[idx] = np.polyval(np.polyfit([ws_rated, (ws_rated + ws_cut_out) / 2,
                                             ws_cut_out], [ct_rated, 0.4, 0.03], 2), ws[idx])
        return ct
    return ct_func

# ----------- SELECT OBJECTIVE & TURN ON/OFF CONSTRAINT --------------
#        obj = False  # objective AEP (True), IRR (False)
#        max_con_on = False  # max installed capacity constraint ON (True), OFF (False)
def main(obj=False, max_con_on=True):
    if __name__ == '__main__':
        start = time.time()
        try:
            import matplotlib.pyplot as plt
            plt.gcf()
            plot = True
        except RuntimeError:
            plot = False

        # ------ DEFINE WIND TURBINE TYPES, LOCATIONS & STORE METADATA -------
        windTurbines = WindTurbines(names=['Ghost_T1', 'T2'],
                                    diameters=[40, 84],
                                    hub_heights=[70, hornsrev1.HornsrevV80().hub_height()],
                                    ct_funcs=[dummy_thrust(ct_rated=0), hornsrev1.HornsrevV80().ct],
                                    power_funcs=[cube_power(power_rated=0), cube_power(power_rated=3000)],
                                    # hornsrev1.HornsrevV80()._power],
                                    power_unit='kW')
        Drotor_vector = windTurbines._diameters
        power_rated_vec = np.array([pcurv(25) / 1000 for pcurv in windTurbines._power_funcs])
        hub_height_vector = windTurbines._hub_heights

        x, y = np.meshgrid(range(-840, 840, 420), range(-840, 840, 420))
        n_wt = len(x.flatten())
        # initial turbine positions and other independent variables
        ext_vars = {'x': x.flatten(), 'y': y.flatten(), 'obj': obj * 1}

        capconst = []
        if max_con_on:
            capconst = [CapacityConstraint(max_capacity=30.01,
                                           rated_power_array=power_rated_vec)]

        # ---------------- DEFINE SITE & SELECT WAKE MODEL -------------------
    #        site = UniformWeibullSite(p_wd=[50, 50], a=[9, 9], k=[2.3, 2.3], ti=.1, alpha=0, h_ref=100)
        site = UniformWeibullSite(p_wd=[100], a=[9], k=[2.3], ti=.1)
        site.default_ws = [9]  # reduce the number of calculations
        site.default_wd = [0]  # reduce the number of calculations

        wake_model = NOJ(site, windTurbines)

        AEPCalc = AEPCalculator(wake_model)

        # ------------- OUTPUTS AEP PER TURBINE & FARM IRR -------------------
        def aep_func(x, y, type, obj, **kwargs):  # TODO fix type as input change to topfarm turbinetype
            out = AEPCalc.calculate_AEP(x_i=x, y_i=y, type_i=type.astype(int)).sum((1, 2))
            if obj:  # if objective is AEP; output the total Farm_AEP
                out = np.sum(out)
            return out * 10**6

        def irr_func(aep, type, **kwargs):
            idx = type.astype(int)
            return economic_evaluation(Drotor_vector[idx],
                                       power_rated_vec[idx],
                                       hub_height_vector[idx],
                                       aep).calculate_irr()

        # ----- WRAP AEP AND IRR INTO TOPFARM COMPONENTS AND THEN GROUP  -----
        aep_comp = CostModelComponent(
            input_keys=[topfarm.x_key, topfarm.y_key, topfarm.type_key, ('obj', obj)],
            n_wt=n_wt,
            cost_function=aep_func,
            output_key="aep",
            output_unit="GWh",
            objective=obj,
            output_val=np.zeros(n_wt),
            income_model=True)
        comps = [aep_comp]  # AEP component is always in the group
        if not obj:  # if objective is IRR initiate/add irr_comp
            irr_comp = CostModelComponent(
                input_keys=[topfarm.type_key, 'aep'],
                n_wt=n_wt,
                cost_function=irr_func,
                output_key="irr",
                output_unit="%",
                objective=True)
            comps.append(irr_comp)

        group = TopFarmGroup(comps)

        # - INITIATE THE PROBLEM WITH ONLY TURBINE TYPE AS DESIGN VARIABLES -
        tf = TopFarmProblem(design_vars={topfarm.type_key: ([0] * n_wt, 0, len(windTurbines._names) - 1)},
                            cost_comp=group,
                            driver=EasyRandomSearchDriver(
                                randomize_func=RandomizeAllUniform([topfarm.type_key]), max_iter=1),
                            # driver=EasySimpleGADriver(max_gen=2, random_state=1),
                            constraints=capconst,
                            # plot_comp=TurbineTypePlotComponent(windTurbines._names),
                            plot_comp=NoPlot(),
                            ext_vars=ext_vars)

        cost, state, rec = tf.optimize()
        # view_model(problem, outfile='ex5_n2.html', show_browser=False)
        end = time.time()
        print(end - start)
        # %%
        # ------------------- OPTIONAL VISUALIZATION OF WAKES ----------------
        post_visual, save = False, False
        if post_visual:
            #        import matplotlib.pyplot as plt
            for cou, (i, j, k, co, ae) in enumerate(zip(rec['x'], rec['y'], rec['type'], rec['cost'], rec['aep'])):
                AEPCalc.calculate_AEP(x_i=i, y_i=j, type_i=k)
                AEPCalc.plot_wake_map(
                    wt_x=i, wt_y=j, wt_type=k, wd=site.default_wd[0], ws=site.default_ws[0], levels=np.arange(
                        2.5, 12, .1))
                windTurbines.plot(i, j, types=k)
                title = f'IRR: {-np.round(co,2)} %, AEP :  {round(np.sum(ae))} GWh, '
                if "totalcapacity" in rec.keys():
                    title += f'Total Capacity: {rec["totalcapacity"][cou]} MW'
                plt.title(title)
                if save:
                    plt.savefig(r'..\..\..\ima2\obj_AEP_{}_MaxConstraint_{}_{}.png'.format(obj, max_con_on, cou))
                plt.show()


for ii in [True, False]:
    for jj in [True, False]:
        main(obj=ii, max_con_on=jj)
