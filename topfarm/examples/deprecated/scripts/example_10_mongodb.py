# %%
from topfarm.deprecated_mongo_recorder import MongoRecorder
import matplotlib.pyplot as plt
import numpy as np
from py_wake.site.xrsite import GlobalWindAtlasSite
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.constraint_components.boundary import CircleBoundaryConstraint, XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent, AEPCostModelComponent
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from topfarm import TopFarmGroup, TopFarmProblem
from topfarm.plotting import XYPlotComp, NoPlot
from topfarm.easy_drivers import EasyRandomSearchDriver, EasyScipyOptimizeDriver, EasySimpleGADriver
# import os
# os.system('bash run_mongo.sh')
import subprocess

def main():
    if __name__ == '__main__':
        subprocess.Popen(['mongod'])
        plot_comp = XYPlotComp()
        # plt.rcParams['figure.dpi'] = 500

        # %% Wind turbine model and site
        wt = V80()
        D = wt.diameter()
        hub_height = wt.hub_height()
        site = Hornsrev1Site()

        # %% wind turbine coordinates (initial) and wake model
        xy = np.array([[423974, 6151447],
                       [424042, 6150891],
                       [424111, 6150335],
                       [424179, 6149779],
                       [424247, 6149224],
                       [424315, 6148668],
                       [424384, 6148112],
                       [424452, 6147556],
                       [424534, 6151447],
                       [424602, 6150891],
                       [424671, 6150335],
                       [424739, 6149779],
                       [424807, 6149224],
                       [424875, 6148668],
                       [424944, 6148112],
                       [425012, 6147556],
                       [425094, 6151447],
                       [425162, 6150891],
                       [425231, 6150335],
                       [425299, 6149779],
                       [425367, 6149224],
                       [425435, 6148668],
                       [425504, 6148112],
                       [425572, 6147556]])

        x = xy[:, 0]
        y = xy[:, 1]

        boundary = [(423500, 6.1474e6), (425700, 6.1474e6),
                    (425200, 6.1515e6), (423000, 6.1515e6)]

        wake_model = IEA37SimpleBastankhahGaussian(
            site, wt)  # select the Gaussian wake model
        simres = wake_model(xy[:, 0], xy[:, 1])
        print(simres.aep().sum())

        # %% AEP function


        def aep_func(x, y, **kwargs):
            wake_model = IEA37SimpleBastankhahGaussian(site, wt)
            simres = wake_model(x, y)
            aep = simres.aep().sum()  # AEP in GWh
            return aep


        # create an openmdao component for aep
        aep_comp = CostModelComponent(input_keys=['x', 'y'],
                                      n_wt=len(xy),
                                      cost_function=aep_func,
                                      output_key="aep",
                                      output_unit="GWh",
                                      objective=True,
                                      output_val=sum(np.zeros(len(xy))),
                                      maximize=True
                                      )

        # %% Definition of the topfarm problem
        problem = TopFarmProblem(design_vars={'x': xy[:, 0], 'y': xy[:, 1]},
                                 cost_comp=aep_comp,
                                 driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=4, tol=1e-6),
                                 constraints=[SpacingConstraint(5 * D), XYBoundaryConstraint(boundary=boundary, boundary_type='polygon')],
                                 plot_comp=NoPlot(),
                                 recorder=MongoRecorder(db_name='data22', case_id='test', clean_up=True),
                                 )

        # run state before optimize
        cost, state = problem.evaluate()

        # %% OPTIMIZE
        cost, state, recorder = problem.optimize(disp=True)
        #additional_recorder = problem._additional_recorders[0]

        # %% the recorder is called in the same way as the TopfarmListRecorder

        keys = recorder.keys()
        print(keys)

        aep = recorder['aep']
        plt.plot(aep)
        plt.show()

        recorder.animate_turbineXY(duration=10, cost='aep', anim_options = {'interval': 20, 'blit': True}, filename='turbine_xy_anim')

main()