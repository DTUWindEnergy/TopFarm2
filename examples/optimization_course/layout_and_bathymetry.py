import numpy as np
import matplotlib.pyplot as plt
import time

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasyRandomSearchDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.examples.data.parque_ficticio_offshore import ParqueFicticioOffshore

from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines

def main():
    if __name__ == '__main__':
        site = ParqueFicticioOffshore()
        x_init, y_init = site.initial_position[:,0], site.initial_position[:,1]
        boundary = site.boundary
        # # # Wind turbines and wind farm model definition
        windTurbines = IEA37_WindTurbines() 
        wfm = IEA37SimpleBastankhahGaussian(site, windTurbines)
        
        wsp = np.asarray([10, 15])
        wdir = np.arange(0,360,45)
        maximum_water_depth = -52
        n_wt = x_init.size
        maxiter = 10
        
        def aep_func(x, y, **kwargs):
            simres = wfm(x, y, wd=wdir, ws=wsp)
            aep = simres.aep().values.sum()
            water_depth = np.diag(wfm.site.ds.interp(x=x, y=y)['water_depth'])
            return [aep, water_depth]
            
        tol = 1e-8
        ec = 1e-2
        min_spacing = 260
        
        cost_comp = CostModelComponent(input_keys=[('x', x_init),('y', y_init)],
                                                  n_wt=n_wt,
                                                  cost_function=aep_func,
                                                  objective=True,
                                                  maximize=True,
                                                  output_keys=[('AEP', 0), ('water_depth', np.zeros(n_wt))]
                                                  )
        problem = TopFarmProblem(design_vars={'x': x_init, 'y': y_init},
                                constraints=[XYBoundaryConstraint(boundary),
                                              SpacingConstraint(min_spacing)],
                                   post_constraints=[('water_depth', {'lower': np.ones(n_wt) * maximum_water_depth})],
                                  cost_comp=cost_comp,
                                    driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=maxiter, tol=tol),
                                   # driver=EasyRandomSearchDriver(RandomizeTurbinePosition()),
                                  plot_comp=XYPlotComp(),
                                  expected_cost=ec)
        
        if 1:
            tic = time.time()
            cost, state, recorder = problem.optimize()
            toc = time.time()
            print('Optimization took: {:.0f}s'.format(toc-tic))
            
            plt.figure()
            plt.plot(recorder['water_depth'].min((1)))
            plt.plot([0,recorder['water_depth'].shape[0]],[maximum_water_depth, maximum_water_depth])
            plt.xlabel('Iteration')
            plt.ylabel('Max depth [m]')
            plt.show()
                
        values = site.ds.water_depth.values
        x = site.ds.x.values
        y = site.ds.y.values
        levels = np.arange(int(values.min()), int(values.max()))
        max_wd_index = int(np.argwhere(levels==maximum_water_depth))
        Y, X = np.meshgrid(y, x)
        
        fig1, ax1 = plt.subplots(1)
        cs = plt.contour(x, y , values.T, levels)
        lines = []
        for line in cs.collections[max_wd_index].get_paths():
            lines.append(line.vertices)
        fig2, ax2 = plt.subplots(1)
        site.ds.water_depth.plot(ax=ax2, levels=100)
        for line in lines:
            ax2.plot(line[:, 0], line[:,1], 'r')
        problem.model.plot_comp.plot_current_position(state['x'], state['y'])
        ax2.set_title(f'Max Water Depth Boundary: {maximum_water_depth} m')

main()
