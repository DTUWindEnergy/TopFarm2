import time
import numpy as np
import warnings
from wetb.utils.timing import print_time
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from openmdao.api import Problem, ScipyOptimizeDriver, IndepVarComp
from topfarm.constraint_components.boundary_component import BoundaryComp,\
    PolygonBoundaryComp
from topfarm.constraint_components.spacing_component import SpacingComp
from topfarm.plotting import PlotComp


class TopFarm(object):
    """Optimize wind farm layout in terms of 
    - Position of turbines
    [- Type of turbines: Not implemented yet]
    [- Height of turbines: Not implemented yet]
    [- Number of turbines: Not implemented yet]
    """

    def __init__(self, turbines, cost_comp, min_spacing, boundary, boundary_type='convex_hull', plot_comp=None,
                 driver_options={'optimizer': 'SLSQP'}):

        self.initial_positions = turbines = np.array(turbines)

        n_wt = turbines.shape[0]
        if boundary_type == 'polygon':
            self.boundardy_comp = PolygonBoundaryComp(boundary, n_wt)
        else:
            self.boundardy_comp = BoundaryComp(boundary, n_wt, boundary_type)
        self.problem = prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('turbineX', turbines[:, 0], units='m')
        indeps.add_output('turbineY', turbines[:, 1], units='m')
        indeps.add_output('boundary', self.boundardy_comp.vertices, units='m')
        prob.model.add_subsystem('cost_comp', cost_comp, promotes=['*'])
        prob.driver = ScipyOptimizeDriver()

        #prob.driver.options['optimizer'] = optimizer
        prob.driver.options.update(driver_options)
        prob.model.add_design_var('turbineX', lower=np.nan, upper=np.nan)
        prob.model.add_design_var('turbineY', lower=np.nan, upper=np.nan)
        prob.model.add_objective('cost')

        prob.model.add_subsystem('spacing_comp', SpacingComp(nTurbines=n_wt), promotes=['*'])
        prob.model.add_subsystem('bound_comp', self.boundardy_comp, promotes=['*'])
        if plot_comp == "default":
            plot_comp = PlotComp()
        if plot_comp:
            plot_comp.n_wt = n_wt
            plot_comp.n_vertices = self.boundardy_comp.vertices.shape[0]
            prob.model.add_subsystem('plot_comp', plot_comp, promotes=['*'])

        self.plot_comp = plot_comp
        prob.model.add_constraint('wtSeparationSquared', lower=np.zeros(int(((n_wt - 1.) * n_wt / 2.))) + (min_spacing)**2)
        prob.model.add_constraint('boundaryDistances', lower=np.zeros(self.boundardy_comp.nVertices * n_wt))

        prob.setup(check=True, mode='fwd')

    def check(self, all=False, tol=1e-3):
        """Check gradient computations"""
        comp_name_lst = [comp.pathname for comp in self.problem.model.system_iter()
                         if comp._has_compute_partials and
                         (comp.pathname not in ['spacing_comp', 'bound_comp', 'plot_comp'] or (all and comp.pathname != 'plot_comp'))]
        print("checking %s" % ", ".join(comp_name_lst))
        res = self.problem.check_partials(comps=comp_name_lst, compact_print=True)
        for comp in comp_name_lst:
            var_pair = list(res[comp].keys())
            worst = var_pair[np.argmax([res[comp][k]['rel error'].forward for k in var_pair])]
            err = res[comp][worst]['rel error'].forward
            if err > tol:
                raise Warning("Mismatch between finite difference derivative of '%s' wrt. '%s' and derivative computed in '%s' is: %f" %
                              (worst[0], worst[1], comp, err))

    def evaluate(self):
        t = time.time()
        self.problem.run_model()
        print("Evaluated in\t%.3fs" % (time.time() - t))
        return self.get_cost(), self.turbine_positions

    def evaluate_gradients(self):
        t = time.time()
        res = self.problem.compute_totals(['cost'], wrt=['turbineX', 'turbineY'], return_format='dict')
        print("Gradients evaluated in\t%.3fs" % (time.time() - t))
        return res

    def optimize(self):
        t = time.time()
        self.problem.run_driver()
        print("Optimized in\t%.3fs" % (time.time() - t))
        return np.array([self.problem['turbineX'], self.problem['turbineY']]).T

    def get_cost(self):
        return self.problem['cost'][0]

    @property
    def boundary(self):
        b = self.boundardy_comp.vertices
        return np.r_[b, b[:1]]

    @property
    def turbine_positions(self):
        return np.array([self.problem['turbineX'], self.problem['turbineY']]).T


def try_me():
    if __name__ == '__main__':
        from topfarm.cost_models.dummy import DummyCostPlotComp, DummyCost

        n_wt = 4
        random_offset = 5
        optimal = [(3, -3), (7, -7), (4, -3), (3, -7), (-3, -3), (-7, -7), (-4, -3), (-3, -7)][:n_wt]
        rotorDiameter = 1.0
        minSpacing = 2.0

        turbines = np.array(optimal) + np.random.randint(-random_offset, random_offset, (n_wt, 2))
        plot_comp = DummyCostPlotComp(optimal)

        boundary = [(0, 0), (6, 0), (6, -10), (0, -10)]
        tf = TopFarm(turbines, DummyCost(optimal), minSpacing * rotorDiameter, boundary=boundary, plot_comp=plot_comp)
        # tf.check()
        tf.optimize()
        # plot_comp.show()


try_me()
