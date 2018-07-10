from topfarm.constraint_components.boundary_component import setup_xy_z_boundary
from topfarm.constraint_components.spacing_component import setup_min_spacing
from topfarm.plotting import PlotComp
from topfarm.utils import pos_from_case, latest_id
from topfarm.utils import shuffle_positions as spos
import os
import time
import numpy as np
import warnings
from openmdao.drivers.doe_generators import FullFactorialGenerator
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.recorders.case_reader import CaseReader
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from openmdao.api import Problem, ScipyOptimizeDriver, IndepVarComp, \
        SqliteRecorder


class TopFarm(object):
    """Optimize wind farm layout in terms of
    - Position of turbines
    [- Type of turbines: Not implemented yet]
    [- Height of turbines: Not implemented yet]
    [- Number of turbines: Not implemented yet]
    """

    def __init__(self, turbines, cost_comp, min_spacing, boundary,
                 boundary_type='convex_hull', plot_comp=None,
                 driver=ScipyOptimizeDriver(),
                 turbine_type_options=None,
                 record_id="Opt_%s" % time.strftime("%Y%m%d_%H%M%S")):
        self.min_spacing = min_spacing
        self.record_id = record_id

        turbines = np.array(turbines)
        n_wt = turbines.shape[0]
        turbines = np.hstack((turbines, np.zeros((n_wt, 4 - turbines.shape[1]))))

        self.initial_positions = turbines.T[:2]

        self.n_wt = n_wt

        self.problem = prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp(),
                                          promotes=['*'])

        setup_min_spacing(prob, min_spacing, n_wt)

        self.boundary_comp, z_boundary = setup_xy_z_boundary(prob, boundary, n_wt, boundary_type)

        do = driver.options
        if 'optimizer' in do and do['optimizer'] == 'SLSQP':
            # Default +/- sys.float_info.max does not work for SLSQP
            design_var_kwargs = {'lower': np.nan, 'upper': np.nan}
        else:
            design_var_kwargs = {}

        if self.boundary_comp is not None:
            min_x, min_y = self.boundary_comp.vertices.min(0)
            mean_x, mean_y = self.boundary_comp.vertices.mean(0)

            indeps.add_output('boundary', self.boundary_comp.vertices, units='m')
            prob.model.add_design_var('turbineX', **design_var_kwargs)
            prob.model.add_design_var('turbineY', **design_var_kwargs)
        else:
            min_x, mean_x, min_y, mean_y = 0, 1, 0, 1

        if z_boundary is not None:
            min_z, mean_z = np.min(z_boundary), np.mean(z_boundary)
            prob.model.add_design_var('turbineZ', lower=z_boundary[:, 0], upper=z_boundary[:, 1])
        else:
            min_z, mean_z = 0, 1

        if 'optimizer' in do and do['optimizer'] == 'SLSQP':
            min_x, min_y, min_z, mean_x, mean_y, mean_z = 0, 0, 0, 1, 1, 1  # scaling disturbs SLSQP

        indeps.add_output('turbineX', turbines[:, 0], units='m', ref0=min_x, ref=mean_x)
        indeps.add_output('turbineY', turbines[:, 1], units='m', ref0=min_y, ref=mean_y)
        indeps.add_output('turbineZ', turbines[:, 2], units='m', ref0=min_z, ref=mean_z)

        indeps.add_output('turbineType', turbines[:, 3].astype(np.int))

        prob.model.add_subsystem('cost_comp', cost_comp, promotes=['*'])

        prob.model.add_objective('cost')
        cost_comp.problem = prob
        prob.driver = driver

        if record_id:
            recorder = SqliteRecorder(record_id + ".sql")
            prob.driver.add_recorder(recorder)
            for n in ['record_desvars', 'record_responses', 'record_objectives', 'record_constraints']:
                prob.driver.recording_options[n] = True

        if plot_comp:
            if plot_comp == "default":
                plot_comp = PlotComp()

            plot_comp.n_wt = n_wt
            plot_comp.n_vertices = self.boundary_comp.vertices.shape[0]
            prob.model.add_subsystem('plot_comp', plot_comp, promotes=['*'])

        self.plot_comp = plot_comp
        prob.setup(check=True, mode='fwd')
        if turbine_type_options is not None:
            self.turbineTypeProblem = TurbineTypeProblem(self, * turbine_type_options)
        else:
            self.turbineTypeProblem = None

    def check(self, all=False, tol=1e-3):
        """Check gradient computations"""
        comp_name_lst = [comp.pathname for comp in self.problem.model.system_iter()
                         if comp._has_compute_partials and
                         (comp.pathname not in ['spacing_comp', 'bound_comp', 'plot_comp'] or (all and comp.pathname != 'plot_comp'))]
        print("checking %s" % ", ".join(comp_name_lst))
        res = self.problem.check_partials(includes=comp_name_lst, compact_print=True)
        for comp in comp_name_lst:
            var_pair = list(res[comp].keys())
            worst = var_pair[np.argmax([res[comp][k]['rel error'].forward for k in var_pair])]
            err = res[comp][worst]['rel error'].forward
            if err > tol:
                raise Warning("Mismatch between finite difference derivative of '%s' wrt. '%s' and derivative computed in '%s' is: %f" %
                              (worst[0], worst[1], comp, err))

    def _set_turbines(self, turbines):
        if turbines is not None:
            for data, n in zip(turbines.T, ['X', 'Y', 'Z', 'Type']):
                self.problem['turbine%s' % n] = data

    def evaluate(self, turbines=None):
        t = time.time()
        self._set_turbines(turbines)
        self.problem.run_model()
        print("Evaluated in\t%.3fs" % (time.time() - t))
        return self.get_cost(), self.turbines

    def evaluate_gradients(self):
        t = time.time()
        res = self.problem.compute_totals(['cost'], wrt=['turbineX',
                                                         'turbineY'], return_format='dict')
        print("Gradients evaluated in\t%.3fs" % (time.time() - t))
        return res

    def optimize(self, turbines=None):
        if self.turbineTypeProblem:
            self.turbineTypeProblem.run_driver()
            return self.get_cost(), self.turbines
        else:
            return self.optimize_xyz(turbines)

    def optimize_xyz(self, turbines):
        if len(self.problem.model._static_design_vars.keys()) == 0:
            return self.evaluate(turbines)
        else:
            t = time.time()
            self._set_turbines(turbines)
            self.problem.run_driver()
            print("Optimized in\t%.3fs" % (time.time() - t))
            return self.get_cost(), self.turbines

    def get_cost(self):
        return self.problem['cost'][0]

    def multistart(self, turbines_lst):
        results = list(map(self.optimize, turbines_lst))
        best = results[np.argmin([r[0] for r in results])]
        return best, results

    @property
    def boundary(self):
        if self.boundary_comp is None:
            return []
        else:
            b = self.boundary_comp.vertices
            return np.r_[b, b[:1]]

    @property
    def turbines(self):
        return np.array([self.problem['turbineX'], self.problem['turbineY'], self.problem['turbineZ'], self.problem['turbineType']]).T

    @property
    def turbine_positions(self):
        return np.array([self.problem['turbineX'], self.problem['turbineY']]).T

    def clean(self):
        for file in os.listdir(self.plot_comp.temp):
            if file.startswith('plot_') and file.endswith('.png'):
                os.remove(os.path.join(self.plot_comp.temp, file))

    def shuffle_positions(self, shuffle_type='rel', n_iter=1000,
                          step_size=0.1, pad=1.1, offset=5, plot=False,
                          verbose=False):
        if shuffle_type is not None:
            turbines = spos(self.boundary, self.n_wt, self.min_spacing,
                            self.turbine_positions, shuffle_type, n_iter,
                            step_size, pad, offset, plot, verbose)
            self.problem['turbineX'] = turbines.T[0]
            self.problem['turbineY'] = turbines.T[1]

    def animate(self, anim_time=10, verbose=False):
        if self.plot_comp.animate:
            self.plot_comp.run_animate(anim_time, verbose)
        else:
            if verbose:
                print('Animation requested but was not enabled for this '
                      'optimization. Set plot_comp.animate = True to enable')

    def load(self, case_id='latest'):
        if case_id is 'latest':
            case_id = latest_id(os.path.dirname(self.record_id))
        turbines = pos_from_case(case_id)
        self.problem.model['turbinesX'] = turbines[:, 0]


class TopFarmComponent(ExplicitComponent):
    def __init__(self, topFarm):
        super().__init__()
        self.topFarm = topFarm
        self.n_wt = topFarm.n_wt

    def setup(self):
        for xyz in 'XYZ':
            self.add_input('turbine%s'%xyz, val=np.zeros(self.n_wt), units='m')
        self.add_input('turbineType', val=np.zeros(self.n_wt, dtype=np.int))
        self.add_output('cost', val=0.0)

    def compute(self, inputs, outputs):
        turbines = np.array([inputs['turbine%s' % xyzt] for xyzt in ['X', 'Y', 'Z', 'Type']]).T
        outputs['cost'] = self.topFarm.optimize_xyz(turbines)[0]


class TurbineTypeProblem(Problem):
    def __init__(self, topFarm, lower, upper, doe_generator):
        Problem.__init__(self)
        indeps = self.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

        turbines = topFarm.turbines
        n_wt = turbines.shape[0]

        lim = np.zeros((n_wt, 2))
        lim[:, 0] += lower
        lim[:, 1] += upper
        assert np.all(lim[:, 0] < lim[:, 1])
        
        self.model.add_constraint('turbineType', lower=lim[:, 0], upper=lim[:, 1])
        indeps.add_output('turbineType', topFarm.turbines[:, 3].astype(np.int))
        self.driver = DOEDriver(doe_generator, )
        self.driver.options['run_parallel'] =  False
        self.model.add_design_var('turbineType', lower=lim[:, 0], upper=lim[:, 1])
        self.model.add_subsystem('cost_comp', TopFarmComponent(topFarm), promotes=['*'])
        self.model.add_objective('cost')
        self.setup()

    def get_DOE_list(self):
        self.driver._set_name()
        case_gen = self.driver.options['generator']
        arr = np.array([c[0][1] for c in case_gen(self.model.get_design_vars(recurse=True), self.model)])
        return np.unique(np.round(arr).astype(np.int), axis=0)


    def optimize(self):
        self.driver.add_recorder(SqliteRecorder("cases2.sql"))
        self.run_driver()
        self.cleanup()
        values = []
        cases = CaseReader("cases1.sql_0").driver_cases
        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            values.append((outputs['x'], outputs['y'], outputs['f_xy']))



def try_me():
    if __name__ == '__main__':
        from topfarm.cost_models.dummy import DummyCostPlotComp, DummyCost

        n_wt = 20
        random_offset = 5
        optimal = [(3, -3), (7, -7), (4, -3), (3, -7), (-3, -3), (-7, -7),
                   (-4, -3), (-3, -7)][:n_wt]
#        optimal = [(3, -3, 5, 1), (7, -7, 5, 1), (4, -3, 5, 1), (3, -7, 5, 1),
#                   (-3, -3, 5, 1), (-7, -7, 5, 1),
#                   (-4, -3, 5, 1), (-3, -7, 5, 1)][:n_wt]
        rotorDiameter = 1.0
        minSpacing = 2.0

        plot_comp = DummyCostPlotComp(optimal)
#        plot_comp.animate = True

        boundary = [[(0, 0), (6, 0), (6, -10), (0, -10)], ]
        tf = TopFarm(optimal, DummyCost(optimal), minSpacing * rotorDiameter,
                     boundary=boundary, plot_comp=plot_comp, record_id=None)
        # topFarm.check()
        tf.shuffle_positions(shuffle_type='abs', offset=random_offset)
        tf.optimize()
        tf.animate()
        tf.clean()


try_me()
