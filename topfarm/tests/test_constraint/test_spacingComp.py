from topfarm.cost_models.dummy import DummyCost, DummyCostPlotComp
from topfarm.plotting import NoPlot
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.tests.test_files import xy3tb
from topfarm.constraint_components.spacing import SpacingConstraint, SpacingComp
from topfarm.tests import npt
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.api import Problem, IndepVarComp
import numpy as np
from topfarm.utils import SmoothMin, LogSumExpMin, StrictMin
import pytest
from topfarm._topfarm import TopFarmProblem


@pytest.mark.parametrize('aggfunc', [  # None,
    StrictMin(),
    SmoothMin(.1),
    SmoothMin(1),
    SmoothMin(10),
    LogSumExpMin(.1),
    LogSumExpMin(1),
    LogSumExpMin(10)
])
@pytest.mark.parametrize('x,y', [([6, 5, -8, 1], [0, -8, -4, 1]),
                                 ([2.84532167, 7.00331189, 3.86523273], [-2.98101466, -6.99667302, -2.98433316])])
@pytest.mark.parametrize('full_aggregation', [True, False])
def test_spacing_4wt_partials(aggfunc, full_aggregation, x, y):

    from topfarm.constraint_components.boundary import XYBoundaryConstraint
    from topfarm.easy_drivers import EasyScipyOptimizeDriver
    import topfarm
    initial = desired = np.array([x, y]).T
    boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
    spacing_constr = SpacingConstraint(2, aggregation_function=aggfunc, full_aggregation=full_aggregation)

    k = {'cost_comp': DummyCost(desired[:, :2], [topfarm.x_key, topfarm.y_key]),
         'design_vars': {topfarm.x_key: initial[:, 0], topfarm.y_key: initial[:, 1]},
         'driver': EasyScipyOptimizeDriver(disp=True, tol=1e-8),
         'plot_comp': NoPlot(),
         'constraints': [spacing_constr, XYBoundaryConstraint(boundary)]}
    if 0:
        k['plot_comp'] = DummyCostPlotComp(desired)
    TopFarmProblem(**k)
    scomp = spacing_constr.constraintComponent
    outputs = {}

    def compute(x, y):
        scomp.compute(dict(x=x, y=y), outputs)
        return outputs['wtSeparationSquared']

    ref = compute(initial[:, 0], initial[:, 1])
    ddx = np.array([(compute(x, initial[:, 1]) - ref) / 1e-6 for x in initial[:, 0] + np.eye(len(x)) * 1e-6]).T
    ddy = np.array([(compute(initial[:, 0], y) - ref) / 1e-6 for y in initial[:, 1] + np.eye(len(x)) * 1e-6]).T

    scomp.compute_partials(dict(x=initial[:, 0], y=initial[:, 1]), outputs)
    npt.assert_array_almost_equal(outputs[('wtSeparationSquared', 'x')].reshape(ddx.shape), ddx, 4)
    npt.assert_array_almost_equal(outputs[('wtSeparationSquared', 'y')].reshape(ddy.shape), ddy, 4)


@pytest.mark.parametrize('aggfunc', [None,
                                     StrictMin(),
                                     SmoothMin(.1),
                                     SmoothMin(.5),
                                     LogSumExpMin(.1),
                                     LogSumExpMin(1),
                                     LogSumExpMin(10)
                                     ])
def test_spacing_4wt(aggfunc):

    from topfarm.constraint_components.boundary import XYBoundaryConstraint
    from topfarm.easy_drivers import EasyScipyOptimizeDriver
    import topfarm
    initial = np.array([[6, 0], [5, -8], [-1, -4], [1, 1]])  # initial turbine layouts
    desired = np.array([[3, -3], [7, -7], [3, -4], [4, -3]])  # desired turbine layouts
    boundary = np.array([(0, 0), (6, 0), (6, -10), (0, -10)])  # turbine boundaries
    # initial = np.array([[6, 0], [1, 1]])  # initial turbine layouts
    # desired = np.array([[3, -3], [4, -3]])  # desired turbine layouts

    k = {'cost_comp': DummyCost(desired[:, :2], [topfarm.x_key, topfarm.y_key]),
         'design_vars': {topfarm.x_key: initial[:, 0], topfarm.y_key: initial[:, 1]},
         'driver': EasyScipyOptimizeDriver(disp=True, tol=1e-8),
         'plot_comp': NoPlot(),
         'constraints': [SpacingConstraint(2, aggregation_function=aggfunc), XYBoundaryConstraint(boundary)]}
    if 0:
        k['plot_comp'] = DummyCostPlotComp(desired)
    tf = TopFarmProblem(**k)
    print(str(aggfunc))
    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing


@pytest.mark.parametrize('aggfunc', [
    # None,
    StrictMin(), SmoothMin(1), LogSumExpMin(1)])
def test_spacing(aggfunc):
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2, aggregation_function=aggfunc)], plot=False)
    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing


def test_spacing_as_penalty():
    driver = SimpleGADriver()
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2)],
                      driver=driver)

    # check normal result if spacing constraint is satisfied
    assert tf.evaluate()[0] == 45
    # check penalized result if spacing constraint is not satisfied
    assert tf.evaluate({'x': [3, 7, 4.], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == 1e10 + 3


def test_satisfy():
    sc = SpacingComp(n_wt=3, min_spacing=2)
    state = sc.satisfy(dict(zip('xy', xy3tb.desired.T)))
    x, y = state['x'], state['y']
    npt.assert_array_less(y, x)


def test_satisfy2():
    n_wt = 5
    sc = SpacingComp(n_wt=n_wt, min_spacing=2)
    theta = np.linspace(0, 2 * np.pi, n_wt, endpoint=False)
    x0, y0 = np.cos(theta), np.sin(theta)

    state = sc.satisfy({'x': x0, 'y': y0})
    x1, y1 = state['x'], state['y']
    if 0:
        import matplotlib.pyplot as plt
        colors = ['b', 'r', 'm', 'c', 'g', 'y', 'orange', 'indigo', 'grey']
        for i, (x0_, y0_, x1_, y1_) in enumerate(zip(x0, y0, x1, y1)):
            c = colors[i]
            plt.plot([x0_], [y0_], '>', color=c)
            plt.plot([x0_, x1_], [y0_, y1_], '-', color=c, label=i)
            plt.plot([x1_], [y1_], '.', color=c)
        plt.axis('equal')
        plt.legend()
        plt.show()

    dist = np.sqrt(sc._compute(x1, y1))
    npt.assert_array_less(2, dist)


@pytest.mark.parametrize('aggfunc', [None, StrictMin(), SmoothMin(1), LogSumExpMin(1)])
def test_partials(aggfunc):
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2, aggregation_function=aggfunc)])
    # if complex numbers work: uncomment tf.setup below and
    # change method='cs' and step=1e-40 in check_partials
    # tf.setup(force_alloc_complex=True)

    # run to get rid of zeros initializaiton, otherwise not accurate partials
    tf.run_model()
    check = tf.check_partials(compact_print=True,
                              includes='spacing*',
                              method='fd',
                              step=1e-6,
                              form='central')
    atol = 1.e-6
    rtol = 1.e-6
    try:
        assert_check_partials(check, atol, rtol)
    except ValueError as err:
        print(str(err))
        raise


@pytest.mark.parametrize('aggfunc', [None,
                                     # StrictMin(), # not working
                                     SmoothMin(1), LogSumExpMin(1)])
def test_partials_many_turbines(aggfunc):
    n_wt = 10
    theta = np.linspace(0, 2 * np.pi, n_wt, endpoint=False)
    sc = SpacingComp(n_wt=n_wt, min_spacing=2, const_id="", aggregation_function=aggfunc)
    tf = Problem()
    ivc = IndepVarComp()
    ivc.add_output('x', val=np.cos(theta))
    ivc.add_output('y', val=np.sin(theta))
    tf.model.add_subsystem('ivc', ivc, promotes=['*'])
    tf.model.add_subsystem('sc', sc, promotes=['x', 'y', 'wtSeparationSquared'])
    tf.setup()
    tf.run_model()

    check = tf.check_partials(compact_print=True,
                              includes='sc*',
                              method='fd',
                              step=1e-6,
                              form='central')

    fil = {'sc': {key: val for key, val in check['sc'].items()
                  if 'constraint_violation' not in key[0]}}

    atol = 1.e-6
    rtol = 1.e-6
    try:
        assert_check_partials(fil, atol, rtol)
    except ValueError as err:
        print(str(err))
        raise
