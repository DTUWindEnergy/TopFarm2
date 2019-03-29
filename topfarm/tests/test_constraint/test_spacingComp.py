from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.tests.test_files import xy3tb
from topfarm.constraint_components.spacing import SpacingConstraint, SpacingComp
from topfarm.tests import npt
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.api import Problem, IndepVarComp
import numpy as np


def test_spacing():
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2)])
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


def test_partials():
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2)])
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


def test_partials_many_turbines():
    n_wt = 10
    theta = np.linspace(0, 2 * np.pi, n_wt, endpoint=False)
    sc = SpacingComp(n_wt=n_wt, min_spacing=2, const_id="")
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
                  if 'penalty' not in key[0]}}

    atol = 1.e-6
    rtol = 1.e-6
    try:
        assert_check_partials(fil, atol, rtol)
    except ValueError as err:
        print(str(err))
        raise


if __name__ == '__main__':
    test_satisfy2()
