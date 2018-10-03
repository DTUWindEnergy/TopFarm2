from topfarm.cost_models.dummy import DummyCost
from topfarm.plotting import NoPlot
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from topfarm.tests.test_files import xy3tb
from topfarm.constraint_components.spacing import SpacingConstraint, SpacingComp
from topfarm.tests import npt


def test_spacing():
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2)])
    tf.optimize()
    tb_pos = tf.turbine_positions[:, :2]
    tf.plot_comp.show()
    tol = 1e-4
    assert sum((tb_pos[2] - tb_pos[0])**2) > 2**2 - tol  # check min spacing


def test_spacing_as_penalty():
    tf = xy3tb.get_tf(constraints=[SpacingConstraint(2)],
                      driver=SimpleGADriver())

    # check normal result if spacing constraint is satisfied
    assert tf.evaluate()[0] == 45
    # check penalized result if spacing constraint is not satisfied
    assert tf.evaluate({'x': [3, 7, 4.], 'y': [-3., -7., -3.], 'z': [0., 0., 0.]})[0] == 1e10 + 3


def test_satisfy():
    sc = SpacingComp(n_wt=3, min_spacing=2)
    state = sc.satisfy(dict(zip('xy', xy3tb.desired.T)))
    x, y = state['x'], state['y']
    npt.assert_array_less(y, x)
