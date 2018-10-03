from topfarm import TopFarm
from topfarm.tests.test_files import xy3tb
from topfarm.cost_models.dummy import DummyCost
from topfarm.tests import npt


def testTopFarm():
    tf = TopFarm(xy3tb.initial,
                 DummyCost(xy3tb.desired, ['x', 'y']),
                 min_spacing=2,
                 boundary=xy3tb.boundary)
    cost, state = tf.evaluate()
    assert cost == 45
    npt.assert_array_equal(state['x'], xy3tb.initial[:, 0])
    npt.assert_array_equal(state['y'], xy3tb.initial[:, 1])
