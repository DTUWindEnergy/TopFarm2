import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import wt_x, wt_y
from topfarm.tests import npt
from topfarm.cost_models.utils.spanning_tree import mst, spanning_tree


def test_spanning_tree():
    x9, y9 = np.array([[50, 90],
                       [100, 150],
                       [200, 150],
                       [300, 140],
                       [340, 110],
                       [300, 55],
                       [200, 50],
                       [100, 50],
                       [160, 110]]).T

    for x, y, ref in [(x9, y9, 561.0566071630595), (wt_x, wt_y, 44232.604068779656)]:
        x = np.asarray(x)
        y = np.asarray(y)
        sp1 = mst(x, y)
        sp2 = spanning_tree(x, y)
        if 0:
            for sp in [sp1, sp2]:
                plt.figure()
                plt.plot(x, y, '.')
                s = []
                for i in sp.keys():
                    if i[0] < i[1] and i[::-1] in sp:
                        continue
                    plt.plot(x[np.array(i)], y[np.array(i)], 'k-')
                    s.append(np.hypot(np.diff(x[np.array(i)]), np.diff(y[np.array(i)])))
                plt.title("%s, %s" % (sum(sp.values()), np.sum(s)))

            plt.show()
        npt.assert_almost_equal(sum(sp1.values()), ref)
        npt.assert_almost_equal(sum([v for k, v in sp2.items() if k[::-1] not in sp2 or k[0] < k[1]]), ref)
