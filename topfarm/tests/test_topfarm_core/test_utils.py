# # -*- coding: utf-8 -*-
# from topfarm import TopFarm
# import numpy as np
# from topfarm.cost_models.cost_model_wrappers import CostModelComponent
# import pytest
# import os
# 
# from topfarm.utils import pos_from_case, latest_id, _shuffle_positions_abs
# 
# thisdir = os.path.dirname(os.path.abspath(__file__))
# turbines = np.array([[ 2.49998371, -2.99999965],
#        [ 6.        , -6.99999467],
#        [ 4.49998371, -3.00002279],
#        [ 3.00001007, -7.00001197]])
# x = np.array([-0.5463264 ,  0.4158521 ,  1.50479727,  3.04121982,  0.82494571,
#             1.48072571,  0.03939927,  2.27593243, -0.18551361,  0.24885285,
#             1.12706339,  2.25472924,  0.04329133,  0.292686  ,  5.18916103,
#             1.76294032,  6.96910295,  4.80383887,  5.93002915,  6.07458626])
# 
# y = np.array([-4.28630451, -1.03701919, -6.11562032,  0.60293213, -6.83330699,
#            -1.9655984 , -7.06706521, -3.56006813,  0.70979837, -2.17497837,
#             0.94819493, -1.94630408, -6.75376048, -6.97213247, -7.11506022,
#            -6.99383667,  0.63581096, -4.57807581, -2.76544057, -8.85507948])
# 
# boundary = [(0, 0), (6, 1), (7, -11), (-1, -10)]
# n_wt, n_iter, step_size, min_space, pad, plot, verbose = \
#     (20, 1000, 0.1, 2.1, 1.01, False, False)
# turbines2_ref = np.array([[-0.53056298, -5.34414632],
#        [ 1.72713409, -1.7339491 ],
#        [ 3.90444365, -5.82606831],
#        [ 4.49089193,  0.74539327],
#        [ 1.26552562, -6.75507526],
#        [ 1.45466343, -4.37313716],
#        [-0.99102804, -9.94909137],
#        [ 3.8725124 , -3.68277265],
#        [-0.02498702, -0.28473405],
#        [-0.25686194, -2.60098671],
#        [ 2.30453881,  0.38010616],
#        [ 3.97302933, -1.39592567],
#        [-0.74326151, -7.47172573],
#        [ 1.4151898 , -9.31374061],
#        [ 5.65712132, -7.44164592],
#        [ 3.56099506, -8.48762334],
#        [ 6.12959435, -0.59355499],
#        [ 6.27224495, -5.30888086],
#        [ 6.34616645, -3.15591179],
#        [ 6.42858562, -9.90795045]])
# 
# def testpos_from_case():
#     crf = "../test_files/recordings/cases_20180703_152607.sql"
#     path = os.path.join(thisdir, crf)
#     np.testing.assert_allclose(turbines, pos_from_case(path))
# 
# 
# def testlatest_id():
#     crd = "../test_files/recordings"
#     path = os.path.join(thisdir, crd)
#     ref_path = os.path.join(path,'cases_20180703_152607.sql')
#     assert latest_id(path) == ref_path
# 
# def test_shuffle_positions_abs():
#     turbines2 = _shuffle_positions_abs(x, y, boundary, n_wt, n_iter, step_size,
#                                  min_space, pad, plot, verbose)
#     np.testing.assert_allclose(turbines2, turbines2_ref)
# 
# if __name__ == '__main__':
# #    testpos_from_case()
# #    testlatest_id()
# #    test_shuffle_positions_abs()
#     pass