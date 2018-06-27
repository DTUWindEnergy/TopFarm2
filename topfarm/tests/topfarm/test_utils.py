# -*- coding: utf-8 -*-
from topfarm import TopFarm
import numpy as np
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
import pytest
import os

from topfarm.utils import pos_from_case, latest_id

thisdir = os.path.dirname(os.path.abspath(__file__))
turbines = np.array([[ 2.4999377 , -2.99987763],
                   [ 6.        , -6.99997496],
                   [ 4.49993771, -2.99985273],
                   [ 3.00004123, -6.9999519 ]])

def test_pos_from_case():
    crf = "../test_files/recordings/cases_20180621_111710.sql"
    path = os.path.join(thisdir, crf)
    np.testing.assert_allclose(turbines, pos_from_case(path))


def test_latest_id():
    crd = "../test_files/recordings"
    path = os.path.join(thisdir, crd)
    ref_path = os.path.join(path,'cases_20180621_111710.sql')
    assert latest_id(path) == ref_path


if __name__ == '__main__':
    test_pos_from_case()
    test_latest_id()
#    pass