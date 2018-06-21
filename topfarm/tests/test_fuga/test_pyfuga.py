import threading
import time
import numpy as np
import pytest
from topfarm.cost_models.fuga.pascal_dll import PascalDLL
from topfarm.cost_models.fuga.py_fuga import PyFuga, fugalib_path
import os
from topfarm.cost_models.fuga import py_fuga
from topfarm import TopFarm


fuga_path = os.path.abspath(os.path.dirname(py_fuga.__file__)) + '/Colonel/'


def check_lib_exists():
    if os.path.isfile(fugalib_path) is False:
        pytest.xfail("Fugalib '%s' not found\n" % fugalib_path)


@pytest.fixture
def get_fuga():
    def _fuga(tb_x=[423974, 424033], tb_y=[6151447, 6150889]):
        check_lib_exists()
        pyFuga = PyFuga()
        pyFuga.setup(farm_name='Horns Rev 1',
                     turbine_model_path=fuga_path + 'LUTs-T/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=70.00]',
                     tb_x=tb_x, tb_y=tb_y,
                     mast_position=(0, 0, 70), z0=0.03, zi=400, zeta0=0,
                     farms_dir=fuga_path + 'LUTs-T/Farms/', wind_atlas_path='MyFarm/north_pm30_only.lib', climate_interpolation=False)
        return pyFuga
    return _fuga


@pytest.fixture
def pyFuga():
    return get_fuga()()


def _test_parallel(i):
    pyFuga = get_fuga()()
    print(pyFuga.stdout_filename, i)
    for _ in range(1):
        print(threading.current_thread(), i)
        np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 0], [0, 200]]).T), [12.124883, 14.900544, 0.3458, 0.813721])
        time.sleep(1)


def testCheckVersion(get_fuga):
    check_lib_exists()
    lib = PascalDLL(fuga_path + "FugaLib/FugaLib.%s" % ('so', 'dll')[os.name == 'nt'])
    with pytest.raises(Exception, match="This version of FugaLib supports interface version "):
        lib.CheckInterfaceVersion(1)
    pyFuga = get_fuga()  # check that current interface version match
    pyFuga.cleanup()


def testSetup(get_fuga):
    pyFuga = get_fuga()
    assert pyFuga.get_no_turbines() == 2
    assert "Loading" in pyFuga.log

    # check that new setup resets number of turbines
    pyFuga = get_fuga()
    assert pyFuga.get_no_turbines() == 2
    pyFuga.cleanup()


def testAEP_one_tb(get_fuga):
    pyFuga = get_fuga([0], [0])
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0], [0]]).T), [8.2896689155874324, 8.2896689155874324, 0.472841, 1.])
    pyFuga.cleanup()


def testAEP(pyFuga):
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 1000], [0, 0]]).T), [
                                         2 * 8.2896689155874324, 2 * 8.2896689155874324, 0.472841, 1.])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 1000], [0, 0]]).T), 0)
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 0], [0, 200]]).T), [14.688347, 16.579338,  0.41891,  0.885943])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 0], [0, 200]]).T), [[-0.003789,  0.003789],
                                                                                                    [-0.007204,  0.007204],
                                                                                                    [0.,  0.]])
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 200], [0, 200]]).T), [20.352901, 16.579338,  0.580462,  1.227606])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 200], [0, 200]]).T), [[-2.033273e-05,  2.033273e-05],
                                                                                                      [7.255895e-06, -7.255895e-06],
                                                                                                      [0.000000e+00,  0.000000e+00]])
    pyFuga.cleanup()


def testLargeOffset(pyFuga):
    o = 1.e16
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0 + o, 0 + o], [0 + o, 200 + o]]).T), [14.688347, 16.579338,  0.41891,  0.885943])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(), [[-0.003789,  0.003789],
                                                                      [-0.007204,  0.007204],
                                                                      [0.,  0.]])


def testAEP_topfarm(get_fuga):
    pyFuga = get_fuga()
    init_pos = [[0, 0], [1000, 0]]
    tf = TopFarm(init_pos, pyFuga.get_TopFarm_cost_component(), 160, init_pos, boundary_type='square')
    tf.evaluate()
    np.testing.assert_array_almost_equal(tf.get_cost(), -16.579337831174865)


def test_pyfuga_cmd():
    check_lib_exists()
    pyFuga = PyFuga()
    pyFuga.execute(r'echo "ColonelInit"')
    assert pyFuga.log.strip().split("\n")[-1] == 'ColonelInit'


# @pytest.mark.xfail
# def test_parallel():
#     from multiprocessing import Pool
#     with Pool(5) as p:
#         print(p.map(_test_parallel, [1, 2]))
