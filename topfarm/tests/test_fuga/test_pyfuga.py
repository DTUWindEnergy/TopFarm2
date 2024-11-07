import os
import threading
import time

import pytest

import numpy as np
import topfarm
from topfarm.cost_models import fuga
from topfarm.tests import uta
from topfarm.plotting import NoPlot
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm._topfarm import TopFarmProblem


def check_lib_exists():
    try:
        import py_colonel
    except ModuleNotFoundError:
        pytest.xfail("Colonel submodule not found\n")
    from py_colonel.py_colonel_lib import fugalib_path
    if os.path.isfile(fugalib_path) is False:
        pytest.xfail("Fugalib '%s' not found\n" % fugalib_path)


def _fuga(tb_x=[423974, 424033], tb_y=[6151447, 6150889], wind_atlas='MyFarm/north_pm15_only.lib'):
    check_lib_exists()
    from topfarm.cost_models.fuga.py_fuga import PyFuga, fuga_path
    pyFuga = PyFuga()
    pyFuga.setup(farm_name='Horns Rev 1',
                 turbine_model_path=fuga_path + 'LUTs-T/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=70.00]',
                 tb_x=tb_x, tb_y=tb_y,
                 mast_position=(0, 0, 70), z0=0.03, zi=400, zeta0=0,
                 farms_dir=fuga_path + 'LUTs-T/Farms/', wind_atlas_path=wind_atlas, climate_interpolation=False)
    return pyFuga


@pytest.fixture
def get_fuga():
    return _fuga


@pytest.fixture
def pyFuga():
    return _fuga()


def _test_parallel(i):
    pyFuga = get_fuga()()
    print(pyFuga.stdout_filename, i)
    for _ in range(1):
        print(threading.current_thread(), i)
        np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 0], [0, 200]]).T), [
                                             12.124883, 14.900544, 0.3458, 0.813721])
        time.sleep(1)


def testCheckVersion(get_fuga):
    check_lib_exists()
    from py_colonel.py_colonel_lib import fugalib_path, PascalDLL
    from topfarm.cost_models.fuga.py_fuga import fuga_path
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
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0], [0]]).T), [
                                         8.2896689155874324, 8.2896689155874324, 0.472841, 1.])
    pyFuga.cleanup()


def testAEP(pyFuga):
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 1000], [0, 0]]).T), [
                                         2 * 8.2896689155874324, 2 * 8.2896689155874324, 0.472841, 1.])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 1000], [0, 0]]).T), [[0, 0],
                                                                                                     [0, 0],
                                                                                                     [1.047718e-002, 9.801237e-003]])
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 0], [0, 200]]).T), [
                                         14.472994, 16.579338, 0.412768, 0.872954])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 0], [0, 200]]).T), [[0, 0],
                                                                                                    [-7.02042144e-03,
                                                                                                        7.02042144e-03],
                                                                                                    [3.099291e-02, -1.459773e-02]])
    np.testing.assert_array_almost_equal(pyFuga.get_aep(np.array([[0, 200], [0, 200]]).T), [
                                         16.543667, 16.579338, 0.471824, 0.997848])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(np.array([[0, 200], [0, 200]]).T), [[-1.679974e-05, 1.679974e-05],
                                                                                                      [7.255895e-06, -7.255895e-06],
                                                                                                      [2.002942e-02, 3.759327e-06]])
    pyFuga.cleanup()


def testLargeOffset(pyFuga):
    o = 1.e16
    np.testing.assert_array_almost_equal(pyFuga.get_aep(
        np.array([[0 + o, 0 + o], [0 + o, 200 + o]]).T), [14.472994, 16.579338, 0.412768, 0.872954])
    np.testing.assert_array_almost_equal(pyFuga.get_aep_gradients(), [[0, 0],
                                                                      [-7.02042144e-03, 7.02042144e-03],
                                                                      [3.099291e-02, -1.459773e-02]])


def get_tf(init_pos, pyFuga, boundary, boundary_type='convex_hull'):
    return TopFarmProblem(
        dict(zip('xy', init_pos.T)),
        pyFuga.get_TopFarm_cost_component(),
        constraints=[SpacingConstraint(160),
                     XYBoundaryConstraint(boundary, boundary_type)],
        driver=EasyScipyOptimizeDriver(disp=False))


def testAEP_topfarm(get_fuga):
    pyFuga = get_fuga()
    init_pos = np.array([[0, 0], [1000, 0]])
    tf = get_tf(init_pos, pyFuga, init_pos, 'square')
    cost, _ = tf.evaluate()
    np.testing.assert_array_almost_equal(cost, -16.579337831174865)


def test_pyfuga_cmd():
    check_lib_exists()
    from topfarm.cost_models.fuga.py_fuga import PyFuga
    pyFuga = PyFuga()
    pyFuga.execute(r'echo "ColonelInit"')
    assert pyFuga.log.strip().split("\n")[-1] == 'ColonelInit'


def testAEP_topfarm_optimization_4tb(get_fuga):
    D = 80.0
    B = 3 * D + 10
    init_pos = np.array([(0, 3 * D), (0, D), (0, -D), (0, -3 * D)])
    init_pos[:, 0] += [-80, 0, 0, 80]

    wind_atlas = 'MyFarm/north_pm45_only.lib'
    pyFuga = get_fuga(init_pos[:1, 0], init_pos[:1, 1], wind_atlas=wind_atlas)
    AEP_pr_tb = pyFuga.get_aep()[1]
    pyFuga = get_fuga(init_pos[:, 0], init_pos[:, 1], wind_atlas=wind_atlas)
    boundary = [(-B, B), (B, B), (B, -B), (-B, -B), (-B, B)]

    plot_comp = NoPlot()
    # plot_comp= PlotComp()
    tf = get_tf(init_pos, pyFuga, boundary)
    cost, _, rec = tf.optimize()
    uta.assertAlmostEqual(-cost, AEP_pr_tb * 4, delta=.2)

    tf.plot_comp.show()

#     # Plot wake map
#     f, a, k = lib_reader.read_lib(fuga_path + 'LUTs-T/Farms/' + wind_atlas)
#     wr = WindResource(f, a, k, np.zeros_like(k))
#     pyFuga.plot_wind_field_with_boundary(10, zip(range(360)[::2], wr.f[::2] * 2), 'XY', 70, boundary)


@pytest.mark.parametrize('scale', [(.001),
                                   (1),
                                   (1000)])
def testAEP_topfarm_optimization_2tb_scale(get_fuga, scale):
    D = 80.0
    B = 2 * D + 10
    init_pos = np.array([(-10, 1 * D), (10, - D)])

    wind_atlas = 'MyFarm/north_pm30_only.lib'
    pyFuga = get_fuga(init_pos[:1, 0], init_pos[:1, 1], wind_atlas=wind_atlas)
    AEP_pr_tb = pyFuga.get_aep()[1]
    pyFuga = get_fuga(init_pos[:, 0], init_pos[:, 1], wind_atlas=wind_atlas)
    boundary = [(-B, B), (B, B), (B, -B), (-B, -B), (-B, B)]

    plot_comp = NoPlot()
    # plot_comp = PlotComp()

    cost_comp = AEPCostModelComponent('xy', init_pos.shape[0],
                                      lambda x, y: scale * pyFuga.get_aep(np.array([x, y]).T)[0],  # only aep
                                      lambda x, y: scale * pyFuga.get_aep_gradients(np.array([x, y]).T)[:2])  # only dAEPdx and dAEPdy

    tf = TopFarmProblem(
        dict(zip('xy', init_pos.T)),
        cost_comp,
        constraints=[SpacingConstraint(2 * D),
                     XYBoundaryConstraint(boundary)],
        plot_comp=plot_comp,
        driver=EasyScipyOptimizeDriver(tol=1e-8, disp=False),
        expected_cost=AEP_pr_tb * 2 * scale)
    cost, _, rec = tf.optimize()
    tf.plot_comp.show()
    uta.assertAlmostEqual(-cost / scale, AEP_pr_tb * 2, delta=.02)


#     # Plot wake map
#     f, a, k = lib_reader.read_lib(fuga_path + 'LUTs-T/Farms/' + wind_atlas)
#     wr = WindResource(f, a, k, np.zeros_like(k))
#     #pyFuga.plot_wind_field_with_boundary(10, zip(range(360)[::2], wr.f[::2] * 2), 'XY', 70, boundary)
#     pyFuga.plot_wind_field_with_boundary(10, 0, 'XY', 70, boundary)

# @pytest.mark.xfail
# def test_parallel():
#     from multiprocessing import Pool
#     with Pool(5) as p:
#         print(p.map(_test_parallel, [1, 2]))
