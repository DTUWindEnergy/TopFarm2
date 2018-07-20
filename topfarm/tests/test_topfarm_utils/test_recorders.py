'''
Created on 17. jul. 2018

@author: mmpe
'''
from topfarm.recorders import ListRecorder, TopFarmListRecorder
import numpy as np
from openmdao.drivers.doe_generators import ListGenerator
import pytest
from topfarm.cost_models.dummy import DummyCost
from topfarm._topfarm import TurbineXYZOptimizationProblem,\
    TurbineTypeOptimizationProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from openmdao.drivers.doe_driver import DOEDriver
from numpy import testing as npt
from topfarm.tests.test_files import tfp
import os
from topfarm.tests.test_fuga import test_pyfuga
from topfarm.plotting import PlotComp, NoPlot
from topfarm.constraint_components.boundary_component import BoundaryComp


@pytest.fixture
def tf_generator():
    def tf(xy_boundary=[(0, 0), (4, 4)], z_boundary=(0, 4), xy_boundary_type='square', **kwargs):
        optimal = [(0, 2, 4), (4, 2, 1)]
        xyz = [(0, 1, 0), (1, 1, 1)]
        boundary_comp = BoundaryComp(2, xy_boundary, z_boundary, xy_boundary_type)
        p1 = DummyCost(optimal_state=optimal,
                       inputs=['turbineX', 'turbineY', 'turbineZ'])

        k = {'cost_comp': p1,
             'turbineXYZ': xyz,
             'min_spacing': 2,
             'driver': EasyScipyOptimizeDriver(optimizer='COBYLA', disp=False, maxiter=10),
             }
        k.update(kwargs)
        return TurbineXYZOptimizationProblem(boundary_comp=boundary_comp, **k)

    return tf


def test_ListRecorder():
    from openmdao.api import Problem, IndepVarComp
    from openmdao.test_suite.components.paraboloid import Paraboloid

    prob = Problem()
    model = prob.model

    model.add_subsystem('p1', IndepVarComp('x', 0.), promotes=['*'])
    model.add_subsystem('p2', IndepVarComp('y', 0.), promotes=['*'])
    model.add_subsystem('comp', Paraboloid(), promotes=['*'])

    model.add_design_var('x', lower=-10, upper=10)
    model.add_design_var('y', lower=-10, upper=10)
    model.add_objective('f_xy')

    xyf = [[0.98, 4.30, 74.1844],
           [2.06, 0.90, 23.7476],
           [-1.53, 2.92, 60.9397],
           [-1.25, 7.84, 145.4481]]
    prob.driver = DOEDriver(ListGenerator([[('x', xy[0]), ('y', xy[1])] for xy in xyf]))
    recorder = ListRecorder()
    prob.driver.add_recorder(recorder)

    prob.setup()
    prob.run_driver()
    prob.cleanup()

    cases = recorder.driver_cases
    assert cases.num_cases == 4
    npt.assert_array_equal(recorder.get('counter'), range(1, 5))
    npt.assert_array_almost_equal(recorder.get(['x', 'y', 'f_xy']), xyf, 4)
    for xyf, k in zip(xyf[0], ['x', 'y', 'f_xy']):
        npt.assert_allclose(cases.get_case(0).outputs[k][0], xyf)

    with pytest.raises(KeyError, match="'missing' not found in meta, input or output"):
        recorder.get('missing')


# @pytest.mark.xfail("RuntimeError: Requested MovieWriter (ffmpeg) not available")
# def test_TopFarmListRecorderAnimation(tf_generator):
#     tf = tf_generator()
#     _, _, recorder = tf.optimize()
#     # Generate test file:
#     # recorder.save('topfarm/tests/test_files/recordings/COBYLA_10iter.pkl')
#     fn = tfp + "/tmp/test.mp4"
#     if os.path.exists(fn):
#         os.remove(fn)
#     recorder.animate_turbineXY(duration=5, filename=fn)
#     assert os.path.isfile(fn)


def test_NestedTopFarmListRecorder(tf_generator):
    optimal = [(0, 2, 4, 1), (4, 2, 1, 0)]
    type_lst = [[0, 0],
                [1, 0],
                [0, 1]]
    p1 = DummyCost(optimal_state=optimal,
                   inputs=['turbineX', 'turbineY', 'turbineZ', 'turbineType'])
    p2 = tf_generator(cost_comp=p1,
                      driver=EasyScipyOptimizeDriver(disp=False))

    tf = TurbineTypeOptimizationProblem(
        cost_comp=p2,
        turbineTypes=[0, 0], lower=0, upper=1,
        driver=DOEDriver(ListGenerator([[('turbineType', t)] for t in type_lst])))

    cost, _, recorder = tf.optimize()
    npt.assert_almost_equal(cost, 0)
    npt.assert_array_almost_equal(recorder.get('turbineType'), type_lst)
    npt.assert_array_almost_equal(recorder.get('cost'), [1, 0, 2])

    for sub_rec in recorder.get('recorder'):
        npt.assert_array_almost_equal(sub_rec.get(['turbineX', 'turbineY', 'turbineZ'])[:, -1], np.array(optimal)[:, :3])


@pytest.mark.parametrize('record_id,filename,load_case', [
    ("test", 'recordings/test.pkl', 'latest'),
    ("test.pkl", 'recordings/test.pkl', 'latest'),
    ("recordings/test.pkl", 'recordings/test.pkl', 'latest'),
    ("topfarm/tests/test_files/tmp/test", 'topfarm/tests/test_files/tmp/test.pkl', 'latest'),
    ("topfarm/tests/test_files/tmp/test.pkl", 'topfarm/tests/test_files/tmp/test.pkl', 'latest'),
    ("c:/tmp/test", 'c:/tmp/test.pkl', 'latest'),
    ("c:/tmp/test:latest", 'c:/tmp/test.pkl', 'latest'),
    ("test:", 'recordings/test.pkl', 'latest'),
    ("test:latest", 'recordings/test.pkl', 'latest'),
    ("test:best", 'recordings/test.pkl', 'best'),
    ("test:none", 'recordings/test.pkl', 'none'),
    (None, "", "")
])
def test_recordid2filename(record_id, filename, load_case):
    fn, lc = TopFarmListRecorder().recordid2filename(record_id)
    assert fn == filename
    assert lc == load_case


@pytest.mark.parametrize('load_case,n,cost',
                         [("latest", 10, 3.4273293099380067),
                          ('best', 8, 2.42732931),
                          ("2", 2, 31)])
def test_TopFarmListRecorderLoad(load_case, n, cost):
    fn = tfp + 'recordings/COBYLA_10iter:%s' % load_case
    rec = TopFarmListRecorder().load(fn)
    npt.assert_equal(len(rec.driver_iteration_lst), n)
    npt.assert_almost_equal(rec.get('cost')[-1], cost)


@pytest.mark.parametrize('load_case', [("none"),
                                       ('0')])
def test_TopFarmListRecorderLoad_none(load_case):
    # load case is "none"
    fn = tfp + 'recordings/COBYLA_10iter:%s' % load_case
    rec = TopFarmListRecorder().load(fn)
    assert len(rec.driver_iteration_lst) == 0
    with pytest.raises(ValueError, match="Driver iteration list empty"):
        rec.get('cost')


@pytest.mark.parametrize('fn', [(None),
                                ('MissingFile')])
def test_TopFarmListRecorderLoad_Nothing(fn):
    # No such file
    with pytest.raises(FileNotFoundError, match=r"No such file '.*'"):
        TopFarmListRecorder().load(fn)
    rec = TopFarmListRecorder().load_if_exists(fn)
    assert len(rec.driver_iteration_lst) == 0


@pytest.mark.parametrize('load_case,n_rec,n_fev', [('', 53, 1),
                                                   ('none', 52, 52),
                                                   (40, 73, 33)])
def test_TopFarmListRecorder_continue(tf_generator, load_case, n_rec, n_fev):

    D = 80.0
    D2 = 2 * D + 10
    init_pos = np.array([(0, 2 * D), (0, 0), (0, -2 * D)])
    init_pos[:, 0] += [-40, 0, 40]

    pyFuga = test_pyfuga.get_fuga()(init_pos[:, 0], init_pos[:, 1], wind_atlas='MyFarm/north_pm45_only.lib')
    boundary = [(-D2, -D2), (D2, D2)]
    plot_comp = PlotComp()
    plot_comp = NoPlot()
    tf = TurbineXYZOptimizationProblem(
        cost_comp=pyFuga.get_TopFarm_cost_component(),
        turbineXYZ=init_pos, min_spacing=2 * D,
        boundary_comp=BoundaryComp(len(init_pos),
                                   xy_boundary=boundary, 
                                   z_boundary=None, 
                                   xy_boundary_type='square'),
        driver=EasyScipyOptimizeDriver(tol=1e-10, disp=False),
        plot_comp=plot_comp, record_id=tfp + 'recordings/test_TopFarmListRecorder_continue:%s' % load_case, expected_cost=25)

    _, _, recorder = tf.optimize()
    # recorder.save() # create test file
    npt.assert_equal(recorder.driver_cases.num_cases, n_rec)
    npt.assert_equal(tf.driver.result['nfev'], n_fev)

    tf.plot_comp.show()


@pytest.mark.parametrize('rec_id,sn,fn', [(tfp + "recordings/tmp", None, tfp + "recordings/tmp"),
                                          (None, tfp + "recordings/tmp2", tfp + "recordings/tmp2")])
def test_TopFarmListRecorder_save(tf_generator, rec_id, sn, fn):
    def remove_file():
        if os.path.isfile(fn + ".pkl"):
            os.remove(fn + ".pkl")
    remove_file()
    tf = tf_generator(record_id=rec_id)
    _, _, recorder = tf.optimize()
    recorder.save(sn)
    npt.assert_array_equal(recorder.get('cost'), TopFarmListRecorder().load(fn).get('cost'))
    remove_file()
