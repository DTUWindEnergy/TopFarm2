'''
Created on 17. jul. 2018

@author: mmpe
'''
from topfarm.recorders import TopFarmListRecorder, split_record_id, \
    recordid2filename
import numpy as np
from openmdao.drivers.doe_generators import ListGenerator
import pytest
from topfarm.cost_models.dummy import DummyCost
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from openmdao.drivers.doe_driver import DOEDriver
from numpy import testing as npt
from topfarm.tests.test_files import tfp
import os
from topfarm.tests.test_fuga import test_pyfuga
from topfarm.plotting import PlotComp, NoPlot, XYPlotComp
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm._topfarm import TopFarmProblem
from topfarm.tests.test_fuga.test_pyfuga import get_fuga
import subprocess


@pytest.fixture
def tf_generator():
    def tf(xy_boundary=[(0, 0), (4, 4)], z_boundary=(0, 4), xy_boundary_type='square', **kwargs):
        optimal = [(0, 2, 4), (4, 2, 1)]
        xyz = np.array([(0, 1, 0), (1, 1, 1)])
        p1 = DummyCost(optimal, 'xyz')
        design_vars = dict(zip('xy', xyz.T))
        design_vars['z'] = (xyz[:, 2], z_boundary[0], z_boundary[1])
        k = {'design_vars': design_vars,
             'cost_comp': p1,
             'driver': EasyScipyOptimizeDriver(optimizer='COBYLA', disp=False, maxiter=10),
             }
        k.update(kwargs)

        return TopFarmProblem(
            constraints=[XYBoundaryConstraint(xy_boundary, xy_boundary_type),
                         SpacingConstraint(2)],
            **k
        )

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
    recorder = TopFarmListRecorder()
    recorder._initialize_database()
    recorder._cleanup_abs2meta()
    # recorder.record_iteration_problem(None, None, None)
    # recorder.record_iteration_system(None, None, None)
    # recorder.record_iteration_solver(None, None, None)
    recorder.record_viewer_data(None)
    recorder.record_metadata_solver(None)
    recorder.record_derivatives_driver(None, None, None)
    recorder.shutdown()

    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.recording_options['includes'] = ['*']
    prob.driver.recording_options['record_inputs'] = True

    prob.setup()
    prob.run_driver()
    prob.cleanup()

    assert recorder.num_cases == 4

    npt.assert_array_equal(recorder.get('counter'), range(1, 5))
    npt.assert_array_equal(recorder['counter'], range(1, 5))

    npt.assert_array_almost_equal(recorder.get(['x', 'y', 'f_xy']), xyf, 4)
    for xyf, k in zip(xyf[0], ['x', 'y', 'f_xy']):
        npt.assert_allclose(recorder[k][0], xyf)

    with pytest.raises(KeyError, match="missing"):
        recorder.get('missing')

    npt.assert_array_equal(recorder.time, recorder['timestamp'] - recorder['timestamp'][0])


def test_TopFarmListRecorderAnimation(tf_generator):
    try:
        from matplotlib import animation
        animation.writers['ffmpeg']
    except Exception:
        pytest.xfail("No matplotlib, animation or ffmpeg writer")

    tf = tf_generator()
    _, _, recorder = tf.optimize()
    # # Generate test file:
#    recorder.save('topfarm/tests/test_files/recordings/COBYLA_10iter.pkl')
    fn = tfp + "/tmp/test.mp4"
    if os.path.exists(fn):
        os.remove(fn)
    recorder.animate_turbineXY(duration=5, filename=fn)
    assert os.path.isfile(fn)


def test_NestedTopFarmListRecorder(tf_generator):
    optimal = [(0, 2, 4, 1), (4, 2, 1, 0)]
    type_lst = [[0, 0],
                [1, 0],
                [0, 1]]
    p1 = DummyCost(optimal_state=optimal,
                   inputs=['x', 'y', 'z', 'type'])
    p2 = tf_generator(cost_comp=p1,
                      driver=EasyScipyOptimizeDriver(disp=False))

    tf = TopFarmProblem(
        {'type': ([0, 0], 0, 1)},
        cost_comp=p2,
        driver=DOEDriver(ListGenerator([[('type', t)] for t in type_lst])))

    cost, _, recorder = tf.optimize()
    npt.assert_almost_equal(cost, 0)
    npt.assert_array_almost_equal(recorder.get('type'), type_lst)
    npt.assert_array_almost_equal(recorder.get('cost'), [1, 0, 2])

    for sub_rec in recorder.get('recorder'):
        npt.assert_array_almost_equal(np.array([sub_rec[k][-1] for k in ['x', 'y', 'z']]).T, np.array(optimal)[:, :3])


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
    ("test:5", 'recordings/test.pkl', '5'),
    ("test:none", 'recordings/test.pkl', 'none'),
    (None, "", "")
])
def test_recordid2filename(record_id, filename, load_case):
    fn, lc = TopFarmListRecorder().recordid2filename(record_id)
    assert fn == filename
    assert lc == load_case
    fn, lc = recordid2filename(record_id)
    assert fn == filename
    assert lc == load_case


@pytest.mark.parametrize('record_id,record_name,load_case', [
    ("test", 'test', 'latest'),
    ("test.pkl", 'test.pkl', 'latest'),
    ("recordings/test.pkl", 'recordings/test.pkl', 'latest'),
    ("topfarm/tests/test_files/tmp/test", 'topfarm/tests/test_files/tmp/test', 'latest'),
    ("topfarm/tests/test_files/tmp/test.pkl", 'topfarm/tests/test_files/tmp/test.pkl', 'latest'),
    ("c:/tmp/test", 'c:/tmp/test', 'latest'),
    ("c:/tmp/test:latest", 'c:/tmp/test', 'latest'),
    ("test:", 'test', 'latest'),
    ("test:latest", 'test', 'latest'),
    ("test:best", 'test', 'best'),
    ("test:none", 'test', 'none'),
    (None, "", "")
])
def test_split_recordid(record_id, record_name, load_case):
    n, lc = TopFarmListRecorder().split_record_id(record_id)
    assert n == record_name
    assert lc == load_case
    n, lc = split_record_id(record_id)
    assert n == record_name
    assert lc == load_case


@pytest.mark.parametrize('load_case,n,cost',
                         [("latest", 11, 2.6187984),
                          ('best', 11, 2.6187984),
                          ("3", 3, 28)])
def test_TopFarmListRecorderLoad(load_case, n, cost):
    fn = tfp + 'recordings/COBYLA_10iter:%s' % load_case
    rec = TopFarmListRecorder().load(fn)
    npt.assert_equal(rec.num_cases, n)
    npt.assert_almost_equal(rec.get('cost')[-1], cost)


@pytest.mark.parametrize('load_case', [("none"),
                                       ('0')])
def test_TopFarmListRecorderLoad_none(load_case):
    # load case is "none"
    fn = tfp + 'recordings/COBYLA_10iter:%s' % load_case
    rec = TopFarmListRecorder().load(fn)
    assert rec.num_cases == 0


@pytest.mark.parametrize('fn', [(None),
                                ('MissingFile')])
def test_TopFarmListRecorderLoad_Nothing(fn):
    # No such file
    with pytest.raises(FileNotFoundError, match=r"No such file '.*'"):
        TopFarmListRecorder().load(fn)
    rec = TopFarmListRecorder().load_if_exists(fn)
    assert rec.num_cases == 0


@pytest.mark.parametrize('load_case,n_rec,n_fev', [('', 55, 1),
                                                   ('none', 53, 52),
                                                   (40, 207, 166)])
def test_TopFarmListRecorder_continue(tf_generator, load_case, n_rec, n_fev, get_fuga):

    D = 80.0
    D2 = 2 * D + 10
    init_pos = np.array([(0, 2 * D), (0, 0), (0, -2 * D)])
    init_pos[:, 0] += [-40, 0, 40]

    pyFuga = get_fuga(init_pos[:, 0], init_pos[:, 1], wind_atlas='MyFarm/north_pm45_only.lib')
    boundary = [(-D2, -D2), (D2, D2)]
    plot_comp = XYPlotComp()
    plot_comp = NoPlot()
    tf = TopFarmProblem(
        dict(zip('xy', init_pos.T)),
        cost_comp=pyFuga.get_TopFarm_cost_component(),
        constraints=[SpacingConstraint(2 * D),
                     XYBoundaryConstraint(boundary, 'square')],
        driver=EasyScipyOptimizeDriver(tol=1e-10, disp=False),
        plot_comp=plot_comp,
        record_id=tfp + 'recordings/test_TopFarmListRecorder_continue:%s' % load_case,
        expected_cost=25)

    _, _, recorder = tf.optimize()
    # Create test file:
    # 1) delete file "test_files/recordings/test_TopFarmListRecorder_continue"
    # 2) Uncomment line below, run and recomment
#    if load_case == "": recorder.save()  # create test file
    npt.assert_equal(recorder.num_cases, n_rec)
    npt.assert_equal(tf.driver.result['nfev'], n_fev)

    tf.plot_comp.show()


def test_TopFarmListRecorder_continue_wrong_recorder(tf_generator):

    tf = TopFarmProblem(
        {'type': ([0, 0, 0], 0, 1)},
        cost_comp=DummyCost(np.array([[0, 1, 0]]).T, ['type']),
        driver=EasyScipyOptimizeDriver(disp=False),
        record_id=tfp + 'recordings/test_TopFarmListRecorder_continue:latest'
    )

    tf.optimize()
    assert 'type' in tf.recorder.keys()
    assert 'x' not in tf.recorder.keys()


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


@pytest.mark.skip(reason="mongodb support is being dropped")
@pytest.mark.parametrize('dn, ci, cu', [('data22', 'test', True), ])
def test_MongoRecorder(tf_generator, dn, ci, cu):
    # subprocess.Popen(['mongod'])
    # tf = tf_generator(recorder=MongoRecorder(db_name=dn, case_id=ci, clean_up=cu))
    # _, _, recorder = tf.optimize()
    # recorder.animate_turbineXY(duration=10, tail=5, cost='cost', anim_options={'interval': 20, 'blit': True})
    pass
