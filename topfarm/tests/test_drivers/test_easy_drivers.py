import pytest
import numpy as np
from topfarm.easy_drivers import (
    EasyScipyOptimizeDriver,
    EasySimpleGADriver,
    EasyRandomSearchDriver,
    EasySGDDriver,
)
from topfarm._topfarm import TopFarmProblem
from topfarm.cost_models.dummy import DummyCost


# ---------------------------------------------------------------------------
# EasyScipyOptimizeDriver
# ---------------------------------------------------------------------------

def test_scipy_default_optimizer_is_slsqp():
    drv = EasyScipyOptimizeDriver()
    assert drv.options['optimizer'] == 'SLSQP'


def test_scipy_cobyla_accepted():
    drv = EasyScipyOptimizeDriver(optimizer='COBYLA')
    assert drv.options['optimizer'] == 'COBYLA'


def test_scipy_unsupported_optimizer_raises():
    with pytest.raises(RuntimeError, match="not supported"):
        EasyScipyOptimizeDriver(optimizer='BFGS')


def test_scipy_ipopt_raises():
    with pytest.raises(RuntimeError, match="IPOPT"):
        EasyScipyOptimizeDriver(optimizer='IPOPT')


def test_scipy_sgd_raises():
    with pytest.raises(RuntimeError, match="SGD"):
        EasyScipyOptimizeDriver(optimizer='SGD')


def test_scipy_options_set_correctly():
    drv = EasyScipyOptimizeDriver(maxiter=50, tol=1e-4, disp=False)
    assert drv.options['maxiter'] == 50
    assert drv.options['tol'] == pytest.approx(1e-4)
    assert drv.options['disp'] is False


def test_scipy_get_name():
    drv = EasyScipyOptimizeDriver(optimizer='SLSQP')
    assert drv._get_name() == 'ScipyOptimize_SLSQP'

    drv2 = EasyScipyOptimizeDriver(optimizer='COBYLA')
    assert drv2._get_name() == 'ScipyOptimize_COBYLA'


def test_scipy_auto_scale_flag():
    drv = EasyScipyOptimizeDriver(auto_scale=True)
    assert drv.auto_scale is True

    drv2 = EasyScipyOptimizeDriver(auto_scale=False)
    assert drv2.auto_scale is False


def test_scipy_get_desvar_kwargs_with_bounds():
    drv = EasyScipyOptimizeDriver()
    optimal = np.array([[1, 0]])
    tf = TopFarmProblem(
        design_vars={'x': (optimal[:, 0], 0.0, 5.0)},
        cost_comp=DummyCost(optimal, ['x']),
        driver=drv,
    )
    kwargs = drv.get_desvar_kwargs(tf.model, 'x', [optimal[:, 0], 0.0, 5.0, None])
    assert 'ref0' in kwargs
    assert 'ref' in kwargs


def test_scipy_get_desvar_kwargs_no_bounds():
    drv = EasyScipyOptimizeDriver()
    optimal = np.array([[1, 0]])
    tf = TopFarmProblem(
        design_vars={'x': optimal[:, 0]},
        cost_comp=DummyCost(optimal, ['x']),
        driver=drv,
    )
    kwargs = drv.get_desvar_kwargs(tf.model, 'x', [optimal[:, 0]])
    assert 'lower' not in kwargs or kwargs.get('lower') is None or 'ref0' in kwargs


# ---------------------------------------------------------------------------
# EasySimpleGADriver
# ---------------------------------------------------------------------------

def test_ga_default_options():
    drv = EasySimpleGADriver()
    assert drv.options['max_gen'] == 100
    assert drv.options['pop_size'] == 25
    assert drv.options['Pc'] == pytest.approx(0.5)
    assert drv.options['elitism'] is True


def test_ga_custom_options():
    drv = EasySimpleGADriver(max_gen=50, pop_size=10, Pc=0.3)
    assert drv.options['max_gen'] == 50
    assert drv.options['pop_size'] == 10
    assert drv.options['Pc'] == pytest.approx(0.3)


def test_ga_does_not_support_inequality_constraints():
    drv = EasySimpleGADriver()
    assert drv.supports['inequality_constraints'] is False


def test_ga_does_not_support_equality_constraints():
    drv = EasySimpleGADriver()
    assert drv.supports['equality_constraints'] is False


def test_ga_random_state():
    drv = EasySimpleGADriver(random_state=42)
    assert drv._randomstate == 42


# ---------------------------------------------------------------------------
# EasyRandomSearchDriver
# ---------------------------------------------------------------------------

def test_random_search_driver_instantiation():
    def dummy_randomize(desvar_dict):
        pass

    drv = EasyRandomSearchDriver(randomize_func=dummy_randomize, max_iter=50)
    assert drv.options['max_iter'] == 50


def test_random_search_driver_max_time():
    def dummy_randomize(desvar_dict):
        pass

    drv = EasyRandomSearchDriver(randomize_func=dummy_randomize, max_time=120)
    assert drv.options['max_time'] == 120


# ---------------------------------------------------------------------------
# EasySGDDriver
# ---------------------------------------------------------------------------

def test_sgd_default_options():
    drv = EasySGDDriver()
    assert drv.learning_rate == pytest.approx(10)
    assert drv.beta1 == pytest.approx(0.1)
    assert drv.beta2 == pytest.approx(0.2)


def test_sgd_custom_options():
    drv = EasySGDDriver(learning_rate=5.0, beta1=0.05)
    assert drv.learning_rate == pytest.approx(5.0)
    assert drv.beta1 == pytest.approx(0.05)
