import numpy as np
import pytest

from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import (
    PyWakeAEPCostModelComponentAdditionalTurbines,
    PyWakeAEPCostModelComponent,
)
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot
from topfarm.tests import npt

from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import wt_x, wt_y, LillgrundSite, SWT2p3_93_65


def test_PyWakeAEPCostModelComponentAdditionalTurbines():
    x2 = np.array([363089.20620581, 362841.19815026])
    y2 = np.array([6154000, 6153854.5244973])
    wind_turbines = SWT2p3_93_65()
    x = wt_x[:4]
    y = wt_y[:4]
    n_wt = len(x)
    site = LillgrundSite()
    wf_model = BastankhahGaussian(site, wind_turbines)
    constraint_comp = XYBoundaryConstraint(np.asarray([x, y]).T)
    cost_comp = PyWakeAEPCostModelComponentAdditionalTurbines(
        windFarmModel=wf_model,
        n_wt=n_wt,
        add_wt_x=x2,
        add_wt_y=y2,
        grad_method=autograd,
    )
    plot_comp = NoPlot()
    problem = TopFarmProblem(
        design_vars={"x": x, "y": y},
        constraints=[
            constraint_comp,
            SpacingConstraint(min_spacing=wind_turbines.diameter() * 2),
        ],
        cost_comp=cost_comp,
        driver=EasyScipyOptimizeDriver(optimizer="SLSQP", maxiter=5),
        plot_comp=plot_comp,
    )

    cost, _, _ = problem.optimize(disp=True)
    npt.assert_almost_equal(cost, -3682.710308568642)


design_vars = {
    "x": np.array([0, 1]),
    "y": np.array([0, 1]),
    "z": np.array([10, 10]),
    "type": np.array([0, 0]),
}
UNHANDLED_PYWAKE_ERROR_MSG = "Unhandeled PyWake error"


class DummyWindTurbines:
    def hub_height(self):
        return 10


class DummyWindFarmModel:
    def __init__(self, mode="normal"):
        # mode: "normal", "raise_same", "raise_other"
        self.mode = mode
        self.windTurbines = DummyWindTurbines()

    def aep(self, x, y, h, type, wd, ws, n_cpu):
        if self.mode == "raise_same":
            raise ValueError("Error: turbines are at the same position")
        elif self.mode == "raise_other":
            raise ValueError(UNHANDLED_PYWAKE_ERROR_MSG)
        else:
            return 100

    # Minimal dummy implementations for gradients, not used in these tests
    def aep_gradients(self, gradient_method, wrt_arg, n_cpu, **kwargs):
        return np.array([[1, 1]])

    def __call__(self, *args, **kwargs):
        return self.aep(*args, **kwargs)


def test_aep_returns_correct_value():
    dummy_model = DummyWindFarmModel(mode="normal")
    comp = PyWakeAEPCostModelComponent(
        windFarmModel=dummy_model, n_wt=2, wd=[0], ws=[10]
    )
    # Call the cost_function defined internally
    result = comp.cost_function(**design_vars)
    assert result == 100


def test_aep_handles_same_position_error():
    dummy_model = DummyWindFarmModel(mode="raise_same")
    comp = PyWakeAEPCostModelComponent(
        windFarmModel=dummy_model, n_wt=2, wd=[0], ws=[10]
    )
    result = comp.cost_function(**design_vars)
    assert result == 0


def test_py_wake_wrapper_brings_up_failed_pywake_aep_call():
    dummy_model = DummyWindFarmModel(mode="raise_other")
    comp = PyWakeAEPCostModelComponent(
        windFarmModel=dummy_model, n_wt=2, wd=[0], ws=[10]
    )
    with pytest.raises(ValueError) as excinfo:
        comp.cost_function(**design_vars)
    # Verify that the exception message is augmented with the specific error string
    assert UNHANDLED_PYWAKE_ERROR_MSG in str(excinfo.value)


def test_additional_turbines_aep_handles_same_position_error():
    dummy_model = DummyWindFarmModel(mode="raise_same")
    comp = PyWakeAEPCostModelComponentAdditionalTurbines(
        windFarmModel=dummy_model,
        n_wt=2,
        add_wt_x=[0],
        add_wt_y=[0],
        grad_method=autograd,
    )
    result = comp.cost_function(**design_vars)
    assert result == 0


def test_additional_turbines_py_wake_wrapper_brings_up_failed_pywake_aep_call():
    dummy_model = DummyWindFarmModel(mode="raise_other")
    comp = PyWakeAEPCostModelComponentAdditionalTurbines(
        windFarmModel=dummy_model,
        n_wt=2,
        add_wt_x=[0],
        add_wt_y=[0],
        grad_method=autograd,
    )
    with pytest.raises(ValueError) as excinfo:
        comp.cost_function(**design_vars)
    assert UNHANDLED_PYWAKE_ERROR_MSG in str(excinfo.value)
