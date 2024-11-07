"""Objects useful for IEA 37 optimizations
"""
from numpy import array as npa

from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windturbine
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian

from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent


def get_iea37_initial(n_wt=9):
    """Initial positions for IEA Task 37 wind farms

    Parameters
    ----------
    n_wt : int, optional
        Number of wind turbines in farm (must be valid IEA 37 farm)
    """
    site = IEA37Site(n_wt)
    return site.initial_position


def get_iea37_constraints(n_wt=9):
    """Constraints for IEA Task 37 wind farms

    Parameters
    ----------
    n_wt : int, optional
        Number of wind turbines in farm (must be valid IEA 37 farm)

    Returns
    -------
    constr : list of topfarm constraints
        Spacing constraint and boundary constraint for IEA 37 model
    """
    diam = read_iea37_windturbine(iea37_path + 'iea37-335mw.yaml')[2]
    spac_constr = SpacingConstraint(2 * diam)
    bound_rad = npa([900, 1300, 2000, 3000])[n_wt == npa([9, 16, 36, 64])][0]
    bound_constr = CircleBoundaryConstraint((0, 0), bound_rad)
    return [spac_constr, bound_constr]


def get_iea37_cost(n_wt=9):
    """Cost component that wraps the IEA 37 AEP calculator"""
    wd = npa(range(16)) * 22.5  # only 16 bins
    site = IEA37Site(n_wt)
    wind_turbines = IEA37_WindTurbines()
    wake_model = IEA37SimpleBastankhahGaussian(site, wind_turbines)
    return PyWakeAEPCostModelComponent(wake_model, n_wt, wd=wd)


def main():
    if __name__ == '__main__':
        from topfarm import TopFarmProblem
        from topfarm.easy_drivers import EasyRandomSearchDriver
        from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
        from topfarm.plotting import XYPlotComp
        n_wt = 16
        x, y = get_iea37_initial(n_wt).T * 0.99999  # make sure the turbines numerically are inside the boundary for random search driver
        constr = get_iea37_constraints(n_wt)
        cost = get_iea37_cost(n_wt)
        driver = EasyRandomSearchDriver(RandomizeTurbinePosition_Circle(max_step=300), max_iter=10, max_time=10)
        problem = TopFarmProblem({'x': x, 'y': y},
                                 cost_comp=cost,
                                 constraints=constr,
                                 driver=driver,
                                 plot_comp=XYPlotComp())
        problem.evaluate()


main()
