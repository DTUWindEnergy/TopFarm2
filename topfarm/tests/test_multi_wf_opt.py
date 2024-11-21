import numpy as np
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import MultiCircleBoundaryConstraint
from topfarm.constraint_components.boundary import MultiXYBoundaryConstraint, Boundary
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.plotting import NoPlot
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import LillgrundSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from topfarm.constraint_components.boundary import MultiWFPolygonBoundaryConstraint


def test_integration_multi_cirle_opt_succeeds():
    wind_turbines = GenericWindTurbine("GenWT", 100.6, 2000, 150)
    site = LillgrundSite()
    wf_model = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)

    grid_side = 2
    wt_x, wt_y = np.meshgrid(
        np.linspace(0, wind_turbines.diameter() * grid_side, grid_side),
        np.linspace(0, wind_turbines.diameter() * grid_side, grid_side),
    )
    wt_x, wt_y = wt_x.flatten(), wt_y.flatten()
    wt_x2 = wt_x + wind_turbines.diameter() * grid_side * 2.33
    wt_y2 = wt_y
    X_full = np.concatenate([wt_x, wt_x2])
    Y_full = np.concatenate([wt_y, wt_y2])
    n_wt = len(X_full)

    n_wt_sf = n_wt // 2
    wf1_mask = np.zeros(n_wt, dtype=bool)
    wf1_mask[:n_wt_sf] = True
    wf2_mask = np.zeros(n_wt, dtype=bool)
    wf2_mask[n_wt_sf:] = True

    def _get_radius(x, y):  # fmt: skip
        return np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2).max() + 100

    def _get_center(x, y):  # fmt: skip
        return np.array([x.mean(), y.mean()])

    constraint_comp = MultiCircleBoundaryConstraint(
        center=[_get_center(wt_x, wt_y), _get_center(wt_x2, wt_y2)],
        radius=[_get_radius(wt_x, wt_y), _get_radius(wt_x2, wt_y2)],
        masks=[wf1_mask, wf2_mask],
    )
    cost_comp = PyWakeAEPCostModelComponent(
        windFarmModel=wf_model,
        n_wt=n_wt,
        grad_method=autograd,
    )
    const = [
        SpacingConstraint(min_spacing=wind_turbines.diameter() * 2),
        constraint_comp,
    ]
    plots = NoPlot()
    problem = TopFarmProblem(
        design_vars={"x": X_full, "y": Y_full},
        n_wt=n_wt,
        constraints=(const),
        cost_comp=cost_comp,
        driver=(EasyScipyOptimizeDriver(optimizer="SLSQP", maxiter=500)),
        plot_comp=plots,
    )

    initial_aep = -problem.evaluate()[0]

    _, state, recorder = problem.optimize(disp=False)
    assert recorder["success"].all()

    final_posx_wf1 = state["x"][wf1_mask]
    final_posy_wf1 = state["y"][wf1_mask]
    final_posx_wf2 = state["x"][wf2_mask]
    final_posy_wf2 = state["y"][wf2_mask]
    center1 = _get_center(wt_x, wt_y)
    center2 = _get_center(wt_x2, wt_y2)
    radius1 = _get_radius(wt_x, wt_y)
    radius2 = _get_radius(wt_x2, wt_y2)

    def __assert_within_boundary(x, y, center, radius):  # fmt: skip
        tolerance = 1  # in meters
        assert np.all(
            np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < radius + tolerance
        )
    __assert_within_boundary(final_posx_wf1, final_posy_wf1, center1, radius1)
    __assert_within_boundary(final_posx_wf2, final_posy_wf2, center2, radius2)

    optimized_aep = -problem.evaluate()[0]
    relative_improvement = (optimized_aep - initial_aep) / initial_aep
    assert relative_improvement > 0.01  # at least 1% improvement


def test_integration_multi_wf_convex_hull_opt_succeeds():
    wind_turbines = GenericWindTurbine("GenWT", 100.6, 2000, 150)
    site = LillgrundSite()
    wf_model = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)

    grid_side = 2
    wt_x, wt_y = np.meshgrid(
        np.linspace(0, wind_turbines.diameter() * grid_side, grid_side),
        np.linspace(0, wind_turbines.diameter() * grid_side, grid_side),
    )
    wt_x, wt_y = wt_x.flatten(), wt_y.flatten()
    wt_x2 = wt_x + wind_turbines.diameter() * grid_side * 3.0
    wt_y2 = wt_y
    X_full = np.concatenate([wt_x, wt_x2])
    Y_full = np.concatenate([wt_y, wt_y2])
    n_wt = len(X_full)

    n_wt_sf = len(wt_x)
    wf1_mask = np.zeros(n_wt, dtype=bool)
    wf1_mask[:n_wt_sf] = True
    wf2_mask = np.zeros(n_wt, dtype=bool)
    wf2_mask[n_wt_sf:] = True

    # construct a rectangle
    def _get_corners(x: np.ndarray, y: np.ndarray, radius):  # fmt: skip
        cx = x.mean()
        cy = y.mean()
        return np.array(
            [
                [cx + radius, cy + radius],
                [cx - radius, cy - radius],
                [cx + radius, cy - radius],
                [cx - radius, cy + radius],
            ]
        )

    radius = np.sqrt((wt_x - wt_x.mean()) ** 2 + (wt_y - wt_y.mean()) ** 2).max() + 150
    wf1_corners = _get_corners(wt_x, wt_y, radius)
    wf2_corners = _get_corners(wt_x2, wt_y2, radius)

    constraint_comp = MultiXYBoundaryConstraint(
        [
            Boundary(wf1_corners, wf1_mask),
            Boundary(wf2_corners, wf2_mask),
        ],
        boundary_type="convex_hull",
    )
    cost_comp = PyWakeAEPCostModelComponent(
        windFarmModel=wf_model, n_wt=n_wt, grad_method=autograd
    )
    problem = TopFarmProblem(
        design_vars={"x": X_full, "y": Y_full},
        n_wt=n_wt,
        constraints=(
            [
                constraint_comp,
                SpacingConstraint(min_spacing=wind_turbines.diameter() * 2),
            ]
        ),
        cost_comp=cost_comp,
        driver=(EasyScipyOptimizeDriver(optimizer="SLSQP", maxiter=500)),
        plot_comp=NoPlot(),
    )

    aep_initial = -problem.evaluate()[0]
    _, state, recorder = problem.optimize(disp=False)
    assert recorder["success"].all()

    final_posx_wf1 = state["x"][wf1_mask]
    final_posy_wf1 = state["y"][wf1_mask]
    final_posx_wf2 = state["x"][wf2_mask]
    final_posy_wf2 = state["y"][wf2_mask]
    wf1_points = np.array([final_posx_wf1, final_posy_wf1]).T
    wf2_points = np.array([final_posx_wf2, final_posy_wf2]).T
    assert are_all_points_in_rectangle(wf1_points, wf1_corners, tolerance=1)
    assert are_all_points_in_rectangle(wf2_points, wf2_corners, tolerance=1)

    aep_optimized = -problem.evaluate()[0]
    relative_improvement = (aep_optimized - aep_initial) / aep_initial
    assert relative_improvement > 0.01  # at least 1% improvement


def reorder_rectangle(rect):
    """
    Ensure the rectangle vertices are ordered in a consistent order (clockwise).

    Parameters:
        rect (np.array): Array of shape (4, 2) representing the rectangle's corners.
    Returns:
        np.array: Reordered rectangle in clockwise order.
    """
    # Sort points first by x-coordinate, then by y-coordinate to get bottom-left first
    rect = rect[np.lexsort((rect[:, 1], rect[:, 0]))]

    # Distinguish points
    bl = rect[0]  # Bottom-left
    tr = rect[3]  # Top-right

    # The two remaining points are either top-left and bottom-right
    other_points = rect[1:3]

    # Sort remaining two points by y to identify bottom-right and top-left
    if other_points[0][1] < other_points[1][1]:  # y-coordinates
        br, tl = other_points
    else:
        tl, br = other_points

    # Return in clockwise order: bottom-left, bottom-right, top-right, top-left
    return np.array([bl, br, tr, tl])


def is_point_in_rectangle(point, rect, tolerance=0):
    """
    Check if a single point is inside a rectangle with some tolerance using the cross product method.

    Parameters:
        point (np.array): Array of shape (2,) with the x and y coordinates of the point.
        rect (np.array): Array of shape (4, 2) representing the rectangle's corners in order.
        tolerance (float): Distance tolerance to expand the rectangle boundaries.

    Returns:
        bool: True if the point is inside the rectangle within the tolerance, False otherwise.
    """
    rect = reorder_rectangle(rect)

    # Shift rectangle corners to start from the origin (0,0)
    rect_shifted = rect - rect[0]
    point_shifted = point - rect[0]

    # Check each pair of rectangle edges
    for i in range(4):
        edge = rect_shifted[(i + 1) % 4] - rect_shifted[i]
        to_point = point_shifted - rect_shifted[i]
        cross_product = np.cross(edge, to_point)

        # Calculate the tolerance adjustment based on the edge length
        edge_length = np.linalg.norm(edge)
        adjusted_tolerance = tolerance * edge_length

        # If the cross product is less than the negative tolerance, point is outside
        if cross_product < -adjusted_tolerance:
            return False

    return True


def are_all_points_in_rectangle(points, rect, tolerance=0):
    """
    Check if all points are inside a rectangle within a given tolerance.

    Parameters:
        points (np.array): Array of shape (n, 2) with the x and y coordinates of the points.
        rect (np.array): Array of shape (4, 2) representing the rectangle's corners in clockwise or counterclockwise order.
        tolerance (float): Distance tolerance to expand the rectangle boundaries.

    Returns:
        bool: True if all points are inside the rectangle within the tolerance, False otherwise.
    """
    return all(is_point_in_rectangle(point, rect, tolerance) for point in points)


def test_integration_multi_wf_polygon_opt_succeeds():
    wind_turbines = GenericWindTurbine("GenWT", 100.6, 2000, 150)
    site = LillgrundSite()
    wf_model = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)

    grid_side = 2
    wt_x, wt_y = np.meshgrid(
        np.linspace(0, wind_turbines.diameter() * grid_side, grid_side),
        np.linspace(0, wind_turbines.diameter() * grid_side, grid_side),
    )
    wt_x, wt_y = wt_x.flatten(), wt_y.flatten()
    wt_x2 = wt_x + wind_turbines.diameter() * grid_side * 4.0
    wt_y2 = wt_y
    X_full = np.concatenate([wt_x, wt_x2])
    Y_full = np.concatenate([wt_y, wt_y2])
    n_wt = len(X_full)

    def _get_corners(x: np.ndarray, y: np.ndarray, radius):  # fmt: skip
        cx = x.mean()
        cy = y.mean()
        return np.array(
            [  # order matters for a arbitrary polygon
                [cx - radius, cy - radius],
                [cx + radius, cy - radius],
                [cx - radius * 0.1, cy],
                [cx + radius, cy + radius],
                [cx - radius, cy + radius],
            ]
        )
    radius = np.sqrt((wt_x - wt_x.mean()) ** 2 + (wt_y - wt_y.mean()) ** 2).max() + 150
    wf1_corners = _get_corners(wt_x, wt_y, radius)
    wf2_corners = _get_corners(wt_x2, wt_y2, radius)

    boundary_coords = {
        0: wf1_corners,
        1: wf2_corners,
    }
    constraint_comp = MultiWFPolygonBoundaryConstraint(
        boundaries=boundary_coords,
        turbine_groups={0: np.arange(n_wt // 2), 1: np.arange(n_wt // 2, n_wt)},
    )
    cost_comp = PyWakeAEPCostModelComponent(
        windFarmModel=wf_model, n_wt=n_wt, grad_method=autograd
    )
    problem = TopFarmProblem(
        design_vars={"x": X_full, "y": Y_full},
        n_wt=n_wt,
        constraints=(
            [
                constraint_comp,
                SpacingConstraint(min_spacing=wind_turbines.diameter() * 2),
            ]
        ),
        cost_comp=cost_comp,
        driver=(EasyScipyOptimizeDriver(optimizer="SLSQP", maxiter=500)),
        plot_comp=NoPlot(),
    )

    aep_initial = -problem.evaluate()[0]
    _, state, recorder = problem.optimize(disp=True)
    assert recorder["success"].all()

    aep_optimized = -problem.evaluate()[0]
    relative_improvement = (aep_optimized - aep_initial) / aep_initial
    assert relative_improvement > 0.01  # at least 1% improvement

    final_posx_wf1 = state["x"][: n_wt // 2]
    final_posy_wf1 = state["y"][: n_wt // 2]
    final_posx_wf2 = state["x"][n_wt // 2:]
    final_posy_wf2 = state["y"][n_wt // 2:]
    wft1_points = np.array([final_posx_wf1, final_posy_wf1]).T
    wft2_points = np.array([final_posx_wf2, final_posy_wf2]).T

    def __assert_all_point_in_non_convex_polygon(points, corners, tolerance=1):
        import shapely.geometry as sg  # fmt: skip
        polygon = sg.Polygon(corners)
        buffred = polygon.buffer(tolerance)
        for point in points:
            assert buffred.contains(sg.Point(point))

    __assert_all_point_in_non_convex_polygon(wft1_points, wf1_corners, tolerance=1)
    __assert_all_point_in_non_convex_polygon(wft2_points, wf2_corners, tolerance=1)
