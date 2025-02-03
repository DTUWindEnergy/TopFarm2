import pytest
import numpy as np

from topfarm.constraint_components.boundary import (
    InclusionZone,
    ExclusionZone,
    MultiPolygonBoundaryComp,
)
from topfarm.constraint_components.boundary import TurbineSpecificBoundaryComp
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt


@pytest.mark.parametrize(
    "point, is_violation",
    [
        ([5, 50], True),  # Too close to left boundary
        ([95, 50], True),  # Too close to right boundary
        ([50, 5], True),  # Too close to bottom boundary
        ([50, 95], True),  # Too close to top boundary
        ([20, 20], False),  # Valid point
        ([80, 80], False),  # Valid point
    ],
)
def test_dist2wt_inclusion_zone(point, is_violation):
    """Test that turbines respect minimum distance from inclusion zone boundary"""
    square_inclusion_zone = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    dist2wt = 10
    zones = [InclusionZone(square_inclusion_zone, dist2wt=lambda: dist2wt)]
    n_wt = 5
    comp = MultiPolygonBoundaryComp(n_wt=n_wt, zones=zones)
    x = point[0]
    y = point[1]
    x_arr = np.array([x] * n_wt)
    y_arr = np.array([y] * n_wt)
    distances = comp.distances(x_arr, y_arr)
    if is_violation:
        assert np.any(
            distances < 0
        ), f"Point ({x},{y}) should violate boundary constraint; Distances {distances};"
    else:
        assert np.all(
            distances >= 0
        ), f"Point ({x},{y}) should not violate boundary constraint"


@pytest.mark.parametrize(
    "point, is_violation",
    [
        ([30, 30], True),  # Too close to exclusion zone
        ([70, 30], True),  # Too close to exclusion zone
        ([50, 70], True),  # Too close to exclusion zone
        ([10, 10], False),  # Valid point
        ([90, 90], False),  # Valid point
        ([50, 20], True),  # Too close to exclusion zone
    ],
)
def test_dist2wt_exclusion_zone(point, is_violation):
    """Test that turbines respect minimum distance from exclusion zone boundary"""
    dist2wt = 10
    square_inclusion_zone = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    triangle_exclusion_zone = np.array([[25, 25], [75, 25], [50, 75]])
    zones = [
        InclusionZone(square_inclusion_zone),
        ExclusionZone(triangle_exclusion_zone, dist2wt=lambda: dist2wt),
    ]
    n_wt = 5

    comp = MultiPolygonBoundaryComp(n_wt=n_wt, zones=zones)

    # Test points near exclusion zone
    (x, y) = point
    x_arr = np.array([x] * n_wt)
    y_arr = np.array([y] * n_wt)
    distances = comp.distances(x_arr, y_arr)

    if is_violation:
        assert np.any(
            distances < 0
        ), f"Point ({x},{y}) should violate exclusion zone constraint"
    else:
        assert np.all(
            distances >= 0
        ), f"Point ({x},{y}) should not violate exclusion zone constraint"


def test_dist2wt_line_exclusion():
    """Test that turbines respect minimum distance from line exclusion zone"""
    dist2wt = 10

    square_inclusion_zone = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    line_exclusion_zone = np.array([[0, 50], [100, 50]])

    zones = [
        InclusionZone(square_inclusion_zone),
        ExclusionZone(
            line_exclusion_zone, dist2wt=lambda: dist2wt, geometry_type="line"
        ),
    ]
    n_wt = 5

    comp = MultiPolygonBoundaryComp(n_wt=n_wt, zones=zones)

    # Test points near line
    test_points = [
        ([50, 55], True),  # Too close to line
        ([50, 45], True),  # Too close to line
        ([50, 70], False),  # Valid point
        ([50, 30], False),  # Valid point
        ([25, 55], True),  # Too close to line
        ([75, 45], True),  # Too close to line
    ]

    for (x, y), should_violate in test_points:
        x_arr = np.array([x] * n_wt)
        y_arr = np.array([y] * n_wt)
        distances = comp.distances(x_arr, y_arr)

        if should_violate:
            assert np.any(
                distances < 0
            ), f"Point ({x},{y}) should violate line exclusion constraint"
        else:
            assert np.all(
                distances >= 0
            ), f"Point ({x},{y}) should not violate line exclusion constraint"


@pytest.mark.parametrize(
    "point, is_violation",
    [
        ([25, 25], True),  # On exclusion vertex
        ([50, 75], True),  # On exclusion vertex
        ([40, 25], True),  # On exclusion edge
        ([24, 24], False),  # Just outside exclusion
        ([51, 76], False),  # Just outside exclusion
    ],
)
def test_dist2wt_zero_distance(point, is_violation):
    """Test behavior when dist2wt is zero"""
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    triangle = np.array([[25, 25], [75, 25], [50, 75]])

    zones = [InclusionZone(square), ExclusionZone(triangle, dist2wt=lambda: 0)]

    n_wt = 5
    comp = MultiPolygonBoundaryComp(n_wt=n_wt, zones=zones)

    (x, y) = point
    x_arr = np.array([x] * n_wt)
    y_arr = np.array([y] * n_wt)
    distances = comp.distances(x_arr, y_arr)

    if is_violation:
        assert np.any(
            distances <= 0
        ), f"Point ({x},{y}) should violate or be exactly on boundary"
    else:
        assert np.all(
            distances > 0
        ), f"Point ({x},{y}) should be strictly outside exclusion zone"





# fmt:off
D1 = 80
D2 = 120
H1 = 70
H2 = 110
def dist2wt_h(D, H):
    return (130 - H) // 4 # type 0 - (130-70)//4 = 15, type 1 - (130-110)//4 = 5
def dist2wt_d(D, H):
    return (20 + D) // 10 # type 0 - (20+80)//10 = 10, type 1 - (20+120)//10 = 14 
def dist2wt_dh(D, H):
    return (D + H) // 20 # type 0 - (80+70)//20 = 7, type 1 - (120+110)//20 = 11
def dist2wt_scalar():
    return 12
# fmt:on


@pytest.mark.parametrize(
    "point, wt_type, is_violation, dist2wt",
    [
        # type 0 violations
        [(10, 10), 0, True, dist2wt_h],  # Too close to inclusion
        [(5, 5), 0, True, dist2wt_d],  # Too close to inclusion
        [(5, 5), 0, True, dist2wt_dh],  # Too close to inclusion
        [(5, 5), 0, True, dist2wt_scalar],  # Too close to inclusion
        [(40, 40), 0, True, dist2wt_h],  # Too close to exclusion
        [(41, 41), 0, True, dist2wt_d],  # Too close to exclusion
        [(45, 45), 0, True, dist2wt_dh],  # Too close to exclusion
        [(39, 39), 0, True, dist2wt_scalar],  # Too close to exclusion
        # type 1 violations
        [(4, 4), 1, True, dist2wt_h],  # Too close to inclusion
        [(5, 5), 1, True, dist2wt_d],  # Too close to inclusion
        [(5, 5), 1, True, dist2wt_dh],  # Too close to inclusion
        [(5, 5), 1, True, dist2wt_scalar],  # Too close to inclusion
        [(46, 46), 1, True, dist2wt_h],  # Too close to exclusion
        [(41, 41), 1, True, dist2wt_d],  # Too close to exclusion
        [(45, 45), 1, True, dist2wt_dh],  # Too close to exclusion
        [(39, 39), 1, True, dist2wt_scalar],  # Too close to exclusion
        # type 0 valid points
        [(16, 16), 0, False, dist2wt_h],
        [(20, 20), 0, False, dist2wt_d],
        [(20, 20), 0, False, dist2wt_dh],
        [(20, 20), 0, False, dist2wt_scalar],
        [(34, 34), 0, False, dist2wt_h],
        [(39, 39), 0, False, dist2wt_d],
        [(35, 35), 0, False, dist2wt_dh],
        [(15, 37), 0, False, dist2wt_scalar],
        # type 1 valid points
        [(10, 10), 1, False, dist2wt_h],
        [(15, 15), 1, False, dist2wt_d],
        [(15, 15), 1, False, dist2wt_dh],
        [(15, 15), 1, False, dist2wt_scalar],
        [(45, 45), 1, False, dist2wt_h],
        [(20, 20), 1, False, dist2wt_d],
        [(87, 87), 1, False, dist2wt_dh],
        [(31, 87), 1, False, dist2wt_scalar],
    ],
)
def test_turbine_specific_dist2wt(point, wt_type, is_violation, dist2wt):
    """Test turbine-specific distances from boundaries"""
    square_incl = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    square_excl = np.array([[50, 50], [75, 50], [75, 75], [50, 75]])
    zones = [
        InclusionZone(square_incl, dist2wt=dist2wt),
        ExclusionZone(square_excl, dist2wt=dist2wt),
    ]
    wts = WindTurbines(
        names=["tb1", "tb2"],
        diameters=[D1, D2],
        hub_heights=[H1, H2],
        powerCtFunctions=[
            CubePowerSimpleCt(
                ws_cutin=3,
                ws_cutout=25,
                ws_rated=12,
                power_rated=2000,
                power_unit="kW",
                ct=8 / 9,
                additional_models=[],
            ),
            CubePowerSimpleCt(
                ws_cutin=3,
                ws_cutout=25,
                ws_rated=12,
                power_rated=3000,
                power_unit="kW",
                ct=8 / 9,
                additional_models=[],
            ),
        ],
    )
    comp = TurbineSpecificBoundaryComp(n_wt=1, wind_turbines=wts, zones=zones)

    x, y = point
    distances = comp.distances(
        np.array([x]),
        np.array([y]),
        type=np.array([wt_type]),
    )
    if is_violation:
        assert np.any(
            distances < 0
        ), f"Point ({x},{y}) should violate boundary with distances {distances}"
    else:
        assert np.all(
            distances >= 0
        ), f"Point ({x},{y}) should not violate boundary with distances {distances}"


def test_non_implemented_zone_geom_type():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    zones = [InclusionZone(square, geometry_type="NON_EXISTENT")]
    with pytest.raises(NotImplementedError):
        comp = MultiPolygonBoundaryComp(n_wt=1, zones=zones)
        point = (50, 50)
        (x, y) = point
        x_arr = np.array([x])
        y_arr = np.array([y])
        _ = comp.distances(x_arr, y_arr)


def test_weird_return_from_dist2wt():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    zones = [InclusionZone(square, geometry_type="polygon", dist2wt=lambda: "weird")]
    with pytest.raises(ValueError):
        comp = MultiPolygonBoundaryComp(n_wt=1, zones=zones)
        point = (50, 50)
        (x, y) = point
        x_arr = np.array([x])
        y_arr = np.array([y])
        _ = comp.distances(x_arr, y_arr)
