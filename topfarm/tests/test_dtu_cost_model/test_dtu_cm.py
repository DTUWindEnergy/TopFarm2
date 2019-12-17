#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:20:09 2019

Full set of dtu cost model tests
"""

from topfarm.cost_models.economic_models.dtu_wind_cm_main import economic_evaluation
from topfarm.tests import npt
import warnings


def test_dtu_cm_capex():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        distance_from_shore = 10          # [km]
        energy_price = 0.2                # [Euro/kWh]
        project_duration = 20             # [years]

        eco_eval = economic_evaluation(distance_from_shore, energy_price, project_duration)

        number_of_turbines = 10
        rated_rpm_vector = [10.0] * number_of_turbines        # [RPM]
        rotor_diameter_vector = [100.0] * number_of_turbines  # [m]
        rated_power_vector = [10.0] * number_of_turbines      # [MW]
        hub_height_vector = [100.0] * number_of_turbines      # [m]
        water_depth_vector = [15.0] * number_of_turbines      # [m]

        eco_eval.calculate_capex(rated_rpm_vector, rotor_diameter_vector, rated_power_vector,
                                 hub_height_vector, water_depth_vector)

        npt.assert_almost_equal(eco_eval.project_costs_sums['CAPEX'] * 1.0E-6, 93.6938301)  # [M Euro]


def test_dtu_cm_devex():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        distance_from_shore = 10          # [km]
        energy_price = 0.2                # [Euro/kWh]
        project_duration = 20             # [years]

        eco_eval = economic_evaluation(distance_from_shore, energy_price, project_duration)

        number_of_turbines = 10
        rated_power_vector = [10.0] * number_of_turbines  # [MW]

        eco_eval.calculate_devex(rated_power_vector)

        npt.assert_almost_equal(eco_eval.project_costs_sums['DEVEX'] * 1.0E-6, 12.3461538)  # [M Euro]


def test_dtu_cm_opex():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        distance_from_shore = 10          # [km]
        energy_price = 0.2                # [Euro/kWh]
        project_duration = 20             # [years]

        eco_eval = economic_evaluation(distance_from_shore, energy_price, project_duration)

        number_of_turbines = 10
        rated_power_vector = [10.0] * number_of_turbines  # [MW]

        eco_eval.calculate_opex(rated_power_vector)

        npt.assert_almost_equal(eco_eval.project_costs_sums['OPEX'] * 1.0E-6, 6.465)  # [M Euro / year]


def test_dtu_cm_irr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        distance_from_shore = 10          # [km]
        energy_price = 0.2                # [Euro/kWh]
        project_duration = 20             # [years]

        eco_eval = economic_evaluation(distance_from_shore, energy_price, project_duration)

        number_of_turbines = 10
        rated_rpm_vector = [10.0] * number_of_turbines        # [RPM]
        rotor_diameter_vector = [100.0] * number_of_turbines  # [m]
        rated_power_vector = [10.0] * number_of_turbines      # [MW]
        hub_height_vector = [100.0] * number_of_turbines      # [m]
        water_depth_vector = [15.0] * number_of_turbines      # [m]
        aep_vector = [8.0e6] * number_of_turbines             # [kWh]

        eco_eval.calculate_irr(rated_rpm_vector, rotor_diameter_vector, rated_power_vector,
                               hub_height_vector, water_depth_vector, aep_vector)

        npt.assert_almost_equal(eco_eval.IRR, 6.2860816)  # [%]


def test_dtu_cm_npv():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        distance_from_shore = 10          # [km]
        energy_price = 0.2                # [Euro/kWh]
        project_duration = 20             # [years]
        discount_rate = 0.062860816       # [-]

        eco_eval = economic_evaluation(distance_from_shore, energy_price,
                                       project_duration, discount_rate)

        number_of_turbines = 10
        rated_rpm_vector = [10.0] * number_of_turbines        # [RPM]
        rotor_diameter_vector = [100.0] * number_of_turbines  # [m]
        rated_power_vector = [10.0] * number_of_turbines      # [MW]
        hub_height_vector = [100.0] * number_of_turbines      # [m]
        water_depth_vector = [15.0] * number_of_turbines      # [m]
        aep_vector = [8.0e6] * number_of_turbines             # [kWh]

        eco_eval.calculate_npv(rated_rpm_vector, rotor_diameter_vector, rated_power_vector,
                               hub_height_vector, water_depth_vector, aep_vector)

        npt.assert_almost_equal(eco_eval.NPV, 0.0437686)  # [Euro]
