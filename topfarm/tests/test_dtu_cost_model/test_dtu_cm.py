#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:20:09 2019

Full set of dtu cost model tests
"""
import numpy as np
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

        npt.assert_almost_equal(eco_eval.NPV, 0.0724087)  # [Euro]


def test_dtu_cm_drivetrain():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        distance_from_shore = 10          # [km]
        energy_price = 0.2                # [Euro/kWh]
        project_duration = 20             # [years]
        discount_rate = 0.062860816       # [-]

        eco_eval = economic_evaluation(distance_from_shore, energy_price,
                                       project_duration, discount_rate)
        rotor_diameter = 100.0
        hub_height = 100.0
        rated_rpm = 10.0
        rated_power = 10.0
        rated_torque = rated_power / (rated_rpm * np.pi / 30.0) * 1.1

        eco_eval.calculate_turbine(rated_rpm, rotor_diameter, rated_power, hub_height)

        eco_eval.high_speed_drivetrain(rotor_diameter=rotor_diameter, rated_torque=rated_torque, rated_rpm=rated_rpm)
        npt.assert_almost_equal(eco_eval.turbine_component_mass_sums['hs_drivetrain'] / 227538.2158, 1)

        eco_eval.direct_drive_drivetrain(rotor_diameter=rotor_diameter, rated_torque=rated_torque)
        npt.assert_almost_equal(eco_eval.turbine_component_mass_sums['dd_drivetrain'] / 179871.67449, 1)
