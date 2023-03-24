"""
2018 - Created by Witold Skrzypinski, wisk@dtu.dk
Implementation of Offshore Turbine Cost Model, ref. Soeren Oemann Lind, soli@dtu.dk
Structure based on 2017 turbine_cost.py by Arvydas Berzonskis, Goldwind
"""
import numpy as np
import numpy_financial as npf


class economic_evaluation():
    '''All masses are in kg, all costs are in 2017 Euro'''

    def __init__(self, distance_from_shore, energy_price, project_duration,
                 discount_rate=0.0909870634):

        # Initialize dictionaries
        self.project_costs = {}
        self.project_costs_sums = {}
        self.discount_rate = discount_rate              # [-]
        self.distance_from_shore = distance_from_shore  # [km]
        self.energy_price = energy_price                # [Euro/kWh]
        self.project_duration = project_duration        # [years]

    def calculate_npv(self, rated_rpm_array, D_rotor_array, Power_rated_array,
                      hub_height_array, water_depth_array, aep_array, cabling_cost=None):
        '''
        Calculate Net Present Value [Euro]

        Input:
        rated_rpm_array   [RPM]
        D_rotor_array     [m]
        Power_rated_array [MW]
        hub_height_array  [m]
        aep_array         [kWh]
        water_depth_array [m]
        '''

        self.calculate_expenditures(
            rated_rpm_array,
            D_rotor_array,
            Power_rated_array,
            hub_height_array,
            aep_array,
            water_depth_array,
            cabling_cost)

        self.calculate_cash_flow()

        self.NPV = npf.npv(self.discount_rate, self.CWF)

        return self.NPV

    def calculate_irr(self, rated_rpm_array, D_rotor_array, Power_rated_array,
                      hub_height_array, water_depth_array, aep_array, cabling_cost=None):
        '''
        Calculate Internal Rate of Return [%]

        Input:
        rated_rpm_array   [RPM]
        D_rotor_array     [m]
        Power_rated_array [MW]
        hub_height_array  [m]
        aep_array         [kWh]
        water_depth_array [m]
        '''

        self.calculate_expenditures(
            rated_rpm_array,
            D_rotor_array,
            Power_rated_array,
            hub_height_array,
            aep_array,
            water_depth_array,
            cabling_cost)

        self.calculate_cash_flow()

        self.IRR = 100 * npf.irr(self.CWF)  # [%]

        return self.IRR

    def calculate_cash_flow(self):
        '''Calculate cash flow every year during the project period'''

        self.CWF = []

        for i in range(self.project_duration + 1):

            if i == 0:
                self.CWF.append(float(-self.project_costs_sums["DEVEX"] - self.project_costs_sums["CAPEX"]))

            elif i == self.project_duration:
                self.CWF.append(float(self.energy_price * sum(self.aep_vector) - self.project_costs_sums["ABEX"] - self.project_costs_sums["OPEX"]))

            else:
                self.CWF.append(self.energy_price * sum(self.aep_vector) - self.project_costs_sums["OPEX"])

    def calculate_expenditures(self, rated_rpm, rotor_diameter, rated_power,
                               hub_height, aep_array, water_depth, cabling_cost=None):
        '''Calculate DEVEX, CAPEX, OPEX and ABEX [Euro]'''

        # Ensure that the variables are numpy arrays
        self.aep_vector = np.array(aep_array)

        # Calculate expenditures
        self.calculate_devex(rated_power)

        self.calculate_capex(rated_rpm, rotor_diameter, rated_power,
                             hub_height, water_depth, cabling_cost)

        self.calculate_opex(rated_power)

        self.calculate_abex()

    def calculate_devex(self, rated_power):
        '''Calculate DEVEX [Euro]'''

        rated_power = np.array(rated_power)
        number_of_turbines = np.size(rated_power)

        myName = "DEVEX"
        magicFactor = 2.0e3 / 39 * 0.3 / 80

        self.project_costs[myName] = {
            "environmental_survey": magicFactor * 1e6 * self.distance_from_shore,
            "sea_bed_survey": 1.5 * 1e5 * number_of_turbines,
            "met_mast": magicFactor * 1e6 * self.distance_from_shore,
            "development_services": 0.7 * 1e5 * np.sum(rated_power)}

        self.project_costs_sums[myName] = sum(
            self.project_costs[myName].values())

    def calculate_capex(self, rated_rpm, rotor_diameter, rated_power,
                        hub_height, water_depth, cabling_cost=None):
        '''Calculate CAPEX [Euro]'''

        rated_rpm = np.array(rated_rpm)
        rotor_diameter = np.array(rotor_diameter)
        rated_power = np.array(rated_power)
        hub_height = np.array(hub_height)
        water_depth = np.array(water_depth)

        self.calculate_turbine(
            rated_rpm,
            rotor_diameter,
            rated_power,
            hub_height)

        self.calculate_foundation(rotor_diameter, rated_power, water_depth)

        # %% BOP  [Euro] (Balance of Plant costs)

        myName = "BOP"

        self.project_costs[myName] = {
            "substation": 6.0e6 +
            1.0e4 *
            np.sum(rated_power) ** 1.5,
            "array_of_cables": 3.5e3 *
            np.sum(rotor_diameter),
            "cables_export": 1.4e6 *
            self.distance_from_shore,
            "onshore_electrical": 5.0e4 *
            np.sum(rated_power) +
            50.0 *
            np.sum(rated_power) ** 2.0}

        if cabling_cost:
            self.project_costs[myName]["array_of_cables"] = cabling_cost

        self.project_costs_sums[myName] = sum(
            self.project_costs[myName].values())

        # %% CAPEX [Euro]

        self.project_costs_sums["CAPEX"] = sum(
            self.turbine_general_costs_sums["bill_of_material"]) + sum(
            self.foundation_general_costs_sums["bill_of_material"]) + self.project_costs_sums["BOP"]

    def calculate_opex(self, rated_power):
        ''' Calculate OPEX [Euro / year]'''

        rated_power = np.array(rated_power)
        number_of_turbines = np.size(rated_power)

        myName = "O&M"

        self.project_costs[myName] = {
            "onshore personnel": 5.5e3 *
            np.sum(rated_power),
            "buildings, habor fees etc.": 3.0e6,
            "mobilization, rental time": 9.6e4 *
            number_of_turbines,
            "jackup personnel": 8.5e3 *
            number_of_turbines,
            "offshore service personnel": 1.2e4 *
            np.sum(rated_power),
            "service, failed components": 2.1e3 *
            np.sum(rated_power),
            "ships, offhsore operations": 4.6e3 *
            number_of_turbines *
            self.distance_from_shore}

        self.project_costs_sums[myName] = sum(
            self.project_costs[myName].values())

        self.project_costs_sums["OPEX"] = self.project_costs_sums["O&M"]

    def calculate_abex(self):
        ''' Calculate ABEX [Euro]'''

        self.project_costs_sums["ABEX"] = 0.03 * \
            self.project_costs_sums["CAPEX"]

    def calculate_turbine(
            self,
            rated_rpm,
            rotor_diameter,
            rated_power,
            hub_height):
        ''' Calculate the part of CAPEX associated with the turbine itself [Euro]'''

        self.turbine_component_mass = {}
        self.turbine_component_mass_sums = {}
        self.turbine_component_costs = {}
        self.turbine_component_costs_sums = {}
        self.turbine_general_costs = {}
        self.turbine_general_costs_sums = {}

        rated_torque = rated_power / (rated_rpm * np.pi / 30.0) * 1.1
        rotor_area = np.pi * (rotor_diameter / 2) ** 2

        # %% Blades

        comp_name = "3_blades"

        self.turbine_component_mass[comp_name] = 0.0 + \
            0.3 * rotor_diameter ** 2.5
        self.turbine_component_costs[comp_name] = 12.0 * \
            self.turbine_component_mass[comp_name]

        self.turbine_component_mass_sums[comp_name] = self.turbine_component_mass[comp_name]
        self.turbine_component_costs_sums[comp_name] = self.turbine_component_costs[comp_name]

        # %% Hub incl. pitch system w/ bearings

        comp_name = "hub"

        self.turbine_component_mass[comp_name] = {
            "structure": 6.0e3 + 0.1 * rotor_diameter ** 2.5,
            "pitch_bearings": 5.0e2 + 0.07 * rotor_diameter ** 2.5,
            "pitch_system": 5.0e2 + 0.03 * rotor_diameter ** 2.5,
            "secondary": 7.0e2 + 15.0 * rotor_diameter ** 1.0}

        self.turbine_component_costs[comp_name] = {
            "structure": 2.5 * self.turbine_component_mass[comp_name]["structure"],
            "pitch_bearings": 8.0 * self.turbine_component_mass[comp_name]["pitch_bearings"],
            "pitch_system": 8.0 * self.turbine_component_mass[comp_name]["pitch_system"],
            "secondary": 8.0 * self.turbine_component_mass[comp_name]["secondary"]}

        self.turbine_component_mass_sums[comp_name] = sum(
            self.turbine_component_mass[comp_name].values())
        self.turbine_component_costs_sums[comp_name] = sum(
            self.turbine_component_costs[comp_name].values())

        # %% High Speed Drivetrain
        # self.high_speed_drivetrain(rotor_diameter, rated_torque, rated_rpm)

        # %% Medium Speed Drivetrain
        self.medium_speed_drivetrain(rotor_diameter, rated_torque, rated_rpm)

        # %% Direct Drive Drivetrain
        # self.direct_drive_drivetrain(rotor_diameter, rated_torque)

        # %% Nacelle

        comp_name = "nacelle"

        self.turbine_component_mass[comp_name] = {
            "cooling": 0.0 + 500.0 * rated_power ** 1.0,
            "converter": 0.0 + 1.0e3 * rated_power ** 1.0,
            "controller": 200.0 + 100.0 * rated_power ** 1.0,
            "yaw": 0.0 + 0.1 * rotor_diameter ** 2.5,
            "canopy": 1.0e3 + 1.5e3 * rated_power ** 1.0,
            "secondary": 1.0e3 + 1.0e3 * rated_power ** 1.0}

        self.turbine_component_costs[comp_name] = {
            "cooling": 8.0 * self.turbine_component_mass[comp_name]["cooling"],
            "converter": 30.0 * self.turbine_component_mass[comp_name]["converter"],
            "controller": 50.0 * self.turbine_component_mass[comp_name]["controller"],
            "yaw": 6.0 * self.turbine_component_mass[comp_name]["yaw"],
            "canopy": 10.0 * self.turbine_component_mass[comp_name]["canopy"],
            "secondary": 10.0 * self.turbine_component_mass[comp_name]["secondary"]}

        self.turbine_component_mass_sums[comp_name] = sum(
            self.turbine_component_mass[comp_name].values())
        self.turbine_component_costs_sums[comp_name] = sum(
            self.turbine_component_costs[comp_name].values())

        # %% Tower

        comp_name = "tower"

        self.turbine_component_mass[comp_name] = {
            "structure": 0.0 + 0.25 * (rotor_area * hub_height) ** 1.0,
            "internal": 1.0e3 + 100.0 * hub_height ** 1.0,
            "cabling": 0.0 + 25.0 * (rated_power * hub_height) ** 1.0,
            "secondary": 1.0e3 + 500.0 * rated_power ** 1.0,
            "transformer": 0.0 + 2.5e3 * rated_power ** 1}

        self.turbine_component_costs[comp_name] = {
            "structure": 1.5 * self.turbine_component_mass[comp_name]["structure"],
            "internal": 8.0 * self.turbine_component_mass[comp_name]["internal"],
            "cabling": 8.0 * self.turbine_component_mass[comp_name]["cabling"],
            "secondary": 10.0 * self.turbine_component_mass[comp_name]["secondary"],
            "transformer": 8.0 * self.turbine_component_mass[comp_name]["transformer"]}

        self.turbine_component_mass_sums[comp_name] = sum(
            self.turbine_component_mass[comp_name].values())
        self.turbine_component_costs_sums[comp_name] = sum(
            self.turbine_component_costs[comp_name].values())

        # %% Turbine Bill of Material (BOM) cost

        self.turbine_general_costs_sums["bill_of_material"] = sum(
            self.turbine_component_costs_sums.values())

        # %% Turbine Direct Production Cost

        myName = "direct_production"

        self.turbine_general_costs[myName] = {
            "direct_labor": 0.03 *
            self.turbine_general_costs_sums["bill_of_material"],
            "production_overhead": 0.1 *
            self.turbine_general_costs_sums["bill_of_material"]}

        self.turbine_general_costs_sums[myName] = sum(self.turbine_general_costs[myName].values(
        )) + self.turbine_general_costs_sums["bill_of_material"]

        # %% Turbine Selling, General and Administrative Expenses (SG&A)

        myName = "SG&A"

        self.turbine_general_costs[myName] = {
            "SG&A_overhead": 0.05 * (self.turbine_general_costs_sums["direct_production"]),
            "R&D": 0.03 * (self.turbine_general_costs_sums["direct_production"]),
            "SG&A": 0.05 * (self.turbine_general_costs_sums["direct_production"])}

        self.turbine_general_costs_sums[myName] = sum(
            self.turbine_general_costs[myName].values())

        # %% Turbine Total Production Cost

        self.turbine_general_costs_sums["total_production"] = self.turbine_general_costs_sums[
            "direct_production"] + self.turbine_general_costs_sums["SG&A"]

        # %% Turbine Project costs

        myName = "project"

        self.turbine_general_costs[myName] = {
            "transportation": 0.2 * sum(self.turbine_component_mass_sums.values()) + 1.0e4,
            "harbor_storage_and_assembly": 2.5e4 * rated_power + 1.5e5,
            "installation_and_comissioning": 5.0e4 * rated_power + 1.0e5,
            "warranty_and_accruals": 0.03 * self.turbine_general_costs_sums["total_production"],
            "financing": 0.02 * self.turbine_general_costs_sums["total_production"]}

        self.turbine_general_costs_sums[myName] = sum(
            self.turbine_general_costs[myName].values())

        # %% Turbine Total Cost

        self.turbine_general_costs_sums["TOTAL"] = self.turbine_general_costs_sums["total_production"] + \
            self.turbine_general_costs_sums["project"]

    def calculate_foundation(self, rotor_diameter, rated_power, water_depth):
        ''' Calculate the part of CAPEX associated with the foundation itself [Euro]'''

        # Initialize dictionaries
        self.foundation_mass = {}
        self.foundation_costs = {}
        self.foundation_general_costs = {}
        self.foundation_general_costs_sums = {}

        # %% Foundation

        self.foundation_mass = {
            "monopile_incl_TP": (
                6.5e4 +
                4.5e3 *
                water_depth +
                40.0 *
                water_depth**2) *
            rated_power}

        self.foundation_costs = {
            "monopile_incl_TP": 1.5 * self.foundation_mass["monopile_incl_TP"]}
        '''
        self.foundation_mass = {
        "jacket_incl_boat_landing" : ( 8.3e4 - 700.0 * water_depth + 110.0 * water_depth**2) * rated_power}

        self.foundation_costs = {
        "jacket_incl_boat_landing" : 2.4 * self.foundation_mass["jacket_incl_boat_landing"]}
        '''
        self.foundation_mass_sum = sum(self.foundation_mass.values())
        self.foundation_cost_sum = sum(self.foundation_costs.values())

        # %% Foundation Bill of Material (BOM) cost

        self.foundation_general_costs_sums["bill_of_material"] = self.foundation_cost_sum

        # %% Foundation Direct Production Cost

        myName = "direct_production"

        self.foundation_general_costs[myName] = {
            "direct_labor": 0.03 *
            self.foundation_general_costs_sums["bill_of_material"],
            "material_overhead": 0.1 *
            self.foundation_general_costs_sums["bill_of_material"]}

        self.foundation_general_costs_sums[myName] = sum(self.foundation_general_costs[myName].values(
        )) + self.foundation_general_costs_sums["bill_of_material"]

        # %% Foundation Selling, General and Administrative Expenses (SG&A)

        myName = "SG&A"

        self.foundation_general_costs[myName] = {
            "SG&A_overhead": 0.05 * (self.foundation_general_costs_sums["direct_production"]),
            "R&D": 0.03 * (self.foundation_general_costs_sums["direct_production"]),
            "SG&A": 0.05 * (self.foundation_general_costs_sums["direct_production"])}

        self.foundation_general_costs_sums[myName] = sum(
            self.foundation_general_costs[myName].values())

        # %% Foundation Total Production Cost

        self.foundation_general_costs_sums["total_production"] = self.foundation_general_costs_sums[
            "direct_production"] + self.foundation_general_costs_sums["SG&A"]

        # %% Foundation Project costs

        myName = "project"

        self.foundation_general_costs[myName] = {
            "transportation": 0.15 *
            self.foundation_general_costs_sums["total_production"],
            "installation": 160.0 *
            self.foundation_mass_sum ** 0.5 +
            2.0e3 *
            self.distance_from_shore,
            "warranty_and_accruals": 0.03 *
            self.foundation_general_costs_sums["total_production"],
            "financing": 0.02 *
            self.foundation_general_costs_sums["total_production"]}

        self.foundation_general_costs_sums[myName] = sum(
            self.foundation_general_costs[myName].values())

        # %% Foundation Total Cost

        self.foundation_general_costs_sums["TOTAL"] = self.foundation_general_costs_sums[
            "total_production"] + self.foundation_general_costs_sums["project"]

    def high_speed_drivetrain(self, rotor_diameter, rated_torque, rated_rpm):
        '''Calculate the cost of high speed drivetrain [Euro]'''

        comp_name = "hs_drivetrain"

        self.turbine_component_mass[comp_name] = {
            "bedplate": 0.0 + 2.4 * rotor_diameter ** 2.0,
            "main_shaft": 0.0 + 0.02 * rotor_diameter ** 2.8,
            "main_bearings": 0.0 + 0.02 * rotor_diameter ** 2.5,
            "bearing_housing": 0.0 + 0.03 * rotor_diameter ** 2.5,
            "gearbox": 0.0 + 15.0e3 * rated_torque ** 1.0,
            "coupling_&_brake": 500.0 + 50.0 * (rated_torque * rated_rpm / 1.5) ** 1.0,
            "generator": 1.0e3 + 400.0 * (rated_torque * rated_rpm / 1.5) ** 1.0}

        self.turbine_component_costs[comp_name] = {
            "bedplate": 2.5 *
            self.turbine_component_mass[comp_name]["bedplate"],
            "main_shaft": 5.0 *
            self.turbine_component_mass[comp_name]["main_shaft"],
            "main_bearings": 15.0 *
            self.turbine_component_mass[comp_name]["main_bearings"],
            "bearing_housing": 2.5 *
            self.turbine_component_mass[comp_name]["bearing_housing"],
            "gearbox": 8.0 *
            self.turbine_component_mass[comp_name]["gearbox"],
            "coupling_&_brake": 8.0 *
            self.turbine_component_mass[comp_name]["coupling_&_brake"],
            "generator": 8.0 *
            self.turbine_component_mass[comp_name]["generator"]}

        self.turbine_component_mass_sums[comp_name] = sum(
            self.turbine_component_mass[comp_name].values())
        self.turbine_component_costs_sums[comp_name] = sum(
            self.turbine_component_costs[comp_name].values())

    def medium_speed_drivetrain(self, rotor_diameter, rated_torque, rated_rpm):
        '''Calculate the cost of medium speed drivetrain [Euro]'''

        comp_name = "ms_drivetrain"

        self.turbine_component_mass[comp_name] = {
            "bedplate": 0.0 + 1.2 * rotor_diameter ** 2.0,
            "main_shaft": 0.0 + 0.02 * rotor_diameter ** 2.8,
            "main_bearings": 0.0 + 0.02 * rotor_diameter ** 2.5,
            "bearing_housing": 0.0 + 0.03 * rotor_diameter ** 2.5,
            "gearbox": 0.0 + 1.0e4 * rated_torque ** 1.0,
            "coupling_&_brake": 0.0 + 30.0 * (rated_torque * rated_rpm / 0.4) ** 1.0,
            "generator": 6.0e3 + 100.0 * (rated_torque * rated_rpm / 0.4) ** 1.0}

        self.turbine_component_costs[comp_name] = {
            "bedplate": 2.5 *
            self.turbine_component_mass[comp_name]["bedplate"],
            "main_shaft": 5.0 *
            self.turbine_component_mass[comp_name]["main_shaft"],
            "main_bearings": 15.0 *
            self.turbine_component_mass[comp_name]["main_bearings"],
            "bearing_housing": 2.5 *
            self.turbine_component_mass[comp_name]["bearing_housing"],
            "gearbox": 8.0 *
            self.turbine_component_mass[comp_name]["gearbox"],
            "coupling_&_brake": 8.0 *
            self.turbine_component_mass[comp_name]["coupling_&_brake"],
            "generator": 8.0 *
            self.turbine_component_mass[comp_name]["generator"]}

        self.turbine_component_mass_sums[comp_name] = sum(
            self.turbine_component_mass[comp_name].values())
        self.turbine_component_costs_sums[comp_name] = sum(
            self.turbine_component_costs[comp_name].values())

    def direct_drive_drivetrain(self, rotor_diameter, rated_torque):
        '''Calculate the cost of direct drive drivetrain [Euro]'''

        comp_name = "dd_drivetrain"

        self.turbine_component_mass[comp_name] = {
            "bedplate": 0.0 + 1.2 * rotor_diameter ** 2.0,
            "generator": 0.0 + 35.0e3 * rated_torque ** 0.666666666667}

        self.turbine_component_costs[comp_name] = {
            "bedplate": 2.5 * self.turbine_component_mass[comp_name]["bedplate"],
            "generator": 8.0 * self.turbine_component_mass[comp_name]["generator"]}

        self.turbine_component_mass_sums[comp_name] = sum(
            self.turbine_component_mass[comp_name].values())
        self.turbine_component_costs_sums[comp_name] = sum(
            self.turbine_component_costs[comp_name].values())
