"""
turbine_costsse_2017.py

Created by Arvydas Berzonskis  Goldwind 2017 based on turbine_costsse_2015.py 2015.
Copyright (c) NREL. All rights reserved.
"""
import numpy as np
import numpy_financial as npf
import math
import matplotlib.pyplot as plt
from openmdao.core.explicitcomponent import ExplicitComponent
plt.close("all")

# Rotor
# -------------------------------------------------------------------------------


class economic_evaluation():

    def __init__(self, D_rotor_array, Power_rated_array, hub_height_array, aep_array, electrical_connection_cost=None, tower_cost=None):
        self.D_rotor_array = D_rotor_array
        self.Power_rated_array = Power_rated_array
        self.hub_height_array = hub_height_array
        self.aep_array = aep_array
        self.electrical_connection_cost = electrical_connection_cost
        self.tower_cost = tower_cost

    def calculate_irr(self):

        #        blade_type_array = [''] * len(D_rotor_array)
        # turbine_type_array = list(map(lambda x, y: self.reflection_turbine_type(x, y),
        #                           D_rotor_array, Power_rated_array))
        # weight_array = list(map(lambda x: self.simple_weight_height_trans(x),
        #                     hub_height_array))
        self.IRR = 0
        aep_vector = np.array([[float(item)] for item in self.aep_array])
        if sum(aep_vector) > 0:
            # turbine_type_vector = [[item] for item in turbine_type_array]
            machine_rating = np.array([int(item) for item in self.Power_rated_array])
            rotor_diameter = np.array([int(item) for item in self.D_rotor_array])
            hub_height = np.array([int(item) for item in self.hub_height_array])
            # weight_vector = [[float(item)] for item in weight_array]
            aep_vector = np.array([float(item) for item in self.aep_array])

        # def __init__(self, rotor_diameter, machine_rating, hub_height, aep_vector):
            # calculate the blade mass and cost
            # Baseline mode
            self.blade_B_mass = 3 * 0.1452 * (rotor_diameter / 2)**2.9158  # all 3 blades
            # Advanced mode
            self.blade_A_mass = 0.4948 * (rotor_diameter / 2)**2.53  # all 3 blades
            # Blade material cost escalator
            self.BCE = 1
            # Labor cost escalator
            self.GDPE = 1
            # Costs
            self.blade_B_costs = 3 * ((0.4019 * (rotor_diameter / 2)**3 - 955.24) * self.BCE +
                                      2.7445 * (rotor_diameter / 2)**2.5025 * self.GDPE) / (1 - 0.28)
            self.blade_A_costs = 3 * ((0.4019 * (rotor_diameter / 2)**3 - 21051) * self.BCE +
                                      2.7445 * (rotor_diameter / 2)**2.5025 * self.GDPE) / (1 - 0.28)

            # calculate the Hub cost and weight
            self.hub_mass = 0.954 * (self.blade_B_mass / 3) + 5680.3
            self.hub_cost = self.hub_mass * 4.25

            # Pitch mechanisms and bearings
            self.pich_bearing_mass = 0.1295 * self.blade_B_mass + 491.31
            self.pitch_system_mass = self.pich_bearing_mass * 1.328 + 555
            # Total pitch costs
            self.pitch_system_cost = 2.28 * (0.2106 * rotor_diameter**2.6578)  # All 3 blades

            # Spinner, nose cone
            self.nose_cone_mass = 18.5 * rotor_diameter - 520.5
            self.nose_cone_cost = self.nose_cone_mass * 5.57

            # Low-speed shaft
            ''' Notes might not be used for direct drive turbine costs'''
            self.low_speed_shaft_mass = 0.0142 * rotor_diameter**2.888
            self.low_speed_shaft_cost = 0.01 * rotor_diameter**2.887

            # Main bearings
            self.bearing_mass = (rotor_diameter * 8 / 600 - 0.033) * 0.0092 * rotor_diameter**2.5
            self.bearing_cost = 2 * self.bearing_mass * 17.6

            # Mecahnical brake, high-speed coupling and associated components
            self.brake_and_coupling_cost = 1.9894 * machine_rating - 0.1141
            self.brake_and_coupling_mass = self.brake_and_coupling_cost / 10.

            # Direct drive Generator
            # self.generator_mass=661.25*self.low_speed_shaft_torque**0.606
            self.generator_cost = machine_rating * 219.33

            # Variable-speed electronics
            self.variablespeed_electronics = machine_rating * 79.

            # Yaw Drive and Bearing
            self.yaw_system_mass = 1.6 * (0.00098 * rotor_diameter**3.314)
            self.yaw_system_cost = 2 * (0.0339 * rotor_diameter**2.964)

            # Mainframe - Direct Drive
            self.mainframe_mass = 1.228 * rotor_diameter**1.953
            self.mainframe_cost = 627.28 * rotor_diameter**0.85

            # Platforms and railings
            self.platform_railing_mass = 0.125 * self.mainframe_mass
            self.platform_railing_cost = self.platform_railing_mass * 8.7

            # Electrical connections
            if not self.electrical_connection_cost:
                self.electrical_connection_cost = machine_rating * 40.

            # Hydraulic and Cooling Systems
            self.hydraulic_cooling_system_mass = 0.08 * machine_rating
            self.hydraulic_cooling_system_cost = machine_rating * 12

            # Nacelle Cover
            self.nacelle_cost = 11.537 * machine_rating + 3849.7
            self.nacelle_mass = self.nacelle_cost / 10.

            # Control, Safety Sytem, Condition Monitoring
            self.control_cost = 35000.0

            # Tower
            # Baseline model
            if not self.tower_cost:
                self.tower_B_mass = 0.3973 * (math.pi * (rotor_diameter / 2)**2) * hub_height - 1414
                # Advanced model
                self.tower_A_mass = 0.2694 * (math.pi * (rotor_diameter / 2)**2) * hub_height - 1779
                self.tower_cost = self.tower_B_mass * 1.5

            # Foundation cost
            self.foundation_cost = 303.24 * (hub_height * (math.pi * (rotor_diameter / 2))**2)**0.4037

            # Transportation cost
            self.trasport_coeff = 1.581e-5 * machine_rating**2 - 0.0375 * machine_rating + 54.7
            self.trasport_cost = machine_rating * self.trasport_coeff

            # Roads and Civil work
            self.roads_civil_cost_fact = 2.17e-6 * machine_rating**2 - 0.0145 * machine_rating + 69.54
            self.roads_civil_cost_cost = self.roads_civil_cost_fact * machine_rating

            # Assembly and Installation
            self.assembly_and_installation_cost = 1.965 * (hub_height * rotor_diameter)**1.1736

            # Electrical Interface and connections
            self.electrical_interface_fact = 3.49e-6 * machine_rating**2 - 0.0221 * machine_rating + 109.7
            self.electrical_interface_cost = machine_rating * self.electrical_interface_fact

            # Annual operating expenses - AOE
            self.LCC = 0.00108 * sum(aep_vector)
            self.OandM = 0.007 * sum(aep_vector)
            self.LRC = sum(10.7 * machine_rating)
            self.AOE = self.LCC + (self.OandM + self.LRC) / sum(aep_vector)

            # I do not know how to calculate the weight
            # self.variable_speed_elec_mass_cost=18.8*variable_speed_elec_mass

            self.cost = 1.33 * (self.blade_B_costs + self.hub_cost + self.pitch_system_cost + self.nose_cone_cost + self.bearing_cost +
                                self.brake_and_coupling_cost + self.generator_cost + self.variablespeed_electronics + self.yaw_system_cost +
                                self.mainframe_cost + self.platform_railing_cost + self.electrical_connection_cost +
                                self.hydraulic_cooling_system_cost + self.nacelle_cost + self.control_cost + self.tower_cost +
                                self.foundation_cost + self.trasport_cost + self.roads_civil_cost_cost + self.assembly_and_installation_cost +
                                self.electrical_interface_cost)  # the ratio of 1.33 is to account for inflation since 2003.

            self.cost[self.aep_array == 0] = 0

            self.CWF = []

            for i in range(1, 20):
                if i == 1:
                    self.CWF.append(int(0.1 * sum(aep_vector) - sum(self.cost) - self.AOE))
                else:
                    self.CWF.append(int(0.1 * sum(aep_vector) - self.AOE))

            self.IRR = 100 * npf.irr(self.CWF)
        return self.IRR


def main():
    if __name__ == '__main__':
        Drotor_vector = np.array([121.0, 115.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0,
                                  121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 121.0, 115.0])
        power_rated_vector = np.array([2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.,
                                       2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.])
        hub_height_vector = np.array([105., 105., 105., 105., 105., 105., 105., 105., 105., 105., 105., 105., 105., 105.,
                                      105., 105., 105., 105., 105., 105., 105., 105., 105., 105.])
        aep_vector = np.array([7719409.76158781, 7703877.645459095, 6034197.878645113, 6494510.809393061, 8147512.144511054,
                               7369147.974147366, 8681084.747836143, 7147269.951808259, 6141634.901501132, 5682294.113683184,
                               6859259.383656042, 7447180.72443833, 5761350.543959712, 6755014.03150229, 6337664.079841179,
                               5151511.151202235, 5456811.610073241, 6722954.6423807535, 7185180.112344236, 7215858.2123201955,
                               7136356.084546783, 6267802.418413456, 5899094.2746176915, 6592877.854876981])
        # initialize class
        Turbine = economic_evaluation(Drotor_vector, power_rated_vector, hub_height_vector, aep_vector)  # Rotor diameter; Power rated; Hub Height
        # run the method for IRR calculations
        Turbine.calculate_irr()

        print('Wind turbine configuration costs', Turbine.cost)
        print('IRR', Turbine.IRR)

        #    fig,ax = plt.subplots()
        #    ax.plot(Drotor_vector,economic_evaluation.cost)
        #    ax.set(xlabel='Rotor diameter [m]', ylabel='Initial costs [$]')

        # just the cost bar lot
        barplotvector = np.asarray([Turbine.blade_B_costs[0], Turbine.hub_cost[0], Turbine.pitch_system_cost[0], Turbine.nose_cone_cost[0],
                                    Turbine.bearing_cost[0], Turbine.brake_and_coupling_cost[0], Turbine.generator_cost[0],
                                    Turbine.variablespeed_electronics[0], Turbine.yaw_system_cost[0], Turbine.mainframe_cost[0],
                                    Turbine.platform_railing_cost[0], Turbine.electrical_connection_cost[0],
                                    Turbine.hydraulic_cooling_system_cost[0], Turbine.nacelle_cost[0], 35000,
                                    Turbine.tower_cost[0], Turbine.foundation_cost[0], Turbine.trasport_cost[0], Turbine.roads_civil_cost_cost[0],
                                    Turbine.assembly_and_installation_cost[0], Turbine.electrical_interface_cost[0]] / Turbine.cost[0])
        N = 21
        ind = np.arange(N)  # the x locations for the groups
        width = 0.5       # the width of the bars

        plt.rcParams.update({'figure.autolayout': True})
        fig, ax = plt.subplots()
        ax.barh(ind, barplotvector, width, color='b')

        # add some text for labels, title and axes ticks
        ax.set_xlabel('Procentage of total costs [%]')
        ax.set_title('Wind turbine components costs')
        ax.set_yticks(ind + width / 2)
        ax.set_yticklabels(('Blades', 'Hub', 'Pitch System', 'Nose cone', 'Bearing', 'Break and coupling', 'Generator',
                            'Variable speed electornics', 'Yaw system', 'Mainframe', 'Platform railing', 'Electical connections',
                            'Hydraulic cooling systems', 'Nacelle cover', 'Control', 'Tower', 'Foundation', 'Transportation',
                            'Roads', 'Assembly and installation', 'Electrical interface'))
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=45, horizontalalignment='right')
        plt.show()


main()
