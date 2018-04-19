'''
Created on 19/04/2018

@author: Mads
'''
import numpy as np


class AEPCalculator(object):

    def __init__(self, wdir=np.arange(360), wsp=np.arange(3, 25), wind_resource, wake_model):
        self.wdir = wdir
        self.wsp = wsp
        self.wind_resource = wind_resource
        self.wake_model = wake_model

    def calculate_aep(self, turbine_positions):
        no_wake_wsp, no_wake_ti, weight = self.wind_resource(turbine_positions, self.wdir, self.wsp)
        wake_wsp, power, ct = self.wake_model(turbine_positions, no_wake_wsp, no_wake_ti)
        return np.sum(power * weight) * 24 * 365
