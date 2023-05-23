# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:55:12 2021

@author: mikf
"""
import numpy as np
from py_wake.examples.data.ParqueFicticio import ParqueFicticio_path
from py_wake.site import WaspGridSite
from py_wake.site.xrsite import XRSite


x = np.asarray([262403., 262553., 262703., 262853., 263003., 263153., 263303.,
                263453., 263603., 263753., 263903., 264053., 264203., 264353.,
                264503., 264653., 264803., 264953., 265103., 265253.])
y = np.asarray([6504239., 6504389., 6504539., 6504689., 6504839., 6504989.,
                6505139., 6505289., 6505439., 6505589., 6505739., 6505889.,
                6506039., 6506189., 6506339., 6506489., 6506639., 6506789.,
                6506939., 6507089.])
wt_x = np.asarray([264904, 264372, 263839, 264904, 264372, 263839, 263306,
                   264638, 264105, 263572, 263039, 264372, 263839, 263039, 264358,
                   263839, 263039, 263839, 263306, 262773, 263306, 262773, 263039])
wt_y = np.asarray([6505613, 6505016, 6504420, 6506063, 6505467, 6504870,
                   6504273, 6506215, 6505619, 6505022, 6504425, 6506368, 6505771,
                   6504876, 6506803, 6506221, 6505326, 6506672, 6506075, 6505478,
                   6506525, 6505929, 6506677])

x_min_d = x.min()
x_max_d = x.max()
y_min_d = y.min()
y_max_d = y.max()

boundary = np.asarray([[x_min_d, y_max_d], [x_max_d, y_max_d],
                       [x_max_d, y_min_d], [x_min_d, y_min_d]])


class ParqueFicticioOffshore(WaspGridSite, XRSite):
    def __init__(self):
        site = self.from_wasp_grd(ParqueFicticio_path, speedup_using_pickle=False)
        site.ds['water_depth'] = - site.ds['Elevation'] / 10
        ds = site.ds.drop_vars(['flow_inc', 'ws_mean', 'orog_spd', 'Turning',
                                'Elevation', 'Speedup'])
        ds['x'] = x
        ds['y'] = y
        XRSite.__init__(self, ds)
        self.boundary = boundary
        self.initial_position = np.array([wt_x, wt_y]).T


def main():
    if __name__ == '__main__':
        ParqueFicticioOffshore()


main()
