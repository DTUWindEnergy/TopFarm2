'''
Created on 23. mar. 2018

@author: mmpe
'''
import numpy as np
from ctypes import *
from topfarm.cost_models.fuga.pascal_dll import PascalDLL
import ctypes
from openmdao.core.explicitcomponent import ExplicitComponent
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent




c_double_p = POINTER(c_double)
c_int_p = POINTER(ctypes.c_int32)


class PyFuga(object):
#    def __init__(self, dll_path, farms_dir, farm_name='Horns Rev 1', turbine_model_path='./LUT/',
#                 mast_position=(0,0,70), z0=0.0001, zi=400, zeta0=0, wind_atlas_path='Horns Rev 1\hornsrev.lib'):
    def __init__(self):
        path = r'C:\Sandbox\Topfarm\Colonel\FugaLib/'
        self.lib = PascalDLL(path + 'FugaLib.dll')
#
        self.lib.setup(path + '../LUT/Farms/', 'Horns Rev 1', path + '../LUT/',
                       0., 0., 70.,
                       0.0001, 400., 0.,
                       'Horns Rev 1\hornsrev.lib')

    def get_no_tubines(self):
        no_turbines_p = c_int_p(c_int(0))
        self.lib.getNoTurbines(no_turbines_p)
        return no_turbines_p.contents.value

    def move_turbines(self, tb_x, tb_y):
        assert len(tb_x) == len(tb_y) == self.get_no_tubines()
        tb_x_ctype = np.array(tb_x, dtype=np.float).ctypes
        tb_y_ctype = np.array(tb_y, dtype=np.float).ctypes

        self.lib.MoveTurbines(tb_x_ctype.data_as(c_double_p), tb_y_ctype.data_as(c_double_p))
        
    def get_aep(self, tb_x, tb_y):
        self.move_turbines(tb_x, tb_y)

        AEPNet_p = c_double_p(c_double(0))
        AEPGros_p = c_double_p(c_double(0))
        capacity_p = c_double_p(c_double(0))
        self.lib.getAEP(AEPNet_p, AEPGros_p, capacity_p)
        #print(tb_x, tb_y, AEPNet_p.contents.value, (15.850434458235156 - AEPNet_p.contents.value) / .000001)
        print(AEPNet_p.contents.value)
        return (AEPNet_p.contents.value, AEPGros_p.contents.value, capacity_p.contents.value)

    def get_aep_gradients(self, tb_x, tb_y):
        self.move_turbines(tb_x, tb_y)

        dAEPdxyz = np.zeros(len(tb_x)), np.zeros(len(tb_x)), np.zeros(len(tb_x))
        dAEPdxyz_ctype = [dAEP.ctypes for dAEP in dAEPdxyz]
        self.lib.getAEPGradients(*[dAEP_ctype.data_as(c_double_p) for dAEP_ctype in dAEPdxyz_ctype])
        #print(tb_x, tb_y, dAEPdxyz)
        return np.array(dAEPdxyz)

    def get_TopFarm_cost_component(self):
        n_wt = self.get_no_tubines()
        return AEPCostModelComponent(n_wt,
                                     lambda *args: self.get_aep(*args)[0],  # only aep
                                     lambda *args: self.get_aep_gradients(*args)[:2])  # only dAEPdx and dAEPdy


#         self.execute("""seed=0
# initialize
# set output file "out.txt"
# load farm "Horns Rev 1"
# 7 point integration off
# meandering off
# insert met mast 0.0 0.0 70.0
# z0=0.0001
# zi=400.0
# zeta0=0.0
# load wakes
# gaussian fit on
# proximity penalty off
# gradients on
# Wind climates interpolation on
# load wind atlas "Horns Rev 1\hornsrev2.lib"
# relative move all turbines -426733.0 -6149501.5
# gradients off
# calculate AEP
# get Farm AEP""")

    def execute(self, script):
        res = self.lib.ExecuteScript(script.encode())
        print("#" + str(res) + "#")


if __name__ == '__main__':

    path = r'C:\Sandbox\Topfarm\Colonel/'
    dll_path = path + 'FugaLib/FugaLib.dll'
    farms_dir = path + 'LUT/Farms/'
    farm_name = 'Horns Rev 1'
    turbine_model_path = path + 'LUT/'
    mast_position = (0., 0., 70.)
    z0, zi, zeta0 = 0.0001, 400., 0.,
    wind_atlas_path = 'Horns Rev 1\hornsrev.lib'
    pyFuga = PyFuga()
#    pyFuga = PyFuga(dll_path, farms_dir, farm_name, turbine_model_path, mast_position, z0,zi,zeta0, wind_atlas_path)
    print(pyFuga.get_no_tubines())
    print(pyFuga.get_aep([0, 0], [0, 1000]))
    print(pyFuga.get_aep([0, 1000], [0, 0]))
    print(pyFuga.get_aep_gradients([0, 0], [0, 100]))
    print(pyFuga.get_aep([0, 0], [0, 100]))
    print(pyFuga.get_aep([0, 0], [0, 101]))
    print(pyFuga.get_aep_gradients([0, 0], [0, 200]))
