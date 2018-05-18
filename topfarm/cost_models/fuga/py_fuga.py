'''
Created on 23. mar. 2018

@author: mmpe
'''
from _ctypes import POINTER
import atexit
from ctypes import c_double, c_int
import ctypes
import os
from tempfile import NamedTemporaryFile

import numpy as np
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from topfarm.cost_models.fuga.pascal_dll import PascalDLL


c_double_p = POINTER(c_double)
c_int_p = POINTER(ctypes.c_int32)

fuga_path = os.path.abspath(os.path.dirname(__file__)) + '/Colonel/'


class PyFuga(object):
    interface_version = 3

    def __init__(self):
        atexit.register(self.cleanup)
        with NamedTemporaryFile() as f:
            self.stdout_filename = f.name + "pyfuga.txt"

        lib_path = os.path.dirname(__file__) + "/Colonel/FugaLib/FugaLib.%s" % ('so', 'dll')[os.name == 'nt']
        if os.path.isfile(lib_path) is False:
            raise Exception("Fuga lib cannot be found: '%s'" % lib_path)

        self.lib = PascalDLL(lib_path)
        self.lib.CheckInterfaceVersion(self.interface_version)
        self.lib.Initialize(self.stdout_filename)

    def setup(self, farm_name='Horns Rev 1',
              turbine_model_path='./LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
              tb_x=[423974, 424033], tb_y=[6151447, 6150889],
              mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
              farms_dir='./LUT/', wind_atlas_path='Horns Rev 1\hornsrev.lib', climate_interpolation=True):
        self.lib.Setup(float(mast_position[0]), float(mast_position[1]), float(mast_position[2]),
                       float(z0), float(zi), float(zeta0))

        tb_x_ctype = np.array(tb_x, dtype=np.float).ctypes
        tb_y_ctype = np.array(tb_y, dtype=np.float).ctypes
        assert len(tb_x) == len(tb_y)

        self.lib.AddWindFarm(farm_name, turbine_model_path, turbine_model_name,
                             len(tb_x), tb_x_ctype.data_as(c_double_p), tb_y_ctype.data_as(c_double_p))

        assert os.path.isfile(farms_dir + wind_atlas_path), farms_dir + wind_atlas_path
        self.lib.SetupWindClimate(farms_dir, wind_atlas_path, climate_interpolation)

        #assert len(tb_x) == self.get_no_tubines(), self.log + "\n%d!=%d" % (self.get_no_tubines(),len(tb_x))

    def cleanup(self):
        if hasattr(self, 'lib'):
            try:
                self.lib.Exit()  # raises exception
            except:
                pass
            del self.lib
        tmp_folder = os.path.dirname(self.stdout_filename)
        for f in os.listdir(tmp_folder):
            if f.endswith('pyfuga.txt'):
                try:
                    os.remove(os.path.join(tmp_folder, f))
                except Exception:
                    pass

    def get_no_tubines(self):
        no_turbines_p = c_int_p(c_int(0))
        self.lib.GetNoTurbines(no_turbines_p)
        return no_turbines_p.contents.value

    def move_turbines(self, tb_x, tb_y):
        assert len(tb_x) == len(tb_y) == self.get_no_tubines(), (len(tb_x), len(tb_y), self.get_no_tubines())
        tb_x_ctype = np.array(tb_x, dtype=np.float).ctypes
        tb_y_ctype = np.array(tb_y, dtype=np.float).ctypes

        self.lib.MoveTurbines(tb_x_ctype.data_as(c_double_p), tb_y_ctype.data_as(c_double_p))

    def get_aep(self, turbine_positions=None):
        if turbine_positions is not None:
            self.move_turbines(turbine_positions[:, 0], turbine_positions[:, 1])

        AEPNet_p = c_double_p(c_double(0))
        AEPGros_p = c_double_p(c_double(0))
        capacity_p = c_double_p(c_double(0))
        self.lib.GetAEP(AEPNet_p, AEPGros_p, capacity_p)
        #print(tb_x, tb_y, AEPNet_p.contents.value, (15.850434458235156 - AEPNet_p.contents.value) / .000001)
        net, gros, cap = [p.contents.value for p in [AEPNet_p, AEPGros_p, capacity_p]]
        return (net, gros, cap, net / gros)

    def get_aep_gradients(self, turbine_positions=None):
        if turbine_positions is not None:
            self.move_turbines(turbine_positions[:, 0], turbine_positions[:, 1])

        n_wt = self.get_no_tubines()
        dAEPdxyz = np.zeros(n_wt), np.zeros(n_wt), np.zeros(n_wt)
        dAEPdxyz_ctype = [dAEP.ctypes for dAEP in dAEPdxyz]
        self.lib.GetAEPGradients(*[dAEP_ctype.data_as(c_double_p) for dAEP_ctype in dAEPdxyz_ctype])
        #print(tb_x, tb_y, dAEPdxyz)
        return np.array(dAEPdxyz)

    def get_TopFarm_cost_component(self):
        n_wt = self.get_no_tubines()
        return AEPCostModelComponent(n_wt,
                                     lambda *args: self.get_aep(*args)[0],  # only aep
                                     lambda *args: self.get_aep_gradients(*args)[:2])  # only dAEPdx and dAEPdy

    @property
    def log(self):
        with open(self.stdout_filename) as fid:
            return fid.read()

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


def try_me():
    if __name__ == '__main__':
        pyFuga = PyFuga()
        pyFuga.setup(farm_name='Horns Rev 1',
                     turbine_model_path=fuga_path + 'LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
                     tb_x=[423974, 424033], tb_y=[6151447, 6150889],
                     mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
                     farms_dir=fuga_path + 'LUT/Farms/', wind_atlas_path='Horns Rev 1\hornsrev.lib')

        print(pyFuga.get_no_tubines())
        print(pyFuga.get_aep(np.array([[0, 0], [0, 1000]])))
        print(pyFuga.get_aep(np.array([[0, 1000], [0, 0]])))
        print(pyFuga.get_aep_gradients(np.array([[0, 0], [0, 100]])))


try_me()
