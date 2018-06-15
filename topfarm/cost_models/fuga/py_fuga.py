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
fugalib_path = os.path.dirname(__file__) + "/Colonel/FugaLib/FugaLib.%s" % ('so', 'dll')[os.name == 'nt']


class PyFuga(object):
    interface_version = 3

    def __init__(self):
        atexit.register(self.cleanup)
        with NamedTemporaryFile() as f:
            self.stdout_filename = f.name + "pyfuga.txt"

        if os.path.isfile(fugalib_path) is False:
            raise Exception("Fuga lib cannot be found: '%s'" % fugalib_path)

        self.lib = PascalDLL(fugalib_path)
        self.lib.CheckInterfaceVersion(self.interface_version)
        self.lib.Initialize(self.stdout_filename)

    def setup(self, farm_name='Horns Rev 1',
              turbine_model_path='./LUT/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=67.00]',
              tb_x=[423974, 424033], tb_y=[6151447, 6150889],
              mast_position=(0, 0, 70), z0=0.0001, zi=400, zeta0=0,
              farms_dir='./LUT/', wind_atlas_path='Horns Rev 1\hornsrev.lib', climate_interpolation=True):
        self.lib.Setup(float(mast_position[0]), float(mast_position[1]), float(mast_position[2]),
                       float(z0), float(zi), float(zeta0))

        assert len(tb_x) == len(tb_y)
        tb_x_ctype, tb_y_ctype = self.tb_ctypes(tb_x, tb_y)

        self.lib.AddWindFarm(farm_name, turbine_model_path, turbine_model_name,
                             len(tb_x), tb_x_ctype.data_as(c_double_p), tb_y_ctype.data_as(c_double_p))

        assert os.path.isfile(farms_dir + wind_atlas_path), farms_dir + wind_atlas_path
        self.lib.SetupWindClimate(farms_dir, wind_atlas_path, climate_interpolation)

        assert len(tb_x) == self.get_no_turbines(), self.log + "\n%d!=%d" % (self.get_no_turbines(), len(tb_x))

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

    def get_no_turbines(self):
        no_turbines_p = c_int_p(c_int(0))
        self.lib.GetNoTurbines(no_turbines_p)
        return no_turbines_p.contents.value

    def tb_ctypes(self, tb_x, tb_y):
        assert len(tb_x) == len(tb_y)
        # remove mean offset to avoid loosing precision due to high offset
        self.tb_x_offset, self.tb_y_offset = np.mean(tb_x), np.mean(tb_y)
        tb_x = np.array(tb_x, dtype=np.float) - self.tb_x_offset
        tb_y = np.array(tb_y, dtype=np.float) - self.tb_y_offset
        return tb_x.ctypes, tb_y.ctypes

    def move_turbines(self, tb_x, tb_y):
        assert len(tb_x) == len(tb_y) == self.get_no_turbines(), (len(tb_x), len(tb_y), self.get_no_turbines())
        tb_x_ctype, tb_y_ctype = self.tb_ctypes(tb_x, tb_y)
        self.lib.MoveTurbines(tb_x_ctype.data_as(c_double_p), tb_y_ctype.data_as(c_double_p))

    def get_aep(self, turbine_positions=None):
        if turbine_positions is not None:
            self.move_turbines(turbine_positions[:, 0], turbine_positions[:, 1])

        AEPNet_p = c_double_p(c_double(0))
        AEPGros_p = c_double_p(c_double(0))
        capacity_p = c_double_p(c_double(0))
        self.lib.GetAEP(AEPNet_p, AEPGros_p, capacity_p)
        net, gros, cap = [p.contents.value for p in [AEPNet_p, AEPGros_p, capacity_p]]
        return (net, gros, cap, net / gros)

    def get_aep_gradients(self, turbine_positions=None):
        if turbine_positions is not None:
            self.move_turbines(turbine_positions[:, 0], turbine_positions[:, 1])

        n_wt = self.get_no_turbines()
        dAEPdxyz = np.zeros(n_wt), np.zeros(n_wt), np.zeros(n_wt)
        dAEPdxyz_ctype = [dAEP.ctypes for dAEP in dAEPdxyz]
        self.lib.GetAEPGradients(*[dAEP_ctype.data_as(c_double_p) for dAEP_ctype in dAEPdxyz_ctype])
        return np.array(dAEPdxyz)

    def get_TopFarm_cost_component(self):
        n_wt = self.get_no_turbines()
        return AEPCostModelComponent(n_wt,
                                     lambda *args: self.get_aep(*args)[0],  # only aep
                                     lambda *args: self.get_aep_gradients(*args)[:2])  # only dAEPdx and dAEPdy

    @property
    def log(self):
        with open(self.stdout_filename) as fid:
            return fid.read()

    def execute(self, script):
        res = self.lib.ExecuteScript(script.encode())
        print("#" + str(res) + "#")


def try_me():
    if __name__ == '__main__':
        pyFuga = PyFuga()
        pyFuga.setup(farm_name='Horns Rev 1',
                     turbine_model_path=fuga_path + 'LUTs-T/', turbine_model_name='Vestas_V80_(2_MW_offshore)[h=70.00]',
                     tb_x=[423974, 424033], tb_y=[6151447, 6150889],
                     mast_position=(0, 0, 70), z0=0.03, zi=400, zeta0=0,
                     farms_dir=fuga_path + 'LUTs-T/Farms/', wind_atlas_path='MyFarm\DEN05JBgr_7.813E_55.489N_7.4_5.lib')

        print(pyFuga.get_no_turbines())
        print(pyFuga.get_aep(np.array([[0, 0], [0, 1000]])))
        print(pyFuga.get_aep(np.array([[0, 1000], [0, 0]])))
        print(pyFuga.get_aep_gradients(np.array([[0, 0], [0, 100]])))


try_me()
