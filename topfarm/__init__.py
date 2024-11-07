from ._topfarm import *
import pkg_resources

plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('topfarm.plugins')
}

__version__ = 'filled by setup.py'
__release__ = 'filled by setup.py'


x_key = 'x'
y_key = 'y'
z_key = 'z'
type_key = 'type'
cost_key = 'cost'
grid_x_key = 'sx'
grid_y_key = 'sy'
