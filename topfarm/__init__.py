from ._topfarm import *
import pkg_resources
from ._version import __version__

__release__ = __version__

plugins = {
    entry_point.name: entry_point.load()
    for entry_point
    in pkg_resources.iter_entry_points('topfarm.plugins')
}
x_key = 'x'
y_key = 'y'
z_key = 'z'
type_key = 'type'
cost_key = 'cost'
grid_x_key = 'sx'
grid_y_key = 'sy'
