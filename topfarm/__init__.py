import sys
from ._topfarm import *
from importlib.metadata import entry_points
from ._version import __version__

__release__ = __version__

entry_pts = (  # change in the entry_points() API between Python 3.9 and 3.10
    entry_points().get("topfarm.plugins", [])
    if sys.version_info < (3, 10)
    else entry_points().select(group="topfarm.plugins")
)

plugins = {entry_point.name: entry_point.load() for entry_point in entry_pts}
x_key = "x"
y_key = "y"
z_key = "z"
type_key = "type"
cost_key = "cost"
grid_x_key = "sx"
grid_y_key = "sy"
