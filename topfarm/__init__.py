import sys
from ._topfarm import TopFarmProblem, TopFarmGroup, TopFarmParallelGroup, TopFarmBaseGroup, ProblemComponent
from .recorders import TopFarmListRecorder
from importlib.metadata import entry_points

__all__ = [
    "TopFarmProblem",
    "TopFarmGroup",
    "TopFarmParallelGroup",
    "TopFarmBaseGroup",
    "ProblemComponent",
    "TopFarmListRecorder",
    "x_key",
    "y_key",
    "z_key",
    "type_key",
    "cost_key",
    "grid_x_key",
    "grid_y_key",
    "plugins",
]
from ._version import __version__
from .constants import x_key, y_key, z_key, type_key, cost_key, grid_x_key, grid_y_key

__release__ = __version__

entry_pts = (  # change in the entry_points() API between Python 3.9 and 3.10
    entry_points().get("topfarm.plugins", [])
    if sys.version_info < (3, 10)
    else entry_points().select(group="topfarm.plugins")
)

plugins = {entry_point.name: entry_point.load() for entry_point in entry_pts}
