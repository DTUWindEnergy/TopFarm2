from py_wake.tests.notebook import Notebook as PyWakeNotebook
import os
from topfarm.easy_drivers import EasyDriverBase


class Notebook(PyWakeNotebook):
    pip_header = """# Install TopFarm2 if needed
try:
    import topfarm
except ModuleNotFoundError:
    !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/TopFarm2.git"""

    def check_code(self):
        try:
            EasyDriverBase.max_iter = 1
            return PyWakeNotebook.check_code(self)
        finally:
            EasyDriverBase.max_iter = None


if __name__ == '__main__':
    nb = Notebook('elements/v80.ipynb')
    nb.check_code()
    nb.check_links()
