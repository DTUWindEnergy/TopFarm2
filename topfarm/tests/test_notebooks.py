import os
from _notebooks.notebook import Notebook
import pytest
import topfarm
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.drivers.random_search_driver import RandomSearchDriver
from topfarm.easy_drivers import EasyDriverBase
from py_wake.tests.check_speed import timeit


def get_notebooks():
    path = os.path.dirname(topfarm.__file__) + "/../_notebooks/elements/"
    return [Notebook(path + f) for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]]


@pytest.mark.parametrize("notebook", get_notebooks())
def test_notebooks(notebook):
    if os.path.basename(notebook.filename) in ['loads.ipynb', 'roads_and_cables.ipynb',
                                               'wake_steering_and_loads.ipynb', 'layout_and_loads.ipynb']:
        pytest.xfail("Notebook, %s, has known issues" % notebook)
    import matplotlib.pyplot as plt

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot
    print(notebook.filename)
    try:
        notebook.check_code()
        notebook.check_links()
    except Exception as e:
        raise Exception(notebook.filename + " failed") from e
    finally:
        plt.close()


if __name__ == '__main__':
    print([os.path.basename(n.filename) for n in get_notebooks()])
