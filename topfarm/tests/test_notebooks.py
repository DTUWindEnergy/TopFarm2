import os

import pytest

from topfarm.tests.notebook import Notebook
import topfarm


def get_notebooks():
    def get(path):
        return [Notebook(path + f) for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]]
    path = os.path.dirname(topfarm.__file__) + "/../docs/notebooks/"
    return get(path)


notebooks = get_notebooks()


@pytest.mark.parametrize("notebook", notebooks, ids=[os.path.basename(nb.filename) for nb in notebooks])
def test_notebooks(notebook):
    if os.path.basename(notebook.filename) in [
        "layout_and_loads.ipynb",  # gives error from tensorflow on synnefo machine
        "roads_and_cables.ipynb",  # fails
        "wake_steering_and_loads.ipynb",  # ok but many warnings from tensorflow
        "wind_farm_cluster.ipynb",  # too long runtime
        "MongoDB_recorder.ipynb",  # deprecated
    ]:
        pytest.xfail("Notebook, %s, has known issues" % notebook)
    import matplotlib.pyplot as plt

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    try:
        plt.rcParams.update({'figure.max_open_warning': 0})
        notebook.check_code()
        notebook.check_links()
        notebook.remove_empty_end_cell()
        notebook.check_pip_header()
        pass
    except Exception as e:
        raise Exception(notebook.filename + " failed") from e
    finally:
        plt.close('all')
        plt.rcParams.update({'figure.max_open_warning': 20})


if __name__ == '__main__':
    print("\n".join([f.filename for f in get_notebooks()]))
