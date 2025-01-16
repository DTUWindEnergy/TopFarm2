from unittest.mock import patch
import matplotlib.pyplot as plt  # fmt:skip
import os
import uuid
import pytest
from topfarm.tests.notebook import Notebook
import topfarm
import sys
import matplotlib  # fmt:skip
matplotlib.use("Agg")


def no_show(*args, **kwargs):  # fmt:skip
    pass


plt.show = no_show  # disable plt show that requires the user to close the plot
plt.ioff()


def get_notebooks():  # fmt:skip
    def get(path):  # fmt:skip
        return [
            Notebook(path + f)
            for f in [f for f in os.listdir(path) if f.endswith(".ipynb")]
        ]
    path = os.path.dirname(topfarm.__file__) + "/../docs/notebooks/"
    return get(path)


excluded = [
    "layout_and_loads.ipynb",  # gives error from tensorflow on synnefo machine
    "wake_steering_and_loads.ipynb",  # ok but many warnings from tensorflow
    "wind_farm_cluster.ipynb",  # too long runtime
    "MongoDB_recorder.ipynb",  # deprecated
]
if sys.version_info > (3, 10):
    excluded += [
        "cables.ipynb",  # ed_win python requirement is python<3.10
    ]

notebooks = get_notebooks()
notebooks = [nb for nb in notebooks if os.path.basename(nb.filename) not in excluded]
grouped_notebooks = [
    pytest.param(nb, marks=pytest.mark.xdist_group(name=f"{uuid.uuid4()}_group"))
    for nb in notebooks
]


@pytest.mark.parametrize(
    "notebook",
    grouped_notebooks,
    ids=[os.path.basename(nb.filename) for nb in notebooks],
)
def test_notebooks(notebook):
    with patch("matplotlib.pyplot.show", no_show):
        try:
            notebook.check_code()
            notebook.check_links()
            notebook.remove_empty_end_cell()
            notebook.check_pip_header()
        except Exception as e:
            raise Exception(notebook.filename + " failed") from e
