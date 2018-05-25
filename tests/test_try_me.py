import importlib
import os
import pkgutil
import warnings
import mock
import pytest
import topfarm


def get_try_me_modules():
    package = topfarm
    modules = []
    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = importlib.import_module(modname)
        if 'try_me' in dir(m):
            modules.append(m)
    return modules


@pytest.mark.parametrize("module", get_try_me_modules())
def test_try_me(module):
    # check that all try_me module examples run without errors 
    if os.name == 'posix' and "DISPLAY" not in os.environ:
        pytest.xfail("No display")
    print("Checking %s.try_me" % module.__name__)
    with mock.patch.object(module, "__name__", "__main__"):
        getattr(module, 'try_me')()
