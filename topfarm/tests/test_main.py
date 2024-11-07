import importlib
import os
import pkgutil
import warnings
import mock
import pytest
import topfarm
import matplotlib.pyplot as plt
import sys
from topfarm.easy_drivers import EasyDriverBase


def get_main_modules():
    package = topfarm
    modules = []
    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if 'Colonel' in modname:
                continue
            if "deprecated" in modname:
                continue
            if "test" in modname:
                continue
            m = importlib.import_module(modname)

        if 'main' in dir(m):
            modules.append(m)
    return modules


def print_main_modules():
    print("\n".join([m.__name__ for m in get_main_modules()]))


@pytest.mark.parametrize("module", get_main_modules())
def test_main(module):
    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    def no_print(s, t=None):
        pass

    try:
        EasyDriverBase.max_iter = 1
        with mock.patch.object(module, "__name__", "__main__"):
            with mock.patch.object(module, "print", no_print):
                getattr(module, 'main')()
    except Exception as e:
        raise type(e)(str(e) + ' in %s.main' % module.__name__).with_traceback(sys.exc_info()[2])
    finally:
        EasyDriverBase.max_iter = None
        plt.close()


if __name__ == '__main__':
    print_main_modules()
