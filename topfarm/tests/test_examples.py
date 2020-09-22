import importlib
import os
import pkgutil
import warnings
import mock
import pytest
import matplotlib.pyplot as plt
import sys
from examples import docs
import subprocess
import contextlib
import io
from topfarm.easy_drivers import EasyDriverBase


def get_main_modules():
    package = docs
    modules = []
    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if 'Colonel' in modname:
                continue
            m = importlib.import_module(modname)

        if 'main' in dir(m):
            modules.append(m)
    return modules


def test_print_main_modules():
    print("\n".join([m.__name__ for m in get_main_modules()]))


@pytest.mark.parametrize("module", get_main_modules())
def test_main(module):
    # check that all main module examples run without errors
    #    if os.name == 'posix' and "DISPLAY" not in os.environ:
    #        pytest.xfail("No display")

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    def no_print(*args, **kwargs):
        pass
    try:
        EasyDriverBase.max_iter = 1
        with mock.patch.object(module, "__name__", "__main__"):
            with contextlib.redirect_stdout(io.StringIO()):

                try:
                    from mpi4py import MPI
                    if hasattr(module, 'N_PROCS'):
                        N_PROCS = getattr(module, 'N_PROCS')
                        use_mpi = N_PROCS > 1
                    else:
                        use_mpi = False
                except ImportError:
                    use_mpi = False
                if use_mpi:
                    path = str(module.__spec__.origin)
                    args = ['mpirun', '--allow-run-as-root', '-n', str(N_PROCS), 'python', path]
                    env = os.environ.copy()
                    process = subprocess.Popen(args,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               universal_newlines=True,
                                               env=env)
                    stdout, stderr = process.communicate()
                    if process.returncode != 0:
                        raise EnvironmentError("%s\n%s" % (stdout, stderr))
                else:
                    getattr(module, 'main')()

    except Exception as e:
        raise type(e)(str(e) + ' in %s.main' % module.__name__).with_traceback(sys.exc_info()[2])
    finally:
        EasyDriverBase.max_iter = None
        plt.close()


if __name__ == '__main__':
    test_main(get_main_modules()[-1])
