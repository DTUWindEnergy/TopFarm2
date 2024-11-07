import importlib
import os
import pkgutil
import warnings
import mock
import pytest
import matplotlib.pyplot as plt
import sys
from examples import scripts
import subprocess
import contextlib
import io
from topfarm.easy_drivers import EasyDriverBase
from topfarm.examples.energy_island import EnergyIsland
import numpy as np


def get_main_modules():
    package = scripts
    modules = []
    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if any(x in modname for x in ["Colonel", "deprecated", "mongo"]):
                continue
            m = importlib.import_module(modname)

        if 'main' in dir(m):
            modules.append(m)
    return modules


def test_print_main_modules():
    print("\n".join([m.__name__ for m in get_main_modules()]))


@pytest.mark.parametrize("module", get_main_modules())
def test_main(module):
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


def test_energy_island():
    TFC = EnergyIsland(random_pct=0)
    power_in_each_farm = 100  # MW (1000 MW is nominal)
    TFC.x_target = TFC.x_target[: int(power_in_each_farm / 15)]
    TFC.y_target = TFC.y_target[: int(power_in_each_farm / 15)]

    RPs = np.arange(10, 16).astype(int)
    n_wt_list = (power_in_each_farm / RPs).astype(int)
    wt_types = [5, 2, 4, 3, 1, 3, 2, 2, 2, 2]
    n_wts = n_wt_list[wt_types].tolist()
    seeds_ss = 10 * [0]
    construction_days = [0, 360, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240]
    df = TFC.run(wt_types,
                 n_wts,
                 construction_days,
                 seeds_ss,)
    ref_mean = {'WS': 10.574318325538643,
                'WD': 212.93790009680293,
                'power': 60954503.60691332,
                'power_no_wake': 61194905.73782971,
                'power_no_neighbours': 61271024.08002765,
                'power_no_neighbours_no_wake': 61523195.35628512,
                'total_wake_loss': 1.5929863559949822,
                'internal_wake_loss': 0.6387653340352578,
                'external_wake_loss': 0.9542210219597246}
    np.testing.assert_allclose(df.mean().values, np.array([*ref_mean.values()]), rtol=2.5e-2)


if __name__ == '__main__':
    test_main(get_main_modules()[-1])
