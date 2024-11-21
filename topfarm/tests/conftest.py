import matplotlib.pyplot as plt
import pytest
import matplotlib  # fmt: skip
matplotlib.use("Agg")

# TODO: requires higher openmdao version
# from openmdao.utils.file_utils import clean_outputs
# def cleanup_openmdao_outputs():
#     clean_outputs(obj='.', recurse=True, prompt=False, pattern='*_out')
# def pytest_sessionfinish(session, exitstatus):
#     cleanup_openmdao_outputs()


@pytest.fixture(autouse=True)
def run_around_tests():
    plt.ioff()
    yield
    plt.close("all")
