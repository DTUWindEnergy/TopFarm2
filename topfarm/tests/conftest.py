import os
import matplotlib.pyplot as plt
import pytest
import matplotlib  # fmt: skip
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def run_around_tests():
    plt.ioff()
    yield
    plt.close("all")


def pytest_generate_tests(metafunc):
    os.environ["OPENMDAO_WORKDIR"] = os.path.join(os.path.dirname(__file__), ".testout")
    os.environ["OPENMDAO_REPORTS"] = "0"
