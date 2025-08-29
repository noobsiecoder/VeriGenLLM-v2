# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption("--rl", action="store", default="PPO")


@pytest.fixture
def rl_alg(request):
    return request.config.getoption("--rl")
