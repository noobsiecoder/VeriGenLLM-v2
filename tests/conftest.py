"""
Configuration file for testing using Pytest
Allows CLI flag like argument passing while testing
"""


def pytest_addoption(parser):
    parser.addoption("--bucket", action="store", default="verilog-llm-evals")


import pytest


@pytest.fixture
def bucket(request):
    return request.config.getoption("--bucket")
