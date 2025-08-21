"""
Configuration file for testing using Pytest
Allows CLI flag like argument passing while testing
"""

import pytest

def pytest_addoption(parser):
    parser.addoption("--bucket", action="store", default="verilog-llm-evals")


@pytest.fixture
def bucket(request):
    return request.config.getoption("--bucket")
