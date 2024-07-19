import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False)


@pytest.fixture
def usegpu(request):
    return request.config.getoption("--gpu")
