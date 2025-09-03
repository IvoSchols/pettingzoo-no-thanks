# tests/conftest.py
import pytest
import pettingzoo_no_thanks as env_module

@pytest.fixture(scope="module")
def module():
    return env_module

@pytest.fixture(params=[3, 5, 7])
def make_env(module, request):
    n = request.param
    def _factory(render_mode=None):
        return module.env(num_players=n, render_mode=render_mode)
    return n, _factory
