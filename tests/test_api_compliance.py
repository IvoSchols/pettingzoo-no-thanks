# tests/test_api_compliance.py
import sys
import importlib
import importlib.util
from pathlib import Path

import pytest
from pettingzoo.test import api_test, seed_test

# --- Helpers to import the env regardless of package name quirks (hyphens/underscores) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = PROJECT_ROOT / "pettingzoo-no-thanks"


def _import_env_module():
    """Import the environment module.
    Tries conventional import first (underscore), then falls back to loading from file.
    """
    # 1) Try a conventional package name with underscores
    try:
        return importlib.import_module("pettingzoo_no_thanks")
    except ModuleNotFoundError:
        pass

    # 2) Fallback to loading from file if not installed
    # Allows testing without having to `pip install .` every time.
    init_file = PKG_DIR / "pettingzoo_no_thanks" / "__init__.py"
    if init_file.exists():
        spec = importlib.util.spec_from_file_location("pettingzoo_no_thanks", init_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["pettingzoo_no_thanks"] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        "Could not import the No Thanks! environment module. Make sure the package is installed."
    )


@pytest.fixture(scope="module")
def env_module():
    """Fixture to provide the imported environment module to tests."""
    return _import_env_module()


@pytest.fixture(params=[3, 5, 7])
def make_env(env_module, request):
    """Factory returning a freshly constructed environment for a given player count."""
    num_players = request.param

    def _factory(render_mode=None):
        return env_module.env(num_players=num_players, render_mode=render_mode)

    return _factory


# --- PettingZoo Standard Tests ---

def test_pettingzoo_api_compliance(make_env):
    """
    Checks the wrapped environment for compliance with the PettingZoo AEC API.
    This test is parameterized by the make_env fixture to run for 3, 5, and 7 players.
    """
    # The 'make_env' fixture provides a factory function to create the environment.
    env_instance = make_env()
    api_test(env_instance, num_cycles=100, verbose=False)


def test_pettingzoo_seed_reproducibility(env_module):
    """
    Checks that the raw environment is reproducible when seeded.
    It requires the `raw_env` constructor, not the wrapped one.
    """
    # The 'env_module' fixture gives us access to the module's contents.
    raw_env_constructor = env_module.raw_env
    seed_test(raw_env_constructor, num_cycles=50)