# tests/test_api_compliance.py
from pettingzoo.test import api_test, seed_test
import pettingzoo_no_thanks as env_module
import pytest


def test_pettingzoo_api_compliance(make_env):
    """
    Checks the wrapped environment for compliance with the PettingZoo AEC API.
    Runs for 3, 5, and 7 players via the make_env fixture.
    """
    _n, factory = make_env
    env_instance = factory()
    api_test(env_instance, num_cycles=100, verbose=False)


def test_pettingzoo_seed_reproducibility(module):
    """
    Checks that the raw environment is reproducible when seeded.
    Uses the `raw_env` constructor (unwrapped).
    """
    raw_env_constructor = module.raw_env
    seed_test(raw_env_constructor, num_cycles=50)
