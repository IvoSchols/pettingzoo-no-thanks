import sys
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# --- Helpers to import the env regardless of package name quirks (hyphens/underscores) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = PROJECT_ROOT / "pettingzoo-no-thanks"


def _import_env_module():
    """Import the environment module as `no_thanks_env`.
    Tries conventional import first (underscore), then falls back to loading from file.
    """
    # 1) Try a conventional package name with underscores
    try:
        return importlib.import_module("pettingzoo_no_thanks")
    except ModuleNotFoundError:
        pass

    # 2) Try package directory with hyphen and a module file with hyphen
    file_candidates = [
        PKG_DIR / "pettingzoo-no-thanks.py",
        PKG_DIR / "__init__.py",
    ]
    for file in file_candidates:
        if file.exists():
            spec = importlib.util.spec_from_file_location("no_thanks_env", file)
            module = importlib.util.module_from_spec(spec)
            # Ensure package dir is importable for intra-package imports like `from board import Board`
            sys.path.insert(0, str(PKG_DIR))
            assert spec and spec.loader
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        "Could not import the No Thanks! environment module. Make sure the package is installed "
        "or the path `pettingzoo-no-thanks/pettingzoo-no-thanks.py` exists."
    )


@pytest.fixture(scope="module")
def env_module():
    return _import_env_module()


@pytest.fixture(params=[3, 5, 7])
def make_env(env_module, request):
    """Factory returning a freshly constructed environment for a given player count."""
    num_players = request.param
    def _factory(render_mode=None):
        return env_module.env(num_players=num_players, render_mode=render_mode)
    return num_players, _factory


def test_reset_returns_first_obs_and_info(make_env):
    num_players, make = make_env
    e = make(render_mode=None)
    obs, info = e.reset(seed=123)

    # PettingZoo AEC: reset returns observation of first agent and its info
    assert isinstance(obs, dict) and "observation" in obs and "action_mask" in obs
    assert isinstance(info, dict)

    # First agent should be player_0 initially
    assert e.agents[0] == "player_0"
    assert e.board.current_player_index == 0


def test_observation_shape_and_dtype(make_env):
    num_players, make = make_env
    e = make()
    obs, _ = e.reset(seed=42)

    # Expected flattened length: 4 scalars + player_chips(num_players) + all hands (num_players * 33)
    expected_len = 4 + num_players + (num_players * 33)

    assert obs["observation"].ndim == 1
    assert obs["observation"].dtype == np.int8
    assert len(obs["observation"]) == expected_len, (
        "Observation length mismatch. Expected formula 4 + N + 33*N; "
        f"got {len(obs['observation'])} for N={num_players}."
    )

    # Action mask shape/dtype
    assert obs["action_mask"].shape == (2,)
    assert obs["action_mask"].dtype == np.int8


def test_action_mask_matches_chip_possession(make_env):
    num_players, make = make_env
    e = make()
    obs, _ = e.reset(seed=7)

    # At start, each player has chips > 0, so action 0 (place chip) should be legal
    assert obs["action_mask"][0] == 1
    assert obs["action_mask"][1] == 1

    # Force current player to have 0 chips and re-observe
    agent = e.agent_selection
    idx = e.agent_name_mapping[agent]
    e.board.player_chip_counts[idx] = 0
    obs2 = e.observe(agent)
    assert obs2["action_mask"][0] == 0  # cannot place chip
    assert obs2["action_mask"][1] == 1  # can always take


def test_step_advances_turn(make_env):
    num_players, make = make_env
    e = make()
    e.reset(seed=9)

    cur_agent = e.agent_selection
    # Take a legal action (if chip available, place a chip; else take)
    mask = e.observe(cur_agent)["action_mask"]
    action = 0 if mask[0] == 1 else 1
    e.step(action)

    assert e.agent_selection != cur_agent


def test_forced_take_when_no_chips(make_env):
    num_players, make = make_env
    e = make()
    e.reset(seed=101)

    agent = e.agent_selection
    idx = e.agent_name_mapping[agent]

    # Ensure current card exists
    assert e.board.current_card is not None
    current = e.board.current_card

    # Remove all chips from current player and try to place a chip (action 0)
    e.board.player_chip_counts[idx] = 0
    e.step(0)

    # Player should have been forced to take the card
    # Hand is a 33-length binary vector; check the index for `current` is set.
    card_index = current - 3
    assert e.board.player_hands[idx, card_index] == 1


def test_game_eventually_terminates_and_rewards_assigned(make_env):
    num_players, make = make_env
    e = make()
    e.reset(seed=2025)

    # Rollout with a simple policy to finish the game fast
    # If no chips, take; else 50/50 between chip and take
    rng = np.random.default_rng(0)
    safety_cap = 2000
    for _ in range(safety_cap):
        agent = e.agent_selection
        if e.terminations[agent] or e.truncations[agent]:
            e.step(None)  # dead step per PettingZoo API
            continue
        mask = e.observe(agent)["action_mask"]
        if mask[0] == 0:
            action = 1
        else:
            action = int(rng.integers(0, 2))
        e.step(action)
        if all(e.terminations[a] or e.truncations[a] for a in e.agents):
            break

    assert all(e.terminations[a] for a in e.agents), "All agents should be terminated at game over."
    # Rewards should be assigned at the end (win=1 or 0 if draw, loss=-1)
    for a in e.agents:
        assert e.rewards[a] in (-1, 0, 1)


def test_seed_reproducibility(make_env):
    num_players, make = make_env

    e1 = make(); e1.reset(seed=999)
    first_card_1 = e1.board.current_card
    deck_snapshot_1 = list(e1.board.card_deck[:10])  # compare prefix

    e2 = make(); e2.reset(seed=999)
    first_card_2 = e2.board.current_card
    deck_snapshot_2 = list(e2.board.card_deck[:10])

    assert first_card_1 == first_card_2
    assert deck_snapshot_1 == deck_snapshot_2


@pytest.mark.parametrize("mode", [None, "human"])  # exclude "rgb_array" until implemented
def test_render_smoke(make_env, mode):
    num_players, make = make_env
    e = make(render_mode=mode)
    e.reset(seed=1)

    # Call render a few times to ensure no crashes
    for _ in range(3):
        agent = e.agent_selection
        mask = e.observe(agent)["action_mask"]
        action = 0 if mask[0] == 1 else 1
        e.step(action)
        if mode == "human":
            e.render()


def test_spaces_match_observations(make_env):
    num_players, make = make_env
    e = make()
    obs, _ = e.reset(seed=5)

    # Check that the declared spaces match the actual data shapes/dtypes
    agent = e.agent_selection
    space = e.observation_spaces[agent]

    # Observation subspace
    obs_arr = obs["observation"]
    assert space["observation"].shape == obs_arr.shape
    assert space["observation"].dtype == obs_arr.dtype

    # Action mask subspace
    mask_arr = obs["action_mask"]
    assert space["action_mask"].shape == mask_arr.shape
    assert space["action_mask"].dtype == mask_arr.dtype
