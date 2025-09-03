# env.py
import gymnasium as gym
import numpy as np
import pygame

from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils import EzPickle

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers

try:
    from pettingzoo.utils import AgentSelector
except ImportError:
    from pettingzoo.utils.agent_selector import agent_selector as AgentSelector


from .board import Board


# The maximum number of moves in a game. Used for truncation.
# (Number of cards) * (Max players)
MAX_MOVES = (35 - 3 + 1 - 9) * 7  # 26 * 7 = 182

def env(num_players=3, render_mode=None):
    """
    Creates the No Thanks! environment. (AEC with wrappers)
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    base = raw_env(num_players=num_players, render_mode=internal_render_mode)
    return base


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "no_thanks_v0",
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(self, num_players=3, render_mode=None):
        EzPickle.__init__(self, num_players=num_players, render_mode=render_mode)

        if not 3 <= num_players <= 7:
            raise ValueError("Number of players must be between 3 and 7.")
        self.num_players = num_players
        self.possible_agents = [f"player_{r}" for r in range(num_players)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # --- PettingZoo API attributes ---
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # ---- Spaces (support both method- and attribute-style access) ----
        # Observation length: 4 scalars + N chips + (N * 33) hand bits
        self._obs_len = 4 + self.num_players + (self.num_players * 33)

        self._obs_space = Dict({
            "observation": Box(low=0, high=55, shape=(self._obs_len,), dtype=np.int8),
            "action_mask": Box(low=0, high=1, shape=(2,), dtype=np.int8),
        })
        self._act_space = Discrete(2)

        # Legacy dict attributes for tests that expect them
        self.action_spaces = {agent: self._act_space for agent in self.possible_agents}
        self.observation_spaces = {agent: self._obs_space for agent in self.possible_agents}

        self.render_mode = render_mode
        self._screen = None
        self._clock = None
        self._surface_size = (600, 400)  # (width, height) for rendering

    # ---- New PettingZoo space methods (required by wrappers) ----
    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def observe(self, agent):
        agent_idx = self.agent_name_mapping[agent]

        # 4 scalars (int8 is safe for these magnitudes)
        header = np.array([
            self.board.current_card if self.board.current_card else 0,
            self.board.current_card_chips,
            len(self.board.card_deck),
            self.board.current_player_index,
        ], dtype=np.int8)

        # chips: length N
        player_chips = np.array([self.board.player_chip_counts[i] for i in range(self.num_players)],
                                dtype=np.int8)

        # hands: (N * 33)
        all_hands_vectors_flat = self.board.player_hands.flatten().astype(np.int8)

        obs = np.concatenate([header, player_chips, all_hands_vectors_flat], dtype=np.int8)

        # Action mask
        legal_moves = self._get_legal_moves(agent_idx)
        return {"observation": obs, "action_mask": legal_moves}

    def _get_legal_moves(self, agent_idx):
        can_place_chip = 1 if self.board.player_chip_counts[agent_idx] > 0 else 0
        return np.array([can_place_chip, 1], dtype=np.int8)

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        agent_idx = self.agent_name_mapping[agent]

        if self.board.is_game_over():
            # Mark terminals to be safe if an edge case triggers step after game end
            self.terminations = {a: True for a in self.agents}
            self._accumulate_rewards()
            return

        # Perform action (respect mask)
        if action is None:
            pass
        else:
            a = int(action)
            mask = self._get_legal_moves(agent_idx)

            if a == 0 and mask[0] == 0:
                # User attempted to place a chip but has none -> forced take
                self.board.play_turn(agent_idx, 1)
            elif mask[a] == 1:
                self.board.play_turn(agent_idx, a)
            else:
                # dead/illegal -> no-op; you could penalize slightly if desired
                pass



        # End of game?
        if self.board.is_game_over():
            scores = self.board.calculate_final_scores()  # {player_idx: score}
            min_score = min(scores.values())
            winners = [i for i, s in scores.items() if s == min_score]
            multi = (len(winners) > 1)
            for i in range(self.num_players):
                name = f"player_{i}"
                if i in winners:
                    self.rewards[name] = 0 if multi else 1
                else:
                    self.rewards[name] = -1
            self.terminations = {a: True for a in self.agents}
        else:
            self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        # Keep returning (obs, info) as your tests expect (AEC supports this pattern)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.board = Board(self.num_players, np.random.default_rng(seed))

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = (
            self._agent_selector.reset() if hasattr(self._agent_selector, "reset")
            else self.agents[0]
        )
        first_agent = self.agents[0]

        if getattr(self, "_screen", None) is not None:
            pygame.quit()
        self._screen = None
        self._clock = None

        return self.observe(first_agent), self.infos[first_agent]

    # --- Rendering ---
    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render() without specifying a render mode.")
            return

        if self.render_mode == "human":
            print("\n" + "=" * 40)
            if self.board.is_game_over():
                print("GAME OVER")
                scores = self.board.calculate_final_scores()
                for i in range(self.num_players):
                    print(f"Player {i}: Score {scores[i]}")
            else:
                print(f"Current Card: {self.board.current_card} "
                      f"({self.board.current_card_chips} chips on it)")
                print(f"Cards left in deck: {len(self.board.card_deck)}")
                print("-" * 20)
                for i in range(self.num_players):
                    turn = ">>>" if i == self.board.current_player_index else "   "
                    chips = self.board.player_chip_counts[i]
                    hand = np.flatnonzero(self.board.player_hands[i]) + 3
                    print(f"{turn} Player {i}: {chips} chips, Hand: {list(hand)}")
            print("=" * 40 + "\n")
            return

        if self.render_mode == "rgb_array":
            if self._screen is None:
                pygame.init()
                self._screen = pygame.Surface(self._surface_size)
                self._clock = pygame.time.Clock()

            # Very simple drawing: background + some text
            self._screen.fill((240, 240, 240))
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.SysFont(None, 24)

            def blit_text(text, y):
                surf = font.render(text, True, (20, 20, 20))
                self._screen.blit(surf, (20, y))

            y = 20
            if self.board.is_game_over():
                blit_text("GAME OVER", y); y += 30
                scores = self.board.calculate_final_scores()
                for i in range(self.num_players):
                    blit_text(f"Player {i}: Score {scores[i]}", y); y += 22
            else:
                blit_text(f"Current Card: {self.board.current_card} "
                          f"(chips: {self.board.current_card_chips})", y); y += 22
                blit_text(f"Cards left in deck: {len(self.board.card_deck)}", y); y += 30
                for i in range(self.num_players):
                    prefix = ">> " if i == self.board.current_player_index else "   "
                    chips = self.board.player_chip_counts[i]
                    hand_vals = list(np.flatnonzero(self.board.player_hands[i]) + 3)
                    blit_text(f"{prefix}Player {i}: {chips} chips, Hand: {hand_vals}", y); y += 22

            # Return as (H, W, 3) ndarray
            arr = pygame.surfarray.array3d(self._screen)  # (W, H, 3)
            arr = np.transpose(arr, (1, 0, 2))            # (H, W, 3)
            if self._clock is not None:
                self._clock.tick(self.metadata.get("render_fps", 5))
            return arr

    def close(self):
        if self._screen is not None:
            pygame.quit()
        self._screen = None
        self._clock = None
