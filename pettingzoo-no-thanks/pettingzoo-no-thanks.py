import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from gymnasium.utils import EzPickle

from board import Board

# The maximum number of moves in a game. Used for truncation.
MAX_MOVES = (35 - 3 + 1 - 9) * 7  # (Number of cards) * (Max players)

def env(num_players=3, render_mode=None):
    """
    Creates the No Thanks! environment.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(num_players=num_players, render_mode=internal_render_mode)
    # This wrapper is required to allow for parallel use. It is fall-through for AEC environments.
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env



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
        self._agent_selector = agent_selector(self.agents)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # --- Action and Observation Spaces ---
        self.action_spaces = {agent: Discrete(2) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Dict({
                "observation": Box(low=0, high=55, shape=(4 + (2 * self.num_players) + 33,), dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(2,), dtype=np.int8),
            }) for agent in self.possible_agents
        }

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def observe(self, agent):
        agent_idx = self.agent_name_mapping[agent]
        
        # Create observation vector
        player_chips = np.array([self.board.player_chip_counts[i] for i in range(self.num_players)], dtype=np.int8)       
        all_hands_vectors_flat = self.board.player_hands.flatten()

        obs = np.concatenate([
            np.array([
                self.board.current_card if self.board.current_card else 0,
                self.board.current_card_chips,
                len(self.board.card_deck),
                self.board.current_player_index
            ], dtype=np.int8),
            player_chips,
            all_hands_vectors_flat, # Own hand (current_player_index) & Opponent hands
        ])
        
        # Create action mask
        legal_moves = self._get_legal_moves(agent_idx)
        
        return {"observation": obs, "action_mask": legal_moves}
    
    def _get_legal_moves(self, agent_idx):
        # Action 0 (place chip) is only legal if the player has chips
        can_place_chip = self.board.player_chip_counts[agent_idx] > 0
        return np.array([can_place_chip, 1], dtype=np.int8)

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        agent_idx = self.agent_name_mapping[agent]

        # An agent can be selected despite the game being over.
        # This is a weird edge case that can happen in PettingZoo.
        if self.board.is_game_over():
            return

        # Perform action
        self.board.play_turn(agent_idx, action)

        # Update rewards and terminations
        if self.board.is_game_over():
            scores = self.board.calculate_final_scores()
            min_score = min(scores.values())
            winners = [i for i, score in scores.items() if score == min_score]

            for i in range(self.num_players):
                agent_name = f"player_{i}"
                if i in winners:
                    self.rewards[agent_name] = 1 if len(winners) == 1 else 0  # Win or Draw
                else:
                    self.rewards[agent_name] = -1  # Loss
            
            self.terminations = {a: True for a in self.agents}
        else:
            # Select next agent
            self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.board = Board(self.num_players, np.random.default_rng(seed))
        
        self.agent_selection = self._agent_selector.reset()

        return self.observe(self.agents[0]), self.infos[self.agents[0]]

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.render_mode == "human":
            print("\n" + "="*40)
            if self.board.is_game_over():
                print("GAME OVER")
                scores = self.board.calculate_final_scores()
                for i in range(self.num_players):
                    print(f"Player {i}: Score {scores[i]}")
            else:
                print(f"Current Card: {self.board.current_card} ({self.board.current_card_chips} chips on it)")
                print(f"Cards left in deck: {len(self.board.card_deck)}")
                print("-" * 20)
                for i in range(self.num_players):
                    turn_indicator = ">>>" if i == self.board.current_player_index else "   "
                    print(f"{turn_indicator} Player {i}: {self.board.player_chip_counts[i]} chips, Hand: {self.board.player_hands[i]}")
            print("="*40 + "\n")
            return

        # --- RGB Array Rendering ---
        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((800, 600))