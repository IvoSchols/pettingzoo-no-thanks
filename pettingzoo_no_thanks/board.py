# board.py
import numpy as np

class Board:
    """
    Represents the game board for the No Thanks! game.
    Tracks current card, chips on it, and players' hands/chips.
    """

    def __init__(self, num_agents, np_random):
        self.np_random = np_random
        self.num_agents = num_agents

        # Player-dependent setup
        if self.num_agents in (3, 4, 5):
            self.starting_chip_count = 11
        elif self.num_agents == 6:
            self.starting_chip_count = 9
        elif self.num_agents == 7:
            self.starting_chip_count = 7
        else:
            raise ValueError("Invalid player count")

        self.player_chip_counts = [self.starting_chip_count] * self.num_agents

        # Binary hand vectors for cards 3..35 inclusive -> 33 cards
        self.player_hands = np.zeros((self.num_agents, 33), dtype=np.int8)

        # Card deck setup (3..35), remove 9 at random
        self.card_deck = list(range(3, 36))
        self.np_random.shuffle(self.card_deck)
        self.card_deck = self.card_deck[:len(self.card_deck) - 9]

        # Game state
        # Start by drawing the first card from the *front* of the shuffled deck
        self.current_card = self.card_deck.pop(0)
        self.current_card_chips = 0
        self.current_player_index = 0

    def play_turn(self, player_index, action):
        """
        0: Place a chip ("No Thanks!")
        1: Take the card (collect card + chips)
        """
        if action == 0:
            if self.player_chip_counts[player_index] == 0:
                # Forced to take the card if no chips left
                self._take_card(player_index)
            else:
                self.player_chip_counts[player_index] -= 1
                self.current_card_chips += 1
                self.current_player_index = (player_index + 1) % self.num_agents
        elif action == 1:
            self._take_card(player_index)
        else:
            raise ValueError("Invalid action (must be 0 or 1)")

    def _take_card(self, player_index):
        """Handle logic of taking the current card (collect chip pot, mark card, draw next)."""
        self.player_chip_counts[player_index] += self.current_card_chips
        card_index = self.current_card - 3  # map value 3..35 -> index 0..32
        self.player_hands[player_index, card_index] = 1

        # Draw a new card if available
        if self.card_deck:
            self.current_card = self.card_deck.pop(0)
            self.current_card_chips = 0
            self.current_player_index = (player_index + 1) % self.num_agents
        else:
            self.current_card = None  # No more cards to draw

    def is_game_over(self):
        """Ended when there are no cards left to show or draw."""
        return not self.card_deck and self.current_card is None

    def calculate_final_scores(self):
        """
        Score per player (lower is better):
        - Sum the smallest card in each run of consecutive cards owned by the player
        - Subtract remaining chips
        Returns dict: {player_index (int): score (int)}
        """
        scores = {}
        for player_idx in range(self.num_agents):
            score = 0
            player_hand = self.player_hands[player_idx]

            # Values of cards owned
            cards = [i + 3 for i, v in enumerate(player_hand) if v == 1]
            cards.sort()

            if cards:
                # Each disjoint run of consecutive cards contributes its first (smallest) value
                score += cards[0]
                for i in range(1, len(cards)):
                    if cards[i] != cards[i - 1] + 1:
                        score += cards[i]

            # Chips reduce score
            score -= self.player_chip_counts[player_idx]
            scores[player_idx] = score

        return scores
