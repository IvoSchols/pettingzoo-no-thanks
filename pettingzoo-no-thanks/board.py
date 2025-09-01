class Board:
    """
    Represents the game board for the No Thanks! game.
    This tracks the state of the game, including the current card, the chips on it, and the players' hands.
    """
    
    def __init__(self, num_agents, np_random):
        """Initialize the game board."""
        self.np_random = np_random
        self.num_agents = num_agents
        
        # Player-dependent setup
        if self.num_agents in [3, 4, 5]:
            self.starting_chip_count = 11
        elif self.num_agents == 6:
            self.starting_chip_count = 9
        elif self.num_agents == 7:
            self.starting_chip_count = 7
        else:
            raise ValueError("Invalid player count")

        self.player_chip_counts = [self.starting_chip_count] * self.num_agents
        # Binary vector representing cards 3-35
        self.player_hands = np.zeros((self.num_agents, 33), dtype=np.int8)
        
        # Card Deck setup (3-35)
        self.card_deck = list(range(3, 36))
        self.np_random.shuffle(self.card_deck)
        self.card_deck = self.card_deck[:len(self.card_deck) - 9]
        
        # Game state
        self.current_card = self.card_deck.pop()
        self.current_card_chips = 0
        self.current_player_index = 0
        
    def play_turn(self, player_index, action):
        """
        Handles the logic for a player's action.
        0: No Thanks! (place a chip)
        1: Take the card
        """
        if action == 0:
            # Place a chip
            if self.player_chip_counts[player_index] == 0:
                # If no chips, the player is forced to take the card
                self._take_card(player_index)
            else:
                self.player_chip_counts[player_index] -= 1
                self.current_card_chips += 1
                self.current_player_index = (player_index + 1) % self.num_agents
        elif action == 1:
            # Take the card
            self._take_card(player_index)
        else:
            raise ValueError("Invalid action")
            
    def _take_card(self, player_index):
        """Helper to handle the logic of taking a card."""
        self.player_chip_counts[player_index] += self.current_card_chips
        card_index = self.current_card - 3 # Convert card value to index for hand vector
        self.player_hands[player_index, card_index] = 1
        
        # Draw a new card if available
        if self.card_deck:
            self.current_card = self.card_deck.pop(0)
            self.current_card_chips = 0
            self.current_player_index = (player_index + 1) % self.num_agents
        else:
            self.current_card = None # No more cards to draw

    def is_game_over(self):
        """Checks if the game has ended."""
        return not self.card_deck and self.current_card is None
        
    def calculate_final_scores(self):
        """Calculates the final score for each player."""
        scores = {}
        for player_idx in range(self.num_agents):
            score = 0
            player_hand = self.player_hands[player_idx]
            
            # Get the list of cards (as values)
            cards = [i + 3 for i, val in enumerate(player_hand) if val == 1]
            cards.sort()
            
            # Calculate the score from cards
            if cards:
                score += cards[0]
                for i in range(1, len(cards)):
                    if cards[i] != cards[i-1] + 1:
                        score += cards[i]
            
            # Subtract the value of remaining chips
            score -= self.player_chip_counts[player_idx]
            scores[f"player_{player_idx}"] = score
            
        return scores