# pettingzoo-no-thanks
No Thanks! is a turn-based card game for three to seven players with simple rules and a deceiving amount of depth. Players try to minimize their score by paying-off the current card with a chip or taking the card along with the pooled chips. The game ends when the draw pile is empty.

## Rules
### Players
- 3-7 players

### Components
- A deck of cards numbered 3 to 35
- A set of chips, initially:
    - 3-5 players: 11 chips per player
    - 6: 9 chips per player
    - 7: 7 chips per player

### Gameplay
1. Shuffle the deck and blindly remove 9 cards.
2. Flip the top card open and place it in the middle.
3. In turn, choose one of two actions:
    1. Pay a chip: Place one of your chips on the card, your turn ends.
    2. Take a card: Take the middle card and all chips that have accumulated on it, flip a new card. Your turn repeats.

### Game End
The game ends when the last card is taken from the middle (i.e., no draw pile and middle card).

### Scoring

- Each card count adds to your score
    - Crucial exception: If you have a sequence of cards (e.g., 20, 21, 22), you only score the lowest value in that run.
- Each chip you have at the end of the game counts as -1 point.
- The player with the lowest score wins.

