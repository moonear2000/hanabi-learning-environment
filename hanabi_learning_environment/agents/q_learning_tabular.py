# Notes regarding a tabular-Q-learining approach.
# Note this may be infeasable in most scenarios due to the large number of states.
# This aims to solve the very-small hanabi game

# Game parameters:
## 1 colour
## 5 ranks
## 10 cards (000 11 22 33 4)
## 2 players
## Hand size = 2
## 3 max hints
## 1 max life

# State consists off (in very small game):
## Cards observed in other players hand
## Known cards in your hand (each card takes 1 hint)
## Cards in discard pile
## Firework card
## Number of info tokens (3 max)
## Number of life tokens (1 max)