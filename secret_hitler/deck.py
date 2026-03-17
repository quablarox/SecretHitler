"""Policy deck for Secret Hitler."""
import random
from typing import List
from .roles import Party


LIBERAL_POLICIES = 6
FASCIST_POLICIES = 11


class PolicyDeck:
    """A shuffled deck of Liberal and Fascist policy tiles."""

    def __init__(self, rng: random.Random = None):
        self._rng = rng or random.Random()
        self._draw_pile: List[Party] = []
        self._discard_pile: List[Party] = []
        self._reset()

    def _reset(self):
        self._draw_pile = (
            [Party.LIBERAL] * LIBERAL_POLICIES
            + [Party.FASCIST] * FASCIST_POLICIES
        )
        self._rng.shuffle(self._draw_pile)
        self._discard_pile = []

    def draw(self, n: int = 3) -> List[Party]:
        """Draw *n* tiles from the top of the deck, reshuffling if needed."""
        while len(self._draw_pile) < n:
            self._draw_pile += self._discard_pile
            self._discard_pile = []
            self._rng.shuffle(self._draw_pile)
        tiles = self._draw_pile[:n]
        self._draw_pile = self._draw_pile[n:]
        return tiles

    def discard(self, tile: Party):
        """Return a tile to the discard pile."""
        self._discard_pile.append(tile)

    def peek(self, n: int = 3) -> List[Party]:
        """Look at the top *n* tiles without drawing them."""
        while len(self._draw_pile) < n:
            self._draw_pile += self._discard_pile
            self._discard_pile = []
            self._rng.shuffle(self._draw_pile)
        return list(self._draw_pile[:n])

    @property
    def draw_pile_size(self) -> int:
        return len(self._draw_pile)

    @property
    def discard_pile_size(self) -> int:
        return len(self._discard_pile)
