"""Random agent — makes uniformly random valid decisions."""
from __future__ import annotations

import random
from typing import List, TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from secret_hitler.game import Game
    from secret_hitler.roles import Party


class RandomAgent(BaseAgent):
    """Baseline agent that takes uniformly random valid actions."""

    def __init__(self, player_id: int, seed: int = None):
        super().__init__(player_id)
        self._rng = random.Random(seed)

    def nominate_chancellor(self, game: "Game") -> int:
        eligible = game.eligible_chancellors()
        return self._rng.choice(eligible).player_id

    def vote(self, game: "Game") -> bool:
        return self._rng.random() < 0.5

    def president_discard(self, game: "Game", tiles: List["Party"]) -> int:
        return self._rng.randrange(len(tiles))

    def chancellor_discard(self, game: "Game", tiles: List["Party"]) -> int:
        return self._rng.randrange(len(tiles))

    def choose_investigate_target(self, game: "Game") -> int:
        candidates = [
            pid for pid in game.alive_ids
            if pid != self.player_id and pid not in game.investigated_players
        ]
        if not candidates:
            candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        return self._rng.choice(candidates)

    def choose_special_election_target(self, game: "Game") -> int:
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        return self._rng.choice(candidates)

    def choose_execution_target(self, game: "Game") -> int:
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        return self._rng.choice(candidates)
