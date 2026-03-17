"""Base agent interface for Secret Hitler AI."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from secret_hitler.game import Game
    from secret_hitler.roles import Party


class BaseAgent(ABC):
    """Abstract base class that all AI agents must implement."""

    def __init__(self, player_id: int):
        self.player_id = player_id

    # ------------------------------------------------------------------
    # Methods the game engine calls — all must be implemented.
    # ------------------------------------------------------------------

    @abstractmethod
    def nominate_chancellor(self, game: "Game") -> int:
        """Return the player_id to nominate as Chancellor."""

    @abstractmethod
    def vote(self, game: "Game") -> bool:
        """Return True (Ja) or False (Nein) for the current government."""

    @abstractmethod
    def president_discard(self, game: "Game", tiles: List["Party"]) -> int:
        """Return the index (0-2) of the tile to discard as President."""

    @abstractmethod
    def chancellor_discard(self, game: "Game", tiles: List["Party"]) -> int:
        """Return the index (0-1) of the tile to discard as Chancellor."""

    def request_veto(self, game: "Game", tiles: List["Party"]) -> bool:
        """Return True to request a veto (Chancellor only, optional override)."""
        return False

    def respond_veto(self, game: "Game") -> bool:
        """Return True to accept the Chancellor's veto request (President only)."""
        return False

    @abstractmethod
    def choose_investigate_target(self, game: "Game") -> int:
        """Return the player_id to investigate."""

    @abstractmethod
    def choose_special_election_target(self, game: "Game") -> int:
        """Return the player_id to be the next President."""

    @abstractmethod
    def choose_execution_target(self, game: "Game") -> int:
        """Return the player_id to execute."""

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def on_policy_peek(self, tiles: List["Party"]) -> None:
        """Called after a policy peek with the top-3 tiles."""

    def on_investigate_result(self, target_id: int, party: "Party") -> None:
        """Called with the result of an investigation."""
