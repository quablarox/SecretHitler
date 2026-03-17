"""Player model for Secret Hitler."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .roles import Role, Party


@dataclass
class Player:
    """Represents a single player in the game."""

    player_id: int
    name: str
    role: Role = field(default=None, repr=False)
    is_alive: bool = True
    investigated: bool = False  # whether this player has been investigated

    @property
    def party(self) -> Optional[Party]:
        return self.role.party if self.role else None

    @property
    def is_liberal(self) -> bool:
        return self.role == Role.LIBERAL

    @property
    def is_fascist(self) -> bool:
        return self.role == Role.FASCIST

    @property
    def is_hitler(self) -> bool:
        return self.role == Role.HITLER

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Player({self.player_id}, {self.name!r})"
