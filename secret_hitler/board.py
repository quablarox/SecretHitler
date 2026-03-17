"""Board tracking enacted policies and presidential powers."""
from typing import Optional
from .roles import Party


# Presidential powers unlocked at each fascist policy count per player size
# Values: None = no power, or power name string
PRESIDENTIAL_POWERS = {
    5: {3: "policy_peek", 4: "execution", 5: "execution"},
    6: {3: "policy_peek", 4: "execution", 5: "execution"},
    7: {2: "investigate", 3: "special_election", 4: "execution", 5: "execution"},
    8: {2: "investigate", 3: "special_election", 4: "execution", 5: "execution"},
    9: {1: "investigate", 2: "investigate", 3: "special_election", 4: "execution", 5: "execution"},
    10: {1: "investigate", 2: "investigate", 3: "special_election", 4: "execution", 5: "execution"},
}

LIBERAL_TRACK_SIZE = 5
FASCIST_TRACK_SIZE = 6


class Board:
    """Tracks enacted policies, election tracker, and veto power."""

    def __init__(self, num_players: int):
        if num_players not in range(5, 11):
            raise ValueError(f"Player count must be 5-10, got {num_players}")
        self.num_players = num_players
        self.liberal_policies: int = 0
        self.fascist_policies: int = 0
        self.election_tracker: int = 0  # resets to 0 on successful government
        self._power_map: dict = PRESIDENTIAL_POWERS.get(num_players, {})

    def enact(self, tile: Party) -> Optional[str]:
        """Enact a policy tile.  Returns the presidential power unlocked (or None)."""
        if tile == Party.LIBERAL:
            self.liberal_policies += 1
            return None
        else:
            self.fascist_policies += 1
            return self._power_map.get(self.fascist_policies)

    def reset_election_tracker(self):
        self.election_tracker = 0

    def advance_election_tracker(self) -> bool:
        """Advance tracker by 1. Returns True if it reaches 3 (chaos)."""
        self.election_tracker += 1
        if self.election_tracker >= 3:
            self.election_tracker = 0
            return True
        return False

    @property
    def veto_unlocked(self) -> bool:
        """Veto power is available once 5 fascist policies are enacted."""
        return self.fascist_policies >= 5

    @property
    def liberal_win(self) -> bool:
        return self.liberal_policies >= LIBERAL_TRACK_SIZE

    @property
    def fascist_win(self) -> bool:
        return self.fascist_policies >= FASCIST_TRACK_SIZE

    def __repr__(self) -> str:
        return (
            f"Board(liberal={self.liberal_policies}/{LIBERAL_TRACK_SIZE}, "
            f"fascist={self.fascist_policies}/{FASCIST_TRACK_SIZE}, "
            f"election_tracker={self.election_tracker})"
        )
