"""Core Secret Hitler game state machine."""
from __future__ import annotations

import random
from enum import Enum, auto
from typing import List, Optional, Tuple

from .board import Board
from .deck import PolicyDeck
from .player import Player
from .roles import Party, Role, ROLE_COUNTS


class Phase(Enum):
    """High-level game phases."""
    NOMINATION = auto()      # President nominates a Chancellor
    VOTE = auto()            # All players vote on the proposed government
    LEGISLATIVE_PRESIDENT = auto()  # President discards one of 3 policy tiles
    LEGISLATIVE_CHANCELLOR = auto() # Chancellor discards one of 2 policy tiles
    PRESIDENTIAL_POWER = auto()     # President uses an executive power
    VETO_REQUESTED = auto()         # Chancellor has requested a veto
    GAME_OVER = auto()


class WinReason(Enum):
    LIBERAL_POLICIES = "5 Liberal policies enacted"
    FASCIST_POLICIES = "6 Fascist policies enacted"
    HITLER_SHOT = "Hitler was executed"
    HITLER_ELECTED = "Hitler elected as Chancellor"


class GameResult:
    def __init__(self, winner: Party, reason: WinReason):
        self.winner = winner
        self.reason = reason

    def __repr__(self) -> str:
        return f"GameResult({self.winner.value} wins — {self.reason.value})"


class Game:
    """
    Complete Secret Hitler game engine.

    Usage::

        game = Game(num_players=6, seed=42)
        # Each step returns a (phase, observation) tuple describing what
        # action is needed.  Callers supply answers via the action methods.
    """

    # ------------------------------------------------------------------
    # Construction & setup
    # ------------------------------------------------------------------

    def __init__(self, num_players: int = 6, seed: int = None):
        if num_players not in range(5, 11):
            raise ValueError("Secret Hitler requires 5-10 players.")
        self.num_players = num_players
        self._rng = random.Random(seed)

        self.players: List[Player] = [
            Player(i, f"Player {i + 1}") for i in range(num_players)
        ]
        self.board = Board(num_players)
        self.deck = PolicyDeck(self._rng)

        self.phase: Phase = Phase.NOMINATION
        self.result: Optional[GameResult] = None

        # Election state
        self.president_idx: int = 0          # current presidential candidate index
        self.chancellor_idx: Optional[int] = None
        self.prev_president_idx: Optional[int] = None
        self.prev_chancellor_idx: Optional[int] = None
        self.votes: List[Optional[bool]] = [None] * num_players  # True=Ja, False=Nein

        # Legislative state
        self._drawn_tiles: List[Party] = []
        self._chancellor_tiles: List[Party] = []
        self._veto_rejected: bool = False  # True if president already rejected veto this round

        # Presidential power state
        self.pending_power: Optional[str] = None
        self.investigated_players: List[int] = []  # player ids investigated
        self.executed_players: List[int] = []

        # Assign roles
        roles = list(ROLE_COUNTS[num_players])
        self._rng.shuffle(roles)
        for player, role in zip(self.players, roles):
            player.role = role

    # ------------------------------------------------------------------
    # Public read-only helpers
    # ------------------------------------------------------------------

    @property
    def alive_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive]

    @property
    def alive_ids(self) -> List[int]:
        return [p.player_id for p in self.alive_players]

    @property
    def president(self) -> Player:
        return self.players[self.president_idx]

    @property
    def chancellor(self) -> Optional[Player]:
        if self.chancellor_idx is None:
            return None
        return self.players[self.chancellor_idx]

    def eligible_chancellors(self) -> List[Player]:
        """Return alive players who can be nominated as Chancellor."""
        blocked = set()
        if self.prev_chancellor_idx is not None:
            blocked.add(self.prev_chancellor_idx)
        # If ≥5 alive players, the previous president is also term-limited
        if len(self.alive_players) >= 5 and self.prev_president_idx is not None:
            blocked.add(self.prev_president_idx)
        return [
            p for p in self.alive_players
            if p.player_id != self.president_idx and p.player_id not in blocked
        ]

    def get_observation(self, player_id: int) -> dict:
        """
        Return what *player_id* can observe.  Fascists see each other;
        Hitler sees the Fascist party members in 5-6 player games.
        """
        player = self.players[player_id]
        known: dict[int, str] = {player_id: player.role.value}

        if player.role in (Role.FASCIST, Role.HITLER):
            fascist_ids = [
                p.player_id for p in self.players if p.role == Role.FASCIST
            ]
            hitler_id = next(
                p.player_id for p in self.players if p.role == Role.HITLER
            )
            if player.role == Role.FASCIST:
                for fid in fascist_ids:
                    known[fid] = Role.FASCIST.value
                known[hitler_id] = Role.HITLER.value
            elif player.role == Role.HITLER and self.num_players <= 6:
                for fid in fascist_ids:
                    known[fid] = Role.FASCIST.value

        return {
            "player_id": player_id,
            "phase": self.phase.name,
            "board": {
                "liberal": self.board.liberal_policies,
                "fascist": self.board.fascist_policies,
                "election_tracker": self.board.election_tracker,
            },
            "alive_players": self.alive_ids,
            "president_idx": self.president_idx,
            "chancellor_idx": self.chancellor_idx,
            "known_roles": known,
        }

    # ------------------------------------------------------------------
    # Phase actions — each method advances the phase if valid
    # ------------------------------------------------------------------

    def nominate_chancellor(self, chancellor_idx: int) -> None:
        """President nominates *chancellor_idx* as Chancellor."""
        if self.phase != Phase.NOMINATION:
            raise RuntimeError(f"Cannot nominate in phase {self.phase}")
        eligible_ids = [p.player_id for p in self.eligible_chancellors()]
        if chancellor_idx not in eligible_ids:
            raise ValueError(
                f"Player {chancellor_idx} is not eligible to be Chancellor. "
                f"Eligible: {eligible_ids}"
            )
        self.chancellor_idx = chancellor_idx
        self.votes = [None] * self.num_players
        self.phase = Phase.VOTE

    def cast_vote(self, player_id: int, vote: bool) -> Optional[Phase]:
        """
        Record *player_id*'s vote (True=Ja, False=Nein).
        Returns the new Phase once all alive players have voted; None otherwise.
        """
        if self.phase != Phase.VOTE:
            raise RuntimeError(f"Cannot vote in phase {self.phase}")
        if player_id not in self.alive_ids:
            raise ValueError(f"Player {player_id} is not alive.")
        self.votes[player_id] = vote

        if all(self.votes[pid] is not None for pid in self.alive_ids):
            return self._resolve_vote()
        return None

    def _resolve_vote(self) -> Phase:
        ja = sum(1 for pid in self.alive_ids if self.votes[pid])
        nein = len(self.alive_ids) - ja
        if ja > nein:
            # Government elected — check Hitler win condition
            chancellor = self.players[self.chancellor_idx]
            if chancellor.is_hitler and self.board.fascist_policies >= 3:
                self.phase = Phase.GAME_OVER
                self.result = GameResult(Party.FASCIST, WinReason.HITLER_ELECTED)
                return self.phase
            # Start legislative session
            self.board.reset_election_tracker()
            self._drawn_tiles = self.deck.draw(3)
            self._veto_rejected = False
            self.phase = Phase.LEGISLATIVE_PRESIDENT
        else:
            chaos = self.board.advance_election_tracker()
            if chaos:
                self._enact_chaos_policy()
            else:
                self._advance_presidency()
                self.phase = Phase.NOMINATION
        return self.phase

    def _enact_chaos_policy(self) -> None:
        """Enact top policy when election tracker reaches 3."""
        tile = self.deck.draw(1)[0]
        power = self.board.enact(tile)
        # term limits reset after chaos
        self.prev_president_idx = None
        self.prev_chancellor_idx = None
        self._advance_presidency()
        if self._check_policy_win():
            return
        # No presidential powers on chaos enactment
        self.phase = Phase.NOMINATION

    def president_discard(self, tile_index: int) -> None:
        """President discards one tile (index 0-2) and passes 2 to Chancellor."""
        if self.phase != Phase.LEGISLATIVE_PRESIDENT:
            raise RuntimeError(f"Cannot discard president tile in phase {self.phase}")
        if tile_index not in range(len(self._drawn_tiles)):
            raise ValueError(f"Invalid tile index {tile_index}")
        discarded = self._drawn_tiles.pop(tile_index)
        self.deck.discard(discarded)
        self._chancellor_tiles = self._drawn_tiles
        self._drawn_tiles = []
        self.phase = Phase.LEGISLATIVE_CHANCELLOR

    def chancellor_discard(self, tile_index: int) -> None:
        """Chancellor discards one tile (index 0-1) and enacts the other."""
        if self.phase != Phase.LEGISLATIVE_CHANCELLOR:
            raise RuntimeError(f"Cannot discard chancellor tile in phase {self.phase}")
        if tile_index not in range(len(self._chancellor_tiles)):
            raise ValueError(f"Invalid tile index {tile_index}")
        discarded = self._chancellor_tiles.pop(tile_index)
        self.deck.discard(discarded)
        enacted = self._chancellor_tiles[0]
        self._chancellor_tiles = []
        self._apply_policy(enacted)

    def chancellor_request_veto(self) -> None:
        """Chancellor requests a veto (only if veto power is unlocked and not already rejected)."""
        if self.phase != Phase.LEGISLATIVE_CHANCELLOR:
            raise RuntimeError(f"Cannot request veto in phase {self.phase}")
        if not self.board.veto_unlocked:
            raise RuntimeError("Veto is not unlocked yet.")
        if self._veto_rejected:
            raise RuntimeError("Veto was already rejected this round.")
        self.phase = Phase.VETO_REQUESTED

    def president_respond_veto(self, accept: bool) -> None:
        """President responds to a veto request."""
        if self.phase != Phase.VETO_REQUESTED:
            raise RuntimeError(f"Not in veto phase.")
        if accept:
            # Veto accepted — discard both tiles
            for tile in self._chancellor_tiles:
                self.deck.discard(tile)
            self._chancellor_tiles = []
            self._veto_rejected = False
            chaos = self.board.advance_election_tracker()
            if chaos:
                self._enact_chaos_policy()
            else:
                self._advance_presidency()
                self.phase = Phase.NOMINATION
        else:
            # Veto rejected — Chancellor must enact (cannot request veto again)
            self._veto_rejected = True
            self.phase = Phase.LEGISLATIVE_CHANCELLOR

    def use_investigate(self, target_id: int) -> Party:
        """President investigates target's party membership card."""
        if self.phase != Phase.PRESIDENTIAL_POWER or self.pending_power != "investigate":
            raise RuntimeError("Not in investigate power phase.")
        if target_id == self.president_idx:
            raise ValueError("Cannot investigate yourself.")
        if target_id not in self.alive_ids:
            raise ValueError(f"Player {target_id} is not alive.")
        self.players[target_id].investigated = True
        self.investigated_players.append(target_id)
        result = self.players[target_id].party
        self.pending_power = None
        self._advance_presidency()
        self.phase = Phase.NOMINATION
        return result

    def use_special_election(self, target_id: int) -> None:
        """President picks the next presidential candidate."""
        if self.phase != Phase.PRESIDENTIAL_POWER or self.pending_power != "special_election":
            raise RuntimeError("Not in special election power phase.")
        if target_id == self.president_idx:
            raise ValueError("Cannot choose yourself.")
        if target_id not in self.alive_ids:
            raise ValueError(f"Player {target_id} is not alive.")
        # The chosen player becomes president; after their term, regular order resumes
        self.prev_president_idx = self.president_idx
        self.president_idx = target_id
        self.pending_power = None
        self.phase = Phase.NOMINATION

    def use_policy_peek(self) -> List[Party]:
        """President peeks at the top 3 policy tiles."""
        if self.phase != Phase.PRESIDENTIAL_POWER or self.pending_power != "policy_peek":
            raise RuntimeError("Not in policy peek power phase.")
        tiles = self.deck.peek(3)
        self.pending_power = None
        self._advance_presidency()
        self.phase = Phase.NOMINATION
        return tiles

    def use_execution(self, target_id: int) -> None:
        """President executes a player."""
        if self.phase != Phase.PRESIDENTIAL_POWER or self.pending_power != "execution":
            raise RuntimeError("Not in execution power phase.")
        if target_id == self.president_idx:
            raise ValueError("Cannot execute yourself.")
        if target_id not in self.alive_ids:
            raise ValueError(f"Player {target_id} is not alive.")
        self.players[target_id].is_alive = False
        self.executed_players.append(target_id)
        if self.players[target_id].is_hitler:
            self.phase = Phase.GAME_OVER
            self.result = GameResult(Party.LIBERAL, WinReason.HITLER_SHOT)
            return
        self.pending_power = None
        self._advance_presidency()
        self.phase = Phase.NOMINATION

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_policy(self, tile: Party) -> None:
        power = self.board.enact(tile)
        self.prev_president_idx = self.president_idx
        self.prev_chancellor_idx = self.chancellor_idx
        if self._check_policy_win():
            return
        if power:
            self.pending_power = power
            self.phase = Phase.PRESIDENTIAL_POWER
        else:
            self._advance_presidency()
            self.phase = Phase.NOMINATION

    def _check_policy_win(self) -> bool:
        if self.board.liberal_win:
            self.phase = Phase.GAME_OVER
            self.result = GameResult(Party.LIBERAL, WinReason.LIBERAL_POLICIES)
            return True
        if self.board.fascist_win:
            self.phase = Phase.GAME_OVER
            self.result = GameResult(Party.FASCIST, WinReason.FASCIST_POLICIES)
            return True
        return False

    def _advance_presidency(self) -> None:
        """Move to the next alive player in seat order."""
        idx = self.president_idx
        for _ in range(self.num_players):
            idx = (idx + 1) % self.num_players
            if self.players[idx].is_alive:
                self.president_idx = idx
                return

    def is_over(self) -> bool:
        return self.phase == Phase.GAME_OVER
