"""Rule-based heuristic agent for Secret Hitler."""
from __future__ import annotations

import random
from typing import List, Optional, TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from secret_hitler.game import Game
    from secret_hitler.roles import Party

from secret_hitler.roles import Party as PartyEnum, Role


class RuleBasedAgent(BaseAgent):
    """
    Heuristic agent that uses simple strategy rules:
    - Liberals try to pass liberal policies and vote against suspected fascists.
    - Fascists try to pass fascist policies and protect Hitler.
    """

    def __init__(self, player_id: int, seed: int = None):
        super().__init__(player_id)
        self._rng = random.Random(seed)
        # Track suspicion levels (0.0 = trusted, 1.0 = fully suspected fascist)
        self._suspicion: dict[int, float] = {}
        self._known_party: dict[int, PartyEnum] = {}  # from investigations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _role(self, game: "Game") -> Role:
        return game.players[self.player_id].role

    def _is_fascist_side(self, game: "Game") -> bool:
        return self._role(game) in (Role.FASCIST, Role.HITLER)

    def _fascist_ids(self, game: "Game") -> List[int]:
        """Return known fascist/Hitler player IDs (known to fascist-side agents)."""
        if not self._is_fascist_side(game):
            return []
        return [
            p.player_id for p in game.players
            if p.role in (Role.FASCIST, Role.HITLER)
        ]

    def _liberal_ids(self, game: "Game") -> List[int]:
        if not self._is_fascist_side(game):
            return []
        return [
            p.player_id for p in game.players
            if p.role == Role.LIBERAL
        ]

    def _suspicion_of(self, pid: int) -> float:
        return self._suspicion.get(pid, 0.3)  # mild default suspicion

    def _most_suspicious(self, game: "Game", exclude: List[int] = None) -> int:
        exclude = exclude or []
        candidates = [
            pid for pid in game.alive_ids
            if pid != self.player_id and pid not in exclude
        ]
        return max(candidates, key=self._suspicion_of, default=candidates[0])

    def _least_suspicious(self, game: "Game", exclude: List[int] = None) -> int:
        exclude = exclude or []
        candidates = [
            pid for pid in game.alive_ids
            if pid != self.player_id and pid not in exclude
        ]
        return min(candidates, key=self._suspicion_of, default=candidates[0])

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def nominate_chancellor(self, game: "Game") -> int:
        eligible = game.eligible_chancellors()
        eligible_ids = [p.player_id for p in eligible]

        if self._is_fascist_side(game):
            # Fascists prefer to nominate Hitler (if safe) or fellow fascists
            fascist_ids = self._fascist_ids(game)
            for fid in fascist_ids:
                if fid in eligible_ids:
                    return fid
        else:
            # Liberals prefer least-suspicious players
            trusted = [
                eid for eid in eligible_ids
                if self._known_party.get(eid) == PartyEnum.LIBERAL
            ]
            if trusted:
                return self._rng.choice(trusted)
            # Avoid most-suspicious
            safe = [eid for eid in eligible_ids if self._suspicion_of(eid) < 0.6]
            if safe:
                return self._rng.choice(safe)

        return self._rng.choice(eligible_ids)

    def vote(self, game: "Game") -> bool:
        if self._is_fascist_side(game):
            # Vote Ja if Chancellor is on fascist side
            if game.chancellor_idx in self._fascist_ids(game):
                return True
            # Otherwise vote Ja somewhat randomly (to avoid suspicion)
            return self._rng.random() < 0.6
        else:
            # Vote Ja if Chancellor is trusted
            if self._known_party.get(game.chancellor_idx) == PartyEnum.LIBERAL:
                return True
            if self._known_party.get(game.chancellor_idx) == PartyEnum.FASCIST:
                return False
            # Suspicious chancellor → lean Nein
            suspicion = self._suspicion_of(game.chancellor_idx)
            return self._rng.random() > suspicion

    def president_discard(self, game: "Game", tiles: List["Party"]) -> int:
        if self._is_fascist_side(game):
            # Fascists discard Liberal policies
            for i, t in enumerate(tiles):
                if t == PartyEnum.LIBERAL:
                    return i
            return 0
        else:
            # Liberals discard Fascist policies
            for i, t in enumerate(tiles):
                if t == PartyEnum.FASCIST:
                    return i
            return 0

    def chancellor_discard(self, game: "Game", tiles: List["Party"]) -> int:
        if self._is_fascist_side(game):
            # Fascists enact Fascist policy (discard Liberal)
            for i, t in enumerate(tiles):
                if t == PartyEnum.LIBERAL:
                    return i
            return 0
        else:
            # Liberals enact Liberal policy (discard Fascist)
            for i, t in enumerate(tiles):
                if t == PartyEnum.FASCIST:
                    return i
            return 0

    def request_veto(self, game: "Game", tiles: List["Party"]) -> bool:
        if not game.board.veto_unlocked:
            return False
        if self._is_fascist_side(game):
            # Veto only if both tiles are Liberal (would hurt fascists)
            return all(t == PartyEnum.LIBERAL for t in tiles)
        else:
            # Veto if both tiles are Fascist
            return all(t == PartyEnum.FASCIST for t in tiles)

    def respond_veto(self, game: "Game") -> bool:
        if self._is_fascist_side(game):
            return True   # Fascist president accepts veto (buys time)
        return False      # Liberal president rejects veto (must enact)

    def choose_investigate_target(self, game: "Game") -> int:
        if self._is_fascist_side(game):
            # Investigate a liberal to "confirm" them as fascist (deception)
            liberal_ids = self._liberal_ids(game)
            uninvestigated = [
                pid for pid in liberal_ids
                if pid in game.alive_ids and pid not in game.investigated_players
            ]
            if uninvestigated:
                return self._rng.choice(uninvestigated)
        else:
            # Investigate most-suspicious player
            uninvestigated = [
                pid for pid in game.alive_ids
                if pid != self.player_id and pid not in game.investigated_players
            ]
            if uninvestigated:
                return max(uninvestigated, key=self._suspicion_of)
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        return self._rng.choice(candidates)

    def choose_special_election_target(self, game: "Game") -> int:
        if self._is_fascist_side(game):
            fascist_ids = [
                fid for fid in self._fascist_ids(game)
                if fid in game.alive_ids and fid != self.player_id
            ]
            if fascist_ids:
                return self._rng.choice(fascist_ids)
        else:
            trusted = [
                pid for pid in game.alive_ids
                if pid != self.player_id
                and self._known_party.get(pid) == PartyEnum.LIBERAL
            ]
            if trusted:
                return self._rng.choice(trusted)
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        return self._rng.choice(candidates)

    def choose_execution_target(self, game: "Game") -> int:
        if self._is_fascist_side(game):
            # Execute most-dangerous liberal
            liberal_ids = self._liberal_ids(game)
            alive_liberals = [pid for pid in liberal_ids if pid in game.alive_ids]
            if alive_liberals:
                return self._rng.choice(alive_liberals)
        else:
            # Execute most-suspicious player (suspected Hitler)
            candidates = [pid for pid in game.alive_ids if pid != self.player_id]
            return max(candidates, key=self._suspicion_of)
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        return self._rng.choice(candidates)

    def on_investigate_result(self, target_id: int, party: "Party") -> None:
        self._known_party[target_id] = party
        if party == PartyEnum.FASCIST:
            self._suspicion[target_id] = 1.0

    def on_policy_peek(self, tiles: List["Party"]) -> None:
        pass  # Could be used in more advanced strategies
